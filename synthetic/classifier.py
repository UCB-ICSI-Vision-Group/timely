from abc import abstractmethod
import fnmatch

from common_imports import *
from common_mpi import *

from synthetic.training import train_svm, svm_predict, save_svm, load_svm
import synthetic.config as config

class Classifier():
  def __init__(self):
    self.name = ''
    self.suffix = ''
  
  def compute_histogram(self, arr, intervals, lower, upper):
    band = upper - lower
    int_width = band/intervals
    hist = np.zeros((intervals,1))
    # first compute the cumulative  histogram
    for i in range(int(intervals)):
      every = sum(arr < (lower + int_width*(i+1)))
      hist[i] =  every
    # and then uncumulate
    for j in range(int(intervals)-1):
      hist[intervals-j-1] -= hist[intervals-j-2]
    if sum(hist) > 0:
      hist = np.divide(hist, sum(hist)) 
    return np.transpose(hist)
  
  @abstractmethod
  def create_vector(self, feats, cls, img, intervals, lower, upper):
    "Create the feature vector."
    # implement in subclasses
  
  def train(self, pos, neg, kernel, C, probab=True):    
    y = [1]*pos.shape[0] + [-1]*neg.shape[0]
    x = np.concatenate((pos,neg))
    model = train_svm(x, y, kernel=kernel, C=C, probab=probab)
    return model
  
  def evaluate(self, pos, neg, model):
    test_set = np.concatenate((pos,neg))
    test_classification = np.matrix([1]*pos.shape[0] + [-1]*neg.shape[0]).reshape((test_set.shape[0],1))  
    result = svm_predict(test_set, model)
     
    return np.multiply(result,test_classification)
        
  def normalize_scores(self, arr):
    scores = arr[:,0:1]
    for idx in range(scores.shape[0]):
      scores[idx] = 1./(math.exp(-2.*scores[idx]) + 1.)  
    return np.hstack((scores, arr[:,1:3]))
      
  def train_for_all_cls(self, train_dataset, feats, intervals, kernel, lower, upper, cls_idx, C, probab=True):
    cls = train_dataset.classes[cls_idx]
    filename = config.get_classifier_svm_learning_filename(
      self,cls,kernel,intervals,lower,upper,C)

    pos_images = train_dataset.get_pos_samples_for_class(cls)
    pos = []
    neg = []
    print comm_rank, 'trains', cls, intervals, kernel, lower, upper, C
    for img in range(len(train_dataset.images)):
      vector = self.create_vector(feats, train_dataset.classes.index(cls), img, intervals, lower, upper)
      if img in pos_images:
        pos.append(vector)
      else:
        neg.append(vector)
        
    pos = np.concatenate(pos)
    neg = np.concatenate(neg)
    # take as many negatives as there are positives
    neg = np.random.permutation(neg)[:pos.shape[0]]
    model = self.train(pos, neg, kernel, C, probab=probab)
   
    save_svm(model, filename)
    
  def classify_image(self, model, dets, cls, img, intervals, lower, upper): 
    vector = self.create_vector(dets, cls, img, intervals, lower, upper)
    result = svm_predict(vector, model)
    # TODO: score
    ret = 0
    if (result > 0)[0][0]:
      ret = 1
    return ret
  
  def load_svm(self, cls):
    model = load_svm(config.get_classifier_filename(self,cls))
    return model
  
  def test_svm(self, test_dataset, feats, intervals, kernel, lower, upper, \
               cls_idx, C, file_out=True,local=False):
    images = test_dataset.images  
  
    cls = test_dataset.classes[cls_idx]
    pos_images = test_dataset.get_pos_samples_for_class(cls)
    pos = []
    neg = []
    print comm_rank, 'evaluates', cls, intervals, kernel, lower, upper, C
    for img in range(len(images)):
      vector = self.create_vector(feats, test_dataset.classes.index(cls), img, intervals, lower, upper)
      if img in pos_images:
        pos.append(vector)
      else:
        neg.append(vector)
        
    pos = np.concatenate(pos)
    neg = np.concatenate(neg)
    neg = np.random.permutation(neg)
    numpos = pos.shape[0]
    numneg = neg.shape[0]
    
    if local:
      filename = config.get_classifier_svm_name(self,cls)
    else:
      filename = config.get_classifier_svm_learning_filename(
        self,cls,kernel,intervals,lower,upper,C)
    model = load_svm(filename)
    evaluation = self.evaluate(pos, neg, model)
    pos_res = evaluation[:pos.shape[0],:]
    neg_res = evaluation[pos.shape[0]:,:]
    tp   = sum(pos_res > 0)
    fn   = sum(pos_res < 0)
    fp   = sum(neg_res < 0)
    tn   = sum(neg_res > 0)
    prec = tp/float(tp+fp)
    rec  = tp/float(tp+fn)
    eval_file = config.get_classifier_svm_learning_eval_filename(
      self,cls,kernel,intervals,lower,upper,C)
    acc = (tp/float(numpos)*numneg + tn)/float(2*neg.shape[0])
    if file_out:
      with open(eval_file, 'a') as myfile:
        myfile.write(cls + ' ' + str(np.array(prec)[0][0]) + ' ' + str(np.array(rec)[0][0]) + \
                     ' ' + str(np.array(acc)[0][0]) + '\n')
    else:
      return np.array(acc)[0][0]
    
  def get_best_svm_choices(self):  
    classes = config.pascal_classes
    maximas = {}
    for i in range(len(classes)):
      maximas[i] = 0  
    best_settings = {}
    kernels = config.kernels
    
    direct = get_classifier_svm_learning_dirname(self)
    for root, dirs, files in os.walk(direct):
      _,kernel =  os.path.split(root)    
      for direc in dirs:
        for filename in os.listdir(os.path.join(root,direc)):
          file_abs = os.path.join(root,direc,filename)
          if fnmatch.fnmatch(str(filename), 'eval_*'):
            infile = open(file_abs, 'r')
            for line in infile:
              words = line.split() 
              cls = words[0]
              cls_idx = classes.index(cls)
              if words[3] > maximas[cls_idx]:
                maximas[cls_idx] = words[3]
                file_spl = filename.split('_')
                lower = file_spl[1]
                upper = file_spl[2]
                C = file_spl[3]
                best_settings[cls] = [kernels.index(kernel),direc,lower,upper,C,words[3]]
            infile.close()  
    best_arr = np.zeros((20,7))
    cols = ['kernel', 'bins', 'lower', 'upper', 'C', 'score', 'cls_ind']
    for idx in range(len(classes)):
      best_arr[idx,:] = best_settings[classes[idx]] + [idx]
    
    # Store the best svms in results
    svm_save_dir = config.get_classifier_svm_dirname(self)
    score_sum = 0
    score_file = open(opjoin(svm_save_dir,'accuracy.txt'),'w')
    for row in best_arr:
      cls = classes[int(row[6])]
      svm_name = cls + '_' + str(row[2]) + '_' + \
        str(row[3]) + '_' + str(row[4])
      os.system('cp ' + config.data_dir + self.name+ '_svm_'+self.suffix+'/' + str(kernels[int(row[0])]) +\
                '/' + str(int(row[1])) + '/' + svm_name + ' ' + svm_save_dir + cls)
      score = row[5]
      score_sum += score
      score_file.writelines(cls + '\t\t\t' + str(score) + '\n')
      print svm_name
    score_file.writelines('mean' + '\t\t\t' + str(score_sum/20.) + '\n')
    
    best_table = ut.Table(best_arr, cols)
    best_table.name = 'Best_'+self.name+'_values'
    best_table.save(opjoin(svm_save_dir,'best_table'))
    print best_table
    
  def get_best_table(self):
    svm_save_dir = config.get_classifier_learning_dirname(self)
    return ut.Table.load(opjoin(svm_save_dir,'best_table'))
    