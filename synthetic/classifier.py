from abc import abstractmethod
import fnmatch

import synthetic.config as config
from common_imports import *
from common_mpi import *
import synthetic.config as config

from synthetic.training import train_svm, svm_predict, save_svm, load_svm,\
  svm_proba
from IPython import embed

class Classifier(object):
  def __init__(self):
    self.name = ''
    self.suffix = ''
    self.cls = ''
    self.tt = ut.TicToc()
    self.bounds = self.load_bounds()
    
  def load_bounds(self):
    filename = config.get_classifier_bounds(self, self.cls)
    if not os.path.exists(filename):
      print 'bounds for %s and class %s dont exist yet'%(self.name, self.cls)
      return None
    bounds = np.loadtxt(filename)
    return bounds
  
  def store_bounds(self, bounds):
    filename = config.get_classifier_bounds(self, self.cls)
    np.savetxt(filename, bounds)
  
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
  
#  @abstractmethod
#  def create_vector(self, img):
#    "Create the feature vector."
#    # implement in subclasses
  
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
        
  def normalize_dpm_scores(self, arr):     
    # TODO from sergeyk: this is silly, this method should take a 1-d array and return the transformed array
    # why are you relying on scores being in a specific column?
    arr[:, 0:1] = np.power(np.exp(-2.*arr[:,0:1])+1,-1)
    return arr
      
  def train_for_cls(self, train_dataset, dets, kernel, cls_idx, C, probab=True):
    cls = train_dataset.classes[cls_idx]
    filename = config.get_classifier_svm_learning_filename(self,cls,kernel,C, self.num_bins)

    pos_imgs = train_dataset.get_pos_samples_for_class(cls)
    neg_imgs = train_dataset.get_neg_samples_for_class(cls, number=len(pos_imgs))
    pos = []
    neg = []    
    #dets.filter_on_column('')
    bounds = ut.importance_sample(dets.subset(['score']).arr, self.num_bins+1)
    self.store_bounds(bounds)
    
    print comm_rank, 'trains', cls
    for img_idx, img in enumerate(pos_imgs):
      vector = self.create_vector_from_dets(dets, img, bounds)
      print 'load image %d/%d on %d'%(img_idx, len(pos_imgs), comm_rank)
      pos.append(vector)
      
    for img_idx, img in enumerate(neg_imgs):
      vector = self.create_vector_from_dets(dets, img, bounds)
      print 'load image %d/%d on %d'%(img_idx, len(pos_imgs), comm_rank)
      neg.append(vector)
              
    pos = np.concatenate(pos)
    neg = np.concatenate(neg)
    
    print '%d trains the model for'%comm_rank, cls
    model = self.train(pos, neg, kernel, C, probab=probab)
   
    save_svm(model, filename)
    
    table_cls = np.zeros((len(train_dataset.images), 1))
    x = np.concatenate((pos, neg))
    prob_t = svm_proba(x, model)
    prob2 =  []
    prob3 = [] 
    for idx in range(x.shape[0]):
      prob2.append(svm_proba(x[idx,:], model))
      if idx >= len(pos_imgs):
        img = neg_imgs[idx-len(pos_imgs)]
      else:
        img = pos_imgs[idx]        
      print 'comp prob3'
      prob3.append(self.classify_image(img, dets))
    prob2 = np.concatenate(prob2)
    
    embed()
    for img_idx, img in enumerate(train_dataset.images):
      score = self.classify_image(img, dets)
      table_cls[img_idx, 0] = score
    return table_cls 
    
  def get_observation(self, image):
    """
    Get the score for given image.
    """
    observation = {}
    self.tt.tic()
    score = self.get_score(image)
    
    observation['score'] = score
    observation['dt'] = self.tt.toc(quiet=True)    
    return observation 
        
  @abstractmethod
  def get_score(self, img): 
    """
    Get the score for the given image.
    """
  
  def load_svm(self):
    svm_file = config.get_classifier_filename(self,self.cls) + '_linear_1.000000_20'
    print svm_file
    if not os.path.exists(svm_file):
      #raise RuntimeWarning("Svm %s is not trained"%svm_file)
      return None
    else:  
      model = load_svm(svm_file)
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
    