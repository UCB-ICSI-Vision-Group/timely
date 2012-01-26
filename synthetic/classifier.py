'''
Created on Nov 20, 2011

@author: Tobias Baumgartner
'''

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
  
  def compute_histogram(self,arr, intervalls, lower, upper):
    band = upper - lower
    int_width = band/intervalls
    hist = np.zeros((intervalls,1))
    # first compute the cumulated  histogram
    for i in range(int(intervalls)):
      every = sum(arr < (lower + int_width*(i+1)))
      hist[i] =  every
    # and then uncumulate
    for j in range(int(intervalls)-1):
      hist[intervalls-j-1] -= hist[intervalls-j-2]
    if sum(hist) > 0:
      hist = np.divide(hist, sum(hist)) 
    return np.transpose(hist)
  
  @abstractmethod
  def create_vector(self, feats, cls, img, intervalls, lower, upper):
    """
    Changes in classifiers. Create the feature vector that is going to be classified 
    """
  
  def train(self, pos, neg, kernel, C):    
    y = [1]*pos.shape[0] + [-1]*neg.shape[0]
    x = np.concatenate((pos,neg))
    model = train_svm(x, y, kernel=kernel, C=C)
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
      
  def train_for_all_cls(self, train_dataset, feats, intervalls, kernel, lower, upper, cls_idx, C):
    cls = train_dataset.classes[cls_idx]
    ut.makedirs(config.save_dir + self.name + '_svm_'+self.suffix+'/' + kernel + '/' + str(intervalls))
    filename = config.save_dir + self.name + '_svm_'+self.suffix+'/'+ kernel + '/' + str(intervalls) + '/'+ \
      cls + '_' + str(lower) + '_' + str(upper) + '_' + str(C)
    
    images = train_dataset.images
    pos_images = train_dataset.get_pos_samples_for_class(cls)
    pos = []
    neg = []
    print comm_rank, 'trains', cls, intervalls, kernel, lower, upper, C
    for img in range(len(images)):
      vector = self.create_vector(feats, train_dataset.classes.index(cls), img, intervalls, lower, upper)
      if img in pos_images:
        pos.append(vector)
      else:
        neg.append(vector)
        
    pos = np.concatenate(pos)
    neg = np.concatenate(neg)
    neg = np.random.permutation(neg)[:pos.shape[0]]
    model = self.train(pos, neg, kernel, C)
   
    save_svm(model, filename)
    
  def classify_image(self, model, dets, cls, img, intervalls, lower, upper): 
    vector = self.create_vector(dets, cls, img, intervalls, lower, upper)
    result = svm_predict(vector, model)
    ret = 0
    if (result > 0)[0][0]:
      ret = 1
    return ret
  
  def load_svm(self, cls):
    filename = config.res_dir + self.name + '_svm_'+self.suffix+'/' + cls
    model = load_svm(filename)
    return model
  
  def test_svm(self, test_dataset, feats, intervalls, kernel, lower, upper, \
               cls_idx, C, file_out=True,local=False):
    images = test_dataset.images  
  
    cls = test_dataset.classes[cls_idx]
    pos_images = test_dataset.get_pos_samples_for_class(cls)
    pos = []
    neg = []
    print comm_rank, 'evaluates', cls, intervalls, kernel, lower, upper, C
    for img in range(len(images)):
      vector = self.create_vector(feats, test_dataset.classes.index(cls), img, intervalls, lower, upper)
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
      filename = config.res_dir + self.name + '_svm_'+self.suffix +'/' +cls
    else:
      filename = config.save_dir + self.name + '_svm_'+self.suffix+'/' + kernel + '/' + str(intervalls) + '/'+ \
        cls + '_' + str(lower) + '_' + str(upper) + '_' + str(C)
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
    eval_file = config.save_dir + self.name + '_svm_'+self.suffix+'/' + kernel + '/' + str(intervalls) + \
      '/'+ 'eval_' + str(lower) + '_' + str(upper) + '_' + str(C)
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
    
    direct = os.path.join(config.save_dir + self.name + '_svm_'+self.suffix+'/')
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
    
    # Store the best svms in resuslts
    svm_save_dir = os.path.join(config.res_dir,self.name)+ '_svm_'+self.suffix+'/'
    os.system('rm ' + svm_save_dir + '*') 
    ut.makedirs(svm_save_dir)
    score_sum = 0
    score_file = open(os.path.join(svm_save_dir,'accuracy.txt'),'a')
    for row in best_arr:
      cls = classes[int(row[6])]
      svm_name = cls + '_' + str(row[2]) + '_' + \
        str(row[3]) + '_' + str(row[4])
      os.system('cp ' + config.save_dir + self.name+ '_svm_'+self.suffix+'/' + str(kernels[int(row[0])]) +\
                '/' + str(int(row[1])) + '/' + svm_name + ' ' + svm_save_dir + cls)
      score = row[5]
      score_sum += score
      score_file.writelines(cls + '\t\t\t' + str(score) + '\n')
      print svm_name
    score_file.writelines('mean' + '\t\t\t' + str(score_sum/20.) + '\n')
    
    
    best_table = ut.Table(best_arr, cols)
    best_table.name = 'Best_'+self.name+'_values'
    save_best_table_name = os.path.join(config.res_dir,'%s_svm_%s'%(self.name,self.suffix),'best_table') 
    best_table.save(save_best_table_name)
    print best_table
    
  def get_best_table(self):
    best_table_name = os.path.join(config.res_dir,'%s_svm_%s'%(self.name,self.suffix),'best_table') 
    return ut.Table.load(best_table_name)
    
    
    