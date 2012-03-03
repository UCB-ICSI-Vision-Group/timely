from abc import abstractmethod
import fnmatch

import synthetic.config as config
from common_imports import *
from common_mpi import *
import synthetic.config as config
from pylab import *

from sklearn.svm import SVC, LinearSVC

from synthetic.training import train_svm, svm_predict, save_svm, load_svm,\
  svm_proba
from synthetic.evaluation import Evaluation

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
      
  def train_for_cls(self, train_dataset, val_dataset, dets, test_det_cls, kernel, C, probab=True, vtype='hist'):
    cls = self.cls
    filename = config.get_classifier_filename(self,cls)

    pos_imgs = train_dataset.get_pos_samples_for_class(cls)
    neg_imgs = train_dataset.get_neg_samples_for_class(cls)#, number=len(pos_imgs))
    pos = []
    neg = []    
      
    dets_arr = dets.subset(['score']).arr
    dets_arr = self.normalize_dpm_scores(dets_arr)
    #bounds = ut.importance_sample(dets_arr, self.num_bins+1)
    bounds = np.linspace(np.min(dets_arr), np.max(dets_arr), self.num_bins+1)
    self.bounds = bounds
    self.store_bounds(bounds)
    
    print comm_rank, 'trains', cls
    pos_det_scores = []
    for idx, img_idx in enumerate(pos_imgs):
      image = train_dataset.images[img_idx]
      vector = self.create_vector_from_dets(train_dataset, dets, image, vtype, bounds,norm=True)
      scores = dets.filter_on_column('img_ind',img_idx).subset_arr('score')
      #scores = np.power(np.exp(-2.*scores)+1,-1)
      pos_det_scores.append(scores)
      print 'load image %d/%d on %d'%(idx, len(pos_imgs), comm_rank)
      pos.append(vector)      
    pos_det_scores = np.concatenate(pos_det_scores)

    neg_det_scores = []
    for idx, img_idx in enumerate(neg_imgs):
      image = train_dataset.images[img_idx]
      vector = self.create_vector_from_dets(train_dataset, dets, image, vtype, bounds,norm=True)
      scores = dets.filter_on_column('img_ind',img_idx).subset_arr('score')
      #scores = np.power(np.exp(-2.*scores)+1,-1)
      neg_det_scores.append(scores)
      print 'load image %d/%d on %d'%(idx, len(neg_imgs), comm_rank)
      neg.append(vector)
    neg_det_scores = np.concatenate(neg_det_scores)
    
    pos = np.concatenate(pos)
    neg = np.concatenate(neg)
    
    print '%d trains the model for'%comm_rank, cls
    x = np.concatenate((pos, neg))
    y = [1]*pos.shape[0] + [-1]*neg.shape[0] 

    model = SVC(kernel='linear', C=C, probability=True)
    model.fit(x, y)#, class_weight='auto')
    print("model.score(C=%d)"%C)
    print model.score(x,y)
    
    table_t = svm_proba(x, model)
    
    y2 = np.array(y)    
    y2 = (y2+1)/2    
    ap,_,_ = Evaluation.compute_cls_pr(table_t[:,1], y2)
    print ap
       
    save_svm(model, filename)
  
    # Classify on val set
    self.svm = model
    print 'evaluate svm'
    table_cls = np.zeros((len(val_dataset.images), 1))
    for idx, image in enumerate(val_dataset.images):
      print '%d eval on img %d/%d'%(comm_rank, idx, len(val_dataset.images))
      score = self.classify_image(val_dataset, image, test_det_cls, probab=probab, vtype=vtype,norm=True)
      table_cls[idx, 0] = score
        
    ap2, _,_ = Evaluation.compute_cls_pr(table_cls, val_dataset.get_cls_ground_truth().subset_arr(cls))
    print 'ap on val for %s: %f'%(self.cls, ap2)

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
        
  def load_svm(self, filename=None):
    if not filename:
      svm_file = config.get_classifier_filename(self,self.cls)
    else:
      svm_file = filename
    print svm_file
    if not os.path.exists(svm_file):
      #raise RuntimeWarning("Svm %s is not trained"%svm_file)
      return None
    else:  
      model = load_svm(svm_file)
      return model
      
  def get_best_table(self):
    svm_save_dir = config.get_classifier_learning_dirname(self)
    return ut.Table.load(opjoin(svm_save_dir,'best_table'))
    