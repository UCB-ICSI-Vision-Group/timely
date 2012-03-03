from abc import abstractmethod
import fnmatch

import synthetic.config as config
from common_imports import *
from common_mpi import *
import synthetic.config as config
from pylab import *


from synthetic.training import train_svm, svm_predict, save_svm, load_svm,\
  svm_proba
from synthetic.evaluation import Evaluation

class Classifier(object):
  def __init__(self):
    self.name = ''
    self.suffix = ''
    self.cls = ''
    self.tt = ut.TicToc()
          
  def train(self, pos, neg, kernel, C):    
    y = [1]*pos.shape[0] + [-1]*neg.shape[0]
    x = np.concatenate((pos,neg))
    model = train_svm(x, y, kernel, C)
    self.svm = model
    print 'model.score(C=%d): %f'%(C, model.score(x,y))
    table_t = svm_proba(x, model)
    y2 = np.array(y)
    y2 = (y2+1)/2 # switch to 0/1
    ap,_,_ = Evaluation.compute_cls_pr(table_t[:,1], y2)
    print 'ap on train set: %f'%ap
    filename = config.get_classifier_filename(self, self.cls, self.train_dataset)
    self.svm = model
    self.save_svm(model, filename)
    return model
  
  def save_svm(self, model, filename):
    save_svm(model, filename)
              
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
      svm_file = config.get_classifier_filename(self, self.cls, self.train_dataset)
    else:
      svm_file = filename
    print svm_file
    if not os.path.exists(svm_file):
      print("Svm %s is not trained"%svm_file)
      return None
    else:  
      model = load_svm(svm_file)
      return model
      
  def get_best_table(self):
    svm_save_dir = config.get_classifier_learning_dirname(self)
    return ut.Table.load(opjoin(svm_save_dir,'best_table'))
    