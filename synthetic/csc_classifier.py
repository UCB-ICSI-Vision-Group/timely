'''
Created on Nov 20, 2011

@author: Tobias Baumgartner
'''

import numpy as np
from mpi4py import MPI
import itertools
import os

from synthetic.dpm_classifier import Classifier
from synthetic.dataset import Dataset
from synthetic.config import Config

comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()


class CSCClassifier(Classifier):
  def __init__(self, suffix):
    self.name = 'csc'
    self.suffix = suffix
    
  def create_vector(self, feats, cls, img, intervalls, lower, upper):
    if feats.arr.size == 0:
      return np.zeros((1,intervalls+1))
    dpm = feats.subset(['score', 'cls_ind', 'img_ind'])
    img_dpm = dpm.filter_on_column('img_ind', img, omit=True)
    if img_dpm.arr.size == 0:
      print 'empty vector'
      return np.zeros((1,intervalls+1))
    cls_dpm = img_dpm.filter_on_column('cls_ind', cls, omit=True)
    hist = self.compute_histogram(cls_dpm.arr, intervalls, lower, upper)
    vector = np.zeros((1, intervalls+1))
    vector[0,0:-1] = hist
    vector[0,-1] = img_dpm.shape()[0]
    return vector

def csc_classifier_train():
  train_set = 'full_pascal_train'
  train_dataset = Dataset(train_set)  
  suffix = 'half'
  filename = Config.get_ext_dets_filename(train_dataset, 'csc_'+suffix)
  csc_train = np.load(filename)
  csc_train = csc_train[()]  
  csc_train = csc_train.subset(['score', 'cls_ind', 'img_ind'])
  score = csc_train.subset(['score']).arr
  csc_classif = CSCClassifier('half')
  csc_train.arr = csc_classif.normalize_scores(csc_train.arr)
  
  val_set = 'full_pascal_val'
  val_dataset = Dataset(val_set)  
  filename = Config.get_ext_dets_filename(val_dataset, 'csc_'+suffix)
  csc_test = np.load(filename)
  csc_test = csc_test[()]  
  csc_test = csc_test.subset(['score', 'cls_ind', 'img_ind'])
  csc_test.arr = csc_classif.normalize_scores(csc_test.arr) 
  
  lowers = [0.]#,0.2,0.4]
  uppers = [1.,0.8,0.6]
  kernels = ['linear']#, 'rbf']
  intervallss = [10, 20, 50]
  clss = range(20)
  Cs = [1., 1.5, 2., 2.5, 3.]  
  list_of_parameters = [lowers, uppers, kernels, intervallss, clss, Cs]
  product_of_parameters = list(itertools.product(*list_of_parameters))
  
  for params_idx in range(mpi_rank, len(product_of_parameters), mpi_size):
    params = product_of_parameters[params_idx] 
    lower = params[0]
    upper = params[1]
    kernel = params[2]
    intervalls = params[3]
    cls_idx = params[4]
    C = params[5]
    cls = Config.pascal_classes[cls_idx]
    filename = Config.save_dir + csc_classif.name + '_svm_'+csc_classif.suffix+'/'+ kernel + '/' + str(intervalls) + '/'+ \
      cls + '_' + str(lower) + '_' + str(upper) + '_' + str(C)
    
    if not os.path.isfile(filename):
      csc_classif.train_for_all_cls(train_dataset, csc_train,intervalls,kernel, lower, upper, cls_idx, C)
      csc_classif.test_svm(val_dataset, csc_test, intervalls,kernel, lower, upper, cls_idx, C)
  
if __name__=='__main__':
  test_set = 'full_pascal_test'
  for suffix in ['half']:#,'default']:
    test_dataset = Dataset(test_set)  
    filename = Config.get_ext_dets_filename(test_dataset, 'csc_'+suffix)
    csc_test = np.load(filename)
    csc_test = csc_test[()]  
    csc_test = csc_test.subset(['score', 'cls_ind', 'img_ind'])
    score = csc_test.subset(['score']).arr
    csc_classif = CSCClassifier(suffix)
    csc_test.arr = csc_classif.normalize_scores(csc_test.arr)
    
    classes = Config.pascal_classes
    
    best_table = csc_classif.get_best_table()
    
    svm_save_dir = os.path.join(Config.res_dir,csc_classif.name)+ '_svm_'+csc_classif.suffix+'/'
    score_file = os.path.join(svm_save_dir,'test_accuracy.txt')
                      
    for cls_idx in range(mpi_rank, 20, mpi_size):
      row = best_table.filter_on_column('cls_ind', cls_idx).arr
      intervalls = row[0,best_table.cols.index('bins')]
      kernel = Config.kernels[int(row[0,best_table.cols.index('kernel')])]
      lower = row[0,best_table.cols.index('lower')]
      upper = row[0,best_table.cols.index('upper')]
      C = row[0,best_table.cols.index('C')]
      acc = csc_classif.test_svm(test_dataset, csc_test, intervalls,kernel, lower, \
                                 upper, cls_idx, C, file_out=False, local=True)
      print acc
      with open(score_file, 'a') as myfile:
          myfile.write(classes[cls_idx] + ' ' + str(acc) + '\n')
      
    
    
    
    
    
    
    
    

  
  
  
  