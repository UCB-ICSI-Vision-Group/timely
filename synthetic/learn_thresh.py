'''
Created on Nov 21, 2011

@author: Tobias Baumgartner
'''

import numpy as np
from mpi4py import MPI
import os

from synthetic.dataset import Dataset
from synthetic.config import Config
from synthetic.classifier import Classifier

comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()

if __name__=='__main__':
  
  train_set = 'full_pascal_train'
  train_dataset = Dataset(train_set)  
  images = train_dataset.images
  classes = Config.pascal_classes
  suffix = 'default'
  filename = Config.get_ext_dets_filename(train_dataset, 'csc_'+suffix)
  csc_train = np.load(filename)
  csc_train = csc_train[()]  
  csc_train = csc_train.subset(['score', 'cls_ind', 'img_ind'])
  score = csc_train.subset(['score']).arr
  classif = Classifier()
  csc_train.arr = classif.normalize_scores(csc_train.arr)

  numpos = train_dataset.get_ground_truth().shape()[0]
  
  threshs = np.arange(0,1.01,0.05)
  
  result_filename = Config.res_dir + 'thresh_classify.txt'
  
  
  result_file = open(result_filename, 'a')
  threshs = np.array([0.15])
  for thrindex in range(mpi_rank, threshs.shape[0], mpi_size):
    
    for cls in range(len(classes)):
      
      tp = 0.
      tn = 0.
      decisions = 0.
      fdec = 0.
      tdec = 0.      
      thresh = threshs[thrindex]    
      for img in range(len(images)):      
        # for each image check whether it con
        decisions += 1.
        cls_img_scores = csc_train.filter_on_column('img_ind', img, omit=True)
        cls_img_scores = cls_img_scores.filter_on_column('cls_ind', cls, omit=True)
        
        if np.sum(cls_img_scores.arr > thresh) > 0:
          # we have a positive. is it true or false?
          if images[img].contains_cls_ind(cls):
            tp += 1.
            tdec += 1.
            print 'class %s found in %s'%(classes[cls], images[img].name)
          else:
            fdec += 1.
            print 'class %s is not in %s'%(classes[cls], images[img].name)
        else:
          if not images[img].contains_cls_ind(cls):
            tn += 1.
            fdec += 1.
          else:
            tdec += 1.
    # we classified all images
      print tn, tp, decisions
      acc_nor = (tp/tdec*fdec+tn) / (fdec*2)
      acc = (tn + tp)/decisions
      print 'acc:', acc
      print 'acc_nor', acc_nor
      result_file.write('%s: %f\n'%(classes[cls], acc_nor))
      
  result_file.close()    
        
    
    
    
    
    
    
    
    
