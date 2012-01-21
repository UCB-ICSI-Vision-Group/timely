'''
Created on Nov 21, 2011

@author: Tobias Baumgartner
'''

import numpy as np
from synthetic.dataset import Dataset

def convert_thresh(cls, tp, tn):
  d=Dataset('full_pascal_train')
  gt = d.get_ground_truth_for_class(cls)
 
  imgs = gt.arr[:,gt.cols.index('img_ind')]
  imgs = np.unique(imgs)
  num_pos = float(imgs.shape[0])
  print num_pos
  num_neg = 2501. - num_pos
  result = (tp/num_pos*num_neg + tn)/(2.*num_neg)
  print result
  return result  
