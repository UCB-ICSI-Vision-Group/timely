'''
Created on Feb 6, 2012

@author: tobibaum
'''

import numpy as np
from synthetic.mean_shift import MeanShiftCluster

def test_mean_shift():
  X = np.matrix([[-323.5, -402.5,   29.5,   95.5],
  [-327.5, -394.5,   25.5,  103.5],
  [-327.5, -402.5,   25.5,   95.5],
  [-331.5, -394.5,   21.5,  103.5],
  [-331.5, -402.5,  21.5 ,  95.5],
  [-335.5, -402.5,   17.5,   95.5]])
  X_clust = MeanShiftCluster(X,.25)
  X_true = np.matrix([[-329.5, -400 + 1/6., 23.5, 98 + 1/6.]])
  #assert(X_true.any() == X_clust.any())
  np.testing.assert_equal(np.asarray(X_clust), np.asarray(X_true))
  
  
  X2 = np.matrix([[ -327.5000, -378.5000,   25.5000,  119.5000],
  [ -331.5000, -382.5000,   21.5000,  115.5000],
  [ -331.5000, -390.5000,   21.5000,  107.5000],
  [ -327.5000, -478.5000,   25.5000,   19.5000]])
  
  X2_true = np.matrix([[-330 - 1/6. ,-384 + 1/6.,   23 - 1/6.,  114 + 1/6.],
  [-327.5      ,  -478.5  ,        25.5       ,   19.5       ]])
  X2_clust = MeanShiftCluster(X2,.25)
  
  X2_clust = np.sort(X2_clust, 0)
  X2_true = np.sort(X2_true, 0)
  np.testing.assert_equal(np.asarray(X2_clust), np.asarray(X2_true))