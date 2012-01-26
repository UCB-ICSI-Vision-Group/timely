import os
import scipy
import numpy as np

import synthetic.util as ut
import synthetic.config as config
from synthetic.image import BoundingBox
from synthetic.dataset import Dataset
from synthetic.dataset_policy import DatasetPolicy
from synthetic.sliding_windows import SlidingWindows

class TestDatasetPolicy:
  def __init__(self):
    self.dataset = Dataset('test_pascal_val')
    self.train_dataset = Dataset('test_pascal_train')
    self.sw = SlidingWindows(self.dataset,self.train_dataset)

  def test_perfect_detector(self):
    policy = DatasetPolicy(self.dataset,self.train_dataset,self.sw,detector='perfect',bounds=None)
    dets = policy.detect_in_dataset()
    gt = self.dataset.get_ground_truth()
    print dets
    print gt
    assert(dets.arr.shape[0] == gt.arr.shape[0])

  def test_load_dpm_detections(self):
    policy = DatasetPolicy(self.dataset,self.train_dataset,self.sw,detector='perfect')
    dets = policy.load_ext_detections(self.dataset,'dpm','dpm_may25')
    # load the same thing that I computed in Matlab, to check that my nms works
    # the same
    filename = os.path.join(config.test_support_dir, 'val_dets.mat')
    dets_correct = scipy.io.loadmat(filename)['dets']
    cols = ['x1','y1','x2','y2','dummy','dummy','dummy','dummy','score','cls_ind','img_ind']
    good_ind = [0,1,2,3,8,9,10]
    dets_correct = dets_correct[:,good_ind]
    dets_correct[:,0:4] = BoundingBox.convert_arr_from_corners(dets_correct[:,0:4])
    # we need to remove the time column from our loaded dets for this comparison
    subset_cols = list(dets.cols)
    subset_cols.remove('time')
    subset_dets = dets.subset(subset_cols)
    print('----mine:')
    print(subset_dets)
    print(subset_dets.shape())
    print('----correct:')
    print(dets_correct)
    print(dets_correct.shape)
    #ut.keyboard()
    assert(np.all(subset_dets.arr == dets_correct))
    print(dets.cols)
    print(policy.get_cols())
    assert(dets.cols == policy.get_cols())

if __name__ == '__main__':
  tdp = TestDatasetPolicy()
  tdp.test_load_dpm_detections()
