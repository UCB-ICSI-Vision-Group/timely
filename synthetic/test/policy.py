import scipy

from synthetic.common_mpi import *
from synthetic.common_imports import *
import synthetic.config as config

from synthetic.image import BoundingBox
from synthetic.dataset import Dataset
from synthetic.dataset_policy import DatasetPolicy
from synthetic.sliding_windows import SlidingWindows

class TestDatasetPolicy:
  def __init__(self):
    self.dataset = Dataset('test_pascal_val')
    self.train_dataset = Dataset('test_pascal_train')

  def test_perfect_detector(self):
    policy = DatasetPolicy(self.dataset,self.train_dataset,detector='perfect',bounds=None)
    dets,clses = policy.run_on_dataset()
    dets = dets.subset(['x', 'y', 'w', 'h', 'cls_ind', 'img_ind'])
    gt = self.dataset.get_ground_truth()
    gt = gt.subset(['x', 'y', 'w', 'h', 'cls_ind', 'img_ind'])
    dets.arr = ut.sort_by_column(dets.arr, 0)
    gt.arr = ut.sort_by_column(gt.arr, 0)
    assert(dets == gt)

  def test_load_dpm_detections(self):
    policy = DatasetPolicy(self.dataset,self.train_dataset,detector='ext')
    dets = policy.load_ext_detections(self.dataset,'dpm_may25',force=True)
    dets = dets.with_column_omitted('time')

    # load the ground truth dets, processed in Matlab
    # (timely/data/test_support/concat_dets.m)
    filename = os.path.join(config.test_support_dir, 'val_dets.mat')
    dets_correct = ut.Table(
        scipy.io.loadmat(filename)['dets'],
        ['x1','y1','x2','y2','dummy','dummy','dummy','dummy','score','cls_ind','img_ind'],
        'dets_correct')
    dets_correct = dets_correct.subset(
        ['x1','y1','x2','y2','score','cls_ind','img_ind'])
    dets_correct.arr[:,:4] -= 1
    dets_correct.arr[:,:4] = BoundingBox.convert_arr_from_corners(
        dets_correct.arr[:,:4])
    dets_correct.cols = ['x','y','w','h','score','cls_ind','img_ind']
    
    print('----mine:')
    print(dets)
    print('----correct:')
    print(dets_correct)
    assert(dets_correct == dets)

if __name__ == '__main__':
  tdp = TestDatasetPolicy()
  tdp.test_load_dpm_detections()
