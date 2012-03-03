import scipy

from synthetic.common_mpi import *
from synthetic.common_imports import *
import synthetic.config as config

from synthetic.image import BoundingBox
from synthetic.dataset import Dataset
from synthetic.belief_state import BeliefState
from synthetic.dataset_policy import DatasetPolicy
from synthetic.sliding_windows import SlidingWindows

class TestDatasetPolicy:
  def __init__(self):
    self.dataset = Dataset('test_pascal_val')
    self.train_dataset = Dataset('test_pascal_train')
    self.weights_dataset_name = 'test_pascal_val'
    self.config = {
      'suffix': 'default',
      'detectors': ['perfect'], # perfect,perfect_with_noise,dpm,csc_default,csc_half
      'policy_mode': 'random',
      'bounds': None,
      'weights_mode': 'manual_1' # manual_1, manual_2, manual_3, greedy, rl
    }
    self.dp = DatasetPolicy(
      self.dataset,self.train_dataset,self.weights_dataset_name,**self.config)

  def test_run_on_dataset(self):
    # run on test dataset
    dets,clses,samples = self.dp.run_on_dataset(force=True) 
    assert(len(samples) == clses.shape()[0])
    assert(len(samples) == self.dp.dataset.num_images()*len(self.dp.actions))
    train_dets,train_clses,train_samples = self.dp.run_on_dataset(train=True,force=True)
    assert(len(train_samples) == train_clses.shape()[0])
    assert(len(train_samples) == self.dp.train_dataset.num_images()*len(self.dp.actions))

  def test_unique_samples(self):
    "Test the correctness of making a list of samples unique."
    dets,clses,samples = self.dp.run_on_dataset()
    new_sample = copy.deepcopy(samples[11])
    new_sample2 = copy.deepcopy(samples[11])
    new_sample2.dt = -40 # an unreasonable value
    assert(new_sample in samples)
    assert(new_sample2 not in samples)

  def test_output_det_statistics(self):
    self.dp.output_det_statistics()

  def test_dp_weights(self):
    modes = ['manual_1','manual_2','manual_3']
    for mode in modes:
      print "%s weights:"%mode
      self.dp.weights_mode=mode
      self.dp.load_weights()
      print self.dp.weights
      assert(self.dp.weights.shape[0] == len(self.dp.actions)*BeliefState.num_features)

  def test_perfect_detector(self):
    dets,clses,samples = self.dp.run_on_dataset()
    dets = dets.subset(['x', 'y', 'w', 'h', 'cls_ind', 'img_ind'])
    gt = self.dataset.get_ground_truth()
    gt = gt.subset(['x', 'y', 'w', 'h', 'cls_ind', 'img_ind'])
    dets.arr = ut.sort_by_column(dets.arr, 0)
    gt.arr = ut.sort_by_column(gt.arr, 0)
    assert(dets == gt)

  def test_load_dpm_detections(self):
    conf = dict(self.config)
    conf['detectors'] = ['dpm']
    policy = DatasetPolicy(self.dataset,self.train_dataset,**conf)
    assert(policy.detectors == ['dpm'])
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
  #tdp.test_run_on_dataset()
  #tdp.test_unique_samples()
  tdp.test_dp_weights()
  #tdp.test_output_det_statistics()
