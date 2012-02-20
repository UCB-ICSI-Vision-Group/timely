import numpy as np

import synthetic.util as ut
from synthetic.dataset import Dataset
from synthetic.dataset_policy import DatasetPolicy
from synthetic.evaluation import Evaluation
from synthetic.sliding_windows import SlidingWindows

class TestEvaluationPerfect:
  def __init__(self):
    self.setup()
    
  def setup(self):
    train_dataset = Dataset('test_pascal_train')
    dataset = Dataset('test_pascal_val')
    sw = SlidingWindows(dataset,train_dataset)
    self.dp = DatasetPolicy(dataset,train_dataset,sw, detector='perfect')
    self.evaluation = Evaluation(self.dp)

  def test_compute_pr_multiclass(self):
    cols = ['x','y','w','h','cls_ind','img_ind','diff'] 
    dets_cols = ['x', 'y', 'w', 'h', 'score', 'time', 'cls_ind', 'img_ind']
    
    # two objects of different classes in the image, perfect detection
    arr = np.array(
        [ [0,0,10,10,0,0,0],
          [10,10,10,10,1,0,0] ])
    gt = ut.Table(arr,cols)

    dets_arr = np.array(
        [ [0,0,10,10,-1,-1,0,0],
          [10,10,10,10,-1,-1,1,0] ]) 
    dets = ut.Table(dets_arr,dets_cols)
    
    # make sure gt and gt_cols aren't modified
    gt_arr_copy = gt.arr.copy()
    gt_cols_copy = list(gt.cols)
    ap,rec,prec = self.evaluation.compute_pr(dets, gt)
    assert(np.all(gt.arr == gt_arr_copy))
    assert(gt_cols_copy == gt.cols)

    correct_ap = 1
    correct_rec = np.array([0.5,1])
    correct_prec = np.array([1,1])
    print((ap, rec, prec))
    assert(correct_ap == ap)
    assert(np.all(correct_rec==rec))
    assert(np.all(correct_prec==prec))

    # some extra detections to generate false positives
    dets_arr = np.array(
        [ [0,0,10,10,-1,-1,0,0],
          [0,0,10,10,0,-1,0,0],
          [10,10,10,10,0,-1,1,0],
          [10,10,10,10,-1,-1,1,0] ]) 
    dets = ut.Table(dets_arr,dets_cols)

    ap,rec,prec = self.evaluation.compute_pr(dets, gt)
    correct_rec = np.array([0.5,1,1,1])
    correct_prec = np.array([1,1,2./3,0.5])
    print((ap, rec, prec))
    assert(np.all(correct_rec==rec))
    assert(np.all(correct_prec==prec))

    # confirm that running on the same dets gives the same answer
    ap,rec,prec = self.evaluation.compute_pr(dets, gt)
    correct_rec = np.array([0.5,1,1,1])
    correct_prec = np.array([1,1,2./3,0.5])
    print((ap, rec, prec))
    assert(np.all(correct_rec==rec))
    assert(np.all(correct_prec==prec))

    # now let's add two objects of a different class to gt to lower recall
    arr = np.array(
        [ [0,0,10,10,0,0,0],
          [10,10,10,10,1,0,0],
          [20,20,10,10,2,0,0],
          [30,30,10,10,2,0,0] ])
    gt = ut.Table(arr,cols)
    ap,rec,prec = self.evaluation.compute_pr(dets, gt)
    correct_rec = np.array([0.25,0.5,0.5,0.5])
    correct_prec = np.array([1,1,2./3,0.5])
    print((ap, rec, prec))
    assert(np.all(correct_rec==rec))
    assert(np.all(correct_prec==prec))

    # now call it with empty detections
    dets_arr = np.array([])
    dets = ut.Table(dets_arr,dets_cols)
    ap,rec,prec = self.evaluation.compute_pr(dets, gt)
    correct_ap = 0
    correct_rec = np.array([0])
    correct_prec = np.array([0])
    print((ap, rec, prec))
    assert(np.all(correct_ap==ap))
    assert(np.all(correct_rec==rec))
    assert(np.all(correct_prec==prec))

if __name__=='__main__':
  tester = TestEvaluationPerfect()
  tester.test_compute_pr_multiclass()
