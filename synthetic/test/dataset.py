import numpy as np

from synthetic.dataset import Dataset
from synthetic.config import Config

class TestDataset:
  def test_load_from_json(self):
    d = Dataset('data1')
    assert(d.num_images() == 4)
    assert(d.classes == ["A","B","C"])

  def test_ground_truth_json(self):
    d = Dataset('data1')
    gt = d.get_ground_truth(include_trun=False)
    arr = np.array(
      [[ 0.,  0.,  0.,  0.,  0.,  0.],
       [ 1.,  1.,  1.,  1.,  1.,  0.],
       [ 2.,  2.,  2.,  2.,  2.,  0.],
       [ 1.,  1.,  1.,  1.,  0.,  1.],
       [ 0.,  0.,  0.,  0.,  1.,  2.],
       [ 0.,  0.,  0.,  0.,  2.,  3.]])
    cols = ['x','y','w','h','cls_ind','img_ind']
    assert(np.all(gt.arr == arr))
    assert(gt.cols == cols)

  def test_get_cls_counts_json(self):
    d = Dataset('data1')
    arr = np.array(
      [ [ 1, 1, 1],
        [ 1, 0, 0],
        [ 0, 1, 0],
        [ 0, 0, 1]])
    print(d.get_cls_counts())
    assert(np.all(d.get_cls_counts() == arr))

  def test_ground_truth_for_class_json(self):
    d = Dataset('data1')
    gt = d.get_ground_truth_for_class("A",include_diff=True,include_trun=True)
    arr = np.array(
      [[ 0.,  0.,  0.,  0.,  0., 0., 0, 0.],
       [ 1.,  1.,  1.,  1.,  0., 0., 0., 1.]])
    cols = ['x','y','w','h','cls_ind','diff','trun','img_ind']
    print(gt.arr)
    assert(np.all(gt.arr == arr))
    assert(gt.cols == cols)

    # no diff or trun
    gt = d.get_ground_truth_for_class("A",include_diff=False,include_trun=False)
    arr = np.array(
      [[ 0.,  0.,  0.,  0.,  0., 0.],
       [ 1.,  1.,  1.,  1.,  0., 1.]])
    cols = ['x','y','w','h','cls_ind','img_ind']
    print(gt.arr)
    assert(np.all(gt.arr == arr))
    assert(gt.cols == cols)
    
  def test_ground_truth_pascal_train(self):
    d = Dataset('test_pascal_train')
    assert(d.num_classes() == 20)
    assert('dog' in d.classes)

  def test_ground_truth_for_class_pascal(self):
    d = Dataset('test_pascal_train')
    correct = np.array(
      [[  48.,  240.,  148.,  132.,    11.,    1., 0.]])
    ans = d.get_ground_truth_for_class("dog").arr
    print ans
    assert np.all(ans == correct)
      
  def test_neg_samples(self):
    d = Dataset('test_pascal_train')

    # unlimited negative examples
    indices = d.get_neg_samples_for_class("dog",include_diff=True,include_trun=True)
    correct = np.array([1,2])
    assert(np.all(indices == correct))

    # maximum 1 negative example
    indices = d.get_neg_samples_for_class("dog",1,include_diff=True,include_trun=True)
    correct1 = np.array([1])
    correct2 = np.array([2])
    print(indices)
    assert(np.all(indices == correct1) or np.all(indices == correct2))

  def test_pos_samples(self):
    d = Dataset('test_pascal_train')
    indices = d.get_pos_samples_for_class("dog")
    correct = np.array([0])
    assert(np.all(indices == correct))
    
  def test_ground_truth_test(self):
    d = Dataset('test_pascal_val')
    gt = d.get_ground_truth(include_trun=False).arr
    correct = np.matrix(
        [ [ 139.,  200.,   69.,  102.,   18.,   0.],
          [ 123.,  155.,   93.,   41.,   17.,   1.],
          [ 239.,  156.,   69.,   50.,    8.,   1.]])
    print(gt)
    assert np.all(gt == correct)

  def test_get_pos_windows(self):
    d = Dataset('test_pascal_val')

