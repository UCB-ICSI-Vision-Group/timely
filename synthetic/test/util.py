from synthetic.common_imports import *
from synthetic.tictoc import TicToc
from synthetic.table import Table
# need this for backwards-compatibility!

from nose.tools import *

def test_random_subset_up_to_N():
  Ns = [1,2,10,100]
  max_nums = [1,50]
  for N,max_num in itertools.product(Ns, max_nums):
    r = ut.random_subset_up_to_N(N, max_num)
    assert(len(r)==min(N,max_num))
    assert(max(r)<=N)
    assert(min(r)>=0)

def test_random_subset_up_to_N_exception():
  Ns = [-2, 0]
  max_nums = [1,50]
  for N,max_num in itertools.product(Ns, max_nums):
    assert_raises(ValueError, ut.random_subset_up_to_N, N, max_num)
  Ns = [1, 10, 100]
  max_nums = [-4, 0]
  for N,max_num in itertools.product(Ns, max_nums):
    assert_raises(ValueError, ut.random_subset_up_to_N, N, max_num)

def test_random_subset():
  l = range(100,120)
  max_num = 10
  r = ut.random_subset(l, max_num)
  assert(len(r)==min(len(l),max_num))
  assert(max(r)<=max(l))
  assert(min(r)>=min(l))

  l = np.array(range(0,120))
  max_num = 10
  r = ut.random_subset(l, max_num)
  assert(len(r)==min(len(l),max_num))
  assert(max(r)<=max(l))
  assert(min(r)>=min(l))

def test_random_subset_ordered():
  l = range(100,120)
  max_num = 10
  r = ut.random_subset(l, max_num, ordered=True)
  assert(len(r)==min(len(l),max_num))
  assert(max(r)<=max(l))
  assert(min(r)>=min(l))
  assert(sorted(r)==r)

def test_determine_bin():
  values = np.array([0, 0.05,0.073,0.0234,0.1,0.13423,0.123534,0.1253,0.212,0.2252,0.43,0.3]).astype(float)
  bounds = np.array([0,0.1,0.2,0.3,np.max(values)])    
  bins = ut.determine_bin(values, bounds, 4)
  bins_gt = np.array([0,0,0,0,1,1,1,1,2,2,3,3])  
  np.testing.assert_equal(bins, bins_gt)

def test_histogram():
  data = np.random.randint(0,10,(5000,)) 
  np.testing.assert_almost_equal(ut.histogram(data, 5), np.tile(1000, (1,5)),-2)  