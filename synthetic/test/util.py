import itertools
from nose.tools import *

import numpy as np
import synthetic.util as ut
from synthetic.util import histogram

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

def test_row_subset_table():
  arr = np.random.rand(5,3)
  cols = ['a','b','c']
  t = ut.Table(arr,cols) 
  assert(t.shape() == (5,3))
  t2 = t.row_subset([0])
  assert(t2.shape() == (1,3))
  t2 = t.row_subset([0., 2.])
  assert(t2.shape() == (2,3))
  t2 = t.row_subset(np.array([0., 2.]))
  assert(t2.shape() == (2,3))

def test_histogram():
  data = np.random.randint(0,10,(5000,))
  
  print histogram(data, 5)
  

if __name__=='__main__':
  test_histogram()