from synthetic.common_imports import *

from synthetic.table import Table

def test_row_subset_table():
  arr = np.random.rand(5,3)
  cols = ['a','b','c']

  t = Table(arr,cols)
  assert(t.shape == (5,3))

  t2 = t.row_subset([0])
  assert(t2.shape == (1,3))

  t2 = t.row_subset([0., 2.])
  assert(t2.shape == (2,3))

  t2 = t.row_subset(np.array([0., 2.]))
  assert(t2.shape == (2,3))