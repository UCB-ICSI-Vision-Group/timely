from synthetic.jumping_windows import *


def test_sub2ind():
  assert(sub2ind([5,10], 1, 7) == 36)
  assert(sub2ind([135,13320], 121, 3437) == 464116)
  #assert(sub2ind([4,4],4,4) == 16)

def test_ind2sub():
  assert(ind2sub(201, 3231) == [14,16])
  
def test_sort_cols():
  a = np.matrix('[0.3,0.4;0.2,0.8;0.1,0.5;0.6,0.7]')
  b, I = sort_cols(a)
  I_true = np.matrix('[3,1;0,3;1,2;2,0]')
  b_true = np.matrix('[0.6,0.8;0.3,0.7;0.2,0.5;0.1,0.4]')
  assert(b.all() == b_true.all())  
  assert(I_true.all() == I.all())

def test_line_up_cols():
  a = np.matrix('[1,2,3,4;5,6,7,8;9,0,1,2]')
  b = np.matrix('[1,5,9,2,6,0,3,7,1,4,8,2]').T
  line = line_up_cols(a)
  assert(line.all() == b.all())