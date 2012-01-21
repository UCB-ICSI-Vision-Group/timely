import numpy as np

from synthetic.bounding_box import BoundingBox

def test_convert_to_and_fro():
  bb = np.array([ 139.,  200.,   69.,  102.])
  bb_c = BoundingBox.convert_arr_to_corners(bb)
  bb2 = BoundingBox.convert_arr_from_corners(bb_c)
  assert(np.all(bb == bb2))

def test_get_overlap():
  bbgt =  np.array([ 139.,  200.,   69.,  102.])
  bb =    np.array([ 139.,  200.,   69.,  102.])
  ov = BoundingBox.get_overlap(bb,bbgt)
  print(ov)
  assert(ov == 1)

  bb =    np.array([ 139.,  200.,   69.,  51.])
  ov = BoundingBox.get_overlap(bb,bbgt)
  print(ov)
  assert(ov == 0.5)

  bb =    np.array([ 139.,  200.,  35.,  51.])
  ov = BoundingBox.get_overlap(bb,bbgt)
  print(ov)
  assert((ov >= 0.24) and (ov <= 0.26))

  bb =    np.array([ 239.,  300.,   69.,  51.])
  ov = BoundingBox.get_overlap(bb,bbgt)
  print(ov)
  assert(ov == 0)

def test_get_overlap_with_array():
  bbgt =  np.array([ 139.,  200.,   69.,  102.])
  bb1 =    np.array([ 139.,  200.,   69.,  102.])
  bb2 =    np.array([ 139.,  200.,   69.,  51.])
  bb3 =    np.array([ 239.,  300.,   69.,  51.])
  bb = np.vstack((bb1,bb2,bb3))
  ov = BoundingBox.get_overlap(bb,bbgt)
  print(ov)
  assert(np.all(ov == np.array([1,0.5,0])))

