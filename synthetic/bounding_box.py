from common_imports import *

class BoundingBox:
  """
  Methods for constructing location in an image and converting between
  different formats of location (e.g. from image space to feature space).
  The native storage format is np.array([x,y,w,h]).
  """

  def __init__(self, seq=None, format='width'):
    """
    Instantiate from a sequence containing either 
    x,y,w,h (format=='width', default) or
    x1,x2,y1,y2 (format!='width').
    """
    if seq:
      x = float(seq[0])
      y = float(seq[1])
      if format == 'width':
        w = float(seq[2])
        h = float(seq[3])
      else:
        w = float(seq[2]-x+1)
        h = float(seq[3]-y+1)
      self.arr = np.array([x,y,w,h])

  def area(self):
    """Returns area."""
    return self.arr[3]*self.arr[2]

  @classmethod
  def clipboxes_arr(cls, arr, bounds):
    """
    Take an arr in (x,y,w,h) format and clip boxes to fit into bounds,
    provided as (min_x,min_y,max_x,max_y).
    """
    # TODO: do we need to check for boxes that should be totally removed
    # TODO: can make slightly faster by not converting to and from corners
    arr = cls.convert_arr_to_corners(arr)
    arr[arr[:,0]<bounds[0], 0] = bounds[0]
    arr[arr[:,1]<bounds[1], 1] = bounds[1]
    arr[arr[:,2]>bounds[2], 2] = bounds[2]
    arr[arr[:,3]>bounds[3], 3] = bounds[3]
    arr = cls.convert_arr_from_corners(arr)
    return arr

  @classmethod
  def convert_arr_from_corners(cls, arr):
    """Take an arr in x1,y1,x2,y2 format, and return arr in x,y,w,h format."""
    if arr.ndim>1:
      width_arr = np.transpose(np.vstack(
          [arr[:,0], arr[:,1], arr[:,2]-arr[:,0]+1, arr[:,3]-arr[:,1]+1]))
    else:
      width_arr = np.array([arr[0], arr[1], arr[2]-arr[0]+1, arr[3]-arr[1]+1])
    return width_arr

  @classmethod
  def convert_arr_to_corners(cls, arr):
    """Take an arr in x1,y1,x2,y2 format, and return arr in x,y,w,h format."""
    if arr.ndim>1:
      corners_arr = np.transpose(np.vstack(
          [arr[:,0], arr[:,1], arr[:,0]+arr[:,2]-1, arr[:,1]+arr[:,3]-1]))
    else:
      corners_arr = np.array([arr[0], arr[1], arr[0]+arr[2]-1, arr[1]+arr[3]-1])
    return corners_arr

  def get_arr(self,format='width'):
    """
    Returns ndarray representation of 
    [x,y,w,h] (format=='width', default) or
    [x1,y1,x2,y2] (format!='width').
    """
    if format == 'width':
      return self.arr
    else:
      return np.array(self.arr[0],self.arr[1],self.arr[0]+self.arr[2]-1,self.arr[1]+self.arr[3]-1)

  @classmethod
  def get_overlap(cls,bb,bbgt,format='width'):
    """
    Return the PASCAL overlap, defined as the area of intersection
    over the area of union.
    bb can be an (n,4) ndarray.
    bbgt must be an (4,) ndarray or a list.
    """
    # TODO: temp to see if fixes bugs
    if format=='width':
      if bb.ndim>1:
        bb2 = cls.convert_arr_to_corners(bb[:,:4])
      else:
        bb2 = cls.convert_arr_to_corners(bb[:4])
      bbgt2 = cls.convert_arr_to_corners(bbgt[:4])
      return cls.get_overlap_corners_format(bb2,bbgt2)

    if not format=='width':
      return self.get_overlap_corners_format(bb,bbgt)
    bb = bb.T
    x_left = np.minimum(bb[0,],bbgt[0])
    x_right = np.maximum(bb[0,],bbgt[0])
    w_left = np.minimum(bb[2,],bbgt[2])
    y_up = np.minimum(bb[1,],bbgt[1])
    y_down = np.maximum(bb[1,],bbgt[1])
    h_up = np.minimum(bb[3,],bbgt[3])
    iw = w_left+x_left-x_right 
    ih = h_up+y_up-y_down
    ua = bb[2,]*bb[3,] + bbgt[2]*bbgt[3] - iw*ih
    if bb.ndim > 1:
      ov = np.zeros(bb.shape[1])
      mask = np.all((iw>0, ih>0),0)
      ov[mask] = iw[mask]*ih[mask]/ua[mask]
    else:
      ov = 0
      if iw>0 and ih>0:
        ov = iw*ih/ua
    return ov

  @classmethod
  def get_overlap_corners_format(cls,bb,bbgt):
    # once we transpose this from Nx4 to 4xN, we can share the code between the
    # 1-D and 2-D arrays
    bb = bb.T
    bi = np.array([np.maximum(bb[0,],bbgt[0]), np.maximum(bb[1,],bbgt[1]), np.minimum(bb[2,],bbgt[2]), np.minimum(bb[3,],bbgt[3])])
    iw = bi[2,]-bi[0,]+1
    ih = bi[3,]-bi[1,]+1
    ua = (bb[2,]-bb[0,]+1)*(bb[3,]-bb[1,]+1)+(bbgt[2]-bbgt[0]+1)*(bbgt[3]-bbgt[1]+1)-iw*ih
    if bb.ndim > 1:
      ov = np.zeros(bb.shape[1])
      mask = np.all((iw>0, ih>0),0)
      ov[mask] = iw[mask]*ih[mask]/ua[mask]
    else:
      ov = 0
      if iw>0 and ih>0:
        ov = iw*ih/ua
    return ov
     
  @classmethod
  def get_cols(cls):
    return ['x','y','w','h']

  def __repr__(self):
    return "BoundingBox: %s" % self.get_arr()

  def generate_random(self, image_bounds):
    """
    Pick a bounding box from a random distribution of bounding boxes in
    an image of given size.
    """
    max_width = image_bounds[0]
    max_height = image_bounds[1]
    # TODO for now, just returning the same bounding box
    x = 0
    y = 0
    w = max_width
    height = max_height
    # We pick the box by picking a random point in the
    # (max_width, max_height, scale, aspect_ratio) space,
    # constraining it to be within the hypercube ascribed by the
    # image dimensions
    #while True:
    #  x = randi(max_width)
    #  y = randi(max_height)

