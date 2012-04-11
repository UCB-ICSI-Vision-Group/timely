from xml.dom.minidom import parseString

from common_imports import *

from synthetic.bounding_box import BoundingBox

class Image:
  """An image has a size and a list of objects."""
  
  def __init__(self,name=None,size=None,objects=None,dataset=None,synthetic=False):
    self.name = name          # just a string identifier 
    self.size = size          # (width,height)
    if not objects:
      self.objects = []
    self.dataset = dataset
    self.synthetic = synthetic
    if synthetic:
      self.gen_cls_ground_truth()

  def gen_cls_ground_truth(self):
    "Hard-coded right now"
    choices = [[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
    probs = np.array([0,8,3,1,6,1,8,3])
    cum_probs = np.cumsum(1.*probs/np.sum(probs))
    choice = np.where(cum_probs>np.random.rand())[0][0]
    # to check that this is right (it is):
    # hist(choices,bins=arange(0,9),normed=True,align='left'); plot(1.*probs/sum(probs))
    self.cls_ground_truth = np.array(choices[choice])

  def get_whole_image_bbox(self):
    """Returns a BoundingBox with (0,0,width,height) of the image."""
    return BoundingBox((0,0,self.size[0],self.size[1]))
      
  @classmethod
  def from_json(cls, dataset, data):
    """Instantiate an Image from a JSON representation."""
    img = Image(size=data['size'])
    img.dataset = dataset
    for obj in data['objects']:
      bbox = BoundingBox(obj['bbox'])
      cls = obj['class']
      cls_ind = dataset.get_ind(cls)
      diff = 0 # just dummy variable, as this data is not in JSON
      trun = 0 # same
      img.objects.append(Object(bbox, cls_ind, diff, trun))
    return img
  
  @classmethod
  def get_data_from_tag(cls, node, tag):
    """Read the entries for a specific tag"""
    if tag is "bndbox":
      x1 = int(node.getElementsByTagName(tag)[0].childNodes[1].childNodes[0].data)
      y1 = int(node.getElementsByTagName(tag)[0].childNodes[3].childNodes[0].data)
      x2 = int(node.getElementsByTagName(tag)[0].childNodes[5].childNodes[0].data)
      y2 = int(node.getElementsByTagName(tag)[0].childNodes[7].childNodes[0].data)
      return (x1, y1, x2, y2)
    else:
      return node.getElementsByTagName(tag)[0].childNodes[0].data

  @classmethod
  def load_from_xml(cls, dataset, filename):
    with open(filename) as f:
      xml = f.read()
    data = parseString(xml)
    name = cls.get_data_from_tag(data, "filename")
    size = data.getElementsByTagName("size")[0]
    im_width = int(cls.get_data_from_tag(size, "width"))
    im_height = int(cls.get_data_from_tag(size, "height"))
    im_depth = int(cls.get_data_from_tag(size, "depth"))
    size = (im_width, im_height)
    img = Image(name,size)
    img.dataset = dataset

    # parse objects
    objs = data.getElementsByTagName("object")
    for obj in objs:
      categ = str(cls.get_data_from_tag(obj, "name")).lower().strip()
      diff = int(cls.get_data_from_tag(obj, "difficult"))
      trun = int(cls.get_data_from_tag(obj, "truncated"))
      rect = cls.get_data_from_tag(obj, "bndbox")
      bbox = BoundingBox(rect, format='corners')
      cls_ind = dataset.get_ind(categ)
      img.objects.append(Object(bbox,cls_ind,diff,trun))
    return img

  def __repr__(self):
    return "Image.name: %s, Image.size: %s\nImage.objects: %s" % (self.name, self.size, self.objects)

  def contains_cls_ind(self,cls_ind):
    if self.synthetic:
      return self.cls_ground_truth[cls_ind]==1
    # TODO: could be made faster, if the whole class was immutable and the
    # array ground truth representation was stored 
    return (len([obj for obj in self.objects if obj.cls_ind == cls_ind]) > 0)

  def get_cls_counts(self, include_diff=False, include_trun=True):
    """
    Return a vector of size num_classes, with the counts of each class in
    the image.
    """
    if self.synthetic:
      return self.cls_ground_truth
    cls_inds = [obj.cls_ind for obj in self.objects]
    bincount = np.bincount(cls_inds)
    # need to pad this with zeros for total length of num_classes
    counts = np.zeros(self.dataset.num_classes())
    counts[:bincount.size] = bincount
    return counts 

  def get_cls_ground_truth(self,include_diff=False,include_trun=False):
    # TODO: hack
    if self.synthetic:
      return self.cls_ground_truth
    counts = self.get_cls_counts(include_diff,include_trun)
    z = np.zeros(counts.shape)
    z[counts>0] = 1
    return z

  def get_ground_truth(self, cls=None, include_diff=False, include_trun=True):
    """
    Return Table of ground truth.
    If cls is given, only return objects of that class.
    """
    gt = ut.Table(arr=self.get_gt_arr(),cols=self.get_gt_cols())
    if not include_diff:
      gt = gt.filter_on_column('diff',0,omit=True)
    if not include_trun:
      gt = gt.filter_on_column('trun',0,omit=True)
    if cls and not cls=='all':
      cls_ind = self.dataset.get_ind(cls)
      gt = gt.filter_on_column('cls_ind',cls_ind)
    return gt

  def get_gt_arr(self):
    """Need this method for convenience of other methods, including in Dataset."""
    return np.array([obj.get_arr() for obj in self.objects])

  @classmethod
  def get_gt_cols(cls):
    return Object.get_cols()

  def get_num_windows(self,window_params):
    """Return the number of windows that self.get_windows will return."""
    return self.get_windows(window_params,just_num=True)

  def get_random_windows(self,window_params,num_windows):
    """
    Return at most num_windows of random windows generated according to params.
    """
    windows = self.get_windows(window_params)
    return windows[ut.random_subset_up_to_N(windows.shape[0],num_windows),:]

  def get_windows(self,window_params,with_time=False,just_num=False):
    """
    Return all windows that can be generated with window_params.
    If with_time=True, return tuple of (windows, time_elapsed).
    If just_num=True, return the number of windows instead of the actual windows.
    """
    t = time.time()
    stride = window_params.stride
    min_width = window_params.min_width
    im_width = self.size[0]
    im_height = self.size[1]
    actual_xs = []
    actual_ys = []
    actual_ws = []
    actual_hs = []
    num_windows = 0
    # we want to be able to capture objects that extend past the image
    # we always iterate over locations in native space, and convert to
    # actual image space when we record the window
    w_pad = int(1.*min_width/2)
    x_min = -w_pad
    for scale in window_params.scales:
      x_max = int(im_width*scale)-w_pad
      if w_pad > 0:
        x_max += stride
      actual_w = int(min_width/scale) + 1
      for ratio in window_params.aspect_ratios:
        h_pad = int(1.*min_width*ratio/2)
        y_min = -h_pad
        y_max = int(im_height*scale)-h_pad
        if h_pad > 0:
          y_max += stride
        actual_h = int(min_width/scale * ratio) + 1
        if just_num:
          num_y = len(range(y_min,y_max,stride))
          num_x = len(range(x_min,x_max,stride))
          num_windows += num_y*num_x
          continue
        for y in range(y_min,y_max,stride):
          for x in range(x_min,x_max,stride):
            actual_ws.append(actual_w)
            actual_hs.append(actual_h)
            actual_xs.append(int(x/scale))
            actual_ys.append(int(y/scale))
    if just_num:
      if with_time:
        time_elapsed = time.time()-t
        return (num_windows,time_elapsed)
      else:
        return num_windows
    windows = np.array([actual_xs,actual_ys,actual_ws,actual_hs]).T
    windows = BoundingBox.clipboxes_arr(windows,(0,0,im_width,im_height))
    if with_time:
      time_elapsed = time.time()-t
      return (windows,time_elapsed)
    else:
      return windows

class Object:
  """An object has a bounding box and a class."""

  def __init__(self, bbox, cls_ind, diff, trun):
    self.diff = diff
    self.bbox = bbox
    self.cls_ind = cls_ind
    self.trun = trun

  def __repr__(self):
    return "Object: %s" % self.get_arr()

  def get_arr(self):
    """Return array with object per row, cols as in get_cols()."""
    return np.hstack((self.bbox.get_arr(), self.cls_ind, self.diff, self.trun))

  @classmethod
  def get_cols(cls):
    return BoundingBox.get_cols() + ['cls_ind', 'diff', 'trun']

