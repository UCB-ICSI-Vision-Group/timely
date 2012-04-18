from synthetic.common_imports import *

import xml.dom.minidom as minidom
from synthetic.bounding_box import BoundingBox
from synthetic.sliding_windows import SlidingWindows

class Image:
  "An image has a name, size and a DataFrame of objects."

  columns = BoundingBox.columns + ['cls_ind', 'diff', 'trun']
  
  def __init__(self,width,height,classes,name):
    self.name = name
    self.width = width
    self.height = height
    self.classes = classes
    self.objects_df = None
    # the above DataFrame is constructed by loader

  def __repr__(self):
    return "Image (%(name)s)\n  W x H: %(width)d x %(height)d\n  Objects:\n %(objects_df)s" % self.__dict__

  def get_objects_df(self,with_diff=False,with_trun=True):
    "Return objects_df filtered with the parameters."
    df = self.objects_df
    ind = []
    if not with_diff:
      df = df[df['diff']==0]
    if not with_trun:
      df = df[df['trun']==0]
    return df

  def get_det_gt(self, cls_name=None, with_diff=False, with_trun=True):
    """
    Return DataFrame of detection ground truth.
    If class_name is given, only includes objects of that class.
    Filter according to with_diff and with_trun.
    """
    df = self.get_objects_df(with_diff,with_trun)
    if cls_name and not cls_name=='all':
      cls_ind = self.classes.index(cls_name)
      df = df.ix[df['cls_ind']==cls_ind]
    return df 

  def get_cls_counts(self, with_diff=False, with_trun=True):
    "Return a Series of the counts of each class in the image."
    counts = np.zeros(len(self.classes))
    objects_df = self.get_objects_df(with_diff,with_trun)
    if objects_df.shape[0]>0:
      cls_inds = objects_df['cls_ind'].astype('int')
      bincount = np.bincount(cls_inds)
      # need to pad this with zeros for total length of num_classes
      counts[:bincount.size] = bincount
    return Series(counts,self.classes)

  def get_cls_gt(self,with_diff=False,with_trun=False):
    "Return a Series of class presence (True/False) ground truth."
    return self.get_cls_counts(with_diff,with_trun)>0

  def contains_class(self, cls_name, with_diff=False, with_trun=True):
    "Return whether the image contains an object of class cls."
    return self.get_cls_gt(with_diff,with_trun)[cls_name]

  ### 
  # Windows
  ###
  def get_whole_image_bbox(self):
    "Return a BoundingBox with (0,0,width,height) of the image."
    return BoundingBox((0,0,self.size[0],self.size[1]))

  def get_windows(self,window_params,with_time=False):
    "Return all windows that can be generated with given params."
    return SlidingWindows.get_windows(self,None,window_params,with_time)

  def get_random_windows(self,window_params,num_windows):
    "Return at most num_windows random windows generated according to params."
    windows = self.get_windows(window_params)
    return windows[ut.random_subset_up_to_N(windows.shape[0],num_windows),:]

  ###
  # Loaders
  ###
  @classmethod
  def load_from_json_data(cls, data, classes):
    "Return an Image instantiated from a JSON representation."
    name = data['name']
    width = data['size'][0]
    height = data['size'][1]
    img = Image(width,height,classes,name)
    objects = []
    for obj in data['objects']:
      bbox = BoundingBox(obj['bbox'])
      cls_name = obj['class']
      cls_ind = classes.index(cls_name)
      diff = obj['diff']
      trun = obj['trun']
      objects.append(np.hstack((bbox.get_arr(), cls_ind, diff, trun)))
    if len(objects)>0:
      img.objects_df = DataFrame(objects, columns=cls.columns)
    else:
      img.objects_df = DataFrame(None, columns=cls.columns)
    return img
  
  @classmethod
  def load_from_pascal_xml_filename(cls, classes, filename):
    "Load image info from a file in the PASCAL VOC XML format."

    def get_data_from_tag(cls, node, tag):
      if tag is "bndbox":
        x1 = int(node.getElementsByTagName(tag)[0].childNodes[1].childNodes[0].data)
        y1 = int(node.getElementsByTagName(tag)[0].childNodes[3].childNodes[0].data)
        x2 = int(node.getElementsByTagName(tag)[0].childNodes[5].childNodes[0].data)
        y2 = int(node.getElementsByTagName(tag)[0].childNodes[7].childNodes[0].data)
        return (x1, y1, x2, y2)
      else:
        return node.getElementsByTagName(tag)[0].childNodes[0].data

    with open(filename) as f:
      data = minidom.parseString(f.read())

    # image info
    name = cls.get_data_from_tag(data, "filename")
    size = data.getElementsByTagName("size")[0]
    im_width = int(cls.get_data_from_tag(size, "width"))
    im_height = int(cls.get_data_from_tag(size, "height"))
    im_depth = int(cls.get_data_from_tag(size, "depth"))
    width = im_width
    height = im_height
    img = Image(width,height,classes,name)

    # per-object info
    objects = []
    for obj in data.getElementsByTagName("object"):
      categ = str(cls.get_data_from_tag(obj, "name")).lower().strip()
      diff = int(cls.get_data_from_tag(obj, "difficult"))
      trun = int(cls.get_data_from_tag(obj, "truncated"))
      rect = cls.get_data_from_tag(obj, "bndbox")
      bbox = BoundingBox(rect, format='corners')
      cls_ind = dataset.get_ind(categ)
      objects.append(np.hstack((bbox.get_arr(), cls_ind, diff, trun)))

    if len(objects)>0:
      img.objects_df = DataFrame(objects, columns=cls.columns)
    else:
      img.objects_df = DataFrame(None, columns=cls.columns)
    return img
