from PIL import Image as PILImage

from common_mpi import *
from common_imports import *

import synthetic.config as config
from synthetic.image import *
from synthetic.sliding_windows import SlidingWindows

class Dataset:
  """
  Methods for constructing, accessing, and evaluating detection performance
  on a dataset.
  """

  # Convenience methods
  def num_classes(self):
    return len(self.classes)
  
  def num_images(self):
    return len(self.images)

  def get_ind(self,cls):
    """
    Return the index of the given cls. If cls is 'all', returns an extra
    index.
    """
    if cls=='all':
      return len(self.classes)
    assert(cls in self.classes)
    return self.classes.index(cls)

  def get_img_ind(self,image):
    """Return the index of the given image."""
    assert(image in self.images)
    return self.images.index(image)

  def get_name(self):
    return "%s_%s"%(self.name,len(self.images))

  def see_image(self,img_ind):
    """Convenience method to display the image at given ind in self.images."""
    # TODO: use scikits-image to load as numpy matrix (matplotlib.imread loads
    # inverted for some reason)
    im = PILImage.open(self.get_image_filename(img_ind))
    im.show()

  def get_image_by_filename(self, filename):
    for img in self.images:
      if img.name == filename:
        return img
    return None

  def get_image_filename(self,img_ind):
    return config.VOC_dir + 'JPEGImages/' + self.images[img_ind].name

  def __repr__(self):
    return self.get_name()
  
  def __init__(self, name=None, force=False):
    self.classes = []
    self.images = []
    self.name = name
    if re.search('pascal', name):
      self.load_from_pascal(name,force)
    elif name == 'data1':
      self.load_from_json(config.data1)
    else:
      print("WARNING: Unknown dataset initialization string, not loading images.")

  def load_from_pascal(self, name, force=False):
    """
    Look up the filename associated with the given name.
    Read image names from provided filename, and construct a dataset from the
    corresponding .xml files.
    Caches self when loaded into conventional filename, such that next time
    loading is faster.
    If force is True, does not look for cached data when loading.
    """
    print("Dataset: Loading from PASCAL...")
    filename = config.get_cached_dataset_filename(name)
    if os.path.exists(filename):
      print("...loading from cached dataset")
      with open(filename) as f:
        cached = cPickle.load(f)
        self.classes = cached.classes
        self.images = cached.images
        print("...done")
        return
    print("...loading from scratch")
    filename = config.pascal_paths[name]
    self.classes = config.pascal_classes 
    with open(filename) as f:
      imgset = [line.strip() for line in f.readlines()]
    t = time.time()
    for i,img in enumerate(imgset):
      ti = time.time()-t
      if ti > 2:
        print("...on image %d/%d"%(i,len(imgset)))
        t = time.time()
      if len(img)>0:
        xml_filename = os.path.join(config.VOC_dir,'Annotations',img+'.xml')
        self.images.append(Image.load_from_xml(self,xml_filename))
    filename = config.get_cached_dataset_filename(name)
    print("...saving to cache file")
    with open(filename, 'w') as f:
      cPickle.dump(self,f)
    print("...done\n")

  def get_pos_windows(self, cls=None, window_params=None, min_overlap=0.6):
    """
    Return array of all ground truth windows for the class, plus windows 
    that can be generated with window_params that overlap with it by more
    than min_overlap.
    * If cls not given, return positive windows for all classes.
    * If window_params not given, use default for the class.
    * Adjust min_overlap to fetch fewer windows.
    """
    sw = SlidingWindows(self, self)
    if not window_params:
      window_params = sw.get_default_window_params(cls)
    overlapping_windows = []
    image_inds = self.get_pos_samples_for_class(cls)
    times = []
    window_nums = []
    for i in image_inds:
      image = self.images[i]
      gts = image.get_ground_truth(cls)
      if gts.arr.shape[0]>0:
        overlap_wins = gts.arr[:,:4]
        overlap_wins = np.hstack((overlap_wins, np.tile(i, (overlap_wins.shape[0],1))))
        overlapping_windows.append(overlap_wins.astype(int))
        windows,time_elapsed = image.get_windows(window_params,with_time=True)
        window_nums.append(windows.shape[0])
        times.append(time_elapsed)
        for gt in gts.arr:
          overlaps = BoundingBox.get_overlap(windows[:,:4],gt[:4])
          overlap_wins = windows[overlaps>=min_overlap,:]
          overlap_wins = np.hstack((overlap_wins, np.tile(i, (overlap_wins.shape[0],1))))
          overlapping_windows.append(overlap_wins.astype(int))
          windows = windows[overlaps<min_overlap,:]
    overlapping_windows = np.concatenate(overlapping_windows,0)
    print("Windows generated per image: %d +/- %.3f, in %.3f +/- %.3f sec"%(
          np.mean(window_nums),np.std(window_nums),
          np.mean(times),np.std(times)))
    return overlapping_windows

  def get_neg_windows(self, num, cls=None, window_params=None, max_overlap=0,
      max_num_images=250):
    """
    Return array of num windows that can be generated with window_params
    that do not overlap with ground truth by more than max_overlap.
    * If cls is not given, returns ground truth for all classes.
    * If max_num_images is given, samples from at most that many images.
    """
    sw = SlidingWindows(self, self)
    if not window_params:
      window_params = sw.get_default_window_params(cls)
    all_windows = []
    image_inds = self.get_pos_samples_for_class(cls)

    max_num = len(image_inds)
    inds = image_inds
    if max_num_images:
      inds = ut.random_subset(image_inds, max_num_images)
    num_per_image = round(1.*num / max_num)
    for ind in inds:
      image = self.images[ind]
      windows = image.get_windows(window_params)
      gts = image.get_ground_truth(cls)
      for gt in gts.arr:
        overlaps = BoundingBox.get_overlap(windows[:,:4],gt[:4])
        windows = windows[overlaps <= max_overlap,:]
      ind_to_take = ut.random_subset_up_to_N(windows.shape[0], num_per_image)
      all_windows.append(np.hstack(
        (windows[ind_to_take,:],np.tile(ind, (ind_to_take.shape[0],1)))))
    all_windows = np.concatenate(all_windows,0)
    return all_windows[:num,:]
  
  def get_pos_samples_for_class(self, cls, include_diff=False,
      include_trun=True):
    """
    Return array of indices of self.images that contain at least one object of
    this class.
    """
    cls_gt = self.get_ground_truth_for_class(cls,include_diff,include_trun)
    img_indices = cls_gt.subset_arr('img_ind')
    return np.sort(np.unique(img_indices)).astype(int)

  def get_neg_samples_for_class(self, cls, number=None,
      include_diff=False, include_trun=True):
    """
    Return array of indices of self.images that contain no objects of this class.
    """
    pos_indices = self.get_pos_samples_for_class(cls,include_diff,include_trun)
    neg_indices = np.setdiff1d(np.arange(len(self.images)),pos_indices,assume_unique=True)
    # TODO tobi: why do these have to be ordered?
    return ut.random_subset(neg_indices, number, ordered=True)
  
  def load_from_json(self, filename):
    """Load all parameters of the dataset from a JSON file."""
    import json
    with open(filename) as f:
      config = json.load(f)
    self.classes = config['classes']
    for image in config['images']:
      self.images.append(Image.from_json(self,image))
      
  def get_ground_truth(self, include_diff=False, include_trun=True):
    """
    Return Table object containing ground truth of the dataset.
    If include_diff or include_trun are False, those column names are omitted.
    """
    gt = ut.Table(arr=self.get_gt_arr(),cols=self.get_gt_cols())
    if not include_diff:
      gt = gt.filter_on_column('diff',0,omit=True)
    if not include_trun:
      gt = gt.filter_on_column('trun',0,omit=True)
    return gt

  def get_ground_truth_for_img_inds(self, img_inds, cls=None, include_diff=False, include_trun=True):
    """
    Return Table object containing ground truth for the given image indices.
    """
    images = (self.images[int(ind)] for ind in img_inds)
    arr = ut.collect(images, Image.get_gt_arr)
    gt = ut.Table(arr,Image.get_gt_cols())
    if not include_diff:
      gt = gt.filter_on_column('diff',0,omit=True)
    if not include_trun:
      gt = gt.filter_on_column('trun',0,omit=True)
    if cls and not cls=='all':
      cls_ind = self.dataset.get_ind(cls)
      gt = gt.filter_on_column('cls_ind',cls_ind)
    return gt

  def get_cls_counts(self):
    """
    Return ndarray of size (num_images,num_classes), with counts of each class
    in each image.
    """
    return ut.collect(self.images, Image.get_cls_counts)

  def get_ground_truth_for_class(self, cls, include_diff=False,
      include_trun=True):
    """
    As get_ground_truth, but filters on cls_ind, without omitting that
    column.
    If cls=='all', returns all ground truth.
    """
    gt = self.get_ground_truth(include_diff, include_trun)
    if cls=='all':
      return gt
    return gt.filter_on_column('cls_ind', self.get_ind(cls))

  def get_gt_arr(self):
    return ut.collect_with_index_column(self.images, Image.get_gt_arr)

  @classmethod
  def get_gt_cols(cls):
    return Image.get_gt_cols() + ['img_ind']

