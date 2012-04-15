from PIL import Image as PILImage
from sklearn.cross_validation import KFold

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
  def __init__(self, name=None, force=False):
    self.classes = []
    self.images = []
    self.name = name
    self.current_fold = -1
    if re.search('pascal', name):
      self.load_from_pascal(name,force)
    elif name == 'test_data1':
      self.load_from_json(config.test_data1)
    else:
      print("WARNING: Unknown dataset initialization string, not loading images.")

  def get_name(self):
    return "%s_%s"%(self.name,self.df.shape[0])

  ###
  # Loaders
  ###
  def load_from_json(self, filename):
    "Load all parameters of the dataset from a JSON file."
    with open(filename) as f:
      config = json.load(f)
    self.classes = config['classes']
    for image in config['images']:
      self.images.append(Image.from_json(self,image))

  def load_from_pascal(self, name, force=False):
    """
    Look up the filename associated with the given name.
    Read image names from provided filename, and construct a dataset from the
    corresponding .xml files.
    Save self to disk when loaded for caching purposes.
    If force is True, does not look for cached data when loading.
    """
    tt = ut.TicToc().tic()
    print("Dataset: %s"%name),
    filename = config.get_cached_dataset_filename(name)
    if opexists(filename) and not force:
      with open(filename) as f:
        cached = cPickle.load(f)
        self.classes = cached.classes
        self.images = cached.images
        print("...loaded from cache in %.2f s"%tt.qtoc())
        return
    print("...loading from scratch")
    filename = config.pascal_paths[name]
    self.classes = config.pascal_classes 
    with open(filename) as f:
      imgset = [line.strip() for line in f.readlines()]
    for i,img in enumerate(imgset):
      tt.tic('2')
      if tt.qtoc('2') > 2:
        print("  on image %d/%d"%(i,len(imgset)))
        tt.tic('2')
      if len(img)>0:
        xml_filename = opjoin(config.VOC_dir,'Annotations',img+'.xml')
        self.images.append(Image.load_from_xml(self,xml_filename))
    filename = config.get_cached_dataset_filename(name)
    print("  ...saving to cache file")
    with open(filename, 'w') as f:
      cPickle.dump(self,f)
    print("  ...done in %.2f s\n"%tt.qtoc())

  ###
  # Misc
  ###
  def see_image(self,img_ind):
    "Convenience method to display the image at given ind in self.images."
    # TODO: use scikits-image to load as numpy matrix (matplotlib.imread loads
    # inverted for some reason)
    im = PILImage.open(self.get_image_filename(img_ind))
    im.show()

  def get_image_filename(self,img_ind):
    return opjoin(config.VOC_dir, 'JPEGImages', self.images[img_ind].name)

  ###
  # Assemble data for training
  ###
  def get_pos_windows(self, cls=None, window_params=None, min_overlap=0.7):
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
      if windows.shape[0] == 0:
        continue
      ind_to_take = ut.random_subset_up_to_N(windows.shape[0], num_per_image)
      all_windows.append(np.hstack(
        (windows[ind_to_take,:],np.tile(ind, (ind_to_take.shape[0],1)))))
    all_windows = np.concatenate(all_windows,0)
    return all_windows[:num,:]

  ###
  # Assemble ground truth
  ###
  def exclude_diff(self, det_gt):
    "Return the passed in DataFrame with difficult objects excluded."
    return det_gt.ix[~det_gt.diff]

  def exclude_trun(self, det_gt):
    "Return the passed in DataFrame with truncated objects excluded."
    return det_gt.ix[~det_gt.trun]

  def get_cls_ground_truth(self,with_diff=True,with_trun=True):
    "Return DataFrame of classification (0/1) ground truth."
    return self.get_cls_counts(with_diff,with_trun)>0

  def get_cls_counts(self, with_diff=True, with_trun=True):
    "Return DataFrame of class presence counts."
    data = ut.collect(self.images, Image.get_cls_counts)
    return DataFrame(data,columns=self.classes)

  def get_det_ground_truth(self, with_diff=True, with_trun=True):
    # TODO: have images return DataFrame as well
    "Return DataFrame of detection ground truth."
    data = ut.collect_with_index(self.images, Image.get_gt_arr)
    columns = Image.get_gt_cols() + ['img_ind']
    return DataFrame(data, columns)

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
      
  def create_folds(self, numfolds):
    """
    Split the images of dataset in numfolds folds,
    Dataset has an inner state about current fold (This is like an implicit 
    generator)
    """
    folds = KFold(len(self.images), numfolds)
    self.folds = []
    for fold in folds:
      self.folds.append(fold)
    self.current_fold = 0
    
  def next_folds(self):
    if self.current_fold < len(self.folds):
      fold = self.folds[self.current_fold]
      self.current_fold += 1
      self.train, self.val = fold
      if type(self.train[0]) == type(np.array([True])[0]):
        self.train = np.where(self.train)[0]
        self.val = np.where(self.val)[0]
      return fold
    if self.current_fold >= len(self.folds):
      self.current_fold = 0
      return 0
  
  def get_fold_by_index(self, ind):
    """
    Random access to folds
    """
    if ind >= len(self.folds):
      raise RuntimeError('Try to access non-existing fold')
    else:
      return self.folds[ind]
  
  def get_pos_samples_for_fold_class(self, cls, include_diff=False,
      include_trun=True):
    if not hasattr(self, 'train'):
      return self.get_pos_samples_for_class(cls, include_diff, include_trun)
    all_pos = self.get_pos_samples_for_class(cls, include_diff, include_trun)
    return np.intersect1d(all_pos, self.train)
  
  def get_neg_samples_for_fold_class(self, cls, num_samples, include_diff=False,
      include_trun=True):
    if not hasattr(self, 'train'):
      return self.get_neg_samples_for_class(cls, include_diff=include_diff, include_trun=include_trun)
    all_neg = self.get_neg_samples_for_class(cls, include_diff=include_diff, include_trun=include_trun)
    intersect = np.intersect1d(all_neg, self.train)
    if intersect.size == 0:
      return np.array([])
    return np.array(ut.random_subset(intersect, num_samples))
