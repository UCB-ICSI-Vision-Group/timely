from synthetic.common_imports import *
import synthetic.config as config

from synthetic.image import Image
from synthetic.sliding_windows import SlidingWindows

""" TODO: temp synthetic code
    self.synthetic = synthetic
    if synthetic:
      self.gen_cls_ground_truth()

  def gen_cls_ground_truth(self):
    
"""

class Dataset(object):
  """
  Representation of a dataset, with methods to construct from different sources
  of data, get ground truth, and construct sets of train/test data.
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
    elif name == 'synthetic':
      self.generate_synthetic()
    else:
      print("WARNING: Unknown dataset initialization string, not loading images.")
    self.image_names = [image.name for image in self.images]
    assert(len(self.image_names)==len(np.unique(self.image_names)))
    self.cached_det_ground_truth = {}

  def get_name(self):
    return "%s_%s"%(self.name,self.num_images())

  def num_images(self):
    return len(self.images)

  ###
  # Loaders / Generators
  ###
  def generate_synthetic(self):
    "Generate a synthetic dataset that follows some simple cooccurence rules."
    # hard-coded 3-class generation
    choices = [[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
    probs = np.array([0,8,3,1,6,1,8,3])
    cum_probs = np.cumsum(1.*probs/np.sum(probs))
    # to check that this is right (it is):
    # hist(choices,bins=arange(0,9),normed=True,align='left'); plot(1.*probs/sum(probs))
    
    self.classes = ['A','B','C']
    num_images = 1000
    for i in range(0,num_images):
      image = Image(100,100,self.classes,str(i))
      choice = np.where(cum_probs>np.random.rand())[0][0]
      objects = []
      for cls_ind,clas in enumerate(choices[choice]):
        if clas == 1:
          objects.append(np.array([0,0,0,0,cls_ind,0,0]))
      image.objects_df = DataFrame(objects, columns=Image.columns)
      self.images.append(image)

  def load_from_json(self, filename):
    "Load all parameters of the dataset from a JSON file."
    with open(filename) as f:
      config = json.load(f)
    self.classes = config['classes']
    for data in config['images']:
      self.images.append(Image.load_from_json_data(self.classes,data))

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
        self.images.append(Image.load_from_pascal_xml_filename(self.classes,xml_filename))
    filename = config.get_cached_dataset_filename(name)
    print("  ...saving to cache file")
    with open(filename, 'w') as f:
      cPickle.dump(self,f)
    print("  ...done in %.2f s\n"%tt.qtoc())

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
  # Ground truth
  ###
  def get_cls_counts(self, with_diff=True, with_trun=True):
    """
    Return DataFrame of class presence counts.
      Index: image.name
      Columns: self.classes
    """
    data = {}
    for image in self.images:
      data[image.name] = image.get_cls_counts()
    return DataFrame(data).T

  def get_cls_ground_truth(self,with_diff=True,with_trun=True):
    "Return DataFrame of classification (0/1) ground truth."
    return self.get_cls_counts(with_diff,with_trun)>0

  def get_det_ground_truth(self, with_diff=True, with_trun=True):
    """
    Return DataFrame of detection ground truth.
      Major index: image.name
      Minor index: object index (irrelevant)
      Columns: Image.columns
    Caches the result in memory.
    """
    name = '%s%s'%(with_diff,with_trun)
    if name not in self.cached_det_ground_truth:
      data = {}
      for image in self.images:
        data[image.name] = image.get_det_gt(with_diff,with_trun)
      df = Panel.from_dict(data,orient='minor').swapaxes().to_frame()
      self.cached_det_ground_truth[name] = df
    return self.cached_det_ground_truth[name]

  ###
  # Statistics
  ###
  def plot_cooccurence(self,with_diff=True,with_trun=True,second_order=False):
    """
    Plot the correlation matrix of class occurence.
      second_order: plot co-occurence of two-class pairs with third class.
    """
    from nitime.viz import drawmatrix_channels
    # TODO: second-order
    df = self.get_cls_ground_truth(with_diff,with_trun)
    f = drawmatrix_channels(df.corr().as_matrix(),df.columns,
      size=(10,10),color_anchor=(-1,1))
    dirname = config.get_dataset_stats_dir(self)
    filename = opjoin(dirname,'cooccur_diff_%s_trun_%s_second_order_%s.png'%(
      with_diff,with_trun,second_order))
    f.savefig(filename)

  ###
  # K-Folds
  ###
  def create_folds(self, numfolds):
    """
    Split the images of dataset in numfolds folds.
    Dataset has an inner state about current fold (This is like an implicit 
    generator).
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
  
  def get_pos_samples_for_fold_class(self, cls, with_diff=False,
      with_trun=True):
    if not hasattr(self, 'train'):
      return self.get_pos_samples_for_class(cls, with_diff, with_trun)
    all_pos = self.get_pos_samples_for_class(cls, with_diff, with_trun)
    return np.intersect1d(all_pos, self.train)
  
  def get_neg_samples_for_fold_class(self, cls, num_samples, with_diff=False,
      with_trun=True):
    if not hasattr(self, 'train'):
      return self.get_neg_samples_for_class(cls, with_diff, with_trun)
    intersect = np.intersect1d(all_neg, self.train)
    if intersect.size == 0:
      return np.array([])
    return np.array(ut.random_subset(intersect, num_samples))
