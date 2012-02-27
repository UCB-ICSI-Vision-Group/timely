  
def split_data(self, data, ratio):
  """
  Take array and ratio, and return tuple of two arrays:
    - the first one is of length 1-ratio
    - the second one is of length ratio
  Both contain rows from the original array in random order.
  """
  num_val = int(ratio*data.shape[0])
  perm_ind = np.random.permutation(data.shape[0])
  val_inds = perm_ind[:num_val]
  train_inds = perm_ind[num_val:]
  train_data = data[train_inds,:]
  val_data = data[val_inds,:]
  return (train_data,val_data)

class SyntheticClassifierDetector(Detector):
  """
  An idealized detector modeled by two distributions.
  P(score=x|class(w)=k) and P(score=x|class(w)!=k).
  These distributions are modeled by half-Gaussian for efficiency.
  Additionally, the detector has an expected_time method which gives an
  estimate of running the detector on the given bounding box in the image.
  """

  def __init__(self, dataset, cls, config=None):
    Detector.__init__(self, dataset, cls, config)
    assert('alpha_given_true' in self.config)
    assert('beta_given_true' in self.config)
    assert('alpha_given_false' in self.config)
    assert('beta_given_false' in self.config)
    self.rv_score_given_true = beta(
        self.config['alpha_given_true'],
        self.config['beta_given_true'])
    self.rv_score_given_false = beta(
        self.config['alpha_given_false'],
        self.config['beta_given_false'])
    # NOTE: if the beta constructor is called with named parameters, the pdf
    # function does not work and compains of taking exactly 4 arguments.

  def detect(self, image):
    """Classify every window in order."""
    windows = image.get_windows_for_detector(self)
    dets = ut.collect(windows, self.classify_window) 
    # TODO: add some noise to the estimated time
    time = self.get_estimated_time(image)
    return (dets, time)

  def predict_value(self, image, priors):
    """
    Compute the expected value of running the detector, given the class priors.
    Do it by first generating fake detections, and then predicting their AP
    score.
    # TODO: combine the two steps
    """
    if not image:
      image = Dataset.get_avg_image()
    expected_detections = self.detect_expected(image, priors)
    ap_value = self.predict_ap(image,expected_detections,priors)
    return ap_value

  def detect_expected(self,image,priors):
    """Generate fake detections given the class priors."""
    # first, generate an image that is like this image, according to the priors
    synthetic_image = Image.generate_like(image,priors)
    # then do synthetic detection in this image
    
  def visualize_distributions(self,filename):
    """
    Method that plots the two distributions and the expected time of the
    detector to provided filename.
    """
    x = np.linspace(0,1)
    plt.clf()
    plt.plot(x, self.rv_score_given_true.pdf(x),label='given true %s'%str(self.rv_score_given_true.args))
    plt.plot(x, self.rv_score_given_false.pdf(x),label='given false %s'%str(self.rv_score_given_false.args))
    plt.legend()
    plt.title('Detector for class %s, expected time is %.2f s'%(self.cls,
      self.config['avg_time_per_image']))
    plt.savefig(filename) 

  def classify_window(self, window):
    """
    Take a window in an image and return a detection in the form of
    self.get_cols().
    """
    bbox = window.bbox
    #img = window.img
    objects = window.objects
    # Use the Synthetic Detector parameters to make a decision
    if len(objects)>0:
      score = self.sample(self.dist_score_given_true)
    else:
      score = self.sample(self.dist_score_given_false)

class TestSyntheticClassifierDetector:
  def test_init(self):
    dataset = Dataset('test_pascal_val')
    config = config.get_default_detector_config()
    config.update(config.get_default_good_synthetic_extra())
    d = SyntheticClassifierDetector(dataset,'dog',config)

  def test_visualize_distributions(self):
    dataset = Dataset('test_pascal_val')
    # good detector
    config = config.get_default_detector_config()
    config.update(config.get_default_bad_synthetic_extra())
    d = SyntheticClassifierDetector(dataset,'dog',config)
    filename = 'good_detector%s.png'
    d.visualize_distributions(filename)
    assert(os.path.exists(filename))
    os.remove(filename)

    # bad detector
    config = config.get_default_detector_config()
    config['avg_time_per_image'] = 10 # half the default
    config.update(config.get_default_bad_synthetic_extra())
    d = SyntheticClassifierDetector(dataset,'dog',config)
    filename = 'bad_detector%s.png'
    d.visualize_distributions(filename)
    assert(os.path.exists(filename))
    os.remove(filename)

  def test_subclass(self):
    dataset = Dataset('test_pascal_val')
    d = Detector(dataset,'dog')
    assert(isinstance(d,Detector))

# Just some initial settings for development
@classmethod
def get_default_good_synthetic_extra(cls):
  return {
    'alpha_given_true': 4,
    'beta_given_true': 1,
    'alpha_given_false': 1,
    'beta_given_false': 4
  }

@classmethod
def get_default_bad_synthetic_extra(cls):
  return {
    'alpha_given_true': 1.5,
    'beta_given_true': 1,
    'alpha_given_false': 1,
    'beta_given_false': 1.5
  }

