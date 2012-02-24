from abc import abstractmethod
from scipy.stats import beta

from common_imports import *

from synthetic.dataset import Dataset
import synthetic.config as config
from synthetic.sliding_windows import WindowParams

class Detector(object):
  """
  Detector takes an image and outputs list of bounding boxes and
  confidences for its class.
  """
  
  # for use when computing the expected time
  AVG_IMAGE_SIZE = (500,375)

  DEFAULT_CONFIG = {
    'avg_time': 10 # in seconds, specified w.r.t Detector.AVG_IMAGE_SIZE
    }

  @classmethod
  def get_cols(cls):
    return ['x','y','w','h','score']

  def __init__(self, dataset, cls, detector_config=None):
    self.dataset = dataset
    self.cls = cls
    self.cls_ind = dataset.get_ind(cls)

    if not detector_config:
      detector_config = Detector.DEFAULT_CONFIG
    self.config = detector_config

  @abstractmethod
  def detect(self, image):
    """
    Return a tuple of
    - array where each row is [x,y,w,h,score]
    - time it took
    This is what ImagePolicy interfaces with.
    """
    # Implement in specific detectors

  def compute_posterior(self, image, dets, oracle=True):
    """
    Return the predicted posterior of the class being present in the image,
    given the detections.
    This implementation returns the ground truth answer to the question of
    object presence--so it's a 100% accurate classifier.
    """
    return image.contains_cls_ind(self.cls_ind)

  def expected_naive_ap(self):
    """
    Return the expected naive ap contribution on an image containing at
    least one such object.
    """
    return self.config['naive_ap']

  def expected_actual_ap(self):
    """
    Return the expected naive ap contribution on an image containing at
    least one such object.
    """
    return self.config['actual_ap']

  def expected_time(self, image, bbox=None):
    """
    Take a subregion of an image and return the expected time that this
    region will take to process according to the window policy of the detector.
    If bbox is not given, use whole image.
    """
    # NOTE: for now, ignoring anything about the image, but in the future can
    # look at some quick statistics to get an estimate of its complexity
    avg_area = np.prod(Detector.AVG_IMAGE_SIZE)
    bbox = image.get_whole_image_bbox()
    expected_time = self.config['avg_time'] * 1.*bbox.area()/avg_area 
    return expected_time

  @classmethod
  def nms_detections(cls,dets,cols,overlap=0.5):
    """
    Non-maximum suppression: Greedily select high-scoring detections and skip
    detections that are significantly covered by a previously selected detection.
    Take
      - dets, an ndarray of detections
      - cols, containing column names for the detections
      - min overlap ratio (0.5 default)
    Return dets that remain after suppression.

    This version is translated from Matlab code by Tomasz Maliseiwicz,
    who sped up a version by Pedro Felzenszwalb.
    """
    #t = time.time()
    if np.shape(dets)[0] < 1:
      return dets 

    x1 = dets[:,cols.index('x')]
    y1 = dets[:,cols.index('y')]

    w = dets[:,cols.index('w')]
    h = dets[:,cols.index('h')]
    x2 = w+x1-1
    y2 = h+y1-1 
    s = dets[:,cols.index('score')]

    area = w*h
    ind = np.argsort(s)

    pick = [] 
    counter = 0
    while len(ind)>0:
      last = len(ind)-1
      i = ind[last] 
      pick.append(i)
      counter += 1
      
      xx1 = np.maximum(x1[i], x1[ind[:last]])
      yy1 = np.maximum(y1[i], y1[ind[:last]])
      xx2 = np.minimum(x2[i], x2[ind[:last]])
      yy2 = np.minimum(y2[i], y2[ind[:last]])
      
      w = np.maximum(0., xx2-xx1+1)
      h = np.maximum(0., yy2-yy1+1)
      
      o = w*h / area[ind[:last]]
      
      to_delete = np.concatenate((np.nonzero(o>overlap)[0],np.array([last])))
      ind = np.delete(ind,to_delete)

    #print("Took %.3f s"%(time.time()-t))
    return dets[pick,:]

class SWDetector(Detector):
  "A detector that must be initialized with a sliding window generator."
  def __init__(self, dataset, cls, sw_generator, detector_config=None):
    Detector.__init__(self,dataset,cls,detector_config)
    self.sw = sw_generator

class PerfectDetector(Detector):
  """
  An idealized detector that returns ground truth.
  Perfect classification performance.
  """

  def detect(self, image):
    """Return the ground truth of the image with perfect confidence."""
    dets = []
    for obj in image.objects:
      if obj.cls_ind == self.cls_ind:
        dets.append(np.hstack((obj.bbox.get_arr(), 1.)))
    return (np.array(dets), self.expected_time(image))

class PerfectDetectorWithNoise(SWDetector):
  """
  Basically PerfectDetector with additional false positives and false
  negatives, generated randomly according to a couple of parameters.
  Still perfect classification!
  """

  def detect(self, image):
    """
    Return all of the ground truth of the image with some high probability,
    but also throw in some random false positives.
    Because of the randomness, needs to generate sliding windows.
    """
    dets = []
    # First, go through all true objects:
    # - with high probability detect the object
    # - draw a score from a right-skewed beta distribution
    for obj in image.objects:
      if obj.cls_ind == self.cls_ind:
        if np.random.rand > 0.2:
          score = np.random.beta(2,1)
          dets.append(np.hstack((obj.bbox.get_arr(), score)))
    # Second, get some false positives:
    # - draw the number of false positives from a Poisson with the expectation
    # of the number of detections already acquired
    # - draw a score for each from a left-skewed beta distribution
    num_true = len(dets)
    num_false = np.random.poisson(num_true)
    windows = self.sw.get_windows_new(image, self.cls)
    for window in windows:
      score = np.random.beta(1,2)
      dets.append(np.hstack((window, score)))
    return (np.array(dets), self.expected_time(image))
