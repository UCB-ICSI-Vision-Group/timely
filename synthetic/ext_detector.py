from common_imports import *
import synthetic.config as config

from synthetic.detector import Detector
from synthetic.csc_classifier import CSCClassifier
from synthetic.dpm_classifier import DPMClassifier

class ExternalDetector(Detector):
  """
  A mock interface to the Felzenszwalb DPM, CSC, or the Pendersoli CtF detector.
  Actually works by pre-loading all the detections and then returning them as
  requested.
  """

  def __init__(self, dataset, cls, dets, detname):
    """
    Expects cached detections in Table format to be passed in.
    The dets should not have the 'cls_ind' column, as they should all be of the
    same class.
    """
    # Check if configs exist and look up the correct config for this detname and cls
    detector_config = None
    filename = os.path.join(config.dets_configs_dir,detname+'.txt')
    if os.path.exists(filename):
      with open(filename) as f:
        configs = json.load(f)
      config_name = detname+'_'+cls
      if config_name in configs:
        detector_config = configs[config_name]

    Detector.__init__(self,dataset,cls,detector_config)
    self.detname = detname
    self.dets = dets
    suffix = detname[4:]

    if self.detname=='dpm':
      self.classif = DPMClassifier()
    else:
      self.classif = CSCClassifier(suffix,cls,dataset)

  def detect(self, image):
    """
    Return the detections that match that image index in cached dets.
    Must return in the same format as the Detector superclass, so we have to
    delete a column.
    """
    img_ind = self.dataset.get_img_ind(image)
    dets = self.dets.filter_on_column('img_ind',img_ind,omit=True)
    time_passed = 0
    if not dets.arr.shape[0]<1:
      time_passed = np.max(dets.subset_arr('time'))
    # Halve the time passed if my may25 DPM detector, to have reasonable times
    # Also halve the time passed by csc_half detector, because we halved its AP
    if self.detname=='dpm_may25' or self.detname=='csc_half':
      time_passed /= 2
    dets = dets.with_column_omitted('time')
    return (dets.arr, time_passed)

  def compute_score(self, image, oracle=False):
    """
    Return the 0/1 decision of whether the cls of this detector is present in
    the image, given the detections table.
    If oracle=True, returns the correct answer (look up the ground truth).
    """
    if oracle:
      return Detector.compute_score(self, image, oracle)
    img_ind = self.dataset.get_img_ind(image)
    dets = self.dets.filter_on_column('img_ind',img_ind)
    score = self.classif.classify_image(img_ind,dets)
    dt = 0
    # TODO: figure out the dt situation above
    return (score,dt)