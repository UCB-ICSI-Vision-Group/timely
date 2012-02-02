import json

from common_imports import *

from synthetic.detector import Detector
import synthetic.config as config
from synthetic.csc_classifier import CSCClassifier

class ExternalDetector(Detector):
  """
  A mock interface to the Felzenszwalb DPM or the Pendersoli CtF detector.
  Actually works by pre-loading all the detections and then returning them as
  requested.
  """
  def __init__(self, dataset, cls, sw, dets, detname):
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
        print("Successfully initialized detector %s with config!"%config_name)

    Detector.__init__(self,dataset,cls,sw,detector_config)
    self.detname = detname
    self.dets = dets
    suffix = detname[4:]
    self.csc_classif = CSCClassifier(suffix)    
    try:
      self.svm = self.csc_classif.load_svm(cls)
      setting_table = ut.Table.load(os.path.join(config.res_dir,'csc_svm_'+suffix,'best_table'))
      settings = setting_table.arr[config.pascal_classes.index(cls),:]
      self.intervalls = settings[setting_table.cols.index('bins')]
      self.lower = settings[setting_table.cols.index('lower')]
      self.upper = settings[setting_table.cols.index('upper')]
    except:
      print("Could not load classifier SVM for class %s"%cls)

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

  def compute_posterior(self, image, dets, oracle=True):
    """
    Return the 0/1 decision of whether the cls of this detector is present in
    the image, given the detections table.
    If oracle=True, returns the correct answer (look up the ground truth).
    """
    if oracle:
      return Detector.compute_posterior(self, image, dets, oracle)
    img = self.dataset.get_img_ind(image)
    cls = config.pascal_classes.index(self.cls)
    return self.csc_classif.classify_image(self.svm,dets,cls,img, self.intervalls, self.lower, self.upper)
