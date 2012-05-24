from common_imports import *
import synthetic.config as config

from synthetic.ext_detector import ExternalDetector

class ExternalDetectorRegions(ExternalDetector):
  
  class RegionModel():
    def __init__(self, type):
      if type == '':
        None
  
  def __init__(self, dataset, train_dataset, cls, dets, detname):
    ExternalDetector.__init__(self, dataset, train_dataset, cls, dets, detname)
    
  @classmethod
  def which_region(self, imparams, windowparams):
    return None
    