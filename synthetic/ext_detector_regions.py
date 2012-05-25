from common_imports import *
import synthetic.config as config

from IPython import embed
from synthetic.ext_detector import ExternalDetector
from synthetic.dataset import Dataset

class RegionModel():
  '''
  There are different approaches to divide into regions. This class offers an 
  interface to allow for different subdivisions.
  Usage: 
  - Create a certain model type by passing in the name and the according 
  number of arguments
  - Call which_region(x, y, scale, aspect_ratio) to get the region for a window   
  '''
  def __init__(self, rtype, *args):
    '''
    Implemented types:
    scale - Divide into regions by a scale threshold only, args: (scale_threshold)
    scale_location - Divide into 4 regions, order: 0-(left small), 1-(right small), 
                     2-(left big), 3-(right big), args: (scale_threshold)
    '''    
    if rtype == 'scale':
      # TODO      
      #self.which_region = eval('self.__which_region_%s'%rtype) # :( doesn't work yet      
      self.which_region = self.__which_region_scale
      args_needed = 1
      self.num_regions = 2
    elif rtype == 'scale_location':
      self.which_region = self.__which_region_scale_location
      args_needed = 1
      self.num_regions = 4
    else:
      raise RuntimeError('Type %s is an unknown RegionModel'%rtype)
    
    if not len(args) == args_needed:
      raise RuntimeError("Not the right number of arguments for type %s. %d needed, %d given"%(rtype, args_needed, len(args)))
    else:
      self.args = args  
    
  def __which_region_scale(self, image_size, x, y, scale, aspect_ratio):
    scale_thresh = self.args[0]
    if scale < scale_thresh:
      result_region = 0
    else:
      result_region = 1
    return result_region
  
  def __which_region_scale_location(self, image_size, x, y, scale, aspect_ratio):
    scale_thresh = self.args[0]
    img_width = image_size[0]
    result_region = 0
    if scale >= scale_thresh:
      result_region += 2
    if x >= img_width/2:
      result_region += 1
    return result_region

  def get_number_regions(self):
    return self.num_regions
  
  
class ExternalDetectorRegions(ExternalDetector):
  '''
  External Detector that also tests for specific regions.
  '''
  def __init__(self, dataset, train_dataset, cls, dets, detname, rtype, args):
    '''
    Also pass in the region-type as rtype and the according arguments as args
    '''
    ExternalDetector.__init__(self, dataset, train_dataset, cls, dets, detname)
    self.region_model = RegionModel(rtype, args)
  
  def detect(self, image, region_id, astable=False):
    None
  
  def compute_score(self, image, region_id, oracle=False):
    return 0 
  
def run():
  d = Dataset('full_pascal_trainval', force=True)
  img = d.images[152]
  region_model = RegionModel('scale', 200)
  region_model.which_region(img.size(), 1, 1, 2, 2)
  
if __name__=='__main__':
  run()
    