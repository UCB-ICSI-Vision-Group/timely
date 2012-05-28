from common_imports import *
import synthetic.config as config

from IPython import embed
from synthetic.ext_detector import ExternalDetector
from synthetic.dataset import Dataset
from synthetic.bounding_box import BoundingBox

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
    scale - Divide into regions by a scale threshold only; order: 0-small, 1-big;
            args: (scale_threshold)
    scale_location - Divide into 4 regions; order: 0-(left small), 1-(right small), 
                     2-(left big), 3-(right big); args: (scale_threshold)
    1big_2small - There are 3 regions, 1 big, 2 small. The small are divided into
                  left and right; order: 0-big, 1-(small, left), 2-(small, right)
    '''    
    if rtype == 'scale':      
      self.which_region = self.__which_region_scale
      args_needed = 1
      self.num_regions = 2
    elif rtype == 'scale_location':
      self.which_region = self.__which_region_scale_location
      args_needed = 1
      self.num_regions = 4
    elif rtype == '1big_2small':
      self.which_region = self.__which_region_1big_2small
      args_needed = 1
      self.num_regions = 3
    else:
      raise RuntimeError('Type %s is an unknown RegionModel'%rtype)
    
    if not len(args) == args_needed:
      raise RuntimeError("Not the right number of arguments for type %s. %d needed, %d given"%(rtype, args_needed, len(args)))
    else:
      self.args = args  
    
  def __which_region_scale(self, image, x, y, scale, aspect_ratio):
    scale_thresh = self.args[0]
    if scale < scale_thresh:
      result_region = 0
    else:
      result_region = 1
    return result_region
  
  def __which_region_scale_location(self, image, x, y, scale, aspect_ratio):
    scale_thresh = self.args[0]
    img_width, _ = image.size
    result_region = 0
    if scale >= scale_thresh:
      result_region += 2
    if x >= img_width/2:
      result_region += 1
    return result_region
  
  def __which_region_1big_2small(self, image, x, y, scale, aspect_ratio):
    '''
    For a window to be on the left side means to overlap with at least 50% with
    the left half of the window.
    '''
    scale_thresh = self.args[0]
    img_width, _ = image.size
    w = img_width*scale
    if scale >= scale_thresh:
      result_region = 0
    else:
      if img_width - 2*x - w > 0: # This does exactly mean more than half the win is left. Work it out.
        result_region = 1
      else:
        result_region = 2
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
    
  def convert_wh2scale(self, img, x, y, w, h):
    W, _ = img.size    
    scale = w/float(W)
    aspect_ratio = w/h
    return (scale, aspect_ratio)
  
  def filter_dets_for_reg_id(self, img, dets, region_id):
    """
    For a given Region ID and detections modify the table such that it holds
    those detections that fall in this region.
    """
    new_dets_arr = []
    for det in dets.arr:
      x = det[dets.cols.index('x')]
      y = det[dets.cols.index('y')]
      w = det[dets.cols.index('w')]
      h = det[dets.cols.index('h')]
      scale, aspect_ratio = self.convert_wh2scale(img, x, y, w, h)
      actual_reg_id = self.region_model.which_region(img, x, y, scale, aspect_ratio)
      if actual_reg_id == region_id:
        new_dets_arr.append(det)
    if len(new_dets_arr) == 0:
      new_dets_arr = np.empty((0, 0))
    else:
      new_dets_arr = np.vstack(new_dets_arr)
    dets.arr = new_dets_arr     
  
  def detect(self, image, region_id, astable=False):
    """
    Return the detections that match that image index in cached dets for a 
    specific region in the image that is determined by the region_id.
    Must return in the same format as the Detector superclass, so we have to
    delete a column.
    """
    img_ind = self.dataset.get_img_ind(image)
    dets = self.dets.filter_on_column('img_ind',img_ind,omit=True) # This function already creates a copy
    self.filter_dets_for_reg_id(image, dets, region_id) # At this position choose only the detections that fall into the desired region
    return ExternalDetector.detect(self, image, astable, dets) # Now call the regular external detector with just those detections
  
  def compute_score(self, image, region_id, oracle=False):
    """
    Return the 0/1 decision of whether the cls of this detector is present in
    the image, given the detections table for a given region id.
    If oracle=True, returns the correct answer (look up the ground truth).
    """
    img_ind = self.dataset.get_img_ind(image)
    dets = self.dets.filter_on_column('img_ind',img_ind,omit=True)
    self.filter_dets_for_reg_id(image, dets, region_id) # At this position choose only the detections that fall into the desired region    
    return ExternalDetector.compute_score(self, image, oracle, dets) # Now call the regular external detector with just those detections 
  
  def get_number_regions(self):
    return self.region_model.get_number_regions()
  

def run():
  dataset = Dataset('full_pascal_test')
  train_dataset = Dataset('full_pascal_trainval')
  cls = 'dog'
  rtype = '1big_2small'
  args = 0.5
  detector = 'csc_default'
  from synthetic.dataset_policy import DatasetPolicy
  all_dets = DatasetPolicy.load_ext_detections(dataset, detector)
  cls_ind = dataset.get_ind(cls)
  dets = all_dets.filter_on_column('cls_ind',cls_ind,omit=True)  
  ext_det = ExternalDetectorRegions(dataset, train_dataset, cls, dets, detector, rtype, args)
  img = dataset.images[13]  # Just some random image...where did the get_image_by_name go?
  print img.size
  print ext_det.detect(img, 0)
  print ext_det.detect(img, 1)
  print ext_det.detect(img, 2)

if __name__=='__main__':
  run()