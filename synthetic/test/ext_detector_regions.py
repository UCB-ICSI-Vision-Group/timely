from common_imports import *
import synthetic.config as config

from synthetic.ext_detector_regions import *
from synthetic.image import Image
from synthetic.dataset_policy import DatasetPolicy

class TestExtDetector(object):
  def setup(self):
    None
    
  def test_region_models(self):  
    img = Image(640, 480)
    
    # First the easy scale model
    rm = RegionModel('scale', 0.5)
    assert(2 == rm.get_number_regions())    
    assert(0 == rm.which_region(img, 100, 200, 0.4, 2))
    assert(1 == rm.which_region(img, 100, 220, 0.6, 2))
    
    # Now the scale and location model. The order of the region ids is: 
    # 0-(left small), 1-(right small), 2-(left big), 3-(right big)    
    rm2 = RegionModel('scale_location', 0.3)
    assert(4 == rm2.get_number_regions())
    assert(0 == rm2.which_region(img, 300, 100, 0.29, 1))
    assert(1 == rm2.which_region(img, 340, 100, 0.29, 1))
    assert(2 == rm2.which_region(img, 300, 100, 0.31, 1))
    assert(3 == rm2.which_region(img, 340, 100, 0.31, 1))
    
  def test_filter_dets(self):
    dataset = Dataset('full_pascal_test')
    train_dataset = Dataset('full_pascal_trainval')
    cls = 'dog'
    rtype = 'scale_location'
    args = 0.5
    detector = 'csc_default'
    all_dets = DatasetPolicy.load_ext_detections(dataset, detector)
    cls_ind = dataset.get_ind(cls)
    dets = all_dets.filter_on_column('cls_ind',cls_ind,omit=True)  
    ext_det = ExternalDetectorRegions(dataset, train_dataset, cls, dets, detector, rtype, args)
    img = dataset.images[133]  # Just some random image...where did the get_image_by_name go?
    split_x = img.size[0]/2
    split_scale = img.size[0]*args
    dets = ext_det.detect(img, 0)
    for det in dets[0]: # left, small      
      assert(det[0] < split_x)
      assert(det[2] < split_scale)
    dets = ext_det.detect(img, 1)
    for det in dets[0]: # right, small      
      assert(det[0] >= split_x)
      assert(det[2] < split_scale)
    dets = ext_det.detect(img, 2)
    for det in dets[0]: # left, big
      assert(det[0] < split_x)
      assert(det[2] >= split_scale)
    dets = ext_det.detect(img, 3)
    for det in dets[0]: # right, big
      assert(det[0] >= split_x)
      assert(det[2] >= split_scale)    
    
    # and those we just run to check that nothing is syntactically going wrong 
    # (no errors/...)
    ext_det.compute_score(img, 0)
    ext_det.compute_score(img, 1)
    ext_det.compute_score(img, 2)
    ext_det.compute_score(img, 3)