from common_imports import *
import synthetic.config as config

from synthetic.ext_detector_regions import *
from synthetic.image import Image

class TestExtDetector(object):
  def setup(self):
    None
    
  def test_region_models(self):  
    img = Image(640, 480)
    
    # First the easy scale model
    rm = RegionModel('scale', 0.5)
    assert(2 == rm.get_number_regions())    
    assert(0 == rm.which_region(img.size(), 100, 200, 0.4, 2))
    assert(1 == rm.which_region(img.size(), 100, 220, 0.6, 2))
    
    # Now the scale and location model. The order of the region ids is: 
    # 0-(left small), 1-(right small), 2-(left big), 3-(right big)    
    rm2 = RegionModel('scale_location', 0.3)
    assert(4 == rm2.get_number_regions())
    assert(0 == rm2.which_region(img.size(), 300, 100, 0.29, 1))
    assert(1 == rm2.which_region(img.size(), 340, 100, 0.29, 1))
    assert(2 == rm2.which_region(img.size(), 300, 100, 0.31, 1))
    assert(3 == rm2.which_region(img.size(), 340, 100, 0.31, 1))
    
