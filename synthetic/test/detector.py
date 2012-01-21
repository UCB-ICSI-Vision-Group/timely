import os

from synthetic.dataset import Dataset
from synthetic.image import Image
from synthetic.detector import *

class TestDetector:
  def test_nms(self):
    """
    The test is actually in TestPolicy::load_dpm_detections().
    """

  def test_subclass(self):
    dataset = Dataset('test_pascal_val')
    d = Detector(dataset,'dog')
    assert(isinstance(d,Detector))

class TestPerfectDetector:
  def test_expected_time(self):
    dataset = Dataset('test_pascal_val')
    d = Detector(dataset,'dog')
    img = Image(size=(640,480))
    print d.expected_time(img)
    assert(d.expected_time(img) == 10)
    img = Image(size=(320,480))
    assert(d.expected_time(img) == 5)

  def test_subclass(self):
    dataset = Dataset('test_pascal_val')
    d = Detector(dataset,'dog')
    assert(isinstance(d,Detector))

