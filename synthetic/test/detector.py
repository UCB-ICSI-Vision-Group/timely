import os

from synthetic.dataset import Dataset
from synthetic.image import Image
from synthetic.sliding_windows import SlidingWindows
from synthetic.detector import *

class TestDetector:
  def test_nms(self):
    """
    The test is actually in TestPolicy::load_dpm_detections().
    """

  def test_subclass(self):
    dataset = Dataset('test_pascal_val')
    train_dataset = Dataset('test_pascal_train')
    sw = SlidingWindows(dataset,train_dataset)
    d = Detector(dataset,'dog',sw)
    assert(isinstance(d,Detector))

class TestPerfectDetector:
  def test_expected_time(self):
    dataset = Dataset('test_pascal_val')
    train_dataset = Dataset('test_pascal_train')
    sw = SlidingWindows(dataset,train_dataset)
    d = Detector(dataset,'dog',sw)
    img = Image(size=(500,375))
    print d.expected_time(img)
    assert(d.expected_time(img) == 10)
    img = Image(size=(250,375))
    assert(d.expected_time(img) == 5)

  def test_subclass(self):
    dataset = Dataset('test_pascal_val')
    train_dataset = Dataset('test_pascal_train')
    sw = SlidingWindows(dataset,train_dataset)
    d = Detector(dataset,'dog',sw)
    assert(isinstance(d,Detector))

