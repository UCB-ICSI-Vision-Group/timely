from synthetic.common_imports import *
from synthetic.csc_classifier import CSCClassifier
from synthetic.dataset import Dataset

class TestCscClassifier:
  def __init__(self):
    self.d = Dataset('full_pascal_trainval')
    cls = 'dog'
    suffix = 'default'
    self.csc = CSCClassifier(suffix, cls, self.d)
    
    
  def test_classify_image(self):
    print self.csc.classify_image(0)
    print self.csc.classify_image(self.d.images[0])

if __name__=='__main__':
  tester = TestCscClassifier()
  tester.test_classify_image()
