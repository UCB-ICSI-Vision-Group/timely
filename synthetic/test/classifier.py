from synthetic.common_imports import *
from synthetic.classifier import Classifier
from synthetic.dataset import Dataset
import synthetic.config as config

class TestClassifier:
  def __init__(self):
    self.clf = Classifier()
    self.d = Dataset('full_pascal_trainval')
    
  def test_load_svm(self):
    self.clf.name = 'csc'
    self.clf.suffix = 'default'
    self.clf.cls = 'dog'
    self.clf.train_dataset = self.d
    self.clf.load_svm()

    