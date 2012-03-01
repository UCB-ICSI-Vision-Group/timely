from synthetic.common_imports import *
from synthetic.classifier import Classifier
from synthetic.csc_classifier import CSCClassifier
from synthetic.dataset import Dataset
import synthetic.config as config

class TestClassifier:
  def test_compute_histogram(self):
    dataset = 'full_pascal_trainval'
    d = Dataset(dataset)
    cls = 'dog'
    cls_idx = d.classes.index(cls)
    csc = CSCClassifier('default', cls, d)
    for img in range(50):
      #img = 1
      image = d.images[img]
      if image.get_cls_counts()[cls_idx] == 0:
        continue
      filename = config.get_ext_dets_filename(d, 'csc_default')
      csc_test = np.load(filename)
      dets = csc_test[()]
      dets = dets.filter_on_column('cls_ind', d.classes.index(cls), omit=True)
      dets = dets.subset(['score', 'img_ind'])
      dets.arr = csc.normalize_dpm_scores(dets.arr)
      img_dpm = dets.filter_on_column('img_ind', img, omit=True)
      
      hist = csc.compute_histogram(img_dpm.arr, csc.intervals, csc.lower, csc.upper)
      vector = np.zeros((1, csc.intervals+1))
      vector[0,0:-1] = hist
      vector[0,-1] = img_dpm.shape()[0]
      print vector

    
if __name__=='__main__':
  tester = TestClassifier()
  tester.test_compute_histogram()

    