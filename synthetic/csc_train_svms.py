from synthetic.common_imports import *
from synthetic.common_mpi import *

from synthetic.dataset import Dataset
from synthetic.dataset_policy import DatasetPolicy
from synthetic.ext_detector import ExternalDetector
from synthetic.csc_classifier import CSCClassifier


def retrain_best_svms():
  d = Dataset('full_pascal_trainval')
  dp = DatasetPolicy(d, d, detectors=['csc_default'])  
  
  kernels = ['linear', 'rbf']  
  num_binss = [5]#,10,20,50]
  Cs = [1.]#, 2, 5, 10]
  settings = list(itertools.product(Cs, range(len(d.classes)), num_binss, kernels))
  
#  kernels = ['chi2']
#  num_binss = [20]
#  settings += list(itertools.product(Cs, range(len(d.classes)), num_binss, kernels))
  
  for set_idx in range(comm_rank, len(settings), comm_size):
    settin = settings[set_idx]
    C = settin[0]
    cls_idx = settin[1]
    num_bins = settin[2]    
    kernel = settin[3]
    cls = d.classes[cls_idx]    
    dets = dp.actions[cls_idx].obj.dets           
    csc = CSCClassifier('default', cls, d, num_bins)
    csc.train_for_all_cls(d, dets, kernel, cls_idx, C, probab=True)     
  
if __name__=='__main__':
  d = Dataset('full_pascal_trainval')
  retrain_best_svms()
  #create_csc_stuff(d, classify_images=False, force_new=False)
