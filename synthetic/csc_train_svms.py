from synthetic.common_imports import *
from synthetic.common_mpi import *

from synthetic.dataset import Dataset
from synthetic.dataset_policy import DatasetPolicy
from synthetic.ext_detector import ExternalDetector
from synthetic.csc_classifier import CSCClassifier
from IPython import embed
from synthetic.evaluation import Evaluation

def train_csc_svms(d_train, d_val, kernel, C):
  # d: trainval
  # d_train: train  |   trainval
  # d_val: val      |   test
  dp = DatasetPolicy(d_train, d_train, detectors=['csc_default'])
      
  for cls_idx in range(comm_rank, len(d_train.classes), comm_size):
    cls = d_train.classes[cls_idx]
    ext_detector = dp.actions[cls_idx].obj
    csc = CSCClassifier('default', cls, d_train, d_val)
    csc.train_for_cls(ext_detector, kernel, C)
    
def test_csc_svm(d_train, d_val): 
  dp = DatasetPolicy(d_val, d_train, detectors=['csc_default'])
  
  table = np.zeros((len(d_val.images), len(d_val.classes)))
  for cls_idx in range(comm_rank, len(d_val.classes), comm_size):
    cls = d_val.classes[cls_idx]
    ext_detector = dp.actions[cls_idx].obj
    # Load the classifier we trained in train_csc_svms
    csc = CSCClassifier('default', cls, d_train, d_val)
    embed()    
    table[:, cls_idx] = csc.eval_cls(ext_detector)
  print '%d_train is at safebarrier'%comm_rank
  safebarrier(comm)

  print 'passed safebarrier'
  table = comm.reduce(table, op=MPI.SUM, root=0)
  if comm_rank == 0:
    print 'save table'
    print table 
    cPickle.dump(table, open('table','w'))
    print 'saved'
  return table

def conv(d_train, table_arr):
  table = ut.Table()
  #table_arr = cPickle.load(open('table_linear_5','r'))
  table.arr = np.hstack((table_arr, np.array(np.arange(table_arr.shape[0]),ndmin=2).T))
  table.cols = d_train.classes + ['img_ind']
  print table
  #cPickle.dump(table, open('tab_linear_5','w'))
  return table
  
if __name__=='__main__':
  d_train = Dataset('full_pascal_train')
  d_val = Dataset('full_pascal_val')

  train_gt = d_train.get_cls_ground_truth()
  val_gt = d_val.get_cls_ground_truth()
  
  kernel = 'linear'
  C = 100

  train_csc_svms(d_train, d_val, kernel, C)
  table_arr = test_csc_svm(d_train, d_val)
  
  if comm_rank == 0:
    #embed()
    table = conv(d_train, table_arr)
    res = Evaluation.compute_cls_map(table, val_gt)
    print res
