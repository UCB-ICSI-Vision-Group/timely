from synthetic.common_imports import *
from synthetic.common_mpi import *

from synthetic.dataset import Dataset
from synthetic.dataset_policy import DatasetPolicy
from synthetic.ext_detector import ExternalDetector
from synthetic.csc_classifier import CSCClassifier
from IPython import embed
from synthetic.evaluation import Evaluation

def retrain_best_svms(d_train, kernel, C, num_bins):
  
  dp = DatasetPolicy(d_train, d_train, detectors=['csc_default'])  
  
  table = np.zeros((len(d_train.images), len(d_train.classes)))
  for cls_idx in range(comm_rank, len(d_train.classes), comm_size):
    cls = d_train.classes[cls_idx]    
    dets = dp.actions[cls_idx].obj.dets           
    csc = CSCClassifier('default', cls, d_train, num_bins)
    col = csc.train_for_cls(d_train, dets, kernel, C, probab=True)
    table[:, cls_idx] = col[:,0]
  
  print '%d_train is at safebarrier'%comm_rank
  safebarrier(comm)
  print 'passed safebarrier'
  table = comm.reduce(table, op=MPI.SUM, root=0)
  if comm_rank == 0:
    print 'save table'
    print table 
    #cPickle.dump(table, open('table_linear_5','w'))
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
  results_filename = 'results.txt'
  w = open(results_filename, 'w')
  kernels = ['linear', 'rbf']
  Cs = [1, 5, 10]
  num_binss = [5, 10, 20]
  settings = list(itertools.product(kernels, Cs, num_binss))
  
  for sets in settings:
    kernel = sets[0]
    C = sets[1]
    num_bins = sets[2]
    
    table_arr = retrain_best_svms(d_train, kernel, C, num_bins)
    if comm_rank == 0:
      table = conv(d_train, table_arr)
      res = Evaluation.compute_cls_map(table, train_gt)
      w.write('%s %f %d_train - %f\n'%(kernel, C, num_bins, res))
    
  #create_csc_stuff(d_train, classify_images=False, force_new=False)
