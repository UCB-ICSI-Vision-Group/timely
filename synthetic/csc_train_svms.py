from synthetic.common_imports import *
from synthetic.common_mpi import *

from synthetic.dataset import Dataset
from synthetic.dataset_policy import DatasetPolicy
from synthetic.ext_detector import ExternalDetector
from synthetic.csc_classifier import CSCClassifier
from IPython import embed
from synthetic.evaluation import Evaluation

def train_csc_svms(d, d_train, d_val, kernel, C, num_bins):
  detector = 'csc_default'
  # d: trainval
  # d_train: train  |   trainval
  # d_val: val      |   test
  dp = DatasetPolicy(d, d_train, detectors=[detector])
  test_dets = dp.load_ext_detections(d_val, detector)
    
  table = np.zeros((len(d_val.images), len(d_val.classes)))
  for cls_idx in range(comm_rank, len(d_train.classes), comm_size):
    cls = d_train.classes[cls_idx]
    dets = dp.actions[cls_idx].obj.dets
    #dets = dp.actions[cls_idx].obj.detect(image)
    csc = CSCClassifier('default', cls, d, num_bins)
    test_det_cls = test_dets.filter_on_column('cls_ind', cls_idx)
    col = csc.train_for_cls(d_train, d_val, dets, test_det_cls, kernel, C, probab=True, vtype='max') # <----IMPORTANT LINE
    table[:, cls_idx] = col[:,0]
  
  print '%d_train is at safebarrier'%comm_rank
  safebarrier(comm)
  print 'passed safebarrier'
  table = comm.reduce(table, op=MPI.SUM, root=0)
  if comm_rank == 0:
    print 'save table'
    print table 
    cPickle.dump(table, open('table_poly','w'))
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
  d = Dataset('full_pascal_trainval')
#  d_train = Dataset('full_pascal_train')
#  d_val = Dataset('full_pascal_val')
  
  d_train = Dataset('full_pascal_trainval')
  d_val = Dataset('full_pascal_test')

  train_gt = d_train.get_cls_ground_truth()
  val_gt = d_val.get_cls_ground_truth()

  if comm_rank == 0:
    results_filename = 'results.txt'
    w = open(results_filename, 'a')
#  kernels = ['linear', 'rbf']
#  Cs = [1, 5, 10]
#  num_binss = [5, 10, 20]
#  kernels = ['linear', 'rbf']
#  Cs = [1, 5, 10]
#  num_binss = [20, 50, 100]
  kernels = ['linear']
  Cs = [100]
  num_binss = [1]

  settings = list(itertools.product(kernels, Cs, num_binss))
  
  for sets in settings:
    kernel = sets[0]
    C = sets[1]
    num_bins = sets[2]
    
    table_arr = train_csc_svms(d, d_train, d_val, kernel, C, num_bins)
    if comm_rank == 0:
      table = conv(d_train, table_arr)
      res = Evaluation.compute_cls_map(table, val_gt)
      print res
      w.write('%s %f %d_train - %f\n'%(kernel, C, num_bins, res))
  
  if comm_rank == 0:
    w.close()
  #create_csc_stuff(d_train, classify_images=False, force_new=False)
