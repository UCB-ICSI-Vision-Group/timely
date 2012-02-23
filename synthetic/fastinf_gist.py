from synthetic.common_imports import *
from synthetic.common_mpi import *

import synthetic.config as config
from synthetic.fastInf import *
from synthetic.gist_detector import GistPriors
from synthetic.dataset import Dataset
import cPickle

def create_gist_model_for_dataset(dataset):
  d = Dataset(dataset)
  gist = GistPriors(dataset)
  
  images = d.images
  
  table = np.zeros((len(images), len(d.classes)))
  
  if comm_rank == 0:
    t = ut.TicToc()
    t.tic()
  # Some map reduce here!
  for idx in range(comm_size, len(images), comm_rank):
    img = images[idx]
    print 'classify image %s on %d'%(img.name, comm_rank)
    classif = gist.get_priors(img)
    table[idx, :] = np.array(classif)
    
  safebarrier(comm)
  if comm_rank == 0:
    print 'computing table took %f seconds'%t.toc(quiet=True)
  comm.Allreduce(table, table)
    
  return table  

def discretize_table(table, num_bins, asInt=True):
  new_table = np.zeros(table.shape)
  for coldex in range(table.shape[1]):
    col = table[:, coldex]
    bounds = ut.importance_sample(col, num_bins+1)
    
    # determine which bin these fall in
    col_bin = np.zeros((table.shape[0],1))
    bin_values = np.zeros(bounds.shape)
    last_val = 0.
    for bidx, b in enumerate(bounds):
      bin_values[bidx] = (last_val + b)/2.
      last_val = b
      col_bin += np.matrix(col < b, dtype=int).T
    bin_values = bin_values[1:]    
    col_bin[col_bin == 0] = 1  
    if asInt:
      a = col_bin-1
      new_table[:, coldex] = a[:,0] 
    else:    
      for rowdex in range(table.shape[0]):
        new_table[rowdex, coldex] = bin_values[int(col_bin[rowdex]-1)]
  if asInt:    
    return new_table.astype(int)
  else:
    return new_table
  
def create_tables():
  datasets = ['full_pascal_trainval','full_pascal_test','full_pascal_train','full_pascal_val']
  for dataset in datasets:
    table = create_gist_model_for_dataset(dataset)
    savefile = config.get_fastinf_data_file(dataset)
    cPickle.dump(table, open(savefile, 'w'))
    print table
    print table.shape

if __name__=='__main__':
  d = Dataset('test_pascal_train_tobi')
  table_gt = d.get_cls_ground_truth().arr.astype(int)
  
  # replace this
  table = np.random.random((table_gt.shape[0],20))
  new_table = discretize_table(table, 5)
  
  print new_table
  
  #print table
  
  
  
    