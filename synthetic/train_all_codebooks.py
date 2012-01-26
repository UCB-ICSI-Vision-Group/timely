from synthetic.dataset import Dataset
from synthetic.training import get_codebook
from mpi4py import MPI
import synthetic.config as config 

comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()

if __name__=='__main__':
  d = Dataset('full_pascal_trainval')
  feature_type = 'dsift'
  numpos = 15
  num_words = 3000
  iterations = 8

  all_classes = config.pascal_classes
  for cls_idx in range(mpi_rank, len(all_classes), mpi_size): # PARALLEL
  #for cls in all_classes:
    cls = all_classes[cls_idx]
    print cls
    get_codebook(d, numpos, num_words, feature_type, cls, iterations, force_new=False,\
                 use_neg=True, kmeansBatch=True)