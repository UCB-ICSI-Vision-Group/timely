from synthetic.dataset import Dataset
from synthetic.extractor import Extractor
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
  e = Extractor()
  all_classes = config.pascal_classes
#  for cls_idx in range(mpi_rank, len(all_classes), mpi_size): # PARALLEL
#  #for cls in all_classes:
#    cls = all_classes[cls_idx]
#    print cls
    #d, feature_type, num_words=3000,iterations=10, force_new=False, kmeansBatch=True
  e.get_codebook(d, feature_type, numpos, iterations, force_new=False, kmeansBatch=True)