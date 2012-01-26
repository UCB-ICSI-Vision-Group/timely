from mpi4py import MPI
from synthetic.safebarrier import safebarrier
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()