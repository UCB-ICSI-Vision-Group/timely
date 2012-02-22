from synthetic.common_imports import *
from synthetic.common_mpi import *


def plausible_assignments(assignments):
  return np.absolute(assignments - np.random.random(assignments.shape)/4.)

def correct_assignments(assignments):
  classif = np.zeros(assignments.shape)
  for i in range(assignments.size):
    None
  None

def create_meassurement_table(num_clss, func):
  """
  Create table containing all meassurements in format
     __________
    /          \ 
   A --- B --- C
   |     |     |
   1     2     3
  """
  table = np.zeros((2**num_clss, num_clss*2))
  
  # do a binary counter to fill up this table (as a ripple counter)
  assignments = np.array(np.zeros((num_clss,)))
  for i in range(2**num_clss):
    classif = func(assignments)
    table[i,:] = np.hstack((assignments, classif))
    
    # done?
    go_on = not np.sum(assignments) == num_clss
    # count up
    assignments[-1] += 1    
    # propagate bit up    
    pos = 1
    while go_on:
      if assignments[-pos] == 2:
        assignments[-pos] = 0
        pos += 1
        assignments[-pos] += 1
      else:
        go_on = False      
   
  return table
  
if __name__=='__main__':
  num_clss = 3
  table = create_meassurement_table(num_clss, plausible_assignments)
  print table
  