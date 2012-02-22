from synthetic.common_imports import *
from synthetic.common_mpi import *
import synthetic.config as config


def plausible_assignments(assignments):
  return np.absolute(assignments - np.random.random(assignments.shape)/4.)

def correct_assignments(assignments):
  classif = np.zeros(assignments.shape)
  for i in range(assignments.size):
    # classify each
    None
  None

def discretize_table(table, num_bins):
  d_table = np.hstack((table[:,:table.shape[1]/2],np.floor(np.multiply(table[:, table.shape[1]/2:],num_bins)))) 
  return d_table

def write_out_mrf(table, num_bins, filename, data_filename):
  """
  Again we assume the table to be of the form displayed below.
  """
  num_vars = table.shape[1]/2
  wm = open(filename, 'w')
   
  #===========
  #= Model
  #===========  
  # ===========Variables==========
  wm.write('@Variables\n')
  for i in range(num_vars):
    wm.write('var%d\t2\n'%i)
  for i in range(num_vars):
    wm.write('var%d\t%d\n'%(i+num_vars, num_bins))
  wm.write('@End\n')
  wm.write('\n')
  
  # ===========Cliques============
  wm.write('@Cliques\n')
  # top clique:
  wm.write('cl0\t%d'%num_vars)
  wm.write('\t')
  for i in range(num_vars):
    wm.write(' %d'%i)
  wm.write('\t%d\t'%num_vars)
  for i in range(num_vars):
    wm.write(' %d'%(i+1))
  wm.write('\n') 
  #pairwise cliques
  for i in range(num_vars):
    wm.write('cl%d\t2\t%d %d\t1\t0\n'%(i+1, i, i+num_vars))  
  wm.write('@End\n')
  wm.close()
  
  #===========
  #= Data
  #===========
  wd = open(data_filename, 'w')
  for rowdex in range(table.shape[0]):
    wd.write('( ')
    for i in range(table.shape[1]):
      wd.write('%d '%table[rowdex, i])    
    wd.write(')\n')
  wd.close()
  
def create_meassurement_table(num_clss, func):
  """
  Create table containing all measurements in format
     __________
    /          \ 
   A --- B --- C
   |     |     |
   1     2     3
   
  => [A, B, C, 1, 2, 3]
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
  num_bins = 8
  filename = config.get_fast_inf_mrf_file()
  data_filename = config.get_fast_inf_data_file()
  
  table = create_meassurement_table(num_clss, plausible_assignments)
  d_table = discretize_table(table, num_bins)
  write_out_mrf(d_table, num_bins, filename, data_filename)
  