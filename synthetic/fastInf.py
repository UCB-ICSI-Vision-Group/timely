from synthetic.common_imports import *
from synthetic.common_mpi import *

import synthetic.config as config
import subprocess as subp

from synthetic.dataset import Dataset
from synthetic.csc_classifier import create_csc_stuff
from synthetic.gist_classifier import cls_for_dataset
import argparse

# TODO: why are these two needed?
def plausible_assignments(assignments):
  return np.absolute(assignments - np.random.random(assignments.shape)/3.)

class FastinfDiscretizer(object):
  def __init__(self,d,suffix):
    # For a given setting return bounds as num_bins x num_cols
    self.d = d
    self.bounds = np.loadtxt(config.get_mrf_bound_filename(d, suffix))

  def discretize_value(self, val):
    """
    For d, suffix discretize val for all 20 classes. 
    Returns (20,) array
    """
    # TODO: why for all classes??????
    discr_val = determine_bin(np.tile(val, (1,len(self.d.classes)))[0,:], self.bounds, self.bounds.shape[0]-1, asInt=True)
    return discr_val.astype(int)
  
def determine_bin(col, bounds, num_bins, asInt=True):
  """ 
  Determine in which bin the values fall
  """
  ret_tab = np.zeros((col.shape[0],1))
  col_bin = np.zeros((col.shape[0],1))
  bin_values = np.zeros(bounds.shape)
  last_val = 0.
  
  for bidx, b in enumerate(bounds):
    bin_values[bidx] = (last_val + b)/2.
    last_val = b
    col_bin += np.matrix(col < b, dtype=int).T
  bin_values = bin_values[1:]    
  col_bin[col_bin == 0] = 1  
  if asInt:
    a = num_bins - col_bin
    ret_tab = a[:,0] 
  else:    
    for rowdex in range(col.shape[0]):
      ret_tab[rowdex, 0] = bin_values[int(col_bin[rowdex]-1)]
  return ret_tab

def discretize_table(table, num_bins, asInt=True):
  """
  discretize the given table and also return the bounds for the column 
  discretization. as num_bins x num_cols
  """
  all_bounds = np.zeros((num_bins+1, table.shape[1]))
  new_table = np.zeros(table.shape)
  
  for coldex in range(table.shape[1]):
    col = table[:, coldex]
     
    if np.where(col==col[0])[0].shape[0] == col.shape[0]:
      bounds = (np.arange(num_bins+1)/num_bins)
    else:
      bounds = ut.importance_sample(col, num_bins+1)
    all_bounds[:, coldex] = bounds
    ut.keyboard()
    
    new_table[:, coldex] = determine_bin(col, bounds, num_bins, asInt)
  if asInt:    
    return (all_bounds, new_table.astype(int))
  else:
    return (all_bounds, new_table)

def write_out_mrf(table, num_bins, filename, data_filename, second_table=None, pairwise=True):
  """
  Again we assume the table to be of the form displayed below.
  """
   
  num_vars = table.shape[1]/2
  wm = open(filename, 'w')
  modelfile = config.get_mrf_model(num_vars)
  print modelfile, os.path.exists(modelfile)
  # TODO!
  if True or not os.path.exists(modelfile):
    #===========
    #= Model
    #===========  
    # ===========Variables==========
    wm.write('@Variables\n')
    for i in range(num_vars):
      wm.write('var%d\t2\n'%i)
    for i in range(num_vars):
      wm.write('var%d\t%d\n'%(i+num_vars, num_bins))
      
    if not second_table == None:
      for i in range(num_vars):
        wm.write('var%d\t%d\n'%(i+2*num_vars, num_bins))    
    wm.write('@End\n')
    wm.write('\n')
    
    # ===========Cliques============
    wm.write('@Cliques\n')
    if not pairwise:
      # top clique:
      wm.write('cl0\t%d'%num_vars)
      wm.write('\t')
      for i in range(num_vars):
        wm.write(' %d'%i)
      wm.write('\t%d\t'%num_vars)
      for i in range(num_vars):
        wm.write(' %d'%(i+1))
      wm.write('\n')
    else:
      combs = list(itertools.combinations(range(num_vars), 2))
      num_combs = len(combs)    
      for idx, comb in enumerate(combs):
        # neighboring cliques:
        neighs = []
        for i, c in enumerate(combs):
          if not c==comb:
            if c[0] in comb or c[1] in comb:
              neighs.append(i)
  
        wm.write('cl%d\t2\t%d %d\t%d\t'%(idx, comb[0], comb[1], len(neighs)))
        for n in neighs:
          wm.write('%d '%n)
        wm.write('\n')
    #pairwise cliques
    for l in range(num_vars):
      neighs = []
      for i, c in enumerate(combs):
        #if not c==[i, i+num_vars]:
        if l in c:
          neighs.append(i)
      wm.write('cl%d\t2\t%d %d\t%d\t'%(l+1+idx, l, l+num_vars, len(neighs)))
      for n in neighs:
        wm.write('%d '%n)
      wm.write('\n')
      
    if not second_table == None:
      for l in range(num_vars):
        neighs = []
        for i, c in enumerate(combs):
          #if not c==[i, i+num_vars]:
          if l in c:
            neighs.append(i)
        wm.write('cl%d\t2\t%d %d\t%d\t'%(l+1+idx+num_vars, l, l+2*num_vars, len(neighs)))
        for n in neighs:
          wm.write('%d '%n)
        wm.write('\n')
    wm.write('@End\n')
    wm.write('\n')
    num_cliques = l+2+idx
    if not second_table == None:
      num_cliques += num_vars
    print num_cliques
      
    # ===========Measures==========
    # Well, there is a segfault if these are empty :/
    wm.write('@Measures\n')
    if not pairwise:
      wm.write('mes0\t%d\t'%(num_vars))
      for _ in range(num_vars):
        wm.write('2 ')
      wm.write('\t')
      for _ in range(2**num_vars):
        wm.write('.1 ')
      wm.write('\n')
    else:
      for j in range(num_combs):
        wm.write('mes%d\t2\t2 2\t.1 .1 .1 .1\n'%j)
      
    for i in range(num_vars):
      wm.write('mes%d\t2\t2 %d'%(i+j+1, num_bins))
      wm.write('\t')
      for _ in range(num_bins*2):
        wm.write('.1 ')
      wm.write('\n')
    if not second_table == None:
      for i in range(num_vars):
        wm.write('mes%d\t2\t2 %d'%(i+j+1+num_vars, num_bins))
        wm.write('\t')
        for _ in range(num_bins*2):
          wm.write('.1 ')
        wm.write('\n')  
    wm.write('@End\n')
    wm.write('\n')
    
    # ===========CliqueToMeasure==========
    wm.write('@CliqueToMeasure\n')
    for i in range(num_cliques):
      wm.write('%(i)d\t%(i)d\n'%dict(i=i))  
    wm.write('@End\n')
    
    wm.close()
    
    # copy to modelfile
    os.system('cp %s %s'%(filename,modelfile))
  else:
    print 'load model...'
    # copy from modelfile
    os.system('cp %s %s'%(modelfile,filename))
  
  print 'reformat data'
  #===========
  #= Data
  #===========
  wd = open(data_filename, 'w')
  if not second_table == None:
    table = np.hstack((table, second_table))
  for rowdex in range(table.shape[0]):
    wd.write('( ')
    for i in range(table.shape[1]):
      wd.write('%.2f '%table[rowdex, i])    
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

def execute_lbp(filename_mrf, filename_data, filename_out, add_settings=[]):
  cmd = ['../fastInf/build/bin/learning', '-i', filename_mrf, 
                         '-e', filename_data, '-o', filename_out] + add_settings

  timefile = filename_out+'_time'
  tt = ut.TicToc()
  tt.tic()        
  cmd = ' '.join(cmd)
  ut.run_command(cmd)
  w = open(timefile, 'w')
  w.write('%s\ntook %f sec'%(cmd, tt.toc(quiet=True)))
  w.close()
  print 'everything done'
  return 

def c_corr_to_a(num_lines, func):
  assignment = np.zeros((3,))
  table = np.zeros((num_lines, 6))
  for i in range(num_lines):
    rand = np.random.random((4,))
    assignment[0] = rand[0] > .7
    assignment[1] = rand[1] > .5
    if rand[2] > 0.2:
      assignment[2] = assignment[0]
    else:
      assignment[2] = rand[3] > .5
    
    classif = func(assignment)
    table[i,:] = np.hstack((assignment, classif))
  return table

def store_bound(d, suffix, bounds):
  bound_file = config.get_mrf_bound_filename(d, suffix)
  if not os.path.exists(bound_file):
    np.savetxt(bound_file, bounds)  

def create_gist_model_for_dataset(d):
  dataset = d.name
  return cls_for_dataset(dataset)

def run_fastinf_different_settings(dataset, ms, rs, suffixs):
  d = Dataset(dataset)
  num_bins = 5
  settings = list(itertools.product(suffixs, ms, rs))
  table_gt = d.get_cls_ground_truth().arr.astype(int)
  print 'run with a total of %d settings'%len(settings)
  
  for setindx in range(comm_rank, len(settings), comm_size):
    second_table = None
    setin = settings[setindx]
    suffix = setin[0]
    m = setin[1]
    r2 = setin[2]
    
    print 'node %d runs %s, m=%s, r2=%s'%(comm_rank, suffix, m, r2)

    filename = config.get_fastinf_mrf_file(d, suffix)
    data_filename = config.get_fastinf_data_file(d, suffix)
    
    if suffix == 'perfect':      
      table = np.hstack((table_gt, table_gt))
      
    elif suffix == 'GIST':
      table = create_gist_model_for_dataset(d)      
      bounds, discr_table = discretize_table(table, num_bins)  
      table = np.hstack((table_gt, discr_table))
      
    elif suffix == 'CSC':
      create_csc_stuff(d)
      filename_csc = os.path.join(config.get_ext_dets_foldname(d),'table')
      table = cPickle.load(open(filename_csc,'r'))
      bounds, discr_table = discretize_table(table, num_bins)
      table = np.hstack((table_gt, discr_table))
      
    elif suffix == 'GIST_CSC':
      create_csc_stuff(d)
      filename_csc = os.path.join(config.get_ext_dets_foldname(d),'table')
      table = cPickle.load(open(filename_csc,'r'))
      bounds, discr_table = discretize_table(table, num_bins)      
      table = np.hstack((table_gt, discr_table))
      store_bound(d, 'CSC', bounds)
      
      second_table = create_gist_model_for_dataset(d)      
      sec_bounds, second_table = discretize_table(second_table, num_bins)      
      store_bound(d, 'GIST', sec_bounds)  
    
    if suffix == 'GIST' or suffix == 'CSC':
      store_bound(d, suffix, bounds)
    
    print 'set up table on %d, write out mrf for %s, m=%s, r2=%s'%(comm_rank, suffix, m, r2)   
      
    write_out_mrf(table, num_bins, filename, data_filename, second_table=second_table)
    
    add_settings = ['-m',m]
    if not r2 == '':
      add_settings += ['-r2', r2]
          
    if not second_table == None:
      sec_bound_file = '%s_secbounds'%filename
      for s in add_sets:
        sec_bound_file += '_'+s
      np.savetxt(sec_bound_file, sec_bounds)
      
    print '%d start running lbp for %s, m=%s, r2=%s'%(comm_rank, suffix, m, r2)
    
    filename_out = config.get_fastinf_res_file(d, suffix, m, r2)
    execute_lbp(filename, data_filename, filename_out, add_settings=add_sets)

def run_all_in_3_parts():
    
  # I run 3 different experiments to be able to abort them separately...
  # that a total of 48 experiments
  parser = argparse.ArgumentParser(
    description="Run fastInf experiments.")

  parser.add_argument('-e',type=int,
    default=0,
    choices=[0,1,2],
    help="""Select which portion of the training is to be run.""")
  
  args = parser.parse_args()
  
  part = args.e
  
  suffixs = ['CSC', 'GIST_CSC', 'perfect', 'GIST']
  ms = ['0', '2', '5']
  rs = ['', '0.5', '1']
  
  if part == 0:
    dataset = 'full_pascal_trainval'
    suffixs = ['CSC', 'GIST_CSC']    
  elif part == 1:
    dataset = 'full_pascal_trainval'
    suffixs = ['perfect', 'GIST']    
  elif part == 2:
    dataset = 'full_pascal_train'
    suffixs = ['CSC', 'GIST_CSC']
    rs = ['', '1']
  print 'Execute fastInf part %d'%part
  print '\tds:', dataset
  print '\tsuff:', suffixs
  print '\tms:', ms
  print '\trs:', rs
     
  run_fastinf_different_settings(dataset, ms, rs, suffixs)
  
def write_out_perfect_bounds():
  None

if __name__=='__main__':
  #run_all_in_3_parts()
  dataset = 'full_pascal_trainval'
  d = Dataset(dataset)
  suffix = 'GIST'
  print discretize_value(.2242, d, suffix)
  