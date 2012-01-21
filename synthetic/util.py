import subprocess
import time, types, os, operator
import numpy as np
import scipy.stats as st
from IPython.parallel import Client

class Table:
  """An ndarray with associated column names."""

  ###################
  # Init/Copy/Repr
  ###################
  def __init__(self,arr=None,cols=None,name=None):
    """
    If arr and cols are passed in, initialize with them by reference. 
    If nothing is passed in, set them to None.
    name is just a place to keep some identifying information about this data.
    """
    self.arr = arr
    self.cols = cols
    self.name = name

  def __deepcopy__(self):
    """Make a deep copy of the Table and return it."""
    ret = Table()
    ret.arr = self.arr.copy() if not self.arr == None else None
    ret.cols = list(self.cols) if not self.cols == None else None
    return ret

  def __repr__(self):
    return "Table:\n  name: %s\n  cols: \n%s\n  arr: \n%s"%(self.name,self.cols,self.arr)

  def shape(self):
    return self.arr.shape

  ###################
  ### Save/Load
  ###################
  def save_csv(self,filename):
    """Writes array out in csv format, with cols on the first row."""
    with open(filename,'w') as f:
      f.write("%s\n"%','.join(self.cols))
      f.write("%s\n"%self.name)
      np.savetxt(f, self.arr, delimiter=',')

  @classmethod
  def load_from_csv(cls,filename):
    """Creates a new Table object by reading in a csv file with header."""
    table = Table()
    with open(filename) as f:
      table.cols = f.readline().strip().split(',')
      table.name = f.readline().strip()
    table.arr = np.loadtxt(filename, delimiter=',', skiprows=2)
    assert(len(table.cols) == table.arr.shape[1])
    return table

  def save(self,filename):
    """
    Writes array out in numpy format, with cols in a separate file.
    Filename should not have an extension; the data will be saved in
    filename.npy and filename_cols.txt.
    """
    # strip extension from filename
    filename, ext = os.path.splitext(filename)
    with open(filename+'_cols.txt','w') as f:
      f.write("%s\n"%','.join(self.cols))
      f.write("%s\n"%self.name)
    np.save(filename,self.arr)

  @classmethod
  def load(cls,filename):
    """
    Create a new Table object, and populate its cols and arr by reading in
    from filename (and derived _cols filename.
    """
    table = Table()
    filename, ext = os.path.splitext(filename)
    with open(filename+'_cols.txt') as f:
      table.cols = f.readline().strip().split(',')
      table.name = f.readline().strip()
    table.arr = np.load(filename+'.npy')
    assert(len(table.cols) == table.arr.shape[1])
    return table

  ###################
  # Filtering
  ###################
  def subset(self,col_names):
    """Return Table with only the specified col_names."""
    return Table(arr=self.subset_arr(col_names), cols=col_names)

  def subset_arr(self,col_names):
    """Return self.arr for only the columns that are specified."""
    if not isinstance(col_names, types.ListType):
      inds = self.cols.index(col_names)
    else:
      inds = [self.cols.index(col_name) for col_name in col_names]
    return self.arr[:,inds]

  def sort_by_column(self,ind_name,descending=False):
    """Modifies self to sort arr by column."""
    if descending:
      sorted_inds = np.argsort(-self.arr[:,self.cols.index(ind_name)])
    else:
      sorted_inds = np.argsort(self.arr[:,self.cols.index(ind_name)])
    self.arr = self.arr[sorted_inds,:]

  def filter_on_column(self,ind_name,val,op=operator.eq,omit=False):
    """
    Take name of column to index by and value to filter by.
    By providing an operator, more than just equality filtering can be done.
    """
    table = Table(cols=self.cols,arr=self.arr)
    table.arr = filter_on_column(table.arr,table.cols.index(ind_name),val,op,omit)
    if omit:
      table.cols = list(table.cols)
      table.cols.remove(ind_name)
    return table

  def with_column_omitted(self,ind_name):
    """Return Table with given column omitted."""
    ind = self.cols.index(ind_name)
    arr = np.hstack((self.arr[:,:ind], self.arr[:,ind+1:]))
    cols = list(self.cols)
    cols.remove(ind_name)
    return Table(arr,cols)

def append_index_column(arr, index):
  """ Take an m x n array, and appends a column containing index. """
  ind_vector = np.ones((np.shape(arr)[0],1)) * index
  arr = np.hstack((arr, ind_vector))
  return arr

def filter_on_column(arr, ind, val, op=operator.eq, omit=False):
  """
  Returns the rows of arr where (arr(:,ind)==val), optionally omitting the ind column.
  """
  arr = arr[op(arr[:,ind], val),:]
  if omit:
    final_ind = range(np.shape(arr)[1])
    final_ind = np.delete(final_ind, ind)
    arr = arr[:,final_ind]
  return arr

# TODO: allow arbitrary arguments to be passed to func
def collect(seq,func,cols=None,with_index=False):
  """
  Take a sequence seq of arguments to function func.
    - func should return an np.array.
    - cols are passed to func if given
  Return the outputs of func concatenated vertically into an np.array
  (thereby making copies of the collected data).
  If with_index is True, append index column to the outputs.
  """
  all_results = []
  for index,image in enumerate(seq):
    results = func(image, cols) if cols else func(image)
    if results != None and max(results.shape)>0:
      if with_index:
        all_results.append(append_index_column(results,index))
      else:
        all_results.append(results)
  return np.vstack(all_results)

def collect_with_index_column(seq, func, cols=None):
  """See collect()."""
  return collect(seq,func,cols,with_index=True)

def makedirs(dirname):
  """Does what mkdir -p does."""
  if not os.path.exists(dirname):
    try:
      os.makedirs(dirname)
    except:
      print("Exception on os.makedirs--what else is new?")

def sort_by_column(arr,ind,mode='ascend'):
  """Return the array row-sorted by column at ind."""
  if mode == 'descend':
    arr = arr[np.argsort(-arr[:,ind]),:]
  else:
    arr = arr[np.argsort(arr[:,ind]),:]
  return arr

def importance_sample(dist, num_points, kde=None):
  """
  dist is a list of numbers drawn from some distribution.
  If kde is given, uses it, otherwise computes own.
  Return num_points points to sample this dist at, spaced such that
  approximately the same area is between each pair of sample points.
  """
  if not kde:
    kde = st.gaussian_kde(dist.T)
  x = np.linspace(np.min(dist),np.max(dist))
  y = kde.evaluate(x)
  ycum = np.cumsum(y)
  points = np.interp(np.linspace(np.min(ycum),np.max(ycum),num_points),xp=ycum,fp=x)
  return points

def fequal(a,b,tol=.0000001):
  """
  Return True if the two floats are very close in value, and False
  otherwise.
  """
  return abs(a-b)<tol

"""From http://vjethava.blogspot.com/2010/11/matlabs-keyboard-command-in-python.html"""
import code
import sys
def keyboard(banner=None):
    ''' Function that mimics the matlab keyboard command '''
    # use exception trick to pick up the current frame
    try:
        raise None
    except:
        frame = sys.exc_info()[2].tb_frame.f_back
    print "# Use quit() to exit :) Happy debugging!"
    # evaluate commands in current namespace
    namespace = frame.f_globals.copy()
    namespace.update(frame.f_locals)
    try:
        code.interact(banner=banner, local=namespace)
    except SystemExit:
        return 

##############################################
# Shell interaction
##############################################
def run_matlab_script(matlab_script_dir, function_string):
  """
  Takes a directory where the desired script is, changes dir to it, runs it with the given function and parameter string, and then chdirs back to where we were.
  """
  if not os.path.exists(matlab_script_dir):
    raise IOError("Cannot find the matlab_script_dir, not doing anything")
  cwd = os.getcwd()
  os.chdir(matlab_script_dir)
  cmd = "matlab -nodesktop -nosplash -r \"%s; exit\"; stty echo" % function_string
  run_command(cmd, loud=True)
  os.chdir(cwd)

def run_command(command, loud=True):
  """
  Runs the passed string as a shell command. If loud, outputs the command and the times. If say exists, outputs it as well. Returns the retcode of the shell command.
  """
  retcode = -1
  if loud:
    print >>sys.stdout, "%s: Running command %s" % (curtime(), command)
    time_start = time.time()
    
  try:
    retcode = subprocess.call(command, shell=True, executable="/bin/bash")
    if retcode < 0:
      print >>sys.stderr, "Child was terminated by signal ", -retcode
    else:
      print >>sys.stderr, "Child returned ", retcode
  except OSError, e:
    print >>sys.stderr, "%s: Execution failed: "%curtime(), e
  
  if loud:
    print >>sys.stdout, "%s: Finished running command. Elapsed time: %f" % (curtime(), (time.time()-time_start))
  return retcode

def curtime():
  return time.strftime("%c %Z")
