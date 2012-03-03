import subprocess
from common_imports import *

class Table:
  "An ndarray with associated column names."

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

  def set_arr(self,arr):
    "Make sure that arr is at least 2d and set self.arr to it."
    self.arr = np.atleast_2d(arr)

  def __deepcopy__(self):
    "Make a deep copy of the Table and return it."
    ret = Table()
    ret.arr = self.arr.copy() if not self.arr == None else None
    ret.cols = list(self.cols) if not self.cols == None else None
    return ret

  def __repr__(self):
    return """
Table name: %(name)s | size: %(shape)s
%(cols)s
%(arr)s
"""%dict(self.__dict__.items()+{'shape':self.shape()}.items())

  def __eq__(self,other):
    "Two Tables are equal if all columns and their names are equal, in order."
    return np.all(self.arr==other.arr) and self.cols == other.cols

  def shape(self):
    return self.arr.shape

  def ind(self,col_name):
    return self.cols.index(col_name)

  ###################
  # Save/Load
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
  def row_subset(self,row_inds):
    "Return Table with only the specified rows."
    return Table(arr=self.row_subset_arr(row_inds), cols=self.cols)

  def row_subset_arr(self,row_inds):
    "Return self.arr with only the specified rows."
    if isinstance(row_inds,np.ndarray):
      row_inds = row_inds.tolist()
    return self.arr[row_inds,:]

  def subset(self,col_names):
    "Return Table with only the specified col_names, in order."
    return Table(arr=self.subset_arr(col_names), cols=col_names)

  def subset_arr(self,col_names):
    "Return self.arr for only the columns that are specified."
    if not isinstance(col_names, types.ListType):
      inds = self.cols.index(col_names)
    else:
      inds = [self.cols.index(col_name) for col_name in col_names]
    return self.arr[:,inds]

  def sort_by_column(self,ind_name,descending=False):
    """
    Modify self to sort arr by column. Return self.
    """
    if descending:
      sorted_inds = np.argsort(-self.arr[:,self.cols.index(ind_name)])
    else:
      sorted_inds = np.argsort(self.arr[:,self.cols.index(ind_name)])
    self.arr = self.arr[sorted_inds,:]
    return self

  def filter_on_column(self,ind_name,val,op=operator.eq,omit=False):
    """
    Take name of column to index by and value to filter by.
    By providing an operator, more than just equality filtering can be done.
    """
    if ind_name not in self.cols:
      return self
    table = Table(cols=self.cols,arr=self.arr)
    table.arr = filter_on_column(table.arr,table.cols.index(ind_name),val,op,omit)
    if omit:
      table.cols = list(table.cols)
      table.cols.remove(ind_name)
    return table

  def with_column_omitted(self,ind_name):
    "Return Table with given column omitted. Name stays the same."
    ind = self.cols.index(ind_name)
    # TODO: why use hstack?
    arr = np.hstack((self.arr[:,:ind], self.arr[:,ind+1:]))
    cols = list(self.cols)
    cols.remove(ind_name)
    return Table(arr,cols,self.name)

###################
# Ndarray manipulations
###################
def append_index_column(arr, index):
  """ Take an m x n array, and appends a column containing index. """
  ind_vector = np.ones((np.shape(arr)[0],1)) * index
  arr = np.hstack((arr, ind_vector))
  return arr

def filter_on_column(arr, ind, val, op=operator.eq, omit=False):
  """
  Returns the rows of arr where arr[:,ind]==val,
  optionally omitting the ind column.
  """
  arr = arr[op(arr[:,ind], val),:]
  if omit:
    final_ind = range(np.shape(arr)[1])
    final_ind = np.delete(final_ind, ind)
    arr = arr[:,final_ind]
  return arr

def collect(seq, func, kwargs=None, with_index=False):
  """
  Take a sequence seq of arguments to function func.
    - func should return an np.array.
    - kwargs is a dictionary of arguments that will be passed to func if given
  Return the outputs of func concatenated vertically into an np.array
  (thereby making copies of the collected data).
  If with_index is True, append index column to the outputs.
  """
  all_results = []
  for index,image in enumerate(seq):
    results = func(image, **kwargs) if kwargs else func(image)
    if results != None and max(results.shape)>0:
      if with_index:
        all_results.append(append_index_column(results,index))
      else:
        all_results.append(results)
  if len(all_results)<1:
    return np.array([])
  return np.vstack(all_results)

def collect_with_index(seq, func, kwargs=None):
  """See collect()."""
  return collect(seq,func,kwargs,with_index=True)

def sort_by_column(arr, ind, mode='ascend'):
  """Return the array row-sorted by column at ind."""
  if mode == 'descend':
    arr = arr[np.argsort(-arr[:,ind]),:]
  else:
    arr = arr[np.argsort(arr[:,ind]),:]
  return arr

###################
# Misc
###################
def random_subset_up_to_N(N, max_num=None):
  """
  Return a random subset of size min(N,max_num) of non-negative integers
  up to N, in permuted order.
  If max_num >= N, order of integers 0..N is not permuted.
  N and max_num must be positive.
  If max_num is not given, max_num=N.
  """
  if max_num == None:
    max_num = N
  if N <= 0 or max_num <= 0:
    raise ValueError("Can't deal with N or max_num <= 0")
  if max_num >= N:
    return range(0,N)
  return np.random.permutation(N)[:max_num]

def random_subset(vals, max_num=None, ordered=False):
  """
  Return a random subset of size min(len(vals),max_num) of a list of
  values, in permuted order (unless ordered=True). If max_num is not given,
  max_num=len(vals).
  If max_num >= N, order is not permuted.
  NOTE: returns a list, not an array
  """
  if max_num == None:
    max_num = len(vals)
  if max_num >= len(vals) and ordered:
    return vals
  arr = np.array(vals)[random_subset_up_to_N(len(vals),max_num)]
  if ordered:
    arr = np.sort(arr)
  return arr.tolist()

def makedirs(dirname):
  "Does what mkdir -p does, and returns dirname."
  if not os.path.exists(dirname):
    try:
      os.makedirs(dirname)
    except:
      print("Exception on os.makedirs")
  return dirname

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

def log2(x):
  "Base-2 log that returns 0 if x==0."
  y = np.atleast_1d(np.copy(x))
  y[y==0]=1
  return np.log2(y)

def mean_squared_error(y_true,y_pred):
  "Because sklearn.metrics is fucking wrong."
  return np.mean((y_pred - y_true) ** 2)

def cartesian(arrays, out=None):
    """
    from http://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out
  

from collections import Counter
def determine_bin(data, bounds, asInt=True):
  """ 
  For data and given bounds, determine in which bin each point falls.
  asInt=True: return number of bin each val falls in
  asInt=False: return representative value for each val (center between bounds)
  """
  num_bins = bounds.shape[0]-1
  ret_tab = np.zeros((data.shape[0],1))
  col_bin = np.zeros((data.shape[0],1))
  bin_values = np.zeros(bounds.shape)
  last_val = 0.
  
  for bidx, b in enumerate(bounds):
    bin_values[bidx] = (last_val + b)/2.
    if bidx == 0:
      continue
    last_val = b
    col_bin += np.matrix(data < b, dtype=int).T
  
  bin_values = bin_values[1:]    
  col_bin[col_bin == 0] = 1  
  
  if asInt:
    a = num_bins - col_bin
    ret_tab = a[:,0]     
  else:    
    for rowdex in range(data.shape[0]):
      ret_tab[rowdex, 0] = bin_values[int(col_bin[rowdex]-1)]
  return ret_tab

def histogram(x, num_bins, normalize=False):
  """
  compute a histogram for x = np.array and num_bins bins
  assumpt: x is already binned up 
  """
  bounds = np.linspace(np.min(x), np.max(x), num_bins+1)
  x = determine_bin(x, bounds)
  return histogram_just_count(x, num_bins, normalize)

def histogram_just_count(x, num_bins, normalize=False):
  if hasattr(x, 'shape'):
    # This is a np object
    if x.ndim > 1:
      x = np.hstack(x)
  counts = Counter(x)
  histogram = [counts.get(x,0) for x in range(num_bins)]
  histogram = np.matrix(histogram, dtype = 'float64')
  if normalize:
    histogram = histogram/np.sum(histogram)
  return histogram

##############################################
# Tic/Toc
##############################################
class TicToc:
  "MATLAB tic/toc."
  def __init__(self):
    self.labels = {}

  def tic(self,label=None):
    "Start timer for given label. Returns self."
    if not label:
      label = '__default'
    self.labels[label] = time.time()
    return self

  def toc(self,label=None,quiet=False):
    "Return elapsed time for given label. Optionally print time."
    if not label:
      label = '__default'
    assert(label in self.labels)
    elapsed = time.time()-self.labels[label]
    if not quiet:
      print "Time elapsed: %.3f"%elapsed
    return elapsed
  
  def qtoc(self,label=None):
    "Quiet toc()"
    return self.toc(label,quiet=True)

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
