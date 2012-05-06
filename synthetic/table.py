from synthetic.util import filter_on_column
import operator
import types
import numpy as np

class Table:
  """
  An ndarray with associated column names.
  Array should be two-dimensional.
  """

  ###################
  # Init/Copy/Repr
  ###################
  def __init__(self,arr=None,cols=None,name=None):
    """
    If arr and cols are passed in, initialize with them by reference.
    If arr is None, initialize with np.array([]) which has shape (0,).
    name is a place to keep some optional identifying information.
    """
    # Passed-in array can be None, or an empty (0,) array, or (M,), or (M,N).
    # Only the last two cases are good, otherwise we set to empty (0,) array.
    self.arr = np.array([])
    if arr != None and arr.shape[0]>0:
      # convert (M,) arrays to (1,M) and leave (M,N) arrays alone
      self.arr = np.atleast_2d(arr)
    self.cols = cols
    self.name = name

  @property
  def shape(self):
    "Return shape of the array."
    return self.arr.shape

  def ind(self,col_name):
    "Return index of the given column name."
    return self.cols.index(col_name)

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
"""%dict(self.__dict__.items()+{'shape':self.shape}.items())

  def __eq__(self,other):
    "Two Tables are equal if all columns and their names are equal, in order."
    return np.all(self.arr==other.arr) and self.cols == other.cols

  def sum(self,dim=0):
    "Return sum of the array along given dimension."
    return np.sum(self.arr,dim)

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
    Modify self to sort arr by column.
    Return self, but the array is a copy due to the sort.
    """
    col = self.arr[:,self.cols.index(ind_name)]
    col = -col if descending else col
    self.arr = self.arr[col.argsort()]
    return self

  def filter_on_column(self,ind_name,val,op=operator.eq,omit=False):
    """
    Take name of column to index by and value to filter by, and return
    copy of self with only the rows that satisfy the filter.
    By providing an operator, more than just equality filtering can be done.
    If omit, removes that column from the returned copy.
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