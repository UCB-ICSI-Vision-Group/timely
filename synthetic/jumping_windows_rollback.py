""" Implementation of Viajayanarasimhan and Graumann's Jumping Windows for
window candidate selection
@author: Tobias Baumgartner
@contact: tobi.baum@gmail.com
"""

from synthetic.common_imports import *
from synthetic.common_mpi import *
import scipy.cluster.vq as sp
from collections import Counter
from dataset import *
from math import floor
import Image, ImageDraw
from synthetic.extractor import Extractor

###############################################################
######################### Utils ###############################
###############################################################

# Gridding factors: NxM grids per window. Good Values are still tbd.
N = 5
M = 5

# -------------------------------------------------- RootWindow
class RootWindow():
  """ A root window containing NxM grid cells, position and size"""
  def __init__(self, bbox):
    self.x = bbox[0]
    self.y = bbox[1]
    self.width = bbox[2]
    self.height = bbox[3]
    self.cell_width = self.width/M
    self.cell_height = self.height/N
  
  def add_feature(self, positions):
    """ We expect a 1x2 positions here: x y """
    # First compute the bin in which this positions belongs. We need the relative
    # coordinates of this feat to the rootWindow. Suppose this positions really
    # belongs here (no check if outsite window.)
    
    x = positions[:,0] - self.x
    y = positions[:,1] - self.y
    x_pos = (x/self.cell_width).astype(int)
    y_pos = (y/self.cell_height).astype(int)
    # return the grid cell it belongs to.
    return self.convert_tuple_to_num((x_pos, y_pos))
  
  def convert_tuple_to_num(self, tuple):
    return N*tuple[1] + tuple[0]
  
  def convert_num_to_tuple(self, num):
    m = num%M
    n = num/M
    return (m, n)

# -------------------------------------------------- LookupTable
class LookupTable():
  """ Data structure for storing the trained lookup table. It's values are
lists of tuples (grid-position, root window)"""
  def __init__(self, codebook = None, filename = None):
    self.codebook = []
    if filename == None:
      # no file given, we train a new lookup table
      self.codebook = codebook
      self.table = {}
    else:
      self.table = self.read_table(filename)
    
    self.windows = []
    self.weights = np.matrix([])
    
  def compute_all_weights(self):
    self.weights = np.zeros((len(self.codebook), N*M))
    for idx in range(len(self.codebook)):
      vect_list = self.table.get(idx)
      if not vect_list == None:
        vect_list = np.array(vect_list)[:,0]
        counts = Counter(vect_list)
        occs_feat = len(vect_list)
        for tup in counts.items():
          self.weights[idx, tup[0]] = float(tup[1])/float(occs_feat)
      else:
        self.weights[idx, :] = np.tile(1./(N*M), N*M)
      
  def add_features(self, features, root_win):
    # First translate features to codebook assignments.
    positions = features[:,0:2]
    assignments = sp.vq(features[:,3:], self.codebook)[0]
    assignments = assignments.reshape(features.shape[0], 1)
       
    # Well....this will obvi always fire.
    if not root_win in self.windows:
      self.windows.append(root_win)
      win_idx = self.windows.index(root_win)
    else:
      win_idx = self.windows.index(root_win)
          
    grid_cell = self.windows[win_idx].add_feature(positions)
    grid_cell = grid_cell.reshape(features.shape[0], 1)
    ass_cell = np.hstack((assignments, grid_cell))
        
    for row in ass_cell:
      self.add_value(row[0], (row[1], win_idx))
      
  def get_bounding_boxes(self, features):
    """ Later we will filter these by the learned weights """
    positions = features[:,0:2]
    assignments = sp.vq(features[:,3:], self.codebook)[0]
    assignments = assignments.reshape(features.shape[0], 1)
    return assignments
    
    #for ass in assignments:
      #print self.table.get(ass[0], -1)
    
  def add_value(self, key, value):
    if self.table.has_key(key):
      self.table[key].append(value)
    else:
      self.table[key] = [value]
      
  def get_value(self, key):
    if self.table.has_key(key):
      return self.table[key]
    else:
      return None
    
  def save_table(self, filename):
    file = open(filename, 'w')
    pickle.dump(self, file)
    file.close()
  
  def read_table(self, filename):
    file = open(filename, 'r')
    content = pickle.load(file)
    file.close()
    return content
         
         
def get_back_indices(num, myM, myN):
  m = num%myM
  n = num/myM
  return (n, m)


###############################################################
######################### Training ############################
###############################################################

# Dataset
d = Dataset('full_pascal_val')

e = Extractor()
codebook = e.get_codebook(d, 'sift')

t = LookupTable(codebook)

# Suppose we do this on just pos bboxes.
# For sake of testing on all boxes in 1 image, which is obviously wrong.
gt = d.get_ground_truth()

for row in gt.arr:
  bbox = row[0:4]
  r = RootWindow(bbox)
  image = d.images[int(row[gt.cols.index('img_ind')])]
  features = e.get_feature_with_pos('sift', image, bbox)
  t.add_features(features, r)
  
t.compute_all_weights()

t.save_table('test.txt')
weights = t.weights.reshape(t.weights.size, 1)
w_width = t.weights.shape[0]
w_height = t.weights.shape[1]

###############################################################
######################### Testing #############################
###############################################################
# We now have a LookupTable and want to determine the root windows
# in a new image.

print 'Jumping Window Testing ...'
  # Dataset
d = Dataset()
filename = os.path.join('test_support','val_tobi.txt')
d.load_from_pascal(filename)
#print d.get_ground_truth().arr

image = d.images[0]
features = e.get_feature_with_size('sift', 'val', image, [0,0,1000000,1000000])
annotations = t.get_bounding_boxes(features)

# create row [weight, pos_x, pos_y, grid, win] for each feature
all_tuples = []
for ann_idx in range(len(annotations)):
  ann = annotations[ann_idx][0]
  if ann in t.table:
  #else: well we haven't observed this feature in training => won't help, no use
    fpos = features[ann_idx, 0:2]
    for tup in t.table[ann]:
      grid = tup[0]
      win = tup[1]
      wght = t.weights[ann, grid]
      all_tuples.append([wght, fpos[0], fpos[1], grid, win])
      
# we collected all pairs and want to sort them by weight now to select top K.
K = 300
all_pairs = np.asarray(all_tuples)
all_weights = all_pairs[:,0]
indices = np.asarray(all_weights.argsort(axis=0))[::-1]
top = indices[:K]
top_selection = all_pairs[top,1:]

# we now got our K best combinations. Compute the according bboxes
bboxes = np.zeros(top_selection.shape)
for i in range(K):
  win = t.windows[top_selection[i,3].astype(int)]
  x = win.x
  y = win.y
  width = win.width
  height = win.height
  cell_width = win.cell_width
  cell_height = win.cell_height
  # compute the middle point of the grid.
  MN = get_back_indices(top_selection[i,2], w_width, w_height)
  
  x_off = cell_width*(MN[0] + 0.5)
  y_off = cell_height*(MN[1] + 0.5)
  bboxes[i,0] = top_selection[i, 0] - x_off
  bboxes[i,1] = top_selection[i, 1] - y_off
  bboxes[i,2] = width
  bboxes[i,3] = height
#print bboxes
image_filename = d.config.VOC_dir + '/JPEGImages/' + image.name
os.system('convert ' + image_filename + ' img.png')
 
im = Image.open('img.png')
draw = ImageDraw.Draw(im)


for k in range(K):
  # TODO: Why are the features selected at these weird points?
  # draw.rectangle(((bboxes[k,0]-5, bboxes[k,1]-5),(bboxes[k,0]+5, bboxes[k,1]+5)))
  draw.rectangle(((bboxes[k,0],bboxes[k,1]),(bboxes[k, 2] + bboxes[k, 0], (bboxes[k, 3] + bboxes[k, 1]))))
del draw
im.save('im-out.png', "PNG")
os.system('xv im-out.png')

print '\nfinished...'