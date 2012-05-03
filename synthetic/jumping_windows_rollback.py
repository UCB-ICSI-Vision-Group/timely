""" Implementation of Viajayanarasimhan and Graumann's Jumping Windows for
window candidate selection
@author: Tobias Baumgartner
@contact: tobi.baum@gmail.com
"""

from synthetic.common_imports import *
from synthetic.common_mpi import *

import scipy.io as sio
from dataset import *
from math import floor
import Image, ImageDraw
from synthetic.extractor import Extractor
from synthetic.mean_shift import MeanShiftCluster

###############################################################
######################### Utils ###############################
###############################################################

# Gridding factors: NxM grids per window. Good Values are still tbd.

N = 4
M = 4
# -------------------------------------------------- RootWindow
class RootWindow():
  """ A root window containing NxM grid cells, position and size"""  
  def __init__(self, bbox):
    None
  
  @classmethod
  def add_feature(self, bbox, positions):
    """ We expect a 1x2 positions here: x y """
    # First compute the bin in which this positions belongs. We need the relative
    # coordinates of this feat to the rootWindow. Suppose this positions really
    # belongs here (no check if outsite window.)    
    x = positions[:,0] - bbox[0]
    y = positions[:,1] - bbox[1]
    x_pos = (x/(bbox[2]/M)).astype(int)
    y_pos = (y/(bbox[3]/N)).astype(int)
    # return the grid cell it belongs to.
    return self.convert_tuple_to_num((x_pos, y_pos))
  
  @classmethod
  def convert_tuple_to_num(self, tup):
    return N*tup[1] + tup[0]
  
  @classmethod
  def convert_num_to_tuple(self, num):
    m = num%M
    n = num/M
    return (m, n)

# -------------------------------------------------- LookupTable
class LookupTable():
  """ Data structure for storing the trained lookup table. It's values are
lists of tuples (grid-position, root window)"""
  def __init__(self, classes = config.pascal_classes, codebook = None):
    self.classes = classes
    self.codebook = []
    
    # no file given, we train a new lookup table
    self.codebook = codebook
    self.tables = [{} for _ in range(len(self.classes))]
    self.weights = [np.matrix([]) for _ in range(len(self.classes))]
    
    
  def compute_all_weights(self):
    for i in range(len(self.classes)):
      tab = self.tables[i]
      weights = self.weights[i]
      weights = np.zeros((len(self.codebook), N*M))
      for idx in tab:
        grid_win = tab[idx]  
        for g in grid_win:     
          weights[idx, g] = len(grid_win[g])
        normalize = np.sum(weights[idx, :])
        weights[idx,:] /= normalize
      self.weights[i] = weights
      
  def perform_mean_shifts(self):
    '''Perform mean_shift of windows for all (cls, v, g) combinations'''
    for cls_ind in range(len(self.classes)):
      table = self.tables[cls_ind]
      for v_ind in table:
        win_grids = table[v_ind]
        for windex in win_grids:
          wins = win_grids[windex]
          wins = np.vstack(wins)
          centers = MeanShiftCluster(wins, 0.25)
          win_grids[windex] = centers
#        table[v_ind] = win_grids
#      self.tables[cls_ind] = table
  
  def get_gridcell(self,positions, bbox):
    # TODO: well obvi here factor in RootWindow
    return RootWindow.add_feature(bbox, positions)
      
  def add_features(self, features, bbox, cls_ind):
    # First translate features to codebook assignments.
    self.cls_ind = cls_ind
    positions = features[:,0:2]
    assignments = features[:,3]
    assignments = assignments.reshape(features.shape[0], 1)
    grid_cell = self.get_gridcell(positions, bbox)
    grid_cell = grid_cell.reshape(features.shape[0], 1)
    ass_cell = np.hstack((assignments, grid_cell))
    for rowdex, row in enumerate(ass_cell):
      pos = positions[rowdex]
      nbbox = np.copy(bbox)
      nbbox[:2] -= pos
      self.add_value(row[0], row[1], nbbox)
          
  def add_value(self, v, g, win):
    '''
    Add one sample point to the lookup table. 
    word v, grid g, bbox win
    Lookuptable schema: {v: {g: [win]}}
    '''
    table = self.tables[self.cls_ind]
    if table.has_key(v):
      tab = table[v]
      if tab.has_key(g):
        tab[g].append(win)
      else:
        tab[g] = [win]
    else:
      table[v] = {g: [win]}
      
  def get_value(self, key):
    if self.table.has_key(key):
      return self.table[key]
    else:
      return None
    
  def save_table(self, filename):
    outfile = open(filename, 'w')
    pickle.dump(self, outfile)
    outfile.close()
  
  def read_table(self, filename):
    outfile = open(filename, 'r')
    content = pickle.load(outfile)
    outfile.close()
    return content
  
  def get_ranked_word_grid(self, words, cls_ind):
    mask = np.zeros((self.codebook.shape[0],M*N))
    mask[words,:] = 1
    weights = self.weights[cls_ind]
    weights = np.multiply(weights, mask)
    shape_orig = weights.shape
    weights = weights.reshape(weights.size,1)
    ranks = np.argsort(-weights, 0)
    width = shape_orig[1]
    x = ranks/width  
    y = ranks%width
    topK = np.hstack((x,y))
    return topK
  
  def get_top_windows(self, K, annotations, cls_ind):
    positions = annotations[:,:2]
    words = annotations[:,3].tolist()
    topK = self.get_ranked_word_grid(words, cls_ind)
    insert_count = 0    
    all_wins = []
    for [word, grid] in topK:
      # First get the windows for this beast.
      table = self.tables[cls_ind]
      if not word in table:
        continue
      win_grid = table[word]
      if not grid in win_grid:
        continue  
      wins = win_grid[grid]
      
      # Now add all those windows for each point that has this annotation
      poss = positions[np.where(annotations[:,3] == word)]
      for pos in poss:
        add_wins = wins + np.tile(pos, (wins.size/4.,2))
        all_wins.append(add_wins)
        insert_count += add_wins.shape[0]
      if insert_count >= K:
        break
    all_wins = np.vstack(all_wins)
    return all_wins
      
         
def get_back_indices(num, myM, myN):
  m = num%myM
  n = num/myM
  return (n, m)


def broken_draw():  
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


def run():
  ###############################################################
  ######################### Training ############################
  ###############################################################
  
  # Dataset
  dataset = 'full_pascal_trainval'
  d = Dataset(dataset)  
  e = Extractor()
  codebook = e.get_codebook(d, 'sift')
    
  if comm_rank == 0:
    t = LookupTable(config.pascal_classes, codebook)  
    gt = d.get_ground_truth()
      
    for row in gt.arr:
      bbox = row[0:4]
      #r = RootWindow(bbox)
      image = d.images[int(row[gt.cols.index('img_ind')])]
      features = e.get_assignments(bbox, 'sift', codebook, image)
      cls_ind = int(row[gt.cols.index('cls_ind')])
      t.add_features(features, bbox, cls_ind)    
      
    t.compute_all_weights()
    # Now also compute the cluster means
    t.perform_mean_shifts()   
    
    t.save_table(config.get_jumping_windows_dir(dataset) + 'lookup_table.pkl')
    
  safebarrier(comm)
  
  ###############################################################
  ######################### Testing #############################
  ###############################################################
  # We now have a LookupTable and want to determine the root windows
  # in a new image.
  
  print 'Jumping Window Testing ...'
    # Dataset
  d = Dataset('full_pascal_test')
  del t
  t = cPickle.load(open(config.get_jumping_windows_dir(dataset) + 'lookup_table.pkl','r'))
  K = 10000
  
  for img_ind in range(comm_rank, len(d.images), comm_size):
    img = d.images[img_ind]  
    annotations = e.get_assignments(None, 'sift', codebook, img)
    for cls_ind, cls in enumerate(config.pascal_classes):
      if not img.get_cls_ground_truth()[cls_ind]:
        continue
      print 'on %s for %s'%(img.name, cls)
      top_wins = t.get_top_windows(K, annotations, cls_ind)
      savename = os.path.join(ut.makedirs(os.path.join(config.res_dir, 'jumping_windows','bboxes')), '%s_%s.mat'%(cls,img.name[:-4]))
      sio.savemat(savename, {'bboxes':top_wins})              
  
  return
  
if __name__=='__main__':
  run()