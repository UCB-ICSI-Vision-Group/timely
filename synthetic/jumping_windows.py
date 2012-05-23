""" Implementation of Viajayanarasimhan and Graumann's Jumping Windows for
window candidate selection
@author: Tobias Baumgartner
@contact: tobi.baum@gmail.com
"""

from synthetic.common_imports import *
from synthetic.common_mpi import *

import scipy.io as sio
from dataset import *
from synthetic.extractor import Extractor
from synthetic.mean_shift import MeanShiftCluster

from synthetic.evaluate_matlab_jws import *

# Gridding factors: NxM grids per window. Good Values are still tbd.
N = 4
M = 4

class LookupTable():
  """ Data structure for storing the trained lookup table. It's values are
lists of tuples (grid-position, root window)"""
  def __init__(self, classes = config.pascal_classes, codebook = None, filename=None):
    self.classes = classes
    self.codebook = codebook
    if filename:
      self.read_table(filename)
    # no file given, we train a new lookup table
    else:
      self.tables = [{} for _ in range(len(self.classes))]
      self.weights = [np.matrix([]) for _ in range(len(self.classes))]
    
    
  def compute_all_weights(self):
    for i in range(len(self.classes)):
      print 'compute weights for class %s...'%(self.classes[i])
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
      print 'perform meanshifts for class %s'%(self.classes[cls_ind])
      table = self.tables[cls_ind]
      for v_ind in table:
        win_grids = table[v_ind]
        for windex in win_grids:
          wins = win_grids[windex]
          wins = np.vstack(wins)
          centers = MeanShiftCluster(wins, 0.25)
          win_grids[windex] = centers
            
  def get_gridcell(self,positions, bbox):
    '''
    Compute the gridcell that these positions fall into within the box and 
    convert it to one number between 0, N*M. Here we numerate the gridcells
    in a row-wise order
    '''
    x = positions[:,0] - bbox[0]
    y = positions[:,1] - bbox[1]
    x_pos = (x/(bbox[2]/M)).astype(int)
    y_pos = (y/(bbox[3]/N)).astype(int)    
    return N*y_pos + x_pos
      
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
    # we also save a just plain file to make search for existence easier. 
    # (I'm lazy)
    open(filename,'w').close()
    for cls_ind, cls in enumerate(self.classes):
      outfile = open(filename+'_'+cls, 'w')
      print 'save %s'%(filename+'_'+cls)
      pickle.dump(self.tables[cls_ind], outfile)
      outfile.close()
      
    outfile = open(filename+'_weights', 'w')
    print 'save %s'%(filename+'_weights')
    pickle.dump(self.weights, outfile)
    outfile.close()
  
  def read_table(self, filename):
    self.tables=[]
    for cls in self.classes:
      infile = open(filename+'_'+cls, 'r')
      print 'read %s'%(filename+'_'+cls)
      table = pickle.load(infile)
      self.tables.append(table)
      infile.close()      
    infile = open(filename+'_weights', 'r')
    print 'read %s'%(filename+'_weights')
    self.weights = pickle.load(infile)
    infile.close()     
  
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
  
  def get_top_windows(self, K, annotations, cls_ind, bounds, outside_overlaps_thresh=0.5, clipped=True):
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
        add_pos = np.hstack((pos, [0,0]))
        try: 
          add_wins = wins + np.tile(add_pos, (wins.size/4.,1))
        except:
          embed()
          raise RuntimeError('break it')
        # A little heuristic to improve everything:
        # - take only boxes that overlap with the image by at least THRESH %
        if clipped:
          overlaps = BoundingBox.get_overlap(add_wins, bounds)
          actual_overlaps = np.multiply(bounds[2]*bounds[3],np.divide(overlaps,add_wins[:,2]*add_wins[:,3]))
          indices = np.where(actual_overlaps>outside_overlaps_thresh)[0]
          if len(indices) == 0:
            continue
          add_wins = add_wins[indices, :]        
        add_wins = BoundingBox.clipboxes_arr(add_wins, bounds)       
        
        all_wins.append(add_wins)
        insert_count += add_wins.shape[0]
      if insert_count >= K:
        break
    # is this usual except for testing data :/ NO: just if none of these objects
    # has been seen during training time
    if len(all_wins) == 0:
      print '##### WARNING: FOUND NO WINS FOR object of class %s ####'%self.classes[cls_ind]
      return all_wins    
    all_wins = np.vstack(all_wins)
    return all_wins      

def run(force=False):
  ###############################################################
  ######################### Training ############################
  ###############################################################
  
  # Dataset
  dataset = 'full_pascal_trainval'
  d = Dataset(dataset)  
  e = Extractor()
  codebook = e.get_codebook(d, 'sift')
  t_train = time.time()
  if comm_rank == 0: # Could I also parallelize training?
    t = LookupTable(config.pascal_classes, codebook)  
    gt = d.get_ground_truth()
    filename_lookuptable = config.get_jumping_windows_dir(dataset) + 'lookup_table.pkl'
    
    if not os.path.exists(filename_lookuptable) and not force:
      for rowdex, row in enumerate(gt.arr):
        bbox = row[0:4]
        print 'add row %d of %d'%(rowdex, gt.shape()[0])
        image = d.images[int(row[gt.cols.index('img_ind')])]
        features = e.get_assignments(bbox, 'sift', codebook, image)
        cls_ind = int(row[gt.cols.index('cls_ind')])
        t.add_features(features, bbox, cls_ind)  
              
      t.compute_all_weights()
      # Now also compute the cluster means
      t.perform_mean_shifts()   
      
      t.save_table(filename_lookuptable)
      outfile = open('times','w')
      t_train = time.time() - t_train
      outfile.writelines('train took %f secs\n'%t_train)  
  
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
  t = LookupTable(config.pascal_classes, codebook, config.get_jumping_windows_dir(dataset) + 'lookup_table.pkl')

  K = 10000
  t_test = time.time()
  for img_ind in range(comm_rank, len(d.images), comm_size):
    img = d.images[img_ind]  
    annotations = e.get_assignments(None, 'sift', codebook, img)
    bounds = [0, 0, img.size[0], img.size[1]]
    for cls_ind, cls in enumerate(config.pascal_classes):
      if not img.get_cls_ground_truth()[cls_ind]:
        continue
      print 'machine %d on %s for %s'%(comm_rank, img.name, cls)
      top_wins = t.get_top_windows(K, annotations, cls_ind, bounds, clipped=False)
      savename = os.path.join(ut.makedirs(os.path.join(config.res_dir, 'jumping_windows','bboxes')), '%s_%s.mat'%(cls,img.name[:-4]))
      sio.savemat(savename, {'bboxes':top_wins})              
  
  safebarrier(comm)
  if comm_rank == 0:
    t_test = time.time() - t_test
    outfile.writelines('test took %f secs\n'%t_test)
    outfile.close()
    
  return
  
if __name__=='__main__':
  force = False
  run(force)