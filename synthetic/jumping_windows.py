""" Implementation of Viajayanarasimhan and Graumann's Jumping Windows for
window candidate selection
@author: Tobias Baumgartner
@contact: tobi.baum@gmail.com
"""

import cPickle
import numpy as np
from collections import Counter
import Image, ImageDraw
import os as os
import time
import synthetic.util as ut
from mpi4py import MPI
from numpy.numarray.numerictypes import Int
import scipy.io as sio
from os.path import join

from synthetic.evaluation import Evaluation
from synthetic.util import Table
from synthetic.extractor import Extractor
from synthetic.dataset import Dataset
import synthetic.config as config
from synthetic.detector import Detector
from numpy.ma.core import floor
from synthetic.bounding_box import BoundingBox
from sklearn.cluster import MeanShift
from synthetic.safebarrier import safebarrier

comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()

class JumpingWindowsDetectorGrid(Detector):
  def __init__(self, warmstart=False,K=3000,M=4):
    """ Detector for jumping windows.
    cbwords - number of words codebooks have been created with
    cbsamp - number of samples codebooks have been trained with
    warmstart - load all lookup tables right away? (takes 5s per table)
    """
    self.cut_tolerance = 0.5
    self.M = M
    self.N = M
    self.K = K
    self.lookupTables = {}
    self.e = Extractor()
    self.all_classes = config.pascal_classes
    if warmstart:
      for cls in config.pascal_classes:
        self.load_lookup_table(cls)
        
  def add_lookup_table(self, cls, table):
    self.lookupTables[cls] = table
    
  def load_lookup_table(self, cls):
    filename = config.save_dir + 'JumpingWindows/' + cls
    t = load_lookup_table(filename)
    t.M = self.M
    t.N = self.N
    self.lookupTables[cls] = t
  
  def get_lookup_table(self, cls):
    if not cls in self.lookupTables:
      self.load_lookup_table(cls)
    return self.lookupTables[cls]
  
  def detect(self,img):
    """Detect bounding-boxes for all classes on image img
    return 5 column table [x,y,w,h,cls_ind]"""
    all_windows = np.zeros((len(self.all_classes)*K,5))
    for cls_idx in range(len(self.all_classes)):
      cls = self.all_classes[cls_idx]
      all_windows[cls_idx*K:(cls_idx+1)*K,:] = self.detect_cls(img,cls,self.K,self.cut_tolerance)      
    return all_windows
  
#  def get_windows(image,cls,with_time=True):
    
  def detect_cls(self, img, cls, K=3000,cut_tolerance=.5):
    """ return the jumping windows for:
    img - Image instance
    cls - class
    K - windows to return
    """
    t = self.get_lookup_table(cls)
    return self.detect_jumping_windows(img, cls, self.e, t, K,cut_tolerance)
  
  def get_windows(self, img, cls, K=3000, cut_tolerance=.5):
    """ return the jumping windows for:
    img - Image instance
    cls - class
    K - windows to return
    """
    t = self.get_lookup_table(cls)
    t_windows = time.time()
    windows = self.detect_jumping_windows(img, cls, self.e, t, K, cut_tolerance)
    time_elapsed = time.time() - t_windows
    return (windows, time_elapsed) 

  def detect_jumping_windows(self,image, cls, e, t, K, cut_tolerance, 
                             feature_type='dsift'):  
    """ detect jumping windows for image and class
    image - Image instance
    cls - class
    e - Extractor
    t - JW LookupTable
    K - number of windows to return
    cut_tolerance - tolerance for window to accept
    returns: (K, 4) matrix of bounding boxes, sorted by score
    """  
    tic()
    pos_bounds = np.array([0,0,image.size[0],image.size[1]]) 
    annotations = t.get_annotations(pos_bounds, feature_type, t.codebook,\
                                    cls, image)
    positions = annotations[:,0:2]
    toc('annotations')    
    tic()
    top_selection = t.compute_top_boxes(annotations, positions,K, cut_tolerance)
    toc('create feature tuples')
    bboxes = np.zeros((top_selection.shape[0],4))
    #print unify_rows(top_selection).shape
    tic()
    for i in range(K):    
      box = t.convert_tuple2bbox(top_selection[i,:], image)
      bboxes[i] = box
      #bboxes[i] = BoundingBox.clipboxes_arr(box,[0,0,image.size[1],image.size[0]])
    toc('compute bboxes')    
    return bboxes
    
  def train_jw_detector(self, all_classes, train_set):
    """Training"""
    d = Dataset(train_set)
    e = Extractor()  
    ut.makedirs(config.save_dir + 'JumpingWindows/')
    ut.makedirs(config.save_dir + 'JumpingWindows/'+str(self.M)+'/')
    for train_cls in all_classes:
      # Codebook
      codebook_file = e.save_dir + 'dsift/codebooks/codebook' 
         
      save_table_file = config.save_dir + 'JumpingWindows/' + train_cls      
      if not os.path.isfile(codebook_file):
        print 'codebook',codebook_file,'does not exist'
        continue
      codebook = np.loadtxt(codebook_file)    
      t = LookupTable_withgrid(codebook, self.M)          
      # Suppose we do this on just pos bboxes.
      gt = d.get_ground_truth_for_class(train_cls)      
      train_gt = gt.arr
      for row in train_gt:
        bbox = row[0:4]
        image = d.images[row[gt.cols.index('img_ind')].astype(Int)]
        r = RootWindow_withgrid(bbox, d.images[row[gt.cols.index('img_ind')].astype(Int)],self.M) 
        features = e.get_feature_with_pos('dsift', image, bbox)
        t.add_features(features, r)         
      t.compute_all_weights()  
      t.save_table(save_table_file)

def load_lookup_table(filename):
  infile = open(filename, 'r')
  content = cPickle.load(infile)    
  infile.close()  
  content.w_width = content.weights.shape[0]
  content.w_height = content.weights.shape[1]
  content.e = Extractor()
  return content

###############################################################
######################### Utils ###############################
###############################################################

# Gridding factors: NxM grids per window. Good Values are still tbd.

# -------------------------------------------------- RootWindow_withgrid
class RootWindow_withgrid():
  """ A root window containing NxM grid cells, position and size"""
  def __init__(self, bbox, image, M):
    self.x = bbox[0]
    self.y = bbox[1] 
    self.M = M
    self.image = image
    self.width = bbox[2]
    self.height = bbox[3]
    self.cell_width = bbox[2]/float(self.M)
    self.cell_height = bbox[3]/float(self.M)
    self.scale = bbox[3]/image.size[1]
    self.ratio = bbox[2]/bbox[3]
  
  def add_feature(self, positions):
    """ We expect a 1x2 positions here: x y """
    # First compute the bin in which this positions belongs. We need the relative
    # coordinates of this feat to the RootWindow_withgrid. 
    
    x = positions[:,0] - self.x
    y = positions[:,1] - self.y
    x_pos = (x/(self.width+1)*self.M).astype(Int)
    y_pos = (y/(self.height+1)*self.M).astype(Int)
    # return the grid cell it belongs to.
    return self.convert_tuple_to_num((x_pos, y_pos))
  
  def convert_tuple_to_num(self, intuple):
    return self.M*intuple[1] + intuple[0]
  
  def convert_num_to_tuple(self, num):
    m = num%self.M
    n = num/self.M
    return (m, n)
  

# -------------------------------------------------- LookupTable_withgrid
class LookupTable_withgrid():
  """ Data structure for storing the trained lookup table. It's values are
  lists of tuples (grid-position, root window)"""
  def __init__(self, M=4, codebook = None, filename = None, use_scale=False):
    self.codebook = []
    if filename == None:
      # no file given, we train a new lookup table
      if codebook == None:
        print 'LookupTable_withgrid - Warning: no codebook given/loaded'
      else:
        self.codebook = codebook
      self.table = {}
    else:
      self = self.read_table(filename)
      print 'codebook:', self.codebook.shape
    self.use_scale = use_scale
    self.M = M
    self.N = M
    self.windows = []
    self.weights = np.matrix([]) 
    self.num_words = codebook.shape[0]
    self.num_grids = self.M*self.N
    self.e = Extractor()
    self.windows_for_vg = {}
    
  def compute_all_weights(self):
    self.weights = np.zeros((len(self.codebook), self.M*self.N))
    t = time.time()
    for idx in range(len(self.codebook)):
      ti = time.time()-t
      if ti > 0.1:
        print idx,'of',len(self.codebook)
        t = time.time()      
      vect_list = self.table.get(idx)
      if not vect_list == None:
        vect_list = np.array(vect_list)[:,0]
        counts = Counter(vect_list)
        occs_feat = len(vect_list)
        for tup in counts.items():
          self.weights[idx, tup[0]] = float(tup[1])/float(occs_feat)
      else:
        self.weights[idx, :] = np.tile(1./(self.N*self.M), self.N*self.M)
    self.w_height = self.weights.shape[1]
    self.w_width = self.weights.shape[0]
    
  def perform_mean_shifts(self):
    # same bandwidth as Grauman
    meanshifter = MeanShift(bandwidth=0.25, bin_seeding=False)
    i = 0
    t = time.time()
    for ann_grid in self.windows_for_vg:
      # for each combination of word and grid do mean shift.
      ti = time.time()-t
      if ti > 2:
        print i,'of',len(self.windows_for_vg)
        t = time.time()      
      i += 1
      
      mat = np.asarray(self.windows_for_vg[ann_grid])
      meanshifter.fit(mat)
      centers = meanshifter.cluster_centers_ 
      self.windows_for_vg[ann_grid] = centers
          
  def add_features(self, bbox, root_win, cls, image):
    # First translate features to codebook assignments.
    
    assignments = self.get_annotations(bbox,'dsift',self.codebook,cls,image)
    positions = assignments[:,0:2]
     
    if not root_win in self.windows:
      self.windows.append(root_win)
    win_idx = self.windows.index(root_win)
            
    grid_cell = self.windows[win_idx].add_feature(positions)
    #print grid_cell

    grid_cell = grid_cell.reshape(assignments.shape[0], 1)
    ass_cell_pos = np.hstack((assignments[:,2:3], grid_cell, positions))
    window = self.windows[win_idx]
    
    for row in ass_cell_pos:
      self.add_window_for_vg(row[0], row[1], row[2:4], window) 
      self.add_value(row[0], [row[1], win_idx])

  def add_value(self, key, value):
    if self.table.has_key(key):
      self.table[key].append(value)
    else:
      self.table[key] = [value]
  
  def add_window_for_vg(self, word, grid, position, window):
    if grid > self.M**2-1:
      return
    x_off = position[0] - window.x
    y_off = position[1] - window.y
    if self.use_scale:
      scale = window.scale
      ratio = window.ratio      
      x_shift = float(x_off) / float(window.width)
      y_shift = float(y_off) / float(window.height)
    else: 
      width = window.width
      height = window.height
    
    key = self.word_grid_2_ind(word, grid)
#    print scale, ratio, x_shift, y_shift
    if self.windows_for_vg.has_key(key):
      if self.use_scale:
        self.windows_for_vg[key].append([scale,ratio,x_shift,y_shift])
      else:
        self.windows_for_vg[key].append([x_off,y_off,width,height])            
    else:
      if self.use_scale:
        self.windows_for_vg[key] = [[scale,ratio,x_shift,y_shift]]
      else:
        self.windows_for_vg[key] = [[x_off,y_off,width,height]]
      
  
  def word_grid_2_ind(self, word, grid):
    return word*self.num_grids + grid
  
  def ind_2_word_grid(self,ind):
    grid = int(ind % self.num_grids)
    word = int(ind) / int(self.num_grids)
    return (word, grid)    
      
  def get_annotations(self, positions, feature_type, codebook, cls, img):
    return self.e.get_assignments(positions, feature_type, codebook, img)
    
  def convert_tuple2bbox(self, intuple, image):
    """
    convert the top tuples to boxes. 
    intuple: [x_v, y_v, x_off, y_off, width, height]
    or: [x_v, y_v, scale, ratio, x_shift, y_shift] NOW
    """ 
    bbox = np.zeros((1,4))
    if self.use_scale:
      width = intuple[2]*image.size[1]
      height = intuple[3]*width
      x_off = intuple[4]*width
      y_off = intuple[5]*height      
      bbox[:,0] = intuple[0] - x_off
      bbox[:,1] = intuple[1] - y_off
      bbox[:,2] = width
      bbox[:,3] = height
    else:
      x_off = intuple[2]
      y_off = intuple[3]
      bbox[:,0] = intuple[0] - x_off
      bbox[:,1] = intuple[1] - y_off
      bbox[:,2] = intuple[4]
      bbox[:,3] = intuple[5]
    return bbox
  
  def compute_top_boxes(self, annots, positions, K, cut_tolerance):
    """ 
    compute the top K boxes in the sense of weights
    """
    weight_vector = self.weights.reshape(self.weights.size, 1)
    indices = np.asarray(weight_vector.argsort(axis=0))[::-1]
    # we will have at most K different weights, so we need the top K weights.
    top_weight_idx = indices
    top_boxes = np.zeros((K,6))
    
    curr_weight_idx = 0
    insert_count = 0
    go_on = True
    
    while go_on:
      if curr_weight_idx >= top_weight_idx.size:
        break
      indices = get_back_indices(top_weight_idx[curr_weight_idx], self.w_height,\
                                 self.w_width)      
      grid = indices[1][0]
      ann = indices[0][0]
      test_annos = np.where(annots == ann)[0]
      test_positions = positions[test_annos][:,0:2]
      if test_positions.size == 0:
        curr_weight_idx += 1
        continue
      # find the root windows that have ann at position grid
      if not ann in self.table:
        # we haven't never seen this annotations before. skip it
        curr_weight_idx += 1
        continue
            
      windex = self.word_grid_2_ind(ann, grid)
      if not windex in self.windows_for_vg:
        curr_weight_idx += 1
        continue
      
      windows = self.windows_for_vg[windex]
      for pos in test_positions:
        if go_on:
          for win in windows:
            top_boxes[insert_count] = [pos[0], pos[1], win[0], win[1], win[2],
                                       win[3]]
            insert_count += 1
            if insert_count == K:
              go_on = False
              break        
        else:
          break
      curr_weight_idx += 1
    return top_boxes
       
  def save_table(self, filename):
    outfile = open(filename, 'w')
    cPickle.dump(self, outfile)
    outfile.close()  
  
  def read_table(self, filename):
    infile = open(filename, 'r')
    content = cPickle.load(infile)    
    infile.close()
    return content

         
def display_bounding_boxes(bbxs, name, d,K):
  """ Draw and display bounding boxes.
  bbxs are samples X (x, y, w, h)
  """  
  image_filename = d.config.VOC_dir + '/JPEGImages/' + name
  os.system('convert ' + image_filename + ' bbox_tmp_img.png')   
  im = Image.open('bbox_tmp_img.png')
  draw = ImageDraw.Draw(im)    
  for k in range(K):
    draw.rectangle(((bbxs[k,0],bbxs[k,1]),(bbxs[k, 2] + bbxs[k, 0], \
                                           (bbxs[k, 3] + bbxs[k, 1]))))
  del draw    
  im.show()
  os.remove('bbox_tmp_img.png')
  

def get_back_indices(num, myM, myN):
  m = num%myM
  n = num/myM
  return (n, m)

ti = 0
def tic():
  global ti
  ti = time.time()
  
def toc(txt):
  tel = time.time() - ti
  print txt,':',tel

def unify_rows(arr):
  return np.array([np.array(x) for x in set(tuple(x) for x in arr)])
  

###############################################################
######################### Training ############################
###############################################################
def train_jumping_windows(train_set,use_scale=True,trun=False, diff=False):
  llc_dir = '../../research/jumping_windows/llc/'
  featdir = '../../research/jumping_windows/sift/'
  
  trainfiles = os.listdir(featdir)
  grids = 4
  
  for file in trainfiles:
    # Load feature positions
    feaSet = sio.loadmat(join(featdir,file))['feaSet']
    x = feaSet['x'][0][0]
    y = feaSet['y'][0][0]
    
    pts = np.hstack((x,y))
  
    codes = sio.loadmat(join(llc_dir,file))['codes'][0][0]
    
  
def count_occurence(annotations, positions):
  None
#  
#  if use_scale:
#    ut.makedirs(config.save_dir + 'JumpingWindows/scale/time/')
#     
#  for train_cls in all_classes:
#    # Codebook    
#    t = LookupTable_withgrid(codebook=codebook,use_scale=use_scale)   
#     
#    if use_scale:
#      save_table_file = config.save_dir + 'JumpingWindows/scale/' + train_cls
#      times_filename = config.save_dir + 'JumpingWindows/scale/time/' + train_cls
#    else:
#      save_table_file = config.save_dir + 'JumpingWindows/' + train_cls
#      times_filename = config.save_dir + 'JumpingWindows/time/' + train_cls
#    # Suppose we do this on just pos bboxes.
##    gt = d.get_ground_truth_for_class(train_cls, include_diff=diff,
##        include_trun=trun)
#    t_feat = time.time()
#    train_gt = d.get_pos_windows(train_cls)
#    print 'train on',train_gt.shape[0],'samples'
#    for row in train_gt:
#      bbox = row[0:4]
#      image = d.images[row[4]]
#      r = RootWindow_withgrid(bbox, image, M) 
#      #features = e.get_feature_with_pos('dsift', image, bbox)
#      t.add_features(bbox, r, train_cls, image) 
#    t_feat = time.time() - t_feat
#    
#    # Here should be the same as in MATLAB
#    
#    t_weight = time.time()
#    print 'collected all boxes, now compute weights'
#    t.compute_all_weights()
#    t_weight = time.time() - t_weight
#    print 'weights computed, perform mean shift'
#    
#    t_meanshift = time.time()    
#    t.perform_mean_shifts()
#    t_meanshift = time.time() - t_meanshift
#    t.save_table(save_table_file)
#    print 'lookup table for',train_cls,'saved. took:',t_weight+t_meanshift+t_feat
#    print save_table_file
#    
#    time_file = open(times_filename,'w')
#    time_file.writelines(['adding feats: '+str(t_feat),'\ncomp weights: '+\
#                          str(t_weight),'\nperform meanshift '+str(t_meanshift)])
    

###############################################################
######################### Testing #############################
###############################################################
# We now have a LookupTable_withgrid and want to determine the root windows
# in a new image.
def detect_jumping_windows_for_set(val_set, K, num_pos, all_classes):
  print 'Jumping Window Testing ...'
  ut.makedirs(config.save_dir + 'JumpingWindows/')
  d = Dataset(val_set)
  
  all_box_list = []  
  jwDect = JumpingWindowsDetectorGrid(warmstart=False, K=K)
  # This is not the cleverest way of parallelizing, but it is one way.
  for cls in all_classes:
    ut.makedirs(config.save_dir + 'JumpingWindows/w_'+cls +'/')
    print 'Testing class',cls
    pos_images = d.get_pos_samples_for_class(cls)
    if not num_pos == 'max':
      #rand = np.random.random_integers(0, len(pos_images) - 1, size=num_pos)
      pos_images = pos_images[0:num_pos]  
    
    bboxes = np.zeros((K*len(pos_images), 6))    
    #for idx in range(len(pos_images)):
    for idx in range(mpi_rank, len(pos_images), mpi_size):
      img_idx = pos_images[idx]
      cls_ind = d.get_ind(cls)
      image = d.images[img_idx.astype(Int)]
      save_file = config.save_dir + 'JumpingWindows/w_'+cls +'/' + image.name[:-4]
      if not os.path.isfile(save_file):
        print 'generate windows for',cls, image.name[:-4]
        bboxes_img = jwDect.detect_cls(image, cls, K)      
        bboxes[idx*K:(idx+1)*K, :] = np.hstack((bboxes_img,np.tile(cls_ind,(K,1)),\
                                                        np.tile(img_idx,(K,1))))        
        np.savetxt(save_file, bboxes[idx*K:(idx+1)*K, :])
      else:
        print 'windows for', cls, image.name[:-4], 'exist'
        bboxes[idx*K:(idx+1)*K, :] = np.loadtxt(save_file)
    all_boxes = np.zeros((K*len(pos_images), 6))
    #comm.barrier()
    #comm.Allreduce(bboxes, all_boxes)
    all_box_list.append(all_boxes)
  
  print 'all boxes:', len(all_box_list)
  all_boxes = np.vstack((all_box_list[i] for i in range(len(all_box_list))))
  print all_boxes.shape 

  return Table(all_boxes, ['x', 'y', 'w', 'h', 'cls_ind', 'img_ind'])
  
  #display_bounding_boxes(bboxes, image.name, d, K)

def mpi_get_sublist(rank, size, all_classes):
  # instead of mpiing with for and lists, create sublists right away
  mpi_bin_size = len(all_classes)/size
  all_nu_classes = all_classes[rank*mpi_bin_size:(rank+1)*mpi_bin_size]    
  if mpi_bin_size*size < len(all_classes):
    missing_size = len(all_classes)-mpi_bin_size*size
    missing_items = all_classes[(-missing_size):(len(all_classes))]
    if rank < missing_size:      
      all_nu_classes.append(missing_items[rank])  
  return all_nu_classes


if __name__=='__main__':
  all_classes = config.pascal_classes
  val_set = 'full_pascal_test'
  train_set = 'full_pascal_trainval'
  K = 3000
  num_pos = 'max'
  #num_pos = 1
  #all_classes = ['bird']
  use_scale = False
    
  train = True
  if train:
    # this is the codebook size
    # This is just for that it broke down during the night
    # MPI this
    all_classes = mpi_get_sublist(mpi_rank, mpi_size, all_classes)
    print all_classes    
    train_jumping_windows(train_set,use_scale=use_scale,trun=True,diff=False)
    
"""  
  classify = False
  if classify:
    print mpi_rank, 'at first barrier'
    print time.strftime('%m-%d %H:%M')
    safebarrier(comm)
    bbox_table = detect_jumping_windows_for_set(val_set, K, num_pos, all_classes)    
    print bbox_table.shape
    
  just_eval = True
  if just_eval:
    
    print 'start testing on node', mpi_rank
    dtest = Dataset('full_pascal_test')
    cls=all_classes[0]
    gt_t = dtest.get_ground_truth_for_class(cls, include_diff=False,
        include_trun=True)
    e = Extractor()
    codebook = e.get_codebook(dtest, 'dsift')
    t = LookupTable_withgrid(codebook=codebook)
    
    if use_scale:
      t = load_lookup_table(config.save_dir + 'JumpingWindows/scale/'+cls)
    else:
      t = load_lookup_table(config.save_dir + 'JumpingWindows/'+cls)
    
    test_gt = gt_t.arr
    npos = test_gt.shape[0]
    test_imgs = test_gt[:,gt_t.cols.index('img_ind')]
    test_imgs = np.unique(test_imgs)
    jwdect = JumpingWindowsDetectorGrid()
    jwdect.add_lookup_table(cls, t)    
    dets = []
    tp = 0
    for i in range(mpi_rank, len(test_imgs), mpi_size):
      img_ind = int(test_imgs[i])
      image = dtest.images[img_ind]
      det = jwdect.detect_cls(image, cls)
      for gt_row in ut.filter_on_column(test_gt,gt_t.cols.index('img_ind'), img_ind):
        for row in det:        
          ov = BoundingBox.get_overlap(row, gt_row[0:4])
          if ov > .5:
            tp += 1.
            break
    print 'tp at',mpi_rank,':', tp
  
    tp = comm.reduce(tp)
    if mpi_rank == 0:  
      rec = tp/npos
      print tp, npos
      print 'rec:' , rec
      
  just_eval2 = False
  if just_eval2:
    e = Extractor()
    d = Dataset(val_set)
    store_table_file = config.save_dir + 'bbox_table.txt'
    infile = open(store_table_file,'r')
    tic()    
    all_boxes = np.loadtxt(store_table_file)
    infile.close()
    bbox_table = Table(all_boxes, ['x', 'y', 'w', 'h', 'score', 'cls_ind', 'img_ind'])
    toc('detections loaded')
    
    if mpi_rank == 0:
      tic()
      evaluator = Evaluation(dataset=d, name='JumpingWindows/')
      evaluator.evaluate_detections_whole(bbox_table)
      toc('Evaluation time:')
  
  debugging = False
  if debugging:    
    cls = 'dog'
    d = Dataset('full_pascal_trainval')
    dtest = Dataset('full_pascal_test')
    e = Extractor()
    codebook = e.get_codebook(d, 'dsift')
    t = LookupTable_withgrid(codebook=codebook, use_scale=False)
        
    gt = d.get_ground_truth_for_class(cls, include_diff=False,
        include_trun=True)
    gt_t = dtest.get_ground_truth_for_class(cls, include_diff=False,
        include_trun=True)
    
    train_gt = gt.arr[:5,:]
    test_gt = gt_t.arr[:5,:]
    
    if mpi_rank == 0:
      for row_idx in range(train_gt.shape[0]):
        print 'learn sample',row_idx,'of',train_gt.shape[0]
        row = train_gt[row_idx,:]
        bbox = row[0:4]
        image = d.images[row[gt.cols.index('img_ind')].astype(Int)]
        r = RootWindow_withgrid(bbox, d.images[row[gt.cols.index('img_ind')].astype(Int)], M) 
        #features = e.get_feature_with_pos('dsift', image, bbox)
        t.add_features(bbox, r, cls, image) 
      
      print 'collected all boxes, now compute weights'
      t.compute_all_weights()  
      print 'weights computed, perform mean shift'
      t.perform_mean_shifts()
      
      #t.save_table(filename)
          
    t = comm.bcast(t)
    
    print t.weights.shape
    print 'start testing on node', mpi_rank
    npos = test_gt.shape[0]
    test_imgs = test_gt[:,gt.cols.index('img_ind')]
    test_imgs = np.unique(test_imgs)
    jwdect = JumpingWindowsDetectorGrid()
    jwdect.add_lookup_table(cls, t)    
    dets = []
    tp = 0
    for i in range(mpi_rank, len(test_imgs), mpi_size):
      img_ind = int(test_imgs[i])
      image = dtest.images[img_ind]
      det = jwdect.detect_cls(image, cls)
      max_ov = 0
      detected = False
      for gt_row in ut.filter_on_column(test_gt,gt.cols.index('img_ind'), img_ind):
        for row in det:        
          ov = BoundingBox.get_overlap(row, gt_row[0:4])
          if ov > .5:
            tp += 1.
            break
    print 'tp at',mpi_rank,':', tp
  
    tp = comm.reduce(tp)
    if mpi_rank == 0:  
      rec = tp/npos
      print tp, npos
      print 'rec:' , rec
#    for cls in d.config.pascal_classes:
#      gt = d.get_ground_truth_for_class(cls)
#      print cls, gt.arr.shape[0]
#    print 'all', d.get_ground_truth().arr.shape[0]
  print mpi_rank, 'finished...'
"""