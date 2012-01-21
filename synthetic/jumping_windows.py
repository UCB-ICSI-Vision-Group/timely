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

#from synthetic.evaluation import Evaluation
from synthetic.util import Table
from synthetic.extractor import Extractor
from synthetic.config import Config
#from synthetic.detector import Detector
#from synthetic.dataset import Dataset
from numpy.ma.core import floor
from synthetic.bounding_box import BoundingBox
from sklearn.cluster import MeanShift
from synthetic.safebarrier import safebarrier

comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()

#TODO: Should inherit from Detector (but probs with cyclic includes)
class JumpingWindowsDetector():
  def __init__(self, warmstart=False, K=3000, M=4, use_scale=True):
    """ Detector for jumping windows.
    cbwords - number of words codebooks have been created with
    cbsamp - number of samples codebooks have been trained with
    warmstart - load all lookup tables right away? (takes 5s per table)
    """
    self.cut_tolerance = 0.5
    self.M = M
    self.N = M
    self.K = K
    self.use_scale = use_scale
    self.lookupTables = {}
    self.e = Extractor()
    self.all_classes = Config.pascal_classes
    if warmstart:
      for cls in Config.pascal_classes:
        self.load_lookup_table(cls)
        
  def add_lookup_table(self, cls, table):
    self.lookupTables[cls] = table
    
  def load_lookup_table(self, cls):
    if self.use_scale:
      filename = Config.save_dir + 'JumpingWindows/scale/' + cls
    else:
      filename = Config.save_dir + 'JumpingWindows/' + cls
    t = load_lookup_table(filename)
    t.M = self.M
    t.N = self.N
    self.lookupTables[cls] = t
  
  def get_lookup_table(self, cls):
    if not cls in self.lookupTables:
      self.load_lookup_table(cls)
    return self.lookupTables[cls]
  
  def detect(self, img):
    """Detect bounding-boxes for all classes on image img
    return 5 column table [x,y,w,h,cls_ind]"""
    all_windows = np.zeros((len(self.all_classes) * K, 5))
    for cls_idx in range(len(self.all_classes)):
      cls = self.all_classes[cls_idx]
      all_windows[cls_idx * K:(cls_idx + 1) * K, :] = self.detect_cls(img, cls, self.K, self.cut_tolerance)      
    return all_windows
  
#  def get_windows(image,cls,with_time=True):
    
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

  def detect_jumping_windows(self, image, cls, e, t, K, cut_tolerance,
                             feature_type='sift'):  
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
    pos_bounds = np.array([0, 0, image.size[0], image.size[1]]) 
    annotations = t.get_annotations(pos_bounds, feature_type, t.codebook, \
                                    cls, image)
    positions = annotations[:, 0:2]
    toc('annotations')    
    tic()
    top_selection = t.compute_top_boxes(annotations, positions, K, cut_tolerance)
    toc('create feature tuples')
    bboxes = np.zeros((top_selection.shape[0], 4))
    print 'unique windows:', unify_rows(top_selection).shape
    tic()
    for i in range(K):    
      box = t.convert_tuple2bbox(top_selection[i, :], image)
      bboxes[i] = box
      #bboxes[i] = BoundingBox.clipboxes_arr(box,[0,0,image.size[1],image.size[0]])
    toc('compute bboxes')
    print bboxes.shape    
    return bboxes
    
#  def train_jw_detector(self, all_classes, train_set):
#    """Training"""
#    d = Dataset(train_set)
#    e = Extractor()  
#    ut.makedirs(Config.save_dir + 'JumpingWindows/')
#    ut.makedirs(Config.save_dir + 'JumpingWindows/'+str(self.M)+'/')
#    for train_cls in all_classes:
#      # Codebook
#      codebook_file = e.save_dir + 'dsift/codebooks/codebook' 
#         
#      save_table_file = Config.save_dir + 'JumpingWindows/' + train_cls      
#      if not os.path.isfile(codebook_file):
#        print 'codebook',codebook_file,'does not exist'
#        continue
#      codebook = np.loadtxt(codebook_file)    
#      t = LookupTable(codebook, self.M)          
#      # Suppose we do this on just pos bboxes.
#      gt = d.get_ground_truth_for_class(train_cls)      
#      train_gt = gt.arr
#      for row in train_gt:
#        bbox = row[0:4]
#        image = d.images[row[gt.cols.index('img_ind')].astype(Int)]
#        r = RootWindow(bbox, d.images[row[gt.cols.index('img_ind')].astype(Int)],self.M) 
#        features = e.get_feature_with_pos('dsift', image, bbox)
#        t.add_features(features, r)         
#      t.compute_all_weights()  
#      t.save_table(save_table_file)

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

# -------------------------------------------------- RootWindow
class RootWindow():
  """ A root window containing NxM grid cells, position and size"""
  def __init__(self, bbox, image, M):
    self.x = bbox[0]
    self.y = bbox[1] 
    self.M = M
    self.image = image
    self.width = bbox[2]
    self.height = bbox[3] 

# -------------------------------------------------- LookupTable
class LookupTable():
  """ Data structure for storing the trained lookup table. It's values are
  lists of tuples (grid-position, root window)"""
  def __init__(self, codebook, M=4, use_scale=True):
    self.codebook = codebook
    self.word_occurences = np.zeros((codebook.shape[0], 1))
    self.pos_word_occurences = np.zeros((codebook.shape[0], 1))
    self.windows_for_word = {}
    self.M = M
    self.N = M
    self.windows = [] 
    self.num_words = codebook.shape[0]
    self.num_grids = self.M * self.N
    self.e = Extractor()
    self.top_words = None
    self.use_scale = True
    
  def compute_all_weights(self):
    
    self.weights = np.nan_to_num(np.divide(self.pos_word_occurences, \
                                           self.word_occurences))    
    self.top_words = np.asarray(self.weights.argsort(axis=0))[::-1]
    
  def perform_mean_shifts(self):
    # same bandwidth as Grauman
    meanshifter = MeanShift(bandwidth=0.25, bin_seeding=False)
    i = 0
    t = time.time()
    for word in self.windows_for_word:
      # for each combination of word and grid do mean shift.
      ti = time.time() - t
      if ti > 2:
        print i, 'of', len(self.windows_for_word)
        t = time.time()      
      i += 1
      mat = np.asarray(self.windows_for_word[word])
      meanshifter.fit(mat)
      centers = meanshifter.cluster_centers_ 
      self.windows_for_word[word] = centers
          
  def build_histogram(self, assignments, words):
    counts = Counter(assignments.reshape(1, assignments.size).astype('float64')[0])
    histogram = [counts.get(x, 0) for x in range(words)]
    histogram = np.transpose(np.matrix(histogram, dtype='float64'))
    return histogram

    
  def add_features(self, bbox, root_win, cls, image, feature_type):
    all_assignments = self.get_annotations(np.array([0, 0, image.size[0], image.size[1]]), \
                                           feature_type, self.codebook, cls, image)
    # update the overall count of features
    self.word_occurences += self.build_histogram(all_assignments, \
                                                 self.codebook.shape[0])
    assignments = self.get_annotations(bbox, feature_type, self.codebook, cls, image)
    hist = self.build_histogram(assignments, self.codebook.shape[0])
    self.pos_word_occurences += hist
      
    for word_idx in range(assignments.shape[0]):
      if feature_type=='sift':
        idx = assignments[word_idx, 3]
        scale_dividend = assignments[word_idx,2] 
      else:
        idx = assignments[word_idx, 2]
        scale_dividend = image.size[1] 
      if hist[int(idx)] > 0:
        width = float(root_win.width)
        height = float(root_win.height)
        scale = float(height) / scale_dividend
        ratio = float(width) / root_win.height
        x_off = assignments[word_idx, 0] - root_win.x
        y_off = assignments[word_idx, 1] - root_win.x
        if self.use_scale:
          wind = np.array([scale, ratio, x_off / width, y_off / height])
        else:
          wind = np.array([x_off, y_off, width, height])
        self.add_window(idx, wind)
  
  def add_window(self, key, value):
    if self.windows_for_word.has_key(key):
      self.windows_for_word[key].append(value)
    else:
      self.windows_for_word[key] = [value]
         
  def get_annotations(self, positions, feature_type, codebook, cls, img):
    return self.e.get_assignments(positions, feature_type, codebook, img)
    
  def convert_tuple2bbox(self, intuple, image):
    """
    convert the top tuples to boxes. 
    intuple: [x_v, y_v, x_off, y_off, width, height]
    or: [x_v, y_v, scale, ratio, x_shift, y_shift] NOW
    """
    bbox = np.zeros((1, 4))
    if self.use_scale:    
      if intuple.shape[0] == 7:  
        width = intuple[2]*intuple[3]
        height = intuple[4]*width
        x_off = intuple[5]*width
        y_off = intuple[6]*height
      else:
        width = intuple[2]*image.size[1]
        height = intuple[3]*width
        x_off = intuple[4]*width
        y_off = intuple[5]*height
      bbox[:, 0] = intuple[0] - x_off
      bbox[:, 1] = intuple[1] - y_off
      bbox[:, 2] = width
      bbox[:, 3] = height
    else: 
      bbox[:, 0] = intuple[0] - intuple[2]
      bbox[:, 1] = intuple[1] - intuple[3]
      bbox[:, 2] = intuple[4]
      bbox[:, 3] = intuple[5]
    return bbox
  
  # TODO: solve all this with tables!!!
  def compute_top_boxes(self, annots, positions, K, cut_tolerance,feature_type='sift'):
    """ compute the top K boxes in the sense of weights"""
    word_idx = 0    
    if feature_type=='sift':
      num_cols = 7
    else:
      num_cols = 6
    top_boxes = np.zeros((K, num_cols))
    insert_count = 0    
    go_on = True
    annots = unify_rows(annots)
    while go_on:
      if word_idx >= self.top_words.shape[0]:
        go_on = False
        continue
      word = self.top_words[word_idx][0]
      if feature_type=='sift':
        test_annos = np.where(annots[:,3] == word)[0]
        test_positions = annots[test_annos][:, 0:3]
      else:
        test_annos = np.where(annots[:,2] == word)[0]
        test_positions = positions[test_annos][:, 0:2]
      
      if test_positions.size == 0:
        word_idx += 1
        continue
      # find the root windows that have ann at position grid
      if not word in self.windows_for_word:
        # we haven't never seen this annotations before. skip it
        word_idx += 1
        continue
      windows = self.windows_for_word[word]
      for pos in test_positions:
        if go_on:
          for win in windows:
            if feature_type=='sift':
              top_boxes[insert_count] = [pos[0], pos[1], pos[2], win[0], win[1], 
                                         win[2], win[3]]
            else:              
              top_boxes[insert_count] = [pos[0], pos[1], win[0], win[1], win[2],
                                       win[3]]
            insert_count += 1
            if insert_count == K:
              go_on = False
              break        
        else:
          break
      word_idx += 1
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

         
def display_bounding_boxes(bbxs, name, d, K):
  """ Draw and display bounding boxes.
  bbxs are samples X (x, y, w, h)
  """  
  image_filename = d.config.VOC_dir + '/JPEGImages/' + name
  os.system('convert ' + image_filename + ' bbox_tmp_img.png')   
  im = Image.open('bbox_tmp_img.png')
  draw = ImageDraw.Draw(im)    
  for k in range(K):
    draw.rectangle(((bbxs[k, 0], bbxs[k, 1]), (bbxs[k, 2] + bbxs[k, 0], \
                                           (bbxs[k, 3] + bbxs[k, 1]))))
  del draw    
  im.show()
  os.remove('bbox_tmp_img.png')
  

def get_back_indices(num, myM, myN):
  m = num % myM
  n = num / myM
  return (n, m)

ti = 0
def tic():
  global ti
  ti = time.time()
  
def toc(txt):
  tel = time.time() - ti
  print txt, ':', tel

def unify_rows(arr):
  return np.array([np.array(x) for x in set(tuple(x) for x in arr)])
  

###############################################################
######################### Training ############################
###############################################################
from synthetic.dataset import Dataset

def train_jumping_windows(all_classes, train_set, num_pos, use_scale=True, trun=False, \
                          diff=False, do_meanshift=True, feature_type='sift'):
  # Dataset
  d = Dataset(train_set)
  
  e = Extractor()
  print 'get cb'
  codebook = e.get_codebook(d, feature_type, force_new=False)
  print 'got cb'
  ut.makedirs(Config.save_dir + 'JumpingWindows/')
  ut.makedirs(Config.save_dir + 'JumpingWindows/time/')
  
  if use_scale:
    ut.makedirs(Config.save_dir + 'JumpingWindows/scale/time/')
     
  for train_cls in all_classes:
    # Codebook    
    t = LookupTable(codebook, use_scale=use_scale)   
     
    if use_scale:
      save_table_file = Config.save_dir + 'JumpingWindows/scale/' + train_cls
      times_filename = Config.save_dir + 'JumpingWindows/scale/time/' + train_cls
    else:
      save_table_file = Config.save_dir + 'JumpingWindows/' + train_cls
      times_filename = Config.save_dir + 'JumpingWindows/time/' + train_cls
    # Suppose we do this on just pos bboxes.
    t_gt = d.get_ground_truth_for_class(train_cls, include_diff=diff,
        include_trun=trun)
    train_gt = t_gt.arr
    if not num_pos=='max':
       train_gt = train_gt[:num_pos]
    #train_gt = d.get_pos_windows(train_cls,min_overlap=.8)
    t_feat = time.time()
    print 'train on', train_gt.shape[0], 'samples'
    for row in train_gt:
      bbox = row[0:4]
      image = d.images[int(row[t_gt.cols.index('img_ind')])]
      #image = d.images[int(row[4])]
      r = RootWindow(bbox, image, M) 
      #features = e.get_feature_with_pos('dsift', image, bbox)
      t.add_features(bbox, r, train_cls, image, feature_type) 
    t_feat = time.time() - t_feat
    
    t_weight = time.time()
    print 'collected all boxes, now compute weights'
    t.compute_all_weights()
    t_weight = time.time() - t_weight
    print 'weights computed'
    
    if do_meanshift:
      print 'perform mean shift'
      t_meanshift = time.time()    
      t.perform_mean_shifts()
      t_meanshift = time.time() - t_meanshift
    else:
      print 'no meanshift'
      
    t.save_table(save_table_file)
    print 'lookup table for', train_cls, 'saved. took:', t_weight + t_meanshift + t_feat
    print save_table_file
    
    time_file = open(times_filename, 'w')
    time_file.writelines(['adding feats: ' + str(t_feat), '\ncomp weights: ' + \
                          str(t_weight), '\nperform meanshift ' + str(t_meanshift)])
    

###############################################################
######################### Testing #############################
###############################################################
# We now have a LookupTable and want to determine the root windows
# in a new image.
def detect_jumping_windows_for_set(val_set, K, num_pos, all_classes):
  print 'Jumping Window Testing ...'
  ut.makedirs(Config.save_dir + 'JumpingWindows/')
  d = Dataset(val_set)
  
  all_box_list = []  
  jwDect = JumpingWindowsDetector(warmstart=False, K=K)
  # This is not the cleverest way of parallelizing, but it is one way.
  for cls in all_classes:
    ut.makedirs(Config.save_dir + 'JumpingWindows/w_' + cls + '/')
    print 'Testing class', cls
    pos_images = d.get_pos_samples_for_class(cls)
    if not num_pos == 'max':
      #rand = np.random.random_integers(0, len(pos_images) - 1, size=num_pos)
      pos_images = pos_images[0:num_pos]  
    
    bboxes = np.zeros((K * len(pos_images), 6))    
    #for idx in range(len(pos_images)):
    for idx in range(mpi_rank, len(pos_images), mpi_size):
      img_idx = pos_images[idx]
      cls_ind = d.get_ind(cls)
      image = d.images[img_idx.astype(Int)]
      save_file = Config.save_dir + 'JumpingWindows/w_' + cls + '/' + image.name[:-4]
      if not os.path.isfile(save_file):
        print 'generate windows for', cls, image.name[:-4]
        ################ MAIN METHOD
        bboxes_img = jwDect.detect_cls(image, cls, K)
        ################ MAIN METHOD
        bboxes[idx * K:(idx + 1) * K, :] = np.hstack((bboxes_img, np.tile(cls_ind, (K, 1)), \
                                                        np.tile(img_idx, (K, 1))))        
        np.savetxt(save_file, bboxes[idx * K:(idx + 1) * K, :])
      else:
        print 'windows for', cls, image.name[:-4], 'exists'
        bboxes[idx * K:(idx + 1) * K, :] = np.loadtxt(save_file)
    all_boxes = np.zeros((K * len(pos_images), 6))
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
  mpi_bin_size = len(all_classes) / size
  all_nu_classes = all_classes[rank * mpi_bin_size:(rank + 1) * mpi_bin_size]    
  if mpi_bin_size * size < len(all_classes):
    missing_size = len(all_classes) - mpi_bin_size * size
    missing_items = all_classes[(-missing_size):(len(all_classes))]
    if rank < missing_size:      
      all_nu_classes.append(missing_items[rank])  
  return all_nu_classes


if __name__ == '__main__':
  all_classes = Config.pascal_classes
  val_set = 'full_pascal_test'
  train_set = 'full_pascal_trainval'
  K = 3000
  num_pos = 'max'
  test_size = 'max'
  #num_pos = 10
  all_classes = ['aeroplane']
  cls = all_classes[0]
  use_scale = True
  M = 4 
  N = M
  train = True
  if train:
    # this is the codebook size
    # This is just for that it broke down during the night
    # MPI this
    all_classes = mpi_get_sublist(mpi_rank, mpi_size, all_classes)
    print all_classes    
    train_jumping_windows(all_classes, train_set, num_pos, use_scale=use_scale, \
                          trun=True, diff=False)
          
  classify = False
  if classify:
    print mpi_rank, 'at first barrier'
    print time.strftime('%m-%d %H:%M')
    safebarrier(comm)
    #os.remove(Config.save_dir + 'JumpingWindows/w_aeroplane/000032')
    bbox_table = detect_jumping_windows_for_set(val_set, K, num_pos, all_classes)    
    #print bbox_table    
  
  just_eval = True
  if just_eval:    
    print 'start testing on node', mpi_rank
    dtest = Dataset(val_set)
    gt_t = dtest.get_ground_truth_for_class(cls, include_diff=False,
        include_trun=True)
    e = Extractor()
    codebook = e.get_codebook(dtest, 'sift')
    t = LookupTable(codebook=codebook)
    
    if use_scale:
      t = load_lookup_table(Config.save_dir + 'JumpingWindows/scale/' + cls)
      print t.top_words
    else:
      t = load_lookup_table(Config.save_dir + 'JumpingWindows/' + cls)
    
    test_gt = gt_t.arr
    if not test_size=='max':      
      test_gt = test_gt[:test_size]
    npos = test_gt.shape[0]
    test_imgs = test_gt[:, gt_t.cols.index('img_ind')]
    test_imgs = np.unique(test_imgs)
    jwdect = JumpingWindowsDetector()
    jwdect.add_lookup_table(cls, t)    
    dets = []
    tp = 0
    for i in range(mpi_rank, len(test_imgs), mpi_size):
      img_ind = int(test_imgs[i])
      image = dtest.images[img_ind]
      det, time_elapsed = jwdect.get_windows(image, cls)
      max_ov = 0
  
      for gt_row in ut.filter_on_column(test_gt, gt_t.cols.index('img_ind'), img_ind):
        for row in det:        
          ov = BoundingBox.get_overlap(row, gt_row[0:4])
          if ov > .5:
            tp += 1.
            break
    print 'tp at', mpi_rank, ':', tp
  
    tp = comm.reduce(tp)
    if mpi_rank == 0:  
      rec = tp / npos
      print tp, npos
      print 'rec:' , rec
      
#  just_eval2 = False
#  if just_eval2:
#    e = Extractor()
#    d = Dataset(val_set)
#    store_table_file = Config.save_dir + 'bbox_table.txt'
#    infile = open(store_table_file,'r')
#    tic()    
#    all_boxes = np.loadtxt(store_table_file)
#    infile.close()
#    bbox_table = Table(all_boxes, ['x', 'y', 'w', 'h', 'score', 'cls_ind', 'img_ind'])
#    toc('detections loaded')
#    
#    if mpi_rank == 0:
#      tic()
#      evaluator = Evaluation(dataset=d, name='JumpingWindows/')
#      evaluator.evaluate_detections_whole(bbox_table)
#      toc('Evaluation time:')
  
  debugging = False
  if debugging:    
    cls = 'dog'
    d = Dataset('full_pascal_trainval')
    dtest = Dataset('full_pascal_test')
    e = Extractor()
    codebook = e.get_codebook(d, 'dsift')
    t = LookupTable(codebook=codebook, use_scale=False)
        
    gt = d.get_ground_truth_for_class(cls, include_diff=False,
        include_trun=True)
    gt_t = dtest.get_ground_truth_for_class(cls, include_diff=False,
        include_trun=True)
    
    train_gt = gt.arr[:5, :]
    test_gt = gt_t.arr[:5, :]
    
    if mpi_rank == 0:
      for row_idx in range(train_gt.shape[0]):
        print 'learn sample', row_idx, 'of', train_gt.shape[0]
        row = train_gt[row_idx, :]
        bbox = row[0:4]
        image = d.images[row[gt.cols.index('img_ind')].astype(Int)]
        r = RootWindow(bbox, d.images[row[gt.cols.index('img_ind')].astype(Int)], M) 
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
    test_imgs = test_gt[:, gt.cols.index('img_ind')]
    test_imgs = np.unique(test_imgs)
    jwdect = JumpingWindowsDetector()
    jwdect.add_lookup_table(cls, t)    
    dets = []
    tp = 0
    for i in range(mpi_rank, len(test_imgs), mpi_size):
      img_ind = int(test_imgs[i])
      image = dtest.images[img_ind]
      det = jwdect.detect_cls(image, cls)
      max_ov = 0
      detected = False
      for gt_row in ut.filter_on_column(test_gt, gt.cols.index('img_ind'), img_ind):
        for row in det:        
          ov = BoundingBox.get_overlap(row, gt_row[0:4])
          if ov > .5:
            tp += 1.
            detected = True
            break

    print 'tp at', mpi_rank, ':', tp
  
    tp = comm.reduce(tp)
    if mpi_rank == 0:  
      rec = tp / npos
      print tp, npos
      print 'rec:' , rec
#    for cls in d.config.pascal_classes:
#      gt = d.get_ground_truth_for_class(cls)
#      print cls, gt.arr.shape[0]
#    print 'all', d.get_ground_truth().arr.shape[0]
  print mpi_rank, 'finished...'
