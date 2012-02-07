""" Implementation of Viajayanarasimhan and Graumann's Jumping Windows for
window candidate selection
@author: Tobias Baumgartner
@contact: tobibaum@gmail.com
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
import synthetic.util as util
from synthetic.extractor import Extractor
from synthetic.dataset import Dataset
import synthetic.config as config
from synthetic.detector import Detector
from numpy.ma.core import floor, ceil
from synthetic.bounding_box import BoundingBox
from sklearn.cluster import MeanShift
from synthetic.safebarrier import safebarrier
import math
from synthetic.mean_shift import MeanShiftCluster
import pickle

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
      t = LookupTable(codebook, self.M)          
      # Suppose we do this on just pos bboxes.
      gt = d.get_ground_truth_for_class(train_cls)      
      train_gt = gt.arr
      for row in train_gt:
        bbox = row[0:4]
        image = d.images[row[gt.cols.index('img_ind')].astype(Int)]
        r = RootWindow(bbox, d.images[row[gt.cols.index('img_ind')].astype(Int)],self.M) 
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
    self.cell_width = bbox[2]/float(self.M)
    self.cell_height = bbox[3]/float(self.M)
    self.scale = bbox[3]/image.size[1]
    self.ratio = bbox[2]/bbox[3]
  
  def add_feature(self, positions):
    """ We expect a 1x2 positions here: x y """
    # First compute the bin in which this positions belongs. We need the relative
    # coordinates of this feat to the RootWindow. 
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


class LookupTable:
  def __init__(self, grids, words, clswords, wordprobs, numcenters, cls):
    self.grids = grids
    self.size = grids*grids*words
    self.bb = {}
    self.clusters = {}
    self.clswords = clswords
    self.cls = cls
    self.wordprobs = wordprobs 
    self.numcenters = numcenters
   
  def insert(self, word, bb):    
    wordid = np.where(self.clswords == word)[0][0,0]
    if not wordid in bb:
      self.bb[wordid] = bb
    else:
      self.bb[wordid] = np.vstack((self.bb[wordid],bb))
      
  def perform_mean_shifts(self):
    # same bandwidth as Grauman
    for bb_all in self.bb.iteritems():            
      bb_mat = bb_all[1]
      bb_key = bb_all[0]
      bb_all = np.asmatrix(bb_mat)

      mat = np.matrix(bb_all)

      if bb_all.shape[0] < 2:
        self.clusters[bb_key] = mat 
        continue
      centers = MeanShiftCluster(mat, 0.25)
      self.clusters[bb_key] = centers
      
  def save(self):
    store_file = open( 'test_' + self.cls, 'w') 
    cPickle.dump(self, store_file)
  
  def compute_top_boxes(self, annots, positions, K, cut_tolerance,feature_type='sift'):
    """ compute the top K boxes in the sense of weights"""
    word_idx = 0    
    top_boxes = np.zeros((K, 4))

    insert_count = 0    
    go_on = True
    #annots = unify_rows(annots)
    
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
        curr_weight_idx += 1
        continue
      # find the root windows that have ann at position grid

      if not word in self.clusters:
        # we haven't never seen this annotations before. skip it
        curr_weight_idx += 1
        continue
     
      windows = self.clusters[word]
      for pos in test_positions[0]:
        pos = np.array(pos)[0]
        if go_on:
          for windex in range(windows.shape[0]):
            win = np.array(windows[windex,:])[0]           
            top_boxes[insert_count,:] = [pos[0] + win[0], pos[1] + win[1], pos[0] + win[2], pos[1] + win[3]]
            insert_count += 1
            if insert_count == K:
              go_on = False
              break        
        else:
          break

      word_idx += 1
    return top_boxes    

def get_indices_for_pos(positions, xmin, xmax, ymin, ymax):
  indices = np.matrix(np.arange(positions.shape[0]))
  indices = indices.reshape(positions.shape[0], 1)
  positions = np.asarray(np.hstack((positions, indices)))
  if not positions.size == 0:  
    positions = positions[positions[:, 0] >= xmin, :]
  if not positions.size == 0:
    positions = positions[positions[:, 0] <= xmax, :]
  if not positions.size == 0:  
    positions = positions[positions[:, 1] >= ymin, :]
  if not positions.size == 0:
    positions = positions[positions[:, 1] <= ymax, :]
  return np.asarray(positions[:, 2], dtype='int32')

def sub2ind(A, x, y):
  return y*A[0] + x

def ind2sub(width, ind):
  y = ind/width
  x = ind%width - 1
  return [x, y]    

def sort_cols(A,top_k=None):
  """
  Sort a matrix column-wise. Return sorted matrix and permuted indices
  """
  n_rows = A.shape[0]
  if not top_k == None:
    min(top_k, n_rows)    
  I = np.argsort(-A, axis=0, kind='mergesort')[:top_k]  
  b = mat_from_col_idx(A, I)
  return [b, I]

def mat_from_col_idx(A, I):
  """
  Generate Submatrix from column index
  """
  n_rows = I.shape[0]
  n_cols = A.shape[1]
  b =  np.asmatrix(np.zeros((n_rows, n_cols)))
  for col in range(n_cols):
    idc = np.asanyarray(I.T[col,:]).tolist()[0]
    if type(A[idc,col]) == type(np.array(None)):
      b[:,col] = np.asmatrix(A[idc,col]).T
    else:
      b[:,col] = A[idc,col]
  return b

def line_up_cols(A):
  return np.reshape(A.T,(A.size,1))

def get_selbins(grids, inds, c_pts, bbox):
  return np.minimum(  grids, 
    np.maximum(  1,
          ceil(
               (c_pts[inds,:] - 
               np.tile(np.matrix([bbox[0], bbox[1]]),(len(inds),1)))/ 
               np.tile(np.matrix([(bbox[2]-1)/grids, (bbox[3]-1)/grids]),(len(inds),1))
              )
        )
  );  
  
def get_idx(inds, codes, c_shape, feats, binidx):
  # Compute grid that each point falls into
    
  if feats == 'llc':
    ind = np.where(codes[:,inds].data > 0)[0]
    words = np.transpose(np.asmatrix(codes[:,inds].nonzero()[0][ind]))
    words += np.ones(words.shape)
    ind = codes.nonzero()[1][ind]
    
  elif feats == 'sift':
    ind = inds
    words = codes[ind]  
  
  idx = sub2ind(c_shape, words, binidx[ind])

  idx = np.unique(np.asarray(idx))
  idx = idx.astype('int32')
  
  return idx


def get_indices_for_pos(positions, xmin, xmax, ymin, ymax):
  indices = np.matrix(np.arange(positions.shape[0]))
  indices = indices.reshape(positions.shape[0], 1)
  positions = np.asarray(np.hstack((positions, indices)))
  if not positions.size == 0:  
    positions = positions[positions[:, 0] >= xmin, :]
  if not positions.size == 0:
    positions = positions[positions[:, 0] <= xmax, :]
  if not positions.size == 0:  
    positions = positions[positions[:, 1] >= ymin, :]
  if not positions.size == 0:
    positions = positions[positions[:, 1] <= ymax, :]
  return np.asarray(positions[:, 2], dtype='int32')


###############################################################
######################### Training ############################
###############################################################
def train_jumping_windows(d, codebook, use_scale=True, trun=False, diff=False, feats='sift'):
  # TODO: convert this to test-env
  llc_dir = '../../research/jumping_windows/llc/'
  featdir = '../../research/jumping_windows/sift/'
    
  trainfiles = os.listdir(featdir)
  grids = 4
  a = sio.loadmat(join(llc_dir,trainfiles[0]))['codes']
  numcenters = a.shape[0]
  ccmat = np.zeros((numcenters, len(d.classes)*grids*grids+1))
  e = Extractor()
  #first_visit = True
  for file in trainfiles:
    print file
    assignment = e.get_assignments([0,0,100000,1000000], 'sift', codebook, d.get_image_by_filename(file))
    # Load feature positions
    if feats == 'sift':
      pts = assignment[:,0:2]
      codes = assignment[2]
      
    elif feats == 'llc':
      feaSet = sio.loadmat(join(featdir,file))['feaSet']
      x = feaSet['x'][0][0]
      y = feaSet['y'][0][0]    
      pts = np.hstack((x,y))
      codes = sio.loadmat(join(llc_dir,file))['codes']
      
    bg = np.ones((pts.shape[0], 1))        
    
          
    image = d.get_image_by_filename(file[:-4]+'.jpg')
    im_ind = d.get_img_ind(image)
    gt = d.get_ground_truth_for_img_inds([im_ind])
    
    for row in gt.arr:
    #for row in gt.arr[0,:]:
      cls = row[gt.cols.index('cls_ind')]
      bbox = row[0:4]
      inds = get_indices_for_pos(pts, bbox[0], bbox[0]+bbox[2], bbox[1], bbox[1]+bbox[3])
      bg[inds] = 0
      
      selbins = get_selbins(grids, inds, pts, bbox) 
      binidx = np.transpose(sub2ind([grids, grids], selbins[:,0], selbins[:,1]) \
          + cls*grids*grids);
      
      idx = get_idx(inds, codes, ccmat.shape, feats, binidx)  
      
      for i in idx:
        [x, y] = ind2sub(ccmat.shape[0], i)        
        ccmat[x, y] = ccmat[x, y] + 1      
    
    # Now record background features
    cls = len(d.classes)*grids*grids
    inds = np.where(bg > 0)[0]

    if feats == 'llc':
      ind = np.where(codes[:,inds].data > 0)[0]
      words = codes[:,inds].nonzero()[0][ind]
      words = np.unique(words)
    elif feats == 'sift':
      words = codes[inds]
    
    for w in words:
      ccmat[w, cls] = ccmat[w, cls] + 1
    #sio.savemat('ccmat', {'ccmat2': ccmat})
    #break
  
  # counted all words for all images&object, now compute weights   
  div = np.sum(ccmat,1)
  for didx in range(len(div)):
    t = div[didx]
    if t == 0:
      ccmat[didx, :] = 2.5
      continue
    ccmat[didx, :] /= t
  
  numwords = 500
  [sortedprob, discwords] = sort_cols(ccmat, numwords)
  return
  # Lookup for every class
  for cls_idx in range(len(d.classes)):
  #for cls_idx in [14]:
    
    cls = d.classes[cls_idx]
    print cls
    clswords = discwords[:, cls_idx*grids*grids:(cls_idx+1)*grids*grids]
    binidx,_ = np.meshgrid(range(grids*grids), np.zeros((clswords.shape[0],1)))
    
    clswords = sub2ind([numcenters, grids*grids], line_up_cols(clswords), line_up_cols(binidx))
    
    # === GOOD! ===
    wordprobs = sortedprob[:, cls_idx*grids*grids:(cls_idx+1)*grids*grids];
    wordprobs = line_up_cols(wordprobs);
    [wordprobs, idx] = sort_cols(wordprobs);
      
    clswords = mat_from_col_idx(clswords, idx);
        
    bbinfo = LookupTable(grids, numwords, clswords, wordprobs, numcenters, cls)
    
    fileids = d.get_ground_truth_for_class(cls)    
    last_filename = '000000'    
    
    for row_idx in range(fileids.shape()[0]):
      row = fileids.arr[row_idx, :]
      filename = d.images[row[fileids.cols.index('img_ind')].astype('int32')].name[:-4]
    
      if not os.path.isfile(join(llc_dir, filename+'.mat')):
        continue
      print filename
      if not last_filename == filename:
        # This is a new file. Load codes and pts.
        feaSet = sio.loadmat(join(featdir,filename))['feaSet']
        x = feaSet['x'][0][0]
        y = feaSet['y'][0][0]    
        pts = np.hstack((x,y))
        bg = np.ones((pts.shape[0], 1))    
        codes = sio.loadmat(join(llc_dir,filename))['codes']
        last_filename = filename
      bbox = row[0:4]
      
      inds = get_indices_for_pos(pts, bbox[0], bbox[0]+bbox[2], bbox[1], bbox[1]+bbox[3])
            
      # Compute grid that each point falls into
      
      selbins = get_selbins(grids, inds, pts, bbox)      
      
      binidx = np.transpose(sub2ind([grids, grids], selbins[:,0]-1, selbins[:,1]-1));
          
      selbins = get_selbins(grids, inds, pts, bbox) 
      binidx = np.transpose(sub2ind([grids, grids], selbins[:,0], selbins[:,1]) \
          + cls*grids*grids);
      
      idx = get_idx(inds, codes, ccmat.shape, feats, binidx)
      
      if feats == 'llc':
        ind = np.where(codes[:,inds].data > 0)[0]
        ind = codes.nonzero()[1][ind]
      elif feats == 'sift':
        ind = inds
      #intersect idx and clswords
            
      clswords = np.unique(np.asarray(clswords))
      I = np.sort(idx)
      J = np.sort(ind)
      idx = np.unique(np.asarray(idx))
      imgwords = np.unique(np.intersect1d(np.asarray(idx), np.asarray(clswords), assume_unique=True))
      for c in imgwords:
        idc = np.where(I == c)[0]        
        featidx = inds[ind[idc]]
        featpts = pts[featidx]
        featpts = np.hstack((featpts, featpts))
        ins = np.tile(bbox, (featpts.shape[0],1)) - featpts + np.matrix([0,0,bbox[0]-1,bbox[1]-1])
        bbinfo.insert(c, ins)         
      
    bbinfo.perform_mean_shifts()
    bbinfo.top_words = np.asarray(bbinfo.wordprobs.argsort(axis=0))[::-1]
    bbinfo.save() 

def generate_jwin(bbinfo, im, cls, codes, pts):
  wordids = []
  pos = []
  
  #[I,J] = codes.nonzero()
  ind = np.where(codes.data > 0)[0]
  
  I = np.transpose(np.asmatrix(codes.nonzero()[0][ind]))
  J = np.transpose(np.asmatrix(codes.nonzero()[1][ind]))
  
  annots = np.hstack((pts[J.T.tolist()[0]], I))
  #print annots.shape
  return bbinfo.compute_top_boxes(annots, annots[:,0:2], K=3000, cut_tolerance = 0.5, feature_type = 'llc')
  
#  sio.savemat('IJ', {'I2': I, 'J2': J})
#  #print I, J
#  binidx = np.tile(range(bbinfo.grids*bbinfo.grids), (I.shape[0],1))
#  binidx = line_up_cols(binidx)
#  
#  print I.shape
#  print 'sumI', np.sum(I)
#  print 'meanI', np.mean(I)
#  print J.shape
#  print 'sumJ', np.sum(J)
#  print 'meanJ', np.mean(J)
#  
#  
#  I = line_up_cols(np.tile(I, (bbinfo.grids*bbinfo.grids, 1)))
#  J = np.tile(J, (bbinfo.grids*bbinfo.grids, 1))
#  I = sub2ind([bbinfo.numcenters, bbinfo.grids*bbinfo.grids], I, binidx)
#  

    
  # np.unique(np.intersect1d(np.asarray(bbinfo.clswords), np.asarray(I), assume_unique=False))
  

#  [c,ib,ia] = intersect(jw.clswords,I);
#  bboxes = [];
#  length(c)
#  [tmp,idx] = sort(jw.wordprobs(ib),'descend');
#  ia = ia(idx);


  None
               
###############################################################
######################### Testing #############################
###############################################################
# We now have a LookupTable and want to determine the root windows
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

  e = Extractor()
  d = Dataset(train_set)
  train = True
  if train:
    # this is the codebook size
    # This is just for that it broke down during the night
    # MPI this
    all_classes = mpi_get_sublist(mpi_rank, mpi_size, all_classes)
    print all_classes    
    codebook = e.get_codebook(d, 'sift')
    train_jumping_windows(d, codebook, use_scale=use_scale,trun=True,diff=False)
    
    store_file = open('test_dog', 'r')
    bbinfo = cPickle.load(store_file)
    
    # ==================
    llc_dir = '../../research/jumping_windows/llc/'
    featdir = '../../research/jumping_windows/sift/'
    d = Dataset('full_pascal_trainval')
    cls = 'dog'
    filename = '000750'
    im = d.get_image_by_filename(filename + '.jpg')
    feaSet = sio.loadmat(join(featdir,filename))['feaSet']
    x = feaSet['x'][0][0]
    y = feaSet['y'][0][0]    
    pts = np.hstack((x,y))
    
    codes = sio.loadmat(join(llc_dir,filename))['codes']
    
#    print codes
#    print pts
    wins = generate_jwin(bbinfo, im, cls, codes, pts)
    
    print wins[1:5,:]
    wins_true = np.matrix([[ 248.,  211.,  586.,  487.],
                           [ -16.,  -59.,  322.,  217.],
                           [  -8.,  -63.,  330.,  213.],
                           [ -88.,  309.,  250.,  585.]])
    np.testing.assert_equal(np.asarray(wins_true), np.asarray(wins[1:5,:]))
    
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
    t = LookupTable(codebook=codebook)
    
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
    t = LookupTable(codebook=codebook, use_scale=False)
        
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