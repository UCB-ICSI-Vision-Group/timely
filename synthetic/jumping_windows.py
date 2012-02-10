""" Implementation of Viajayanarasimhan and Graumann's Jumping Windows for
window candidate selection
@author: Tobias Baumgartner
@contact: tobibaum@gmail.com
"""

import cPickle
import numpy as np
import Image
import os as os
import synthetic.util as ut
from mpi4py import MPI
from numpy.numarray.numerictypes import Int
import scipy.io as sio
from os.path import join

from synthetic.extractor import Extractor
from synthetic.dataset import Dataset
import synthetic.config as config
from numpy.ma.core import ceil
from synthetic.mean_shift import MeanShiftCluster

comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()

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
    filename_lookup = join(config.data_dir, 'jumping_window','lookup', self.cls)
    store_file = open( filename_lookup, 'w')
    print 'save ', filename_lookup 
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
        word_idx += 1
        continue
      # find the root windows that have ann at position grid

      if not word in self.clusters:
        # we haven't never seen this annotations before. skip it
        word_idx += 1
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
    ind = range(binidx.size)
    words = codes[ind]
    #words = np.ones(words.shape)
  if binidx.size == 0:
    return []
  
  idx = sub2ind(c_shape, words, binidx[ind])

  idx = np.unique(np.asarray(idx))
  idx = idx.astype('int32')
  
  return idx


###############################################################
######################### Training ############################
###############################################################
def train_jumping_windows(d, codebook, use_scale=True, trun=False, diff=False, feature='sift'):
  tocer = ut.TicToc()
  llc_dir = '../../research/jumping_windows/llc/'
  featdir = '../../research/jumping_windows/sift/'
  
  if feature == 'sift':
    trainfile = join(config.VOC_dir, 'ImageSets','Main','trainval.txt')
    trainfiles = open(trainfile,'r').readlines()
  elif feature == 'llc':
    trainfiles = os.listdir(featdir)
    
  grids = 4
  if feature == 'sift':
    numcenters = codebook.shape[0]
  elif feature == 'llc':
    a = sio.loadmat(join(llc_dir,trainfiles[0]))['codes']
    numcenters = a.shape[0]
  ccmat = np.zeros((numcenters, len(d.classes)*grids*grids+1))
  e = Extractor()
  #first_visit = True
  print 'Read all features to create weights'
  tocer.tic()
  for filename in trainfiles:
    print filename

    # Load feature positions
    if feature == 'sift':
      filename = filename[:-1] + '.jpg'
      assignment = e.get_assignments(np.asarray([0,0,100000,1000000]), 'sift', codebook, d.get_image_by_filename(filename))
      pts = assignment[:,0:2]
      codes = assignment[:,2]
      
    elif feature == 'llc':
      feaSet = sio.loadmat(join(featdir,filename))['feaSet']
      x = feaSet['x'][0][0]
      y = feaSet['y'][0][0]    
      pts = np.hstack((x,y))
      codes = sio.loadmat(join(llc_dir,filename))['codes']
      
    bg = np.ones((pts.shape[0], 1))        
    
          
    image = d.get_image_by_filename(filename[:-4]+'.jpg')
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
      if feature == 'sift':
        binidx -= np.tile(grids, binidx.shape)
            
            
      idx = get_idx(inds, codes, ccmat.shape, feature, binidx)  
      
      for i in idx:        
        [x, y] = ind2sub(ccmat.shape[0], i)  
        #print x, y      
        ccmat[x, y] = ccmat[x, y] + 1      
    
    # Now record background features
    cls = len(d.classes)*grids*grids
    inds = np.where(bg > 0)[0]

    if feature == 'llc':
      ind = np.where(codes[:,inds].data > 0)[0]
      words = codes[:,inds].nonzero()[0][ind]
      words = np.unique(words)
    elif feature == 'sift':
      words = codes[inds]
    
    for w in words:
      ccmat[w, cls] = ccmat[w, cls] + 1
    #sio.savemat('ccmat', {'ccmat2': ccmat})
    #break
  
  print 'features counted'
  tocer.toc()
  tocer.tic()
  # counted all words for all images&object, now compute weights   
  div = np.sum(ccmat,1)
  for didx in range(len(div)):
    ta = div[didx]
    if ta == 0:
      ccmat[didx, :] = 2.5
      continue
    ccmat[didx, :] /= ta
  
  print 'computed weights'
  tocer.toc()
  
  tocer.tic()
  numwords = 500
  [sortedprob, discwords] = sort_cols(ccmat, numwords)
  #return
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
        # This is a new filename. Load codes and pts.
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
      
      idx = get_idx(inds, codes, ccmat.shape, feature, binidx)
      
      if feature == 'llc':
        ind = np.where(codes[:,inds].data > 0)[0]
        ind = codes.nonzero()[1][ind]
      elif feature == 'sift':
        ind = inds
      #intersect idx and clswords
            
      clswords = np.unique(np.asarray(clswords))
      I = np.sort(idx)
#      J = np.sort(ind)
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
    
  print 'computed all lookup tables' 
  tocer.toc()
  
def generate_jwin(bbinfo, im, cls, codes, pts, feature = 'sift'):
  if feature == 'llc':
    ind = np.where(codes.data > 0)[0]
    
    I = np.transpose(np.asmatrix(codes.nonzero()[0][ind]))
    J = np.transpose(np.asmatrix(codes.nonzero()[1][ind]))
    annots = np.hstack((pts[J.T.tolist()[0]], I))
    
  elif feature == 'sift':
    annots = codes
  
  return bbinfo.compute_top_boxes(annots, annots[:,0:2], K=3000, cut_tolerance = 0.5, feature_type = 'llc')
  

              
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

  use_scale = False

  e = Extractor()
  d = Dataset(train_set)
  train = False
  if train:
    # this is the codebook size
    # This is just for that it broke down during the night
    # MPI this
    feature = 'sift'        
    codebook = e.get_codebook(d, 'sift')
    ut.makedirs(join(config.data_dir, 'jumping_window','lookup'))
    train_jumping_windows(d, codebook, use_scale=use_scale,trun=True,diff=False, feature=feature)
  
  debug = True
  just_eval = True
  if just_eval:
    basedir = join(config.data_dir, 'jumping_window')
    foldname_det = join(basedir, 'detections')    
    foldname_lookup = join(basedir, 'lookup')
    ut.makedirs(foldname_det)
    
    print 'start testing on node', mpi_rank
    dtest = Dataset('full_pascal_test')
    #for cls_idx, cls in enumerate(all_classes):
    for cls_idx, cls in enumerate([all_classes[0]]):
      #cls=all_classes
      gt_t = dtest.get_ground_truth_for_class(cls, include_diff=False,
          include_trun=True)
      e = Extractor()
      codebook = e.get_codebook(dtest, 'sift')
            
      filename_lookup = join(foldname_lookup,cls)
      store_file = open(filename_lookup, 'r')
      bbinfo = cPickle.load(store_file)
            
      test_gt = gt_t.arr
      npos = test_gt.shape[0]
      test_imgs = test_gt[:,gt_t.cols.index('img_ind')]
      
      #test_imgs = np.unique(test_imgs)
      test_imgs = np.unique(test_imgs)[:1]
          
      for i in range(mpi_rank, len(test_imgs), mpi_size):
        img_ind = int(test_imgs[i])
        image = dtest.images[img_ind]
        codes = e.get_assignments(np.array([0,0,100000,100000]), 'sift', codebook, image)
        dets = generate_jwin(bbinfo, image, cls, codes, codes[:,0:2])
      
        bbox_curr = np.zeros((dets.shape[0], 7))
        ent_idx = 0
        num_rows = dets.shape[0]
        print dets
        for i, det in enumerate(dets):
          bbox_curr[i,:] = np.hstack((det, 1 - i/float((num_rows - 1)), cls_idx, img_ind))
        
        # save this detections
        
        filename = join(foldname_det, image.name + "_" + cls)
        np.savetxt(filename, bbox_curr)