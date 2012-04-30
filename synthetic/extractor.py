import numpy as np
import time
import subprocess as subp
from string import atoi
from numpy.numarray.numerictypes import Int
import scipy.cluster.vq as sp
from sklearn import cluster
from IPython import embed

import util as ut

from synthetic.config import *
from dataset import *
from common_mpi import *
from collections import Counter
from numpy.ma.core import floor

class Extractor():
  def __init__(self):

    self.data_dir = join(config.data_dir, 'features/')   
    
    if not os.path.isdir(self.data_dir):
      os.mkdir(self.data_dir)
    self.feature = ''
    
  def get_codebook(self,d, feature_type, num_words=3000,iterations=10, force_new=False, kmeansBatch=True):
    """ Load or create a codebook
    input:
      d - Dataset
      numpos - Number of positive samples ['max' for as much as existent]
      num_words - Number of num_words
      feature_type - type of feature_type to use
      train_class - pascal-class to train on
      force_new - delete existing codebook
      use_neg - Create with negative samples?
    """
    filename = config.get_codebook_path(feature_type)
    if (not os.path.isfile(filename)) or force_new:
      if force_new:
        if os.path.isfile(filename):
          os.remove(filename)
      print 'compute codebook:', filename 
      # select the windows to draw feature_type from
      gt = d.get_ground_truth()
      pos_wins = np.random.permutation(gt.arr)[:400]
      pos_wins = np.hstack((pos_wins[:,0:4],pos_wins[:,gt.cols.index('img_ind'):]))
      
      # dog here is just some random class, to read the window param from.
      neg_win = d.get_neg_windows(700,'dog')
      
      #featureList = []
      all_wins = np.concatenate((pos_wins, neg_win))
      all_wins = np.random.permutation(all_wins)
      num_feats = 150000
      
      idx = 0
      pers_max = num_feats/comm_size
      feat_mat = np.zeros((num_feats,131))
      for windex in range(comm_rank, all_wins.shape[0], comm_size):
        win = all_wins[windex,:]
        img = d.images[win[4].astype(Int)]
        feat = self.get_feature_with_pos(feature_type, img, win[0:4])
        feat = np.random.permutation(feat)[:min(200,pers_max - idx),:]
        feat_mat[idx+pers_max*comm_rank:idx+pers_max*comm_rank+feat.shape[0],:] = feat
          
        idx += feat.shape[0]
        if idx >= pers_max:
          break
        #featureList.append(feature_type)
           
      #print len(featureList), 'feature_type selected'
      #feature_matrix = np.vstack(featureList)
      feature_matrix = np.zeros((num_feats,131))
      comm.Reduce(feat_mat,feature_matrix)
            
      if not comm_rank == 0:
        return
        
      print feature_matrix.shape
      
      # remove 0 rows
      sums = np.sum(feature_matrix, 1)
      feature_matrix = feature_matrix[sums > 0,:]
      print feature_matrix.shape
      
      if feature_matrix.shape[0] > 100000:
        feature_matrix = np.random.permutation(feature_matrix)[:100000]
      
      feature_matrix = feature_matrix[:,3:]
      print "feat_mat: ", feature_matrix.shape
      print 'feat: ', feature_type
      
      time_filename = self.data_dir + feature_type + '/' + 'codebooks/time' 
  
      print 'start computing codebook...'
      #if comm_rank == 0: infile = open(filename, 'w')
      t_codebook = time.time()
      codebook = self.create_codebook(feature_matrix, num_words, iterations,\
                                 kmeansBatch=kmeansBatch)        
      t_codebook = time.time() - t_codebook
      if comm_rank == 0 or kmeansBatch:
        print 'saving codebook...' 
        print codebook.shape
        np.savetxt(filename, codebook.view(float))
        print 'stored in', filename
        #infile.close()
      if comm_rank == 0 or kmeansBatch:
        time_file = open(time_filename, 'w')
        time_file.write(str(t_codebook))
        time_file.close() 
    else:
      print 'loading codebook '
      codebook = np.loadtxt(filename)  
    
    return codebook  
  
  def create_codebook(self,feature_list, means, iterations,kmeansBatch=True):
    #feature_type = np.matrix([feature_list[i][j][4:] for j in arange(len(feature_list[i])) ])
  
    feature_type = sp.whiten(feature_list)
    print comm_rank,'starts k-means'
    
    
    if kmeansBatch:
      print 'kmeansBatch started, chunksize:', means*10, ', iterations: ', iterations
      kmeaner = cluster.MiniBatchKMeans(means,max_iter=iterations, tol=0.01,chunk_size=means*10)
      codebook = np.matrix(kmeaner.fit(feature_type).cluster_centers_)
    else:
      print 'k-means with',comm_size,'nodes and',iterations/comm_size+1,'iterations each'
      codebook = sp.kmeans(feature_type, means, iter=iterations/comm_size+1)  
      distortion = codebook[1]
      print comm_rank, distortion
      codebook = np.matrix(codebook[0])
      min_codebook = 10
      minimum = comm.allreduce(distortion, min_codebook,op=MPI.MIN)
      codebook = codebook[0]
    
      if distortion == minimum:
        comm.bcast(codebook[0], root=comm_rank)
    return codebook
  
  def get_image_feature(self, d , feature_type, img_ind, bound_box):
    image = d.images[img_ind.astype(Int)]
    print image.name
    # meassure the time it took and print to file
    t = time.time()
    feature = self.get_feature(feature_type, image, bound_box)
    t = time.time() - t
    filename = self.data_dir + feature_type + '/times/' + \
      image.name[0:-4]
    if not os.path.isfile(filename):
      f = open(filename, 'w')
      f.write(str(t))
      f.close()
    return feature

  
  def process_img(self, img, feature, sizes=[16,24,32], step_size=4):
    filename = self.data_dir + feature + '/' + img.name[0:-4]               

    # since our parallelism just uses different images, we don't check the file 
    # for correct size.
    if not os.path.isfile(filename):
      # extract featues and write to file
      print 'extracting',feature,'for',img.name[0:-4],'...'
      feature_type = self.dense_extract(config.get_image_path(img),
                      feature, (0, 0, img.size[0], img.size[1]),sizes, step_size)
      if feature[0:4] == 'phog':
        np.savetxt(filename, feature_type.view(float))
      else:  
        np.savetxt(filename, feature_type, fmt='%d')  
    else:
      print img.name[0:-4], 'already exists'
 
  def get_assignments(self, positions, feature_type, codebook, img, sizes=[16,24,32],step_size=4):
    filename = config.get_assignments_path(feature_type, img)
    if not os.path.isfile(filename):
      print 'compute assignment for ' + img.name[0:-4]      
      feature = self.get_feature_with_pos(feature_type,img,[0,0,img.size[0],img.size[1]], sizes=sizes, step_size=step_size)
      features_white = sp.whiten(feature[:,3:])
      assignments = sp.vq(features_white, codebook)[0]
      assignments = assignments.reshape(feature.shape[0], 1)
      if feature_type=='sift':
        assignments = np.hstack((feature[:,0:3], assignments))
      else:
        assignments = np.hstack((feature[:,0:2], assignments))
      np.savetxt(filename, assignments, fmt='%d')
    else:
      print 'load assignment:',img.name[0:-4]
      assignments = np.loadtxt(filename)
    if not positions:
      positions = [0,0,10000000,10000000]
    if type(positions) == type([]):
      bbox = [positions[0],positions[1],positions[0]+positions[2],positions[1]+\
              positions[3]]
    else:
      if not positions.size == 4: 
        bbox = [np.amin(positions[:,0]), np.amin(positions[:,1]), 
              np.amax(positions[:,0]), np.amax(positions[:,1])]
      else:
        bbox = [positions[0],positions[1],positions[0]+positions[2],positions[1]+\
              positions[3]]
      
    if not assignments.size == 0:  
      assignments = assignments[assignments[:, 0] >= bbox[0], :]
    if not assignments.size == 0:
      assignments = assignments[assignments[:, 0] <= bbox[2], :]
    if not assignments.size == 0:  
      assignments = assignments[assignments[:, 1] >= bbox[1], :]
    if not assignments.size == 0:
      assignments = assignments[assignments[:, 1] <= bbox[3], :]
    return assignments
 
  def get_feature(self, feature_type, img, bound_box):
    ret = self.get_feature_with_pos(feature_type, img, bound_box)
    return ret[:,3:]
  
  def get_feature_with_pos(self, feature_type, img, bound_box, sizes=[16,24,32],step_size=4):
    """ feature_type = 'sift', 'dsift', 'phow', 'cphow', 'phog180', 'phog360'
        data_collection = 'val', 'train', 'test' 
        img: Image instance
        bound_box: [x, y, w, h]"""
        
    self.feature = feature_type
    self.process_img(img, feature_type, sizes=sizes,step_size=step_size)
    filename = self.data_dir + feature_type + '/' + img.name[0:-4]
    try:
      feature = np.loadtxt(filename)
    except ValueError:
      print 'loading feature for', img.name[0:-4],'failed, redoing it!...'
      os.remove(filename)
      return self.get_feature_with_pos(feature, img, bound_box)
    
    # Crop the matrix to fit the boundingbox
    x_min = bound_box[0]
    y_min = bound_box[1]
    x_max = bound_box[0] + bound_box[2] - 1
    y_max = bound_box[1] + bound_box[3] - 1
       
    if not feature.size == 0:
      feature = feature[feature[:,0] >= x_min,:]
    if not feature.size == 0:
      feature = feature[feature[:,0] <= x_max,:]
    if not feature.size == 0:
      feature = feature[feature[:,1] >= y_min,:]
    if not feature.size == 0:
      feature = feature[feature[:,1] <= y_max,:]
    # Forget spatial information
    return feature
  
  def dense_extract(self,infile, feature, bbox, sizes, step_size):
    """Extract specified feature_type from image."""
    feature = feature.lower()   
      
    if feature[0:4] == "phog":
      level = 2
      process = subp.Popen(['./features_cpp/feature_type', infile, feature, 
                         str(bbox[0]), str(bbox[1]), str(bbox[2]), str(bbox[3]), 
                         str(bin), str(level)], shell=False, stdout=subp.PIPE)
      result = np.matrix(process.communicate()[0].split())
    else:
      if feature[0:5] == "dsift":
        mats = []
        for size in sizes:
          
          process = subp.Popen(['./extract_features/extract_sift.ln', '-sift','-i', infile,\
                                '-dense', str(step_size), str(step_size), '-dradius', str(size)], shell=False, \
                                stdout=subp.PIPE)
          result = process.communicate()[0].split()
          result_file = result[len(result)-1]
          result = open(result_file).read()
          os.remove(result_file)
          num_feat = atoi(result.split()[1])
          len_mat = len(result.split())-2
          result = result.split()[2:]
          # The first two lines of the infile describe number of feature_type and feat size
          mat = np.matrix(result, dtype=float)
          # Get the correct shape to work with [SAMPLES, FEAT-DIMS]
          mat = mat.reshape(num_feat, len_mat/num_feat)
          # Cut out columns 3-5. These are meaningless to us, add size and round to
          # integers
          mat = (np.hstack((mat[:,0:2],np.tile(size,(num_feat,1)),mat[:,5:])).\
                 round()).astype(Int)    
          mats.append(mat)
        mat = np.vstack(mats)

      elif feature[0:4] == "sift":
        # ./extract_sift.ln -heslap -sift -i  ~/Documents/Vision/data/input.jpg -o2 out.desc
        mats = []
      
        process = subp.Popen(['./extract_features/extract_sift.ln', '-heslap',\
                              '-sift','-noangle','-i', infile, '-o2', infile+'.heslap.sift'], shell=False, \
                              stdout=subp.PIPE)
        process.communicate()
        result_file = infile + '.heslap.sift'
        result = open(result_file).read()
        os.remove(result_file)
        result_lines = result.split('\n')
        num_feat = atoi(result_lines[1])
        result = result[65:].split()
        len_mat = len(result)
        mat = np.matrix(result, dtype=float)
        # Get the correct shape to work with [SAMPLES, FEAT-DIMS]
        mat = mat.reshape(num_feat, len_mat/num_feat)
        # There are 128 + 38 = 166 columns. We only want 131.
        # select columns for x, y and size.
        x_column = mat[:,2]
        y_column = mat[:,3]
        size_column = mat[:,5]
        mat = mat[:,38:]        
        mat = np.hstack((x_column, y_column, size_column, mat))
        mats.append(mat)
        mat = np.vstack(mats)
        print mat.shape
      else:
        print 'feature_type not understood: ', feature     
      result = mat
          
    if not result.shape[1] == 131:
      print 'something went wrong, probably feature unknown, give it another try'
    return result
  
  def create_image_feature(self,image, feature, sizes, step_size):
    print image.name
    # meassure the time it took and print to file
    t = time.time()
    self.feature = feature
    self.process_img(image, feature, sizes, step_size)
    t = time.time() - t
    filename = self.data_dir + feature + '/times/' + image.name[0:-4]
    if not os.path.isfile(filename):
      f = open(filename, 'w')
      f.write(str(t))
      f.close()
    
  def extract_all(self,feature_type,image_sets,sizes, step_size):    
    if not os.path.isdir(self.data_dir):
      os.mkdir(self.data_dir)    
    for imset in image_sets:#'train.txt', 'test.txt']:
      d = Dataset(imset)
      images = d.images    
      for feature in feature_type:#, 'phow', 'cphow', 'phog180', 'phog360']:          
        ut.makedirs(self.data_dir + feature + '/' )
        ut.makedirs(self.data_dir + feature + '/times/')               
        for img in range(comm_rank, len(images), comm_size): # PARALLEL
          image = images[img]
          self.create_image_feature(image, feature, sizes, step_size)
          
  def extract_all_assignments(self, feature, image_sets,all_classes,numpos=15, \
                              num_words=3000, iterations = 10, sizes=[16,24,32],step_size=4):
    if not os.path.isdir(self.data_dir):
      os.mkdir(self.data_dir)
    for imset in image_sets:
      d = Dataset(imset)
      codebook = self.get_codebook(d, feature, num_words=3000, iterations=iterations)
      images = d.images
      for img in range(comm_rank, len(images), comm_size): # PARALLEL
        print ''
        image = images[img]
        pos_bounds = [0,0,image.size[0],image.size[1]]
        self.get_assignments(pos_bounds, feature, codebook, image, sizes=sizes,step_size=step_size)
        embed()
            
  def get_bow_for_image(self, d, num_words, ass, image):
    width = image.size[0]
    height = image.size[1]
    _, histogram = count_histogram_for_bin(ass[:,0:2], ass, width, height, \
                                        num_bins=1, i=0, j=0, \
                                        num_words=num_words)
    return np.asmatrix(histogram)/float(ass.shape[0])    


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

def count_histogram(indices, assignments, num_words):  
  if indices.size == 0:
    bin_ass = np.matrix([])
  else:
    bin_ass = assignments[indices][:,2]    
  bin_ass = bin_ass.reshape(1,bin_ass.size)[0] - 1
  counts = Counter(bin_ass)
  histogram = [counts.get(x,0) for x in range(num_words)]
  return bin_ass, histogram

def count_histogram_for_bin(positions, assignments, im_width, im_height, num_bins, i, j, num_words):
  xmin = floor(im_width / float(num_bins) * i)
  xmax = floor(im_width / float(num_bins) * (i + 1))
  ymin = floor(im_height / float(num_bins) * j)
  ymax = floor(im_height / float(num_bins) * (j + 1))
  indices = get_indices_for_pos(positions, xmin, xmax, ymin, ymax)
  return count_histogram(indices, assignments, num_words)

def count_histogram_for_slice(assignments, im_width, im_height, num_bins, i, num_words):
  positions = assignments[:, 0:2]
  xmin = 0
  xmax = im_width
  ymin = (im_height-1) / float(num_bins) * i
  ymax = (im_height-1) / float(num_bins) * (i + 1)
  indices = get_indices_for_pos(positions, xmin, xmax, ymin, ymax)
  return count_histogram(indices, assignments, num_words)

if __name__ == '__main__':
  e = Extractor()
  image_sets = ['full_pascal_trainval','full_pascal_test']

  feature_type = 'dsift'
  sizes = [16,24,32]
  step_size = 4
  
  #e.extract_all(feature_type,image_set, sizes, step_size)
  all_classes = config.pascal_classes
#  all_classes = ['cat']
  e.extract_all_assignments(feature_type, image_sets, all_classes, sizes=sizes, \
                            step_size=step_size)
   
#  e.get_codebook(d, 'dsift', 3000, True, True)
