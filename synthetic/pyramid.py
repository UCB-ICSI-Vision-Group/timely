""" Lazebnik's Spatial Pyramid Matching
@author: Tobias Baumgartner
@contact: tobi.baum@gmail.com
@date: 10/27/11
"""

import scipy.cluster.vq as sp
from scipy import io 

from common_imports import *
from common_mpi import *
import synthetic.config as config

from synthetic.dataset import Dataset
from synthetic.image import Image
from synthetic.extractor import count_histogram_for_bin,\
  count_histogram_for_slice

def get_pyr_feat_size(L, M):
  return (4**(L+1)-1)/3.*M

# This code should be in Extractor!
def extract_pyramid(L, positions, assignments, codebook, image):
  im_width = image.size[0]
  im_height = image.size[1]
  num_bins = 2**  L
  histogram_levels = []
  M = codebook.shape[0]
  histogram_level = np.zeros((num_bins,num_bins,M))
  #finest level
  for i in range(num_bins):
    for j in range(num_bins):
      [bin_ass, histogram] = count_histogram_for_bin(positions, assignments, im_width, im_height, num_bins, i, j, M)
      if not len(bin_ass) == 0:
        histogram_level[i,j,:] = np.divide(histogram,float(assignments.shape[0]))
      else:
        histogram_level[i,j,:] = histogram
        
  histogram_levels.append(histogram_level)
  
  # the other levels (just sum over lower levels respectively)  
  for level in range(L):
    num_bins = num_bins/2  
    level = level + 1
    lower_histogram = histogram_levels[level - 1]
    histogram_level = np.zeros((num_bins,num_bins,M))  
    for i in range(num_bins):
      for j in range(num_bins):
        histogram_level[i,j,:] = lower_histogram[2*i,2*j + 1,:] + \
          lower_histogram[2*i + 1,2*j,:] + lower_histogram[2*i,2*j,:] + \
          lower_histogram[2*i + 1,2*j + 1,:]
    histogram_levels.append(histogram_level)
    
  pyramid = []
  num_bins = 2**L
  for lev in range(L+1):
    if lev == L:
      power = -L
    else:
      power = -lev-1
    for m in range(M):
      for j in range(num_bins):         
        for i in range(num_bins):         
          pyramid.append(histogram_levels[lev][i,j,m]*2**(power))
    num_bins = num_bins/2
  pyramid = np.matrix(pyramid)
  return pyramid

def extract_horiz_sclices(num_bins, assignments, image, num_words):
  im_width = image.size[0]
  im_height = image.size[1]
  slices = []
  for i in range(num_bins):
    hist = np.matrix(count_histogram_for_slice(assignments, im_width, im_height, \
                        num_bins, i, num_words)[1])
    hist_sum = float(np.sum(hist))
    slices.append(hist/hist_sum)
  return slices

if __name__ =='__main__':
  assignments = np.zeros((25,3))
  ind = 0
  for i in range(5):
    for j in range(5):
      ass = 1
      if (i, j) == (1,1) or (i, j) == (2,1) or (i, j) == (3,1):
        ass = 2
      if (i, j) == (1,2) or (i, j) == (2,2) or (i, j) == (3,2):
        ass = 3
      if (i, j) == (1,3) or (i, j) == (2,3) or (i, j) == (3,3):
        ass = 4
      assignments[ind, :] = np.matrix([[i, j, ass]])
      ind += 1
  image = Image(size=(5,5))
  num_words = 4
  slices = extract_horiz_sclices(3, assignments, image, num_words)
  corr_stack = np.matrix([[7, 3, 0, 0],[2, 0, 3, 0],[7, 0, 0, 3]])
  slice_stack = np.vstack(slices)
  assert(corr_stack.all() == slice_stack.all())
  #print slices
    
    