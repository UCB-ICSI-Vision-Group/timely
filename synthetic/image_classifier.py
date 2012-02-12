import numpy as np
from synthetic.classifier import *
from synthetic.pyramid import extract_pyramid, get_pyr_feat_size
import synthetic.config
from synthetic.dataset import Dataset
from synthetic.extractor import Extractor

def train_image_classifiers(dataset):
  e = Extractor()
  d = Dataset(dataset)
  cls = 'dog'
  pos_images = d.get_pos_samples_for_class(cls)
  neg_images = d.get_neg_samples_for_class(cls, pos_images.size)
  dense_codebook = e.get_codebook(d, 'dsift')
  local_codebook = e.get_codebook(d, 'sift')
  L = 2
  pyr_feat_size = get_pyr_feat_size(L, dense_codebook.shape[0])
  # 1. extract all the pyramids
  
  pos_pyrs = np.zeros((len(pos_images),pyr_feat_size + local_codebook.shape[0]))  
  for idx, img in enumerate(pos_images):
    print 'image', img
    image = d.images[img]
    assignments = e.get_assignments(np.array([0, 0, image.size[0]+1, image.size[1]+1]),\
                                    'dsift', dense_codebook, image, \
                                    sizes=[16,24,32],step_size=4)
    positions = assignments[:, 0:2]
    pyramid = extract_pyramid(L, positions, assignments, dense_codebook, image)
    bow = e.get_bow_for_image(image, 'sift')
    bow_pyr = np.hstack((bow,pyramid))
    pos_pyrs[idx, :] = bow_pyr

  neg_pyrs = np.zeros((len(neg_images),pyr_feat_size + local_codebook.shape[0]))  
  for idx, img in enumerate(neg_images):
    print 'image', img
    image = d.images[img]
    assignments = e.get_assignments(np.array([0, 0, image.size[0]+1, image.size[1]+1]),\
                                    'dsift', dense_codebook, image, \
                                    sizes=[16,24,32],step_size=4)
    positions = assignments[:, 0:2]
    pyramid = extract_pyramid(L, positions, assignments, dense_codebook, image)
    bow = e.get_bow_for_image(image, 'sift')
    bow_pyr = np.hstack((bow,pyramid))
    neg_pyrs[idx, :] = bow_pyr

if __name__=='__main__':
  
  train_image_classifiers('test_pascal_train')