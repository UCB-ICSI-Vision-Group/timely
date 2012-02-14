import numpy as np
from synthetic.classifier import *
from synthetic.pyramid import extract_pyramid, get_pyr_feat_size
import synthetic.config
from synthetic.dataset import Dataset
from synthetic.extractor import Extractor
from util import TicToc

comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()

def train_image_classifiers(dataset):
  e = Extractor()
  d = Dataset(dataset)
  tictocer = TicToc()
  dense_codebook = e.get_codebook(d, 'dsift')
  local_codebook = e.get_codebook(d, 'sift')
  L = 2
  pyr_feat_size = get_pyr_feat_size(L, dense_codebook.shape[0])
  all_classes = d.classes
  
  tictocer.tic('overall')
  for cls_idx in range(mpi_rank, len(all_classes), mpi_size): # PARALLEL
  #for cls in all_classes:
    cls = all_classes[cls_idx]
    print 'compute classifier for class', cls
    pos_images = d.get_pos_samples_for_class(cls)
    neg_images = d.get_neg_samples_for_class(cls, pos_images.size)
      
    # 1. extract all the pyramids    
    # ======== POSTIVE IMAGES ===========
    pos_pyrs = np.zeros((len(pos_images),pyr_feat_size + local_codebook.shape[0]))
    print 'compute feature vector for positive images'  
    for idx, img in enumerate(pos_images):
      tictocer.tic('image')
      print 'Pos image', img
      image = d.images[img]
      tictocer.tic()
      dense_assignments = e.get_assignments(np.array([0, 0, image.size[0]+1, image.size[1]+1]),\
                                      'dsift', dense_codebook, image, \
                                      sizes=[16,24,32],step_size=4)
      print '\t %f'%tictocer.toc(quiet=True)
      tictocer.tic()
      sparse_assignments = e.get_assignments([0,0,image.size[0]+1,image.size[1]+1], \
                                             'sift', local_codebook, image)
      print '\t %f'%tictocer.toc(quiet=True)
      positions = dense_assignments[:, 0:2]
      
      tictocer.tic()
      pyramid = extract_pyramid(L, positions, dense_assignments, dense_codebook, image)
      print '\textr pyramid %f'%tictocer.toc(quiet=True)
      
      tictocer.tic()
      bow = e.get_bow_for_image(d, local_codebook.shape[0], sparse_assignments, image)
      print '\textr bow %f'%tictocer.toc(quiet=True)
      
      bow_pyr = np.hstack((bow,pyramid))
      pos_pyrs[idx, :] = bow_pyr
      
      print '\t%f seconds for image %s'%(tictocer.toc('image', quiet=True),img)
  
    # ======== NEGATIVE IMAGES ===========
    print 'compute feature vector for negative images'
    neg_pyrs = np.zeros((len(neg_images),pyr_feat_size + local_codebook.shape[0]))  
    for idx, img in enumerate(neg_images):
      print 'Neg image', img
      image = d.images[img]
      dense_assignments = e.get_assignments(np.array([0, 0, image.size[0]+1, image.size[1]+1]),\
                                      'dsift', dense_codebook, image, \
                                      sizes=[16,24,32],step_size=4)
      sparse_assignments = e.get_assignments([0,0,image.size[0]+1,image.size[1]+1], \
                                             'sift', local_codebook, image)
      positions = dense_assignments[:, 0:2]
      
      tictocer.tic()
      pyramid = extract_pyramid(L, positions, dense_assignments, dense_codebook, image)
      print '\textr pyramid', tictocer.toc(quiet=True)
      
      tictocer.tic()
      bow = e.get_bow_for_image(d, local_codebook.shape[0], sparse_assignments, image)
      print '\textr bow', tictocer.toc(quiet=True)
      
      bow_pyr = np.hstack((bow,pyramid))
      neg_pyrs[idx, :] = bow_pyr
    
    
    X = np.vstack((pos_pyrs, neg_pyrs))
    Y = [1]*pos_pyrs.shape[0] + [-1]*neg_pyrs.shape[0] 
    #print Y.shape 
    
    print 'train svm for class', cls
    clf = train_svm(X, Y, kernel='rbf')
    
    filename = config.get_classifier_svm_name(cls)
    print 'save as', filename
    save_svm(clf, filename)
    
  print 'that all took:', tictocer.toc('overall', quiet=True), 'seconds on', mpi_rank
  
if __name__=='__main__':
  
  train_image_classifiers('full_pascal_trainval')