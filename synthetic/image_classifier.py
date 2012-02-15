from synthetic.common_imports import *
from synthetic.common_mpi import *

from synthetic.pyramid import extract_pyramid, get_pyr_feat_size
from synthetic.dataset import Dataset
from synthetic.extractor import Extractor
from util import TicToc
from synthetic.training import train_svm, save_svm, load_svm, svm_proba
from synthetic import config
from sklearn.cross_validation import KFold

comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()

class ClassifierConfig():
  def __init__(self, dataset, L, numfolds=4):
    self.d = Dataset(dataset)
    self.e = Extractor()
    self.dense_codebook = self.e.get_codebook(self.d, 'dsift')
    self.local_codebook = self.e.get_codebook(self.d, 'sift')
    self.tictocer = TicToc()
    self.L = L
    self.numfolds = numfolds
    
  def kfold(self):
    train_idx, val_idx = KFold(len(len(self.d.images), self.numfolds))
    self.train = self.d.images.

def get_feature_vector(cc, img, quiet=False):
  """
  return feature vector for given image, load precomputed vector if possible
  """
  savefilename = config.get_classifier_featvect_name(cc.d.images[img], L)  
  if os.path.isfile(savefilename):
    feat_vect = cPickle.load(open(savefilename,'r'))
  else:
    feat_vect = compute_feature_vector(cc, img, quiet=quiet)
    cPickle.dump(feat_vect, open(savefilename,'w'))
  return feat_vect
    
def compute_feature_vector(cc, img, quiet=False):
  cc.tictocer.tic('image')
  if not quiet:
    print 'Pos image', img
  image = cc.d.images[img]
  cc.tictocer.tic()
  dense_assignments = cc.e.get_assignments(np.array([0, 0, image.size[0]+1, image.size[1]+1]),\
                                  'dsift', cc.dense_codebook, image, \
                                  sizes=[16,24,32],step_size=4)
  if not quiet:
    print '\t %f'%cc.tictocer.toc(quiet=True)
  cc.tictocer.tic()
  sparse_assignments = cc.e.get_assignments([0,0,image.size[0]+1,image.size[1]+1], \
                                         'sift', cc.local_codebook, image)
  if not quiet:
    print '\t %f'%cc.tictocer.toc(quiet=True)
  positions = dense_assignments[:, 0:2]  
  cc.tictocer.tic()
  pyramid = extract_pyramid(cc.L, positions, dense_assignments, cc.dense_codebook, image)
  if not quiet:
    print '\textr pyramid %f'%cc.tictocer.toc(quiet=True)  
  cc.tictocer.tic()
  bow = cc.e.get_bow_for_image(cc.d, cc.local_codebook.shape[0], sparse_assignments, image)
  if not quiet:
    print '\textr bow %f'%cc.tictocer.toc(quiet=True)  
    print '\t%f seconds for image %s'%(cc.tictocer.toc('image', quiet=True),img)  
  return np.hstack((bow,pyramid))

def train_image_classifier():
  None

def train_image_classify_svm(cc, cls, C=1.0, gamma=0.0):  
  pyr_feat_size = get_pyr_feat_size(cc.L, cc.dense_codebook.shape[0])  
  
  cc.tictocer.tic('overall')
  
  filename = config.get_classifier_svm_name(cls)
  if os.path.exists(filename):
    continue
  print 'compute classifier for class', cls
  pos_images = cc.d.get_pos_samples_for_class(cls)
  neg_images = cc.d.get_neg_samples_for_class(cls, pos_images.size)
    
  # 1. extract all the pyramids    
  # ======== POSTIVE IMAGES ===========
  print 'compute feature vector for positive images'
  pos_pyrs = np.zeros((len(pos_images),pyr_feat_size + cc.local_codebook.shape[0]))      
  for idx, img in enumerate(pos_images):
    pos_pyrs[idx, :] = get_feature_vector(cc, img)

  # ======== NEGATIVE IMAGES ===========
  print 'compute feature vector for negative images'
  neg_pyrs = np.zeros((len(neg_images),pyr_feat_size + cc.local_codebook.shape[0]))  
  for idx, img in enumerate(neg_images):
    neg_pyrs[idx, :] = get_feature_vector(cc, img)
  
  # 2. Compute SVM for class
  X = np.vstack((pos_pyrs, neg_pyrs))
  Y = [1]*pos_pyrs.shape[0] + [-1]*neg_pyrs.shape[0] 
  
  if X.shape[0] > 0:
    print 'train svm for class %s'%cls
    cc.tictocer.tic()    
    clf = train_svm(X, Y, kernel='rbf', probab=True, gamma=gamma, C=C)
    print '\ttook %f seconds'%cc.tictocer.toc(quiet=True)
    
    print 'save as', filename
    save_svm(clf, filename)
  else:
    print 'Don\'t compute SVM, no examples given'
  
  print 'training all classifier SVMs took:', cc.tictocer.toc('overall', quiet=True), 'seconds on', mpi_rank
  
def classify_image(cc, img, cls=None):
  """
  Input: ClassifierConfig, img
  Output: Score for given class; 20list of scores if cls=None
  """
  feat_vect = get_feature_vector(cc, img)
  if cls == None:
    score = {}
    for cls_idx, cls in enumerate(cc.d.classes):
      print '\tclassify %s for %s on %d'%(cc.d.images[img].name, cls, comm_rank)
      path = config.get_classifier_svm_name(cls)
      if os.path.exists(path):
        clf = load_svm(path)
        score[cls_idx] = svm_proba(feat_vect, clf)
  else:
    clf = load_svm(config.get_classifier_svm_name(cls))
    score = svm_proba(feat_vect, clf)
  return score

def classify_all_images(cc):
  """
  For the given dataset in the ClassifierConfig cc compute and store all scores 
  """
  print 'Classify images'
  images = cc.d.images
  for img_idx in range(mpi_rank, len(images), mpi_size): # PARALLEL
    print 'classify image %d/%d at %d'%(img_idx/mpi_size, len(images)/mpi_size, comm_rank)
    img = images[img_idx]
    scores = classify_image(cc, img_idx)
    savefile = config.get_classifier_score_name(img, cc.L)
    cPickle.dump(scores, open(savefile,'w'))  
        
if __name__=='__main__':
  tictocer = TicToc()
  tictocer.tic('overall')
  
  test = False
  if test:
    train_dataset = 'test_pascal_train'
    eval_dataset = 'test_pascal_val'
  else:
    train_dataset = 'full_pascal_trainval'
    eval_dataset = 'full_pascal_test'
  
  # Train
  train = False
  L = 2
  if train:       
    cc = ClassifierConfig(train_dataset, L)  
    train_image_classify_svm(cc)
  
  safebarrier(comm)  
  # Evaluate  
  cc = ClassifierConfig(eval_dataset, L)    
  classify_all_images(cc)
  
  print 'Everything done in %f seconds'%tictocer.toc('overall',quiet=True)
  