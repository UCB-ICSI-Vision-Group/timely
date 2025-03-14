from synthetic.common_imports import *
from synthetic.common_mpi import *

from synthetic.pyramid import extract_pyramid, get_pyr_feat_size,\
  extract_horiz_sclices
from synthetic.dataset import Dataset
from synthetic.extractor import Extractor
from util import TicToc
from synthetic.training import train_svm, save_svm, load_svm, svm_predict
from synthetic import config
from sklearn.cross_validation import KFold
from synthetic.image import Image
import itertools

class ClassifierConfig():
  def __init__(self, dataset, L, numfolds=4):
    self.d = Dataset(dataset)
    self.e = Extractor()
    self.dense_codebook = self.e.get_codebook(self.d, 'dsift')
    self.sparse_codebook = self.e.get_codebook(self.d, 'sift')
    self.L = L
    self.numfolds = numfolds
    
  def kfold(self):
    train_idx, val_idx = KFold(len(len(self.d.images), self.numfolds))
    self.d.create_folds(self.numfolds)

def get_feature_vector(cc, img, quiet=False):
  """
  return feature vector for given image, load precomputed vector if possible
  """
  savefilename = config.get_classifier_featvect_name(cc.d.images[img])  
  if os.path.isfile(savefilename):
    print 'load feat_vect %s'%(cc.d.images[img].name)
    feat_vect = cPickle.load(open(savefilename,'r'))
  else:
    feat_vect = compute_feature_vector(cc, img, quiet=quiet)
    cPickle.dump(feat_vect, open(savefilename,'w'))
  return feat_vect
    
def compute_feature_vector(cc, img_idx, quiet=False):
  cc.tictocer.tic('image')
  if not quiet:
    print 'Image', cc.d.images[img_idx].name
  image = cc.d.images[img_idx]
  
  cc.tictocer.tic()
  dense_assignments = cc.e.get_assignments(np.array([0, 0, image.size[0]+1, image.size[1]+1]),\
                                  'dsift', cc.dense_codebook, image, \
                                  sizes=[16,24,32],step_size=4)
  if not quiet:
    print '\tdns_ass %d took %f'%(img_idx, cc.tictocer.toc(quiet=True))
    
  cc.tictocer.tic()
  sparse_assignments = cc.e.get_assignments([0,0,image.size[0]+1,image.size[1]+1], \
                                         'sift', cc.sparse_codebook, image)
  if not quiet:
    print '\tsp_ass %d took %f'%(img_idx, cc.tictocer.toc(quiet=True))
    
  positions = dense_assignments[:, 0:2]  
  
  # Extract dense Pyramid
  cc.tictocer.tic()
  pyramid = extract_pyramid(cc.L, positions, dense_assignments, cc.dense_codebook, image)
  if not quiet:
    print '\textract pyramid %f'%cc.tictocer.toc(quiet=True)  
  
  # Extract sparse Bag of Words
  cc.tictocer.tic()
  bow = cc.e.get_bow_for_image(cc.d, cc.sparse_codebook.shape[0], sparse_assignments, image)
  if not quiet:
    print '\textract bow %f'%cc.tictocer.toc(quiet=True)  
        
  # Extract dense slices
  num_slices = 3
  cc.tictocer.tic()
  slices = extract_horiz_sclices(num_slices, dense_assignments, image, cc.sparse_codebook.shape[0])
  slices = np.hstack(slices)
  if not quiet:
    print '\textract slices %f'%cc.tictocer.toc(quiet=True)
  
  if not quiet:
    print '\t%f seconds for image %s'%(cc.tictocer.toc('image', quiet=True),img_idx)
  return np.hstack((bow, pyramid, slices))

def train_all_svms(cc, Cs, gammas, numfolds=4):
  cc.d.create_folds(numfolds)
  all_settings = list(itertools.product(Cs, gammas,cc.d.classes))

  for set_idx in range(comm_rank, len(Cs)*len(gammas), comm_size): # Parallel
    curr_set = all_settings[set_idx]
    C = curr_set[0]
    gamma = curr_set[1]
    cls = curr_set[2]
    
    for _ in range(numfolds):
      cc.d.next_folds()
      train_image_classify_svm(cc, cls=cls, C=C, gamma=gamma)
    cc.d.create_folds(numfolds)


def cross_valid_training(cc, Cs, gammas, numfolds=4, train=True):
  if train:
    train_all_svms(cc, Cs, gammas, numfolds=numfolds)
  
  all_settings = list(itertools.product(Cs, gammas))

  for set_idx in range(comm_rank, len(all_settings), comm_size): # Parallel
    curr_set = all_settings[set_idx]
    C = curr_set[0]
    gamma = curr_set[1]
    class_corr = 0
    overall = 0
        
    for _ in range(numfolds):
      cc.d.next_folds()
      val_set = cc.d.val     
      
      classification = classify_images(cc, val_set, C, gamma)
      overall += classification.size
      class_corr += validate_images(cc, val_set, classification)
      print 'correct:', class_corr
      #break
    accuracy = float(class_corr)/float(overall)
    filename = config.get_classifier_crossval()
    writef = open(filename, 'a')
    writef.write('%f %f - %f\n'%(C, gamma, accuracy))
    cc.d.create_folds(numfolds)

def classify_images(cc, images, C, gamma):
  res = np.zeros((images.shape[0], len(cc.d.classes)))
  for cls_idx, cls in enumerate(cc.d.classes):
    filename = config.get_classifier_svm_name(cls, C, gamma, cc.d.current_fold)
    if not os.path.exists(filename):
      continue
    clf = load_svm(filename, probability=False)
    for idx2, img_idx in enumerate(images):
      x = get_feature_vector(cc, img_idx)
      pred = svm_predict(x, clf)
      if pred.size > 0:
        res[idx2, cls_idx] = 1
  return res*2-1

def validate_images(cc, image_inds, classifications):
  """
  images: list of image indidces 
  classifications: binar classifications (num_img x num_classes) 
  return accuracy
  """
  gt = get_gt_classification(cc, image_inds)
  print gt
  print classifications
  print gt+classifications
  comb = np.where(gt + classifications)[0].size
  return comb
  

def get_gt_classification(cc, image_inds):
  """
  For given image_inds return classification matrix: num_img x num_classes
  with 1 and -1
  """
  res = np.zeros((len(image_inds), len(cc.d.classes)))
  images = (cc.d.images[int(ind)] for ind in image_inds)
  gts = []
  for img_idx, img in enumerate(images): 
    coll =  ut.collect([img], Image.get_gt_arr)
    gt_app = np.hstack((coll, np.tile(img_idx,(coll.shape[0], 1))))    
    gts.append(gt_app)
  gts = np.vstack(gts)
  for row_idx in range(gts.shape[0]):
    row = gts[row_idx, :]
    res[row[-1], row[4]] = 1  
  res = res*2 - 1
  return res
  
  
def train_image_classify_svm(cc, cls, C=1.0, gamma=0.0, force_new=False):
  current_fold = cc.d.current_fold
  filename = config.get_classifier_svm_name(cls, C, gamma, current_fold)
  if not force_new and os.path.exists(filename):
    print 'svm for class %s, C=%f, gamma=%f exists already'%(cls,C,gamma)
    return
      
  pyr_feat_size = get_pyr_feat_size(cc.L, cc.dense_codebook.shape[0])  
  
  cc.tictocer.tic('overall') 
  
  print 'compute classifier(C=%f, gamma=%f) for class %s'%(C, gamma, cls)
  pos_images = cc.d.get_pos_samples_for_fold_class(cls)
  if pos_images.size == 0:
    return
  neg_images = cc.d.get_neg_samples_for_fold_class(cls, pos_images.size)
    
  # 1. extract all the pyramids    
  # ======== POSTIVE IMAGES ===========
  print 'compute feature vector for positive images'
  num_slices = 3
  pos_pyrs = np.zeros((len(pos_images),pyr_feat_size + cc.sparse_codebook.shape[0]*(1+num_slices)))      
  for idx, img in enumerate(pos_images):
    pos_pyrs[idx, :] = get_feature_vector(cc, img)

  # ======== NEGATIVE IMAGES ===========
  print 'compute feature vector for negative images'
  neg_pyrs = np.zeros((len(neg_images),pyr_feat_size + cc.sparse_codebook.shape[0]*(1+num_slices)))  
  for idx, img in enumerate(neg_images):
    neg_pyrs[idx, :] = get_feature_vector(cc, img)
  
  # 2. Compute SVM for class
  X = np.vstack((pos_pyrs, neg_pyrs))
  Y = [1]*pos_pyrs.shape[0] + [-1]*neg_pyrs.shape[0] 
  
  if X.shape[0] > 0:
    print 'train svm for class %s, C=%f, gamma=%f'%(cls,C,gamma)
    cc.tictocer.tic()    
    clf = train_svm(X, Y, kernel='rbf', gamma=gamma, C=C)
    print '\ttook %f seconds'%cc.tictocer.toc(quiet=True)
    
    print 'save as', filename
    save_svm(clf, filename)
  else:
    print 'Don\'t compute SVM, no examples given'
  
  print 'training all classifier SVMs took:', cc.tictocer.toc('overall', quiet=True), 'seconds on', comm_rank
  
def classify_image(cc, img_idx, C=1.0, gamma=0.0, cls=None):
  """
  Input: ClassifierConfig, img_idx
  Output: Score for given class; 20list of scores if cls=None
  """
  feat_vect = get_feature_vector(cc, img_idx)
  if cls == None:
    score = {}
    for cls_idx, cls in enumerate(cc.d.classes):
      print '\tclassify %s for %s on %d'%(cc.d.images[img_idx].name, cls, comm_rank)
      path = config.get_classifier_svm_name(cls, C, gamma)
      if os.path.exists(path):
        clf = load_svm(path)
        score[cls_idx] = svm_predict(feat_vect, clf)
  else:
    clf = load_svm(config.get_classifier_svm_name(cls))
    score = svm_predict(feat_vect, clf)
  return score

def classify_all_images(cc):
  """
  For the given dataset in the ClassifierConfig cc compute and store all scores 
  """
  print 'Classify images'
  images = cc.d.images
  for img_idx in range(comm_rank, len(images), comm_size): # PARALLEL
    print 'classify image %d/%d at %d'%(img_idx/comm_size, len(images)/comm_size, comm_rank)
    img = images[img_idx]
    scores = classify_image(cc, img_idx)
    savefile = config.get_classifier_score_name(img, cc.L)
    cPickle.dump(scores, open(savefile,'w'))  
        
if __name__=='__main__':
  tictocer = TicToc()
  tictocer.tic('overall')
  
  test = False
  if test:
    train_dataset = 'test_pascal_train_tobi'
    eval_dataset = 'test_pascal_val_tobi'
    numfolds = 2
  else:
    train_dataset = 'full_pascal_trainval'
    eval_dataset = 'full_pascal_test'
    numfolds = 4
  
  # Train
  L = 1  
  
  safebarrier(comm)  
  # Evaluate  
  cc = ClassifierConfig(eval_dataset, L)
  Cs = [1, 2, 5, 10, 50, 100, 200, 500]
  gammas = [0, 0.4, 0.8, 1.2, 2.0, 2.4, 3.0]
  
  #train_all_svms(cc, Cs, gammas, numfolds=4)
  cross_valid_training(cc, Cs, gammas, numfolds, train=False)
#  gt = get_gt_classification(cc, [0,1])
#  classific = -np.ones(gt.shape)
#  
#  val = validate_images(cc, [0,1], classific)
#  print gt
#  print val
  
  
  print 'Everything done in %f seconds'%tictocer.toc('overall',quiet=True)
  