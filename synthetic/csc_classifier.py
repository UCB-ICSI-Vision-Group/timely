from common_imports import *
from common_mpi import *
import synthetic.config as config
from IPython import embed

from synthetic.classifier import Classifier
from synthetic.dataset import Dataset
from synthetic.training import svm_predict, svm_proba
#import synthetic.config as config
from synthetic.config import get_ext_dets_filename
from synthetic.image import Image
from synthetic.util import Table
from synthetic.evaluation import Evaluation
#from synthetic.dpm_classifier import create_vector

class CSCClassifier(Classifier):
  def __init__(self, suffix, cls, train_dataset, val_dataset, num_bins=5):
    self.name = 'csc'
    self.suffix = suffix
    self.cls = cls
    self.train_dataset = train_dataset
    self.val_dataset = val_dataset
    self.svm = self.load_svm()
    self.num_bins = num_bins
        
  def classify_image(self, image, scores):
    """
    Return score as a probability [0,1] for this class.
    Scores should be a vector of scores of the detections for this image.
    """
    # TODO: rename classify_scores(), does not use image at all!
    vector = self.create_vector_from_scores(scores)
    return svm_proba(vector, self.svm)[0][1]
  
  def create_vector_from_scores(self, scores):
    "scores should be filtered for the index of the image"
    scores = self.normalize_dpm_scores(scores)
    
    vect = np.ones((1,3))
    if scores.shape[0] == 0:
      vect[0,:2] = 0
    elif scores.shape[0] == 1:
      vect[0,:2] = [np.max(scores), 0]
    else:
      vect[0,:2] = np.sort(scores)[:2].T
    return vect
          
  def normalize_dpm_scores(self, arr):     
    return np.power(np.exp(-2.*arr)+1,-1)
    
  def train_for_cls(self, ext_detector, kernel, C):
    dataset = ext_detector.dataset
    assert(dataset.name in ['full_pascal_train','full_pascal_trainval'])
    print dataset.name

    print '%d trains %s'%(comm_rank, self.cls)
    # Positive samples
    pos_imgs = dataset.get_pos_samples_for_class(self.cls)
    pos = []
    for idx, img_idx in enumerate(pos_imgs):
      image = dataset.images[img_idx]
      img_dets, _ = ext_detector.detect(image)
      img_scores = img_dets.subset_arr('score')
      vector = self.create_vector_from_scores(img_scores)
      print 'load image %d/%d on %d'%(idx, len(pos_imgs), comm_rank)
      pos.append(vector)
    pos = np.concatenate(pos)

    # Negative samples
    neg_imgs = dataset.get_neg_samples_for_class(self.cls)
    neg = []
    for idx, img_idx in enumerate(neg_imgs):
      image = dataset.images[img_idx]
      img_dets, _ = ext_detector.detect(image)
      img_scores = img_dets.subset_arr('score')
      vector = self.create_vector_from_scores(img_scores)
      print 'load image %d/%d on %d'%(idx, len(neg_imgs), comm_rank)
      neg.append(vector)
    neg = np.concatenate(neg)
    
    print '%d trains the model for'%comm_rank, self.cls
    self.train(pos, neg, kernel, C)
    
  def eval_cls(self, ext_detector):
    print 'evaluate svm for %s'%self.cls
    dataset = ext_detector.dataset
    assert(dataset.name in ['full_pascal_val','full_pascal_test'])
    print dataset.name

    table_cls = np.zeros(len(dataset.images))
    for img_idx, image in enumerate(dataset.images):
      print '%d eval on img %d/%d'%(comm_rank, img_idx, len(dataset.images))
      img_dets, _ = ext_detector.detect(image)
      img_scores = img_dets.subset_arr('score')
      score = self.classify_image(image, img_scores)
      table_cls[img_idx] = score
      
    ap, _,_ = Evaluation.compute_cls_pr(table_cls, dataset.get_cls_ground_truth().subset_arr(self.cls))
    print 'ap on val for %s: %f'%(self.cls, ap)

    return table_cls

def classify_all_images(d_train, force_new=False):
  suffix = 'default'
  tt = ut.TicToc()
  tt.tic()
  print 'start classifying all images on %d_train...'%comm_rank
  table = np.zeros((len(d_train.images), len(d_train.classes)))
  i = 0
  for cls_idx, cls in enumerate(d_train.classes):
    csc = CSCClassifier(suffix, cls, d_train)
    for img_idx in range(comm_rank, len(d_train.images), comm_size):    
      if i == 5:
        print 'image %d_train on %d_train/%d_train'%(comm_rank, 20*i, 20*len(d_train.images)/comm_size)  
        i = 0
      i += 1  
      
      score = csc.get_score(img_idx, probab=True)
      table[img_idx, cls_idx] = score
              
  dirname = ut.makedirs(os.path.join(config.get_ext_dets_foldname(d_train), 'agent_wise'))
  filename = os.path.join(dirname,'table_%d_train'%comm_rank)
  np.savetxt(filename, table)
            
  print 'Classified all images in %f secs on %d_train'%(tt.toc(quiet=True), comm_rank)
  
def compile_table_from_classifications(d_train):  
  errors = 0
  table = np.zeros((len(d_train.images), len(d_train.classes)))
  dirname = ut.makedirs(os.path.join(config.get_ext_dets_foldname(d_train), 'agent_wise'))
  
  for i in range(comm_size):
    filename = os.path.join(dirname,'table_%d_train'%i)
    table += np.loadtxt(filename)
  dirname = ut.makedirs(os.path.join(config.get_ext_dets_foldname(d_train)))
  filename = os.path.join(dirname,'table')
  np.savetxt(filename, table)
    
  print 'errors: %d_train'%errors
  return table

def create_csc_stuff(d_train, classify_images=True, force_new=False):
        
  dirname = ut.makedirs(os.path.join(config.get_ext_dets_foldname(d_train)))
  print dirname
  filename = os.path.join(dirname,'table')
  
  if not os.path.exists(filename):
    if classify_images:
      classify_all_images(d_train, force_new=force_new)

    safebarrier(comm)    
    table = compile_table_from_classifications(d_train)
    
    if comm_rank == 0:      
      print 'save table as %s'%filename
      
      csc_table = Table()
      csc_table.cols = d_train.classes + ['img_ind']
      csc_table.arr = np.hstack((table, np.array(np.arange(table.shape[0]),ndmin=2).T))      
      print csc_table
      cPickle.dump(csc_table, filename)                 
