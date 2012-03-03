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
#from synthetic.dpm_classifier import create_vector

class CSCClassifier(Classifier):
  def __init__(self, suffix, cls, dataset, num_bins=5):
    self.name = 'csc'
    self.suffix = suffix
    self.cls = cls
    self.dataset = dataset
    self.svm = self.load_svm()
    self.num_bins = num_bins
    
    self.bounds = self.load_bounds()
    
  def classify_image(self, image, dets=None, probab=True, vtype='hist'):
    result = self.get_score(image, dets=dets, probab=probab, vtype=vtype)    
    return result
    
  def get_score(self, image, dets=None, probab=True, vtype='hist'):
    """
    with probab=True returns score as a probability [0,1] for this class
    without it, returns result of older svm
    """
    if not dets:
      vector = self.get_vector(image)
    else:
      vector = self.create_vector_from_dets(dets, image, vtype=vtype)
    
    if probab:
      return svm_proba(vector, self.svm)[0][1]
    return svm_predict(vector, self.svm)#[0,0]
  
  def create_vector_from_dets(self, dets, image, vtype='hist',bounds=None, w_count=False,norm=False):
    if not isinstance(image, Image):
      raise RuntimeWarning("Create feat vector should get an Image instance")
    
    # Find the correct img_idx for this image and dets
    image_trainval = self.dataset.get_image_by_filename(image.name)
    img_idx = self.dataset.images.index(image_trainval)
    
    if 'cls_ind' in dets.cols:
      dets = dets.filter_on_column('cls_ind', self.dataset.classes.index(self.cls), omit=True)
    
    if bounds == None:
      bounds = self.bounds
    dets = dets.subset(['score', 'img_ind'])
    if norm:
      dets.arr = self.normalize_dpm_scores(dets.arr)
  
    if dets.arr.size == 0:
      img_dpm = np.array([])
    else:
      img_dpm = dets.filter_on_column('img_ind', img_idx, omit=True)
        
    if vtype == 'hist':
      vect = self.create_hist_vector_from_dets(img_dpm, bounds=bounds, w_count=w_count)
    elif vtype == 'max':
      vect = self.create_max2_vector_from_dets(img_dpm)
      
    return vect
      
  def create_max2_vector_from_dets(self, dets):
    vect = np.ones((1,2))
    if dets.shape()[0] == 0:
      vect[0,:1] = 0
    elif dets.shape()[0] == 1:
      vect[0,:1] = np.matrix([np.max(dets.arr)])
    else:
      vect[0,:1] = np.sort(dets.arr)[:1].T
    return vect
      
  def create_hist_vector_from_dets(self, dets, bounds=None, w_count=False):
    img_dpm = dets
    if img_dpm.arr.size == 0:
      if w_count:
        return np.zeros((1,self.num_bins+1))
      else:
        return np.zeros((1,self.num_bins))
    bins = ut.determine_bin(img_dpm.arr.T[0], bounds)
    hist = ut.histogram_just_count(bins, self.num_bins, normalize=True)
    if w_count:
      hist = np.hstack((hist, np.array(img_dpm.shape()[0], ndmin=2)))
    return hist
     
  def get_vector(self, image):
    #image = self.dataset.images[image]  
    filename = os.path.join(config.get_ext_dets_vector_foldname(self.dataset),image.name[:-4])
    if os.path.exists(filename):
      return np.load(filename)[()]
    else:
      vector = self.create_vector(image)
      np.save(filename, vector)
      return vector
    
  def create_vector(self, image):
    filename = config.get_ext_dets_filename(self.dataset, 'csc_'+self.suffix)
    csc_test = np.load(filename)
    dets = csc_test[()]
    return self.create_vector_from_dets(dets, image)    
  
  def get_all_vectors(self):
    for img_idx in range(comm_rank, len(self.dataset.images), comm_size):
      print 'on %d_train get vect %d_train/%d_train'%(comm_rank, img_idx, len(self.dataset.images))
      self.get_vector(img_idx)
      
  def csc_classifier_train(self, parameters, suffix, dets, train_dataset, probab=True, test=True, force_new=False):      
    kernels = ['linear', 'rbf', 'chi2']       
    for params_idx in range(comm_rank, len(parameters), comm_size):
      params = parameters[params_idx] 
      
      kernel = params[2]
      if not type(kernel) == type(''):
        kernel = kernels[int(kernel)]
      
      C = params[5]
      
      print kernel, C

      filename = config.get_classifier_svm_learning_filename(self.csc_classif, self.cls, kernel, C)
      
      if not os.path.isfile(filename) or force_new:
        bounds = ut.importance_sample(dets.subset(['score']).arr, self.num_bins+1)
        self.store_bounds(bounds)

        self.train_for_cls(train_dataset, dets, kernel, self.cls, C, probab=probab)
        if test:
          #self.test_svm(val_dataset, csc_test, kernel, cls_idx, C)
          None  

  def csc_classifier_train_all_params(self,suffix):
    lowers = [0.]#,0.2,0.4]
    uppers = [1.,0.8,0.6]
    kernels = ['linear']#, 'rbf']
    intervallss = [10, 20, 50]
    clss = range(20)
    Cs = [1., 1.5, 2., 2.5, 3.]  
    list_of_parameters = [lowers, uppers, kernels, intervallss, clss, Cs]
    product_of_parameters = list(itertools.product(*list_of_parameters))  
    self.csc_classifier_train(product_of_parameters)
  
def old_training_stuff(): 
  test_set = 'full_pascal_test'
  for suffix in ['half']:#,'default']:
    test_dataset = Dataset(test_set)  
    filename = config.get_ext_dets_filename(test_dataset, 'csc_'+suffix)
    csc_test = np.load(filename)
    csc_test = csc_test[()]  
    csc_test = csc_test.subset(['score', 'cls_ind', 'img_ind'])
    score = csc_test.subset(['score']).arr
    csc_classif = CSCClassifier(suffix)
    csc_test.arr = csc_classif.normalize_dpm_scores(csc_test.arr)
    
    classes = config.pascal_classes
    
    best_table = csc_classif.get_best_table()
    
    svm_save_dir = os.path.join(config.res_dir,csc_classif.name)+ '_svm_'+csc_classif.suffix+'/'
    score_file = os.path.join(svm_save_dir,'test_accuracy.txt')
                      
    for cls_idx in range(comm_rank, 20, comm_size):
      row = best_table.filter_on_column('cls_ind', cls_idx).arr
      intervals = row[0,best_table.cols.index('bins')]
      kernel = config.kernels[int(row[0,best_table.cols.index('kernel')])]
      lower = row[0,best_table.cols.index('lower')]
      upper = row[0,best_table.cols.index('upper')]
      C = row[0,best_table.cols.index('C')]
      acc = csc_classif.test_svm(test_dataset, csc_test, intervals,kernel, lower, \
                                 upper, cls_idx, C, file_out=False, local=True)
      print acc
      with open(score_file, 'a') as myfile:
          myfile.write(classes[cls_idx] + ' ' + str(acc) + '\n')

def get_best_parameters():
  parameters = []
  d_train = Dataset('full_pascal_trainval')
  
  # this is just a dummy, we don't really need it, just to read best vals
  csc = CSCClassifier('default', 'dog', d_train)
  best_table = csc.get_best_table()
  for row_idx in range(best_table.shape()[0]):
    row = best_table.arr[row_idx, :]
    params = []
    for idx in ['lower', 'upper', 'kernel', 'bins', 'cls_ind', 'C']:
      params.append(row[best_table.ind(idx)])
    parameters.append(params)
  return parameters

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
