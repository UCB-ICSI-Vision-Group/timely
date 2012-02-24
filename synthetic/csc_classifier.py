from common_imports import *
from common_mpi import *

from synthetic.classifier import Classifier
from synthetic.dataset import Dataset
from synthetic.training import svm_predict

class CSCClassifier(Classifier):
  def __init__(self, suffix, cls, dataset):
    self.name = 'csc'
    self.suffix = suffix
    self.cls = cls
    self.dataset = dataset
    self.svm = self.load_svm()
    
    setting_table = ut.Table.load(opjoin(config.get_classifier_dirname(self),'best_table'))
    settings = setting_table.arr[config.pascal_classes.index(cls),:]
    self.intervals = settings[setting_table.cols.index('bins')]
    self.lower = settings[setting_table.cols.index('lower')]
    self.upper = settings[setting_table.cols.index('upper')]
    
  def classify_image(self, img):
    model = self.svm
    #, dets, cls, img, intervals, lower, upper 
    vector = self.create_vector(img)
    result = svm_predict(vector, model)
    return result
    
  def create_vector(self, img):
    filename = config.get_ext_dets_filename(self.dataset, 'csc_'+self.suffix)
    csc_test = np.load(filename)
    csc_test = csc_test[()]  
    csc_test = csc_test.subset(['score', 'cls_ind', 'img_ind'])
    csc_test.arr = self.normalize_scores(csc_test.arr)
    feats = csc_test
    
    if feats.arr.size == 0:
      return np.zeros((1,self.intervals+1))
    dpm = feats.subset(['score', 'cls_ind', 'img_ind'])
    img_dpm = dpm.filter_on_column('img_ind', img, omit=True)
    if img_dpm.arr.size == 0:
      print 'empty vector'
      return np.zeros((1,self.intervals+1))
    cls_dpm = img_dpm.filter_on_column('cls_ind', self.cls, omit=True)
    hist = self.compute_histogram(cls_dpm.arr, self.intervals, self.lower, self.upper)
    vector = np.zeros((1, self.intervals+1))
    vector[0,0:-1] = hist
    vector[0,-1] = img_dpm.shape()[0]
    return vector

def csc_classifier_train_all_params(suffix):
  lowers = [0.]#,0.2,0.4]
  uppers = [1.,0.8,0.6]
  kernels = ['linear']#, 'rbf']
  intervallss = [10, 20, 50]
  clss = range(20)
  Cs = [1., 1.5, 2., 2.5, 3.]  
  list_of_parameters = [lowers, uppers, kernels, intervallss, clss, Cs]
  product_of_parameters = list(itertools.product(*list_of_parameters))  
  csc_classifier_train(product_of_parameters)
  
def csc_classifier_train(parameters, suffix, probab=True, test=True, force_new=False):
  train_set = 'full_pascal_train'
  train_dataset = Dataset(train_set)  
  filename = config.get_ext_dets_filename(train_dataset, 'csc_'+suffix)
  csc_train = np.load(filename)
  csc_train = csc_train[()]  
  csc_train = csc_train.subset(['score', 'cls_ind', 'img_ind'])
  score = csc_train.subset(['score']).arr
  csc_classif = CSCClassifier(suffix)
  csc_train.arr = csc_classif.normalize_scores(csc_train.arr)
  kernels = ['linear', 'rbf']
  
  val_set = 'full_pascal_val'
  val_dataset = Dataset(val_set)  
  filename = config.get_ext_dets_filename(val_dataset, 'csc_'+suffix)
  csc_test = np.load(filename)
  csc_test = csc_test[()]  
  csc_test = csc_test.subset(['score', 'cls_ind', 'img_ind'])
  csc_test.arr = csc_classif.normalize_scores(csc_test.arr)   
  
  for params_idx in range(comm_rank, len(parameters), comm_size):
    params = parameters[params_idx] 
    lower = params[0]
    upper = params[1]
    kernel = params[2]
    if not type(kernel) == type(''):
      kernel = kernels[int(kernel)]
    intervals = params[3] 
    cls_idx = int(params[4])
    C = params[5]
    cls = config.pascal_classes[cls_idx]
    filename = config.get_classifier_svm_learning_filename(csc_classif, cls, kernel, intervals, lower, upper, C)
#    filename = config.data_dir + csc_classif.name + '_svm_'+csc_classif.suffix+'/'+ kernel + '/' + str(intervals) + '/'+ \
#      cls + '_' + str(lower) + '_' + str(upper) + '_' + str(C)
    
    if not os.path.isfile(filename) or force_new:
      csc_classif.train_for_all_cls(train_dataset, csc_train,intervals,kernel, lower, upper, cls_idx, C, probab=probab)
      if test:
        csc_classif.test_svm(val_dataset, csc_test, intervals,kernel, lower, upper, cls_idx, C)
  
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
    csc_test.arr = csc_classif.normalize_scores(csc_test.arr)
    
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
  best_table = csc.get_best_table()
  for row_idx in range(best_table.shape()[0]):
    row = best_table.arr[row_idx, :]
    params = []
    for idx in ['lower', 'upper', 'kernel', 'bins', 'cls_ind', 'C']:
      params.append(row[best_table.ind(idx)])
    parameters.append(params)
  return parameters

if __name__=='__main__':
  csc = CSCClassifier('default')  
  
  # list of lists of svm settings
  # [lowers, uppers, kernels, intervallss, clss, Cs]
  parameters = get_best_parameters()
  csc_classifier_train(parameters, 'default', probab=True, test=False, force_new=True)
    