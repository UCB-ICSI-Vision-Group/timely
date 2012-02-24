from sklearn.cross_validation import KFold

from common_imports import *
from common_mpi import *

from synthetic.ngram_model import NGramModel
from synthetic.image import Image
from synthetic.training import *
from synthetic.classifier import Classifier
from synthetic.fastInf import write_out_mrf, execute_lbp, discretize_table

class GistClassifier(Classifier):
  """
  Compute a likelihood-vector for the classes given a (precomputed) gist detection
  """
  def __init__(self, cls, dataset):
    """
    Load all gist features right away
    """
    if type(dataset) == type(''):
      self.dataset = Dataset(dataset)
      dataset_name = dataset
    else:
      self.dataset = dataset
      dataset_name = dataset.name
      
    Classifier.__init__(self)
    print("Started loading GIST")
    self.tt.tic()
    self.gist_table = np.load(config.get_gist_dict_filename(dataset_name))
    print("Time spent loading gist: %.3f"%self.tt.qtoc())
    self.cls = cls
    self.svm = self.load_svm()
  
  def get_scores_for_image_set(self, image_idc):
    """
    Get a list of image indices (of own dataset) and return all scores as column-vect
    """
    scores = np.zeros((len(image_idc), 1))    
    for idx, img_idx in enumerate(image_idc):
      img = self.dataset.images[img_idx]
      scores[idx] = self.get_score(img)      
    return scores
    
  def get_score(self, img):
    return self.get_proba(img)[0][1]
        
  def load_svm(self):
    filename = config.get_gist_svm_filename(self.cls)
    if os.path.exists(filename):
      print 'load svm %s'%filename
      svm = load_svm(config.get_gist_svm_filename(self.cls))
    else:
      print 'gist svm for',self.cls,'does not exist'
    return svm
    
  def get_proba(self, img):
    image = self.dataset.get_image_by_filename(img.name)
    index = self.dataset.get_img_ind(image)
    gist = np.array(self.gist_table[index])
    return svm_proba(gist, self.svm)
     
  def compute_obj_func(self, gist, truth):
    diff = gist - truth
    sqr = np.multiply(diff,diff)
    sqrsum = np.sum(sqr)
    return sqrsum
  
  def get_gists_for_imgs(self, imgs, dataset):
    images = dataset.images
    num = imgs.size
    print num
    gist = np.zeros((num, 960))
    ind = 0    
    for img in imgs:
      image = dataset.get_image_by_filename(images[img].name)
      index = dataset.get_img_ind(image)
      gist[ind] = self.gist_table[index]
      ind += 1
    return gist
  
  def train_all_svms(self, dataset,C=1.0):
    """
    Train classifiers to 
    """
    for cls in config.pascal_classes:
      print 'train class', cls
      t = time.time()
      pos = dataset.get_pos_samples_for_class(cls)
      num_pos = pos.size
      neg = dataset.get_neg_samples_for_class(cls)
      neg = np.random.permutation(neg)[:num_pos]
      print '\tload pos gists'
      pos_gist = self.get_gists_for_imgs(pos, dataset)
      print '\tload neg gists'       
      neg_gist = self.get_gists_for_imgs(neg, dataset)
      x = np.concatenate((pos_gist, neg_gist))
      y = [1]*num_pos + [-1]*num_pos
      print '\tcompute svm'
      svm = train_svm(x, y, kernel='linear',C=C,probab=True)
      svm_filename = config.get_gist_svm_filename(cls)+'_'+str(C)
      save_svm(svm, svm_filename)     
      print '\ttook', time.time()-t,'sec'

  def evaluate_svm(self, cls, dataset, C):
    svm_filename = config.get_gist_svm_filename(cls)+'_'+str(C)
    svm = load_svm(svm_filename)
    print 'evaluate class', cls
    t = time.time()
    pos = dataset.get_pos_samples_for_class(cls)
    num_pos = pos.size 
    neg = dataset.get_neg_samples_for_class(cls)
    neg = np.random.permutation(neg)[:num_pos]
    print '\tload pos gists'
    pos_gist = self.get_gists_for_imgs(pos, dataset)
    print '\tload neg gists'       
    neg_gist = self.get_gists_for_imgs(neg, dataset)
    x = np.concatenate((pos_gist, neg_gist))
    y = [1]*num_pos + [-1]*num_pos
    result = svm_predict(x, svm)
    test_classification = np.matrix([1]*pos_gist.shape[0] + [-1]*neg_gist.shape[0]).reshape((result.shape[0],1))  
    acc = sum(np.multiply(result,test_classification) > 0)/float(2.*num_pos)
    outfile_name = os.path.join(config.gist_dir, cls)
    outfile = open(outfile_name,'a')
    outfile.writelines(str(C) + ' ' + str(acc[0][0])+'\n') 
    
  def cross_val_lambda(self, lam):
    images = self.dataset.images
    num_folds = 4
    loo = KFold(len(images), num_folds)
    errors = []
    fold_num = 0
    for train,val in loo:
      print 'compute error for fold', fold_num
      indices = np.arange(len(images))[train]
      print len(indices)
      data = np.zeros((indices.size,20))
      ind = 0
      for idx in indices:
        data[ind] = images[idx].get_cls_counts()
        ind += 1        
      model = NGramModel(data)
      priors = model.get_probabilities()
      error = 0
      indices = np.arange(len(images))[val]
       
      for idx in indices:
        img = images[idx]
        print 'evaluate img', img.name
        t = time.time()
        gist = self.get_priors_lam(img,  priors, lam)
        t = time.time() - t
        print 'gist took %f secs'%t
        error += self.compute_obj_func(gist, img.get_cls_counts()>0)
      error = error/indices.shape[0]
      
      errors.append(error)
      print 'error:', error
    avg_error = sum(errors)/4
    return avg_error
    
    
def gist_evaluate_best_svm():
  train_d = Dataset('full_pascal_train')
  train_dect = GistClassifier(train_d.name)
  val_d = Dataset('full_pascal_val')
  val_dect = GistClassifier(val_d.name)  
  
  ranges = np.arange(0.5,10.,0.5)
  for C_idx in range(comm_rank, len(ranges), comm_size):
    C = ranges[C_idx] 
    train_dect.train_all_svms(train_d, C)
    for cls in config.pascal_classes:
      val_dect.evaluate_svm(cls, val_d, C)
    
def test_gist_one_sample(dataset):    
  dect = GistClassifier(dataset)
  d = Dataset(dataset)
  vect = dect.get_priors(d.images[1])
  for idx in range(len(vect)):
    if vect[idx] > 0.5:
     print config.pascal_classes[idx], vect[idx]
     
def save_gist_differently(datasets):
  gist_dict = cPickle.load(open(join(config.res_dir,'gist_features','features'),'r'))
  for dataset in datasets:
    d = Dataset(dataset)
    print 'converting set', dataset
    save_file = os.path.join(config.res_dir,'gist_features',dataset)
    images = d.images
    gist_tab = np.zeros((len(images), 960))
    for idx in range(len(images)):
      img = images[idx]
      print 'on \t', img.name
      gist_tab[idx,:] = gist_dict[img.name[:-3]]
    np.save(save_file, gist_tab)
    
def convert_gist_datasets(dataset_origin, datasets_new):
  data = np.load(config.get_gist_dict_filename(dataset_origin))
  d_orig = Dataset(dataset_origin)
  for dataset in datasets_new:
    savefile = config.get_gist_dict_filename(dataset)
    d = Dataset(dataset)
    images = d.images
    new_data = np.zeros((len(images), data.shape[1]))
    for img_idx, img in enumerate(images):
      orig_img = d_orig.get_image_by_filename(img.name)
      row = d_orig.images.index(orig_img)
      new_data[img_idx, :] = data[row, :] 
      
    np.save(savefile, new_data)
  
def crossval():
  #save_gist_differently()
  #test_gist_one_sample('full_pascal_test')
  #gist_evaluate_best_svm()
  dect = GistClassifier('full_pascal_trainval')
  lams = np.arange(0,1,0.025)
  errors = np.zeros((lams.shape[0],1))
  for idx in range(comm_rank, len(lams),comm_size):
    lam = lams[idx]
    err = dect.cross_val_lambda(lam)
    errors[idx, 0] = err
  errors = comm.reduce(errors)
  
  if comm_rank == 0:
    result_file = config.res_dir + 'cross_val_lam_gist.txt'
    outfile = open(result_file,'w')
    print errors
    for idx in range(lams.shape[0]):
      outfile.write(str(lams[idx]) + ' ' + str(errors[idx]) + '\n')    
    outfile.close() 
    
  
def convert():
  datasets = ['test_pascal_train_tobi','test_pascal_val_tobi']
  dataset_origin = 'full_pascal_trainval'
  convert_gist_datasets(dataset_origin, datasets)

def cls_for_dataset(dataset):
  d = Dataset(dataset)
  classes = d.classes
  table = np.zeros((len(d.images), len(classes)))
  savefile = config.get_gist_fastinf_table_name(dataset, None)
   
  print savefile
  if os.path.exists(savefile):
    return cPickle.load(open(savefile, 'r'))
  
  for cls_idx in range(comm_rank, len(classes), comm_size):
    cls = classes[cls_idx]
    
    savefile = config.get_gist_fastinf_table_name(dataset, cls)
    if os.path.exists(savefile):
      table = cPickle.load(open(savefile,'r'))
      continue    
    gist = GistClassifier(cls, d)
    d = gist.dataset
    images = d.images
    table[:, cls_idx] = gist.get_scores_for_image_set(range(len(images)))[:,0]    
    cPickle.dump(table, open(savefile,'w'))
    
  safebarrier(comm)
  table = comm.allreduce(table)  
  if comm_rank == 0:
    savefile = config.get_gist_fastinf_table_name(dataset, None)
    cPickle.dump(table, open(savefile,'w'))
  return table

if __name__=='__main__':
  dataset = 'full_pascal_trainval'
  table = cls_gt_for_dataset(dataset)
  d = Dataset(dataset)
  num_bins = 5
  suffix = 'gist_pair'
  filename = config.get_fastinf_mrf_file(dataset, suffix)
  data_filename = config.get_fastinf_data_file(dataset, suffix)
  filename_out = config.get_fastinf_res_file(dataset, suffix)
  
  table_gt = d.get_cls_ground_truth().arr.astype(int)
  print table.shape
  
  table = np.hstack((table_gt, table))
  
  discretize_table(table, num_bins)
  if comm_rank == 0:  
    write_out_mrf(table, num_bins, filename, data_filename)  
    result = execute_lbp(filename, data_filename, filename_out)
  