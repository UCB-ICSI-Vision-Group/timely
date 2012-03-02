from common_imports import *
from common_mpi import *
import synthetic.config as config

import scipy.cluster.vq as sp
from sklearn import cluster
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from collections import Counter
from string import atoi
from numpy.numarray.numerictypes import Int
from numpy.ma.core import ceil
from scipy import io

from synthetic.extractor import Extractor
from synthetic.dataset import Dataset
from synthetic.pyramid import *
import synthetic.util as ut
from synthetic.jumping_windows import *

def save_to_mat(filename, X, Y, testX):
  Y = Y.astype('float64')
  X = X.astype('float64')
  print X
  print Y
  testX = testX.astype('float64')
  io.savemat(config.repo_dir + 'fast-additive-svms/tmp/' + filename,{'X':X,'Y':Y,'testX':testX})
  
def load_from_mat(filename, value):
  mdict = {}
  io.loadmat(config.repo_dir +'fast-additive-svms/tmp/'+ filename, mdict)
  return mdict[value]

def mat_train_test_svm(filename):
  ut.run_matlab_script(config.repo_dir + 'fast-additive-svms/', \
         'train_test_svm(\'' +config.repo_dir +'fast-additive-svms/tmp/'+ filename + '\')')  

def chi_square_kernel(x, y):
  """
  Create a custom chi-square kernel  
  """
  chi_sum = 0  
  for i in range(x.size):
    if not (x[0, i] + y[0, i]) == 0 and not x[0, i] == y[0, i]: 
      chi_sum += 2*(x[0, i] - y[0, i])**2/(x[0, i] + y[0, i])
  
  return chi_sum

def train_svm(x, y, kernel='chi2',C=1.0, gamma=0.0, probab=True):
  """
  Train a svm.
  x - n x features data
  y - n x 1 labels
  kernel - kernel to be used in ['linear', 'rbf', 'chi2']
  """
  if kernel == 'chi2':
    clf = SVC(kernel='precomputed',C=C, probability=probab)
    gram = np.zeros((x.shape[0],x.shape[0]))
    t_gram = time.time()
    inner_total = x.shape[0]**2/2
    inner_act = 0
    for i in range(x.shape[0]):
      for j in range(x.shape[0]-i-1):
        j += i + 1
        kern = chi_square_kernel(x[i,:], x[j,:])
        gram[i,j] = kern
        gram[j,i] = kern
        inner_act += 1
        if inner_act%5000 == 0:
          print '%d is in gram on: %d / %d'%(comm_rank, inner_act, inner_total)
          print '\t%f seconds passed'%(time.time() - t_gram)
    t_gram = time.time() - t_gram
    print 'computed gram-matrix in',t_gram,'seconds'
    clf.fit(gram, y)
  elif kernel == 'rbf':
    clf = SVC(kernel=kernel, C=C, probability=probab, gamma=gamma)
    clf.fit(x, y)
  elif kernel == 'linear':
    clf = SVC(kernel=kernel, C=C, probability=probab)
    clf.fit(x, y)
  else:
    raise RuntimeError("Unknown kernel passed to train_svm")  
  
  return clf

def svm_predict(x, clf):
  x = np.array(x, dtype='float')
  result = clf.decision_function(x)
  return result

def svm_proba(x, clf):
  return clf.predict_proba(x)

def save_svm(model, filename):
  dump = cPickle.dumps(model)
  f = open(filename, 'w')
  f.write(dump)
  f.close()

def load_svm(filename, probability=True):
  dump = open(filename).read()
  model = cPickle.loads(dump)
  return model

def get_hist(assignments, M):
  counts = Counter(assignments.reshape(1,assignments.size).astype('float64')[0])
  histogram = [counts.get(x+1,0) for x in range(M)]
  histogram = np.matrix(histogram, dtype = 'float64')
  histogram = histogram/np.sum(histogram)
  return histogram

def get_pyr(d,e,table,pyr_size,L,codebook,feature_type,cls):
#  pyrs = np.zeros((table.arr.shape[0], pyr_size))
  pyrs = []   
  for pos_idx in range(table.arr.shape[0]):
    pos = table.arr[pos_idx,:]
    image = d.images[pos[table.cols.index('img_ind')].astype(Int)]
    positions = np.array([pos[0],pos[1],pos[2],pos[3]])
    assignments = e.get_assignments(positions,feature_type, codebook,image)
    pyr = extract_pyramid(L, assignments[:,0:2], assignments, codebook, image)
    ass = get_hist(assignments[:,2:3], codebook.shape[0])
    pyr = np.asarray(ass)
    pyrs.append(pyr)
  pyrs = np.vstack(pyrs)
  return pyrs

def get_test_windows(testsize,dtest,e,pyr_size,L,codebook,feature_type,cls,\
                     cols, randomize=False):
  pos_test_arr = dtest.get_pos_windows(cls)
  if testsize == 'max':
    testsize = pos_test_arr.shape[0]
  neg_test_arr = dtest.get_neg_windows(pos_test_arr.shape[0],cls)
  
  if not testsize == 'max':    
    if randomize:
      rand = np.random.random_integers(0, pos_test_arr.shape[0] - 1, size=testsize)
      pos_test_arr = pos_test_arr[rand]
      rand = np.random.random_integers(0, neg_test_arr.shape[0] - 1, size=testsize)
      neg_test_arr = neg_test_arr[rand]
    else:
      pos_test_arr = pos_test_arr[:testsize]
      neg_test_arr = neg_test_arr[:testsize]
  
  pos_test_table = ut.Table(pos_test_arr, cols)
  neg_test_table = ut.Table(neg_test_arr, cols)
  
  print 'get pos test pyrs'
  pos_test_pyr = get_pyr(dtest,e,pos_test_table,pyr_size,L,codebook,feature_type,cls)
  print 'get neg test pyrs'
  neg_test_pyr = get_pyr(dtest,e,neg_test_table,pyr_size,L,codebook,feature_type,cls)
  
  test_classification = np.asarray([1]*pos_test_arr.shape[0] + [-1]*neg_test_arr.shape[0])

  test_pyr = np.concatenate((pos_test_pyr,neg_test_pyr))
  return (test_pyr, test_classification)
  
def train_with_hard_negatives(d, dtest,cbwords, cbsamps, codebook, cls, pos_table, neg_table,feature_type,\
                               iterations, kernel='chi2', L=2, \
                               testsize = 'max',randomize=False): 
  """ 
    An iterative training with hard negatives
    -input: 
      d - training Dataset
      dtest - test Dataset
      codebook - dsift codebook for pyramid (recommended size 3000)
      cls - the class to be trained
      pos_table - Table with cols [x,y,w,h,img_ind]
      neg_table - same
      kernel - 'linear', 'rbf', 'chi2'
      iterations - number of rounds of training
      L - levels for pyramids    
      testsize - size of initial test 
  """
  # Featurize and pyramidize the input
  e = Extractor()
  M = codebook.shape[0]
  pyr_size = M*1./3.*(4**(L+1)-1)

  print 'get pos train pyrs'
  pos_pyrs = get_pyr(d,e,pos_table,pyr_size,L,codebook,feature_type,cls)
  print 'get neg train pyrs'
  neg_pyrs = get_pyr(d,e,neg_table,pyr_size,L,codebook,feature_type,cls)
    
  print 'built all positive pyramids'
  
  classification = np.asarray([1]*pos_table.arr.shape[0] + [-1]*neg_table.arr.shape[0])
  
  filename = config.data_dir + 'features/' + feature_type + '/svms/' + kernel + \
        '/'+ cls
  ut.makedirs(config.data_dir + 'features/' + feature_type + '/svms/' + kernel)
  
    
  # with that we now determined our whole dataset D  
  #D = np.concatenate((pos_pyrs, neg_pyrs, pos_test_pyr, neg_test_pyr))
  #Y = np.concatenate((classification, test_classification))
  #idx_train = np.arange(pos_pyrs.shape[0] + neg_pyrs.shape[0])
  #idx_test = np.arange(pos_test_pyr.shape[0] + neg_test_pyr.shape[0]) + idx_train.size
  train_pyrs = np.concatenate((pos_pyrs,neg_pyrs))  
  for i in range(iterations):
    # train new model - according to hard mining algorithm by Felszenswalb et al.
    # "Object Detection with Discriminatively Trained Part Based Modles"
    
    [test_pyrs, test_classification] = get_test_windows(testsize,dtest,e,\
                                          pyr_size,L,codebook,feature_type,cls,\
                                          pos_table.cols, randomize=randomize)
    print 'node',comm_rank,'training in round', i, 'with', np.sum(classification==1),'pos and',\
      np.sum(classification == -1),'negs'
    print 'testing', test_pyrs.shape[0], 'new samples'
    print time.strftime('%m-%d %H:%M')
    
    # get new test samples
    
    
    # ----------1-------------
    model = train_svm(train_pyrs, classification, kernel)
  
    testY = svm_predict(np.concatenate((train_pyrs,test_pyrs)), model)
    result = testY
    print result
    
    res_train = testY[:train_pyrs.shape[0]]
    res_test = testY[train_pyrs.shape[0]:]
          
    # ----------3-------------        
    # remove easy samples from train-set
    idx_tr_list = []
    for s_ind in range(res_train.shape[0]):
      if res_train[s_ind]*classification[s_ind] <= 1:
        idx_tr_list.append(s_ind)
    indices = np.matrix(idx_tr_list).reshape(1,len(idx_tr_list))
    indices = np.array(indices.astype(Int))[0]        
    train_pyrs = train_pyrs[indices]
    classification = classification[indices]
    
    # ----------4-------------
    idx_hn_list = []
    new_hards = False
    for s_ind in range(res_test.shape[0]):
      if res_test[s_ind]*test_classification[s_ind] < 1:
        new_hards = True
        idx_hn_list.append(s_ind)    

    nu_train_idc = np.matrix(idx_hn_list).reshape(1,len(idx_hn_list))
    nu_train_idc = np.array(nu_train_idc.astype(Int))[0]      
    train_pyrs = np.vstack((train_pyrs, test_pyrs[nu_train_idc]))
    classification = np.concatenate((classification, test_classification[nu_train_idc]))
        
    test_result = result[-test_pyrs.shape[0]:]
    fp = np.sum(np.multiply(test_result < 0, np.transpose(np.matrix(test_classification == 1))))
    tp = np.sum(np.multiply(test_result > 0, np.transpose(np.matrix(test_classification == 1))))
    fn = np.sum(np.multiply(test_result > 0, np.transpose(np.matrix(test_classification == -1))))
    
    # save these to the training file
    prec = tp/float(tp+fp)
    rec = tp/float(tp+fn)
    print 'tp, fp:',tp,fp
    print 'prec, rec:', prec,rec
    with open(filename + '_train', "a") as myfile:
      myfile.write(str(prec) + ' ' + str(rec)+'\n')    
  
    # ----------2-------------
    if not new_hards:
      # no new_hards from test set,we want to quit.
      break
  # save the trained svm. 
  #save_svm(model, filename)
  

if __name__=='__main__':
  #ut.run_matlab_script(  config.repo_dir + 'fast-additive-svms/','demo')
 
#  x = np.matrix([[0,2],[0,4],[2,2],[2,4]])
#  x = np.concatenate((np.random.random((250,5)),\
#                     np.random.random((250,5))))
#  y = [-1]*250  + [1]*250
#  x0 = np.concatenate((np.random.random((5,5)),\
#                       np.random.random((5,5))))
#  print x0
#  filename = 'test.mat'
#  
#  save_to_mat(filename, x,np.array(y),x0)
#  mat_train_test_svm(filename)
#  testY = load_from_mat('testY.mat', 'testY')
#  print testY
   
#  model = train_svm(x, y)         
#  print 'result:'  
#  result = svm_predict(x0, model)
#  print result
  

#  d = Dataset('full_pascal_trainval')
#  d.evaluate_get_pos_windows(0.5)

#if False:
  randomize = not os.path.exists('/home/tobibaum')
  
  d = Dataset('full_pascal_train')
  dtest = Dataset('full_pascal_val')
  
  e = Extractor()
  
  classes = config.pascal_classes  
  num_words = 3000
  iters = 10
  feature_type = 'dsift'
  codebook_samples = 15
  num_pos = 'max'
  testsize = 'max' 
  kernel = 'chi2'
  
#  num_pos = 3
#  testsize = 4
  
  # For my local testings
  classes = ['dog']
  #classes = ['bicycle','bird','boat','bottle','bus','car','cat']
#  testsize = 1
#  num_pos = 1

  if comm_rank == 0:
    ut.makedirs(config.data_dir + 'features/' + feature_type + '/times/')
    ut.makedirs(config.data_dir + 'features/' + feature_type + '/codebooks/times/')
    ut.makedirs(config.data_dir + 'features/' + feature_type + '/svms/train_times/')
    
  for cls_idx in range(comm_rank, len(classes), comm_size): 
  #for cls in classes:
    cls = classes[cls_idx]
    codebook = e.get_codebook(d, feature_type)
    pos_arr = d.get_pos_windows(cls)[10:]
    
    neg_arr = d.get_neg_windows(pos_arr.shape[0], cls, max_overlap=0)
    
    if not num_pos == 'max':    
      if not randomize:
        pos_arr = pos_arr[:num_pos]
        neg_arr = pos_arr[:num_pos]
      else:
        rand = np.random.random_integers(0, pos_arr.shape[0] - 1, size=num_pos)
        pos_arr = pos_arr[rand]
        rand = np.random.random_integers(0, neg_arr.shape[0] - 1, size=num_pos)
        neg_arr = neg_arr[rand]   
    

    pos_table = ut.Table(pos_arr, ['x','y','w','h','img_ind'])
    neg_table = ut.Table(neg_arr, pos_table.cols)
    
    train_with_hard_negatives(d, dtest,  num_words,codebook_samples,codebook,\
                              cls, pos_table, neg_table,feature_type, iterations=iters, \
                              kernel=kernel, L=2, testsize=testsize)
    # Testing implementation here (visually confirm imgs):

#    for set_vis in [neg_vis]:
#      for row in set_vis:
#        image_idx = row[pos_table.cols.index('img_ind')]
#        image = d.images[image_idx]
#        filename = config.VOC_dir + 'JPEGImages/' + image.name
#        os.system('convert ' + filename + ' bbox_tmp_img.png')
#        im = Image.open('bbox_tmp_img.png')
#        os.remove('bbox_tmp_img.png')
#        draw = ImageDraw.Draw(im)  
#        draw.rectangle(((row[0],row[1]),(row[0]+row[2],row[1]+row[3])))
#        del draw
#        im.show()
