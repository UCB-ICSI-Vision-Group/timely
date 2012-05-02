from synthetic.common_imports import *
from synthetic.common_mpi import *
from synthetic.dataset import Dataset
import synthetic.config as config
import scipy.io as sio
from synthetic.evaluation import Evaluation
from synthetic.dataset_policy import DatasetPolicy
from synthetic.bounding_box import BoundingBox
import matplotlib.pyplot as plt


def evaluate_matlab_jws(dataset, suffix):
  d_train = Dataset('full_pascal_trainval')
  d = Dataset(dataset)
  dp = DatasetPolicy(d, d_train)
  e = Evaluation(dp, d)
  jwdir = os.path.join(config.res_dir, 'jumping_windows')
        
  picklename = os.path.join(jwdir, 'all_bboxes_'+suffix)
  t = ut.TicToc()
  if not os.path.exists(picklename):
    bboxes_table = pickle_matlab_jws(d, e, picklename)
  else:
    print 'loading boxes...'
    t.tic()
    bboxes_table = cPickle.load(open(picklename, 'r'))
    t.toc()
  gt = d.get_ground_truth(include_diff=True)
  
  filename = os.path.join(jwdir, 'recall_vs_jws_'+suffix)
  (x, y) = e.plot_recall_vs_windows(bboxes_table, gt, filename)
    
  # compute area under recall_vs_windows curve
  auc = np.dot(x,y)/np.max(x)
  return auc
     

def pickle_matlab_jws(d, e, picklename):
  # First get the files for recorded bboxes
  path = os.path.join(config.res_dir,'jumping_windows','bboxes') 
  full_arr = []
  filelist = os.listdir(path)
  
  for idx, filename in enumerate(filelist):
    [cls, imgname] = filename.split('_')
    imgname = imgname[:-4]+'.jpg'
    print '%d of %d: %s - %s'%(idx, len(filelist), imgname, cls)
    cls_ind = config.pascal_classes.index(cls)
   
    img = d.get_image_by_filename(imgname)
    img_ind = d.get_img_ind(img)
    
    # if this cls is not in the original image, we are not interested.
    if not img.get_cls_counts()[cls_ind]:
      os.remove(os.path.join(path,filename))
      continue
    
    # load these bounding boxes
    bboxes = sio.loadmat(os.path.join(path,filename))['bboxes']    
    bboxes[:,2:] -= bboxes[:,:2]-1
    n = bboxes.shape[0]
       
    print 'bboxes:', n
    arr = np.hstack((bboxes,(n-np.arange(n)).reshape(n,1)/float(n),cls_ind*np.ones((bboxes.shape[0],1)), img_ind*np.ones((bboxes.shape[0],1))))
    if arr.shape[0] == 0:
      continue
    full_arr.append(arr)        
  full_arr = np.vstack(full_arr)  
  cols = ['x','y','w','h','score','cls_ind','img_ind']
  bboxes_table = ut.Table(full_arr, cols, 'all_bboxes')  
  cPickle.dump(bboxes_table, open(picklename, 'w'))
  return bboxes_table      

if __name__=='__main__':
  dataset = 'full_pascal_trainval'
  suffix = 'small'
  evaluate_matlab_jws(dataset, suffix)