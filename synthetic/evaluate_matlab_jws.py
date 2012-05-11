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
  
  gt = d.get_ground_truth(include_diff=True)
        
#  # No point in pickling this....to big!
#  picklename = os.path.join(jwdir, 'all_bboxes_'+suffix)  
#  if not os.path.exists(picklename):
#    bboxes_table = pickle_matlab_jws(d, e, picklename, suffix)
#  else:
#    print 'loading boxes...'
#    t.tic()
#    bboxes_table = cPickle.load(open(picklename, 'r'))
#    t.toc()
  
  
  filename = os.path.join(jwdir, 'recall_vs_jws_'+suffix)
  nsamples_total = 0
  auc = 0
  bbox_generator = generate_bboxes(d, e, suffix)
  for bboxes_table in bbox_generator:
    nsamples = bboxes_table.shape()[0]
    nsamples_total += nsamples
    (x, y) = e.evaluate_recall_vs_jws(bboxes_table, gt)
    print x, y ,' are the x ys' 
    auc += np.dot(x,y)/np.sum(x)*nsamples
    print auc, 'for rank %d'%comm_rank    
    break
  
  print 'auc on rank %d is %f'%(comm_rank, auc)
  nsamples_total = comm.reduce(nsamples_total)
  auc = comm.reduce(auc)
  print 'samples on rank %d: %d'%(comm_rank, nsamples_total)
  
  if comm_rank == 0:
    auc /= float(nsamples_total)    
  # compute area under recall_vs_windows curve
  
  print 'auc on rank %d is %f'%(comm_rank, auc)
  
  return auc
     

def generate_bboxes(d, e, suffix):
  # First get the files for recorded bboxes
  path = os.path.join(config.res_dir,'jumping_windows','bboxes_'+suffix) 
  full_arr = []
  filelist = os.listdir(path)
  max_file_count = 0
  
  for idx in range(comm_rank, len(filelist), comm_size):
    filename = filelist[idx]
    
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
    max_file_count += 1
    
    if max_file_count >= 500/comm_size:
      # We run into memory errors otherwise :/
      full_arr = np.vstack(full_arr)  
      cols = ['x','y','w','h','score','cls_ind','img_ind']
      bboxes_table = ut.Table(full_arr, cols, 'all_bboxes')
      full_arr = []
      yield bboxes_table      
            

if __name__=='__main__':
  dataset = 'full_pascal_test'
  suffix = 'mine'
  auc = evaluate_matlab_jws(dataset, suffix)
  
  if comm_rank == 0:
    print 'auc is %f'%auc