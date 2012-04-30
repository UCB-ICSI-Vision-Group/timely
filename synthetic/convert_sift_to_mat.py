from common_imports import *
from common_mpi import *
import synthetic.config as config
from synthetic.extractor import Extractor
from synthetic.dataset import Dataset
from IPython import embed
import scipy.cluster.vq as sp
import scipy.io as sio
import os

def save_assignment_as_mat(file_name, ass, img):
  width = img.size[0]
  height = img.size[1]
  x = ass[:,0]
  y = ass[:,1]
  feaArr = sp.whiten(ass[:,3:]).T
  feaSet = {}
  feaSet['width'] = width
  feaSet['height'] = height
  feaSet['x'] = x
  feaSet['y'] = y
  feaSet['feaArr'] = feaArr
  print file_name
  sio.savemat(file_name, {'feaSet':feaSet})
  
def convert_all_assignment(dataset, feature_type):
  e = Extractor()
  d = Dataset(dataset)
  savedir = os.path.join(config.res_dir,'../jumping_windows/jumping_windows/sift')
  ut.makedirs(savedir)
  
  for imgdex in range(comm_rank, len(d.images), comm_size): # parallel
    img = d.images[imgdex]
    file_name = os.path.join(savedir,img.name[:-4])
    print '%d is on %s'%(comm_rank, img.name)
    if not os.path.exists(file_name):
      ass = e.get_feature_with_pos(feature_type, img, [0,0,100000,100000])
      save_assignment_as_mat(file_name, ass, img)

if __name__=='__main__':
  feat_type = 'dsift'
  dset = 'full_pascal_test'
  convert_all_assignment(dset, feat_type)