from synthetic.common_imports import *
from synthetic.common_mpi import *

import synthetic.config as config
from synthetic.fastInf import *
from synthetic.gist_detector import GistPriors
from synthetic.dataset import Dataset

def create_gist_model_for_dataset(dataset):
  d = Dataset(dataset)
  gist = GistPriors(dataset)
  savefile = config.get_fastinf_data_file(dataset)
  images = d.images
  
  table = np.zeros((len(images), len(d.classes)))
  for idx, img in enumerate(images):
    classif = gist.get_priors(img)
    table[idx, :] = np.array(classif)
    
  return table  
  

if __name__=='__main__':
  dataset = 'full_pascal_trainval'
  table = create_gist_model_for_dataset(dataset)
  
  print table
