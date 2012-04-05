from common_mpi import *
from common_imports import *
import synthetic.config as config
from synthetic.dataset import Dataset
from synthetic.fastinf_model import FastinfModel
from IPython import embed

def evaluate_error_vs_iterations():
  # assemble truth
  dataset = 'full_pascal_trainval'
  d = Dataset(dataset)
  truth = d.get_cls_ground_truth().arr  
  num_classes = truth.shape[1]
  # leave out perc % per image of data as unobserved
  perc = .1  
  all_taken = np.apply_along_axis(np.random.permutation, 1, np.asmatrix(np.tile(np.arange(num_classes),(truth.shape[0],1))))[:,int(perc*num_classes):]
#  unobs = np.copy(truth).astype(int)  
#  for rowdex in range(truth.shape[0]):
#    unobs[rowdex, set_quest[rowdex,:]] = -1
    
   
  # do inference
  fm = FastinfModel(d, 'perfect', 20)
  rowdex = 0
  obs = truth[rowdex,:]
  taken = all_taken[rowdex,:]
  fm.update_with_observations(taken, obs)
  embed()
  # compute AP 
  
  # plot it


if __name__=='__main__':
  evaluate_error_vs_iterations()
