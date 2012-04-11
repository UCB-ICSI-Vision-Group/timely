from common_mpi import *
from common_imports import *
import synthetic.config as config
from synthetic.dataset import Dataset
from synthetic.fastinf_model import FastinfModel
from IPython import embed
import matplotlib.pyplot as plt

def compute_score(desired, ranking, num_classes):
  # First compute max and min score, used to normalize later
  num_unobs = np.sum(desired)
  if num_unobs == 0:
    # If there is nothing to find, every answer is equally good.
    return 1  
  min_score = (num_unobs-1)*num_unobs/2
  max_score = (num_classes-1)*num_unobs - min_score
  eval_at = np.nonzero(desired)[0]
  score = (np.sum(ranking[eval_at])-min_score)/(max_score-min_score) 
  return score

def compute_sq_error(desired, inferred):
  sq_error = np.power(desired - inferred,2)  
  return np.sum(sq_error)

def compute_error_vs_iterations(suffix):
  # assemble truth
  num_images = 100
  dataset = 'full_pascal_trainval'
  d = Dataset(dataset)
  truth = d.get_cls_ground_truth().arr
  truth = np.random.permutation(truth)[:num_images,:]  
  num_classes = truth.shape[1]
  
  # leave out perc % per image of data as unobserved
  full_grid = np.asmatrix(np.tile(np.arange(num_classes),(truth.shape[0],1)))  
  tt = ut.TicToc()
  percs = np.arange(20)/20.
  #percs = np.array([0.2, 0.4])
#  lbp_times = [0] + [10**x for x in range(3)]
#  lbp_times += [1000+1000*x for x in range(10)]
#  lbp_times += [10**x for x in [5]]
  lbp_times = range(5)
  
  #lbp_times = [0] + [10**x for x in range(2)]
  #lbp_times = [1000*x + 1000 for x in range(10)]  

  all_scores = np.zeros((percs.shape[0], len(lbp_times)))
  all_times = np.zeros((percs.shape[0], len(lbp_times))) 
  
  for percdex in range(comm_rank, len(percs), comm_size):
    perc = percs[percdex]
    num_unobs = int(perc*num_classes)  
    all_taken = np.apply_along_axis(np.random.permutation, 1, full_grid)[:,num_unobs:]
    all_taken = np.apply_along_axis(np.sort, 1, all_taken)
    counter = 0

    # do inference
    for itdex in range(len(lbp_times)): # parallel
      fm = FastinfModel(d, 'perfect', 20, lbp_time=lbp_times[itdex])
      score = 0      
      utime = 0
      for rowdex in range(truth.shape[0]):
        counter += 1
        obs = truth[rowdex,:].astype(int)
        taken = np.zeros(num_classes).astype(int)
        taken[all_taken[rowdex,:]] = 1
        tt.tic()
        fm.update_with_observations(taken, obs)
        utime += tt.toc(quiet=True)     
        curr_score = compute_sq_error(obs, fm.p_c) 
        score += curr_score
        print '%d is at %d / %d : %f'%(comm_rank, counter, len(lbp_times)*num_images,curr_score)
      
      score /= num_images      
      utime /= num_images
      all_scores[percdex, itdex] = score
      all_times[percdex, itdex] = utime
    
  safebarrier(comm)
  all_scores = comm.reduce(all_scores)
  all_times = comm.reduce(all_times)
  if comm_rank == 0:
    np.savetxt('all_scores_'+suffix, all_scores)  
    np.savetxt('all_times_'+suffix, all_times)    
    
def evaluate_error_vs_iterations(suffix):
  # all_scores: num_unobs x iterations
  lbp_times = [-1] + range(3)
  lbp_times += [math.log10(1000+1000*x) for x in range(10)]
  lbp_times += [5]
  all_scores = np.loadtxt('all_scores_'+suffix)
  all_times = np.loadtxt('all_times_'+suffix)

  plt.plot(np.mean(all_scores, 0))
  plt.ylabel('Avg Error')
  plt.xlabel('Time-paramter')
  
  plt.figure()  
  plt.plot(np.mean(all_times, 0))
  plt.ylabel('Computation Time')
  plt.xlabel('Time-paramter')
  
  plt.figure()
  plt.plot(np.mean(all_scores,1)[::-1])
  plt.ylabel('Avg Error')
  plt.xlabel('# Observed')
  
  plt.show()

if __name__=='__main__':
  suffix = 'secs'
  compute_error_vs_iterations(suffix)
  if comm_rank == 0:
    evaluate_error_vs_iterations(suffix)
  
