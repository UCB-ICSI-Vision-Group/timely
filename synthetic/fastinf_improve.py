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
  return sq_error

def compute_error_vs_iterations(suffix, num_images, dataset):
  # assemble truth
  
  d = Dataset(dataset)
  truth = d.get_cls_ground_truth().arr
  truth = np.random.permutation(truth)[:num_images,:]  
  num_classes = truth.shape[1]  
  tt = ut.TicToc()
  
  lbp_times = [0] + [10**x for x in range(3)]
  lbp_times += [1000+1000*x for x in range(10)]
  lbp_times += [10**x for x in [5]]  
  #lbp_times = [3000]

  all_scores = np.zeros((num_classes, len(lbp_times), num_classes))
  all_times = np.zeros((num_classes, len(lbp_times)))
      
  counter = 0
  # do inference
  for itdex in range(len(lbp_times)):
    fm = FastinfModel(d, 'perfect', num_classes, lbp_time=lbp_times[itdex])
    for rowdex in range(comm_rank, truth.shape[0], comm_size): # parallel
      obs = truth[rowdex,:].astype(int)
      taken = np.zeros(num_classes).astype(int)
      
      for num_obser in range(num_classes):            
        counter += 1        
        taken[np.argmax(fm.p_c-taken)] = 1
        
        tt.tic()
        fm.update_with_observations(taken, obs)
        utime = tt.toc(quiet=True)     
        curr_score = compute_sq_error(obs, fm.p_c) 
        all_scores[num_obser, itdex, :] = np.add(all_scores[num_obser, itdex, :], curr_score)
          
        all_times[num_obser, itdex] += utime
        print '%d is at %d / %d :'%(comm_rank, counter, len(lbp_times)* \
                                       num_classes*num_images/float(comm_size)),curr_score
    
  all_scores /= num_images
  all_times /= num_images        
    
  safebarrier(comm)
  all_scores = comm.reduce(all_scores)
  all_times = comm.reduce(all_times)
  if comm_rank == 0: #parallel
    outfile = open('all_scores_'+suffix,'w')
    cPickle.dump(all_scores,outfile)
    outfile.close()
    outfile = open('all_times_'+suffix,'w')
    cPickle.dump(all_times,outfile)
    outfile.close()    
    
def evaluate_error_vs_iterations(suffix):
  # all_scores: num_unobs x iterations
  lbp_times = [-1] + range(3)
  lbp_times += [math.log10(1000+1000*x) for x in range(10)]
  lbp_times += [5]
  infile = open('all_scores_'+suffix,'r')
  all_scores = cPickle.load(infile)
  infile.close()
  infile = open('all_times_'+suffix,'r')
  all_times = cPickle.load(infile)
  infile.close()
 
#  plt.plot( np.mean(all_scores, 0))
#  plt.ylabel('Avg Error')
#  plt.xlabel('Time-paramter')
#  
#  plt.figure()  
#  plt.plot(np.mean(all_times, 0))
#  plt.ylabel('Computation Time')
#  plt.xlabel('Time-paramter')
  
  plt.figure()
  plt.plot(np.mean(all_scores,1))
  plt.plot(np.mean(np.mean(all_scores,1),1), 'r--', linewidth=2)
  plt.ylabel('Avg Error')
  plt.xlabel('# Observed')
  
  plt.show()

if __name__=='__main__':
  suffix = 'synth'
  num_images = 1000
  dataset = 'full_pascal_trainval' # 'synthetic'  
  #compute_error_vs_iterations(suffix, num_images, dataset)
  if comm_rank == 0:
    evaluate_error_vs_iterations(suffix)
