from common_imports import *
from common_mpi import *

import synthetic.config as config

class ClassPriors:
  """
  Encapsulation of the part of the belief state that keeps track of the object
  class priors.
  Methods to initialize the model, update with an observed posterior,
  condition on observed values, and compute expected information gain.
  """

  accepted_modes = [
    	'random','oracle','fixed_order',
    	'no_smooth','backoff', 'fastinf']

  def __init__(self, dataset, mode='random'):
    self.dataset = dataset
    data = self.dataset.get_cls_counts()

    assert(mode in ClassPriors.accepted_modes)
    self.mode = mode

    if mode=='random':
      self.model = RandomModel(data)
    elif mode=='oracle':
      self.model = RandomModel(data)
    elif mode in ['fixed_order','no_smooth','backoff']:
      self.model = NGramModel(data,self.mode)
    elif mode=='fastinf':
    	self.model = FastInfModel(dataset)
    else:
      raise RuntimeError("Unknown mode")
    self.priors = self.model.get_probabilities()
    self.observed_inds = []
    self.observed_vals = []

  def __repr__(self):
    return "ClassPriors: \n%s\n%s"%(self.priors,zip(self.observed_inds,self.observed_vals))

  def update_with_posterior(self, cls_ind, posterior):
    """
    Update the priors given the observed posterior of one class.
    """
    # TODO: for now, just set to hard 0/1
    self.observed_inds.append(cls_ind)
    self.observed_vals.append(posterior)
    self.priors = self.model.get_probabilities(self.observed_inds,self.observed_vals)
    
  def update_with_gist(self, gist_priors):
    """
    Update the priors given the GIST-conditioned class probabilities.
    """
    # TODO: not quite clear what to do
    new_priors = []
    for i in range(len(self.priors)):
      new_priors.append(self.priors[i]*gist_priors[i])
    self.priors = new_priors

class RandomModel:
  def __init__(self,data):
    self.num_classes = data.shape[1]

  def get_probabilities(self, cond_inds=None, cond_vals=None): 
    return np.random.rand(self.num_classes)

class FastinfModel:
	def __init__(self,dataset):
		self.fname = config.get_fastinf_filename(dataset)

class NGramModel:
  accepted_modes = ['fixed_order','no_smooth','smooth','backoff']

  def __init__(self,data,mode='fixed_order'):
    self.data = data
    self.cache = {}
    assert(mode in NGramModel.accepted_modes)
    self.mode = mode
    self.cls_inds = range(self.data.shape[1])
    print("NGramModel initialized with %s mode"%self.mode)

  def shape(self): return self.data.shape

  def get_probabilities(self, cond_inds=None, cond_vals=None):
    """Return list of the values of each cls_ind."""
    if self.mode=='fixed_order':
      return [self.cond_prob(cls_ind) for cls_ind in self.cls_inds]
    else:
      return [self.cond_prob(cls_ind, cond_inds, cond_vals) for cls_ind in self.cls_inds]

  def marg_prob(self, cls_inds, vals=None):
    """
    Returns the marginal probability of a given list of cls_inds being assigned
    the given vals.
    """
    if not isinstance(cls_inds, types.ListType):
      cls_inds = [cls_inds]

    # If vals aren't given, set all to 1
    if vals == None:
      vals = [1. for cls_ind in cls_inds]
    else:
      if not isinstance(vals, types.ListType):
        vals = [vals]

    # check if we've already computed value for this
    hash_string = ' '.join(["%d:%d"%(int(x),int(y)) for x,y in zip(cls_inds,vals)])
    if hash_string in self.cache:
      return self.cache[hash_string]

    num_total = self.data.shape[0]
    rep_vals = np.tile(vals,(num_total,1))
    inds = np.all(self.data[:,cls_inds]==rep_vals,1)[:]
    filtered = self.data[inds,:]
    num_filtered = filtered.shape[0]

    ans = 1.*num_filtered/num_total
    #print("The marginal probability of %s is %s"%(cls_inds,ans))
    self.cache[hash_string] = ans
    return ans

  def cond_prob(self, cls_inds, cond_cls_inds=None, vals=None, lam1=0.05, lam2=0.):
    """
    Returns the conditional probability of a given list of cls_inds, given
    cond_cls_inds. If these are None, then reverts to reporting the marginal.
    Arguments must be lists, not ndarrays!
    Accepted modes:
      - 'no_smooth': the standard, allows 0 probabilities if no data is present
      - 'backoff': P(C|A) is approximated by P(C,A)/P(A) + \lambda*P(C)
      - 'smooth': NOT IMPLEMENTED
    """
    if not isinstance(cls_inds, types.ListType):
      cls_inds = [cls_inds]

    # if no conditionals given, return marginal
    if cond_cls_inds == None:
      return self.marg_prob(cls_inds)

    if not isinstance(cond_cls_inds, types.ListType):
      cond_cls_inds = [cond_cls_inds]

    # If vals aren't given, set all to 1
    if vals == None:
      vals = [1. for cls_ind in cond_cls_inds]
    else:
      if not isinstance(vals, types.ListType):
        vals = [vals]

    # check if cls_inds are in cond_cls_inds and return their vals if so
    # TODO: generalize to the list case
    assert(len(cls_inds)==1)
    if cls_inds[0] in cond_cls_inds:
      ind = cond_cls_inds.index(cls_inds[0])
      return vals[ind]

    # check if we've cached this conditional
    sorted_inds = np.argsort(cond_cls_inds)
    sorted_cond_cls_inds = np.take(cond_cls_inds, sorted_inds)
    sorted_vals = np.take(vals, sorted_inds)
    hash_string = ' '.join([str(x) for x in sorted(cls_inds)])
    hash_string += '|' + ' '.join(["%d:%d"%(int(x),int(y)) for x,y in zip(cond_cls_inds,vals)])
    if hash_string in self.cache:
      return self.cache[hash_string]

    cls_ind_vals = [1. for cls_ind in cls_inds]
    joint_vals = cls_ind_vals+vals
    joint_prob = self.marg_prob(cls_inds+cond_cls_inds, joint_vals)
    prior_prob = self.marg_prob(cond_cls_inds, vals)
    if prior_prob == 0:
      ans = 0.
    else:
      ans = joint_prob/prior_prob

    # TODO: cross-validate this parameter
    if self.mode=='backoff' and len(cond_cls_inds) > 0 :      
      ans = (1-lam1-lam2)*ans + lam2*self.marg_prob(cls_inds)
      # k pairwise cond
      sum_cond = 0.
      for other_ind in range(len(cond_cls_inds)):
        joint = self.marg_prob([cls_inds[0], cond_cls_inds[other_ind]], [1, vals[other_ind]])
        prior = self.marg_prob(cond_cls_inds[other_ind], vals[other_ind]) 
        if prior == 0:
          sum_cond += 0.
        else:
          sum_cond += joint/prior
      ans += lam1*sum_cond*(1./len(cond_cls_inds))
    self.cache[hash_string] = ans
    return ans

def find_best_lam1_lam2(filename):
  infile = open(filename,'r').readlines()
  minimum = 100000.
  bestvals = 0
  for line in infile:
    entries = line.split()
    score = atof(entries[2])
    if score < minimum:
      minimum = score
      bestvals = entries
  print bestvals

if __name__ == '__main__':
  NGramModel.test_self()

  from synthetic.dataset import Dataset
  dataset = Dataset('full_pascal_trainval')
  cp = ClassPriors(dataset, 'no_smooth')
  
  find_best_lam1_lam2(os.path.join(config.res_dir,'cross_val_lam1_lam2/')+'auc.txt')

  #cp.evaluate_marginal_prob_error()
  lams = []
  for lam1 in np.arange(0,1,0.05):
    for lam2 in np.arange(0,1-lam1,0.05):
      lams.append((lam1,lam2))
  cp.evaluate_method(0.05, 0)
#  auc_file = open(os.path.join(config.res_dir,'cross_val_lam1_lam2/')+'auc.txt','a')
#  for lamdex in range(mpi_rank, len(lams), mpi_size):
#    lam = lams[lamdex]
#    auc = cp.evaluate_method(lam[0], lam[1])
#    auc_file.write(str(lam[0])+' '+str(lam[1])+ ' '+str(auc)+'\n')
