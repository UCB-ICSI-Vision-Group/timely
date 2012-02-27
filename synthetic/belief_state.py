from common_mpi import *
from common_imports import *
import synthetic.config as config

from synthetic.fastinf_model import FastinfModel
from synthetic.ngram_model import NGramModel,FixedOrderModel

class BeliefState(object):
  """
  Encapsulates stuff that we keep track of during policy execution.
  Methods to initialize the model, update with an observed posterior,
  condition on observed values, and compute expected information gain.
  """

  ngram_modes = ['no_smooth','backoff']
  accepted_modes = ngram_modes+['fixed_order','fastinf']

  def __init__(self,dataset,actions,mode='fixed_order',bounds=None):
    assert(mode in self.accepted_modes)
    self.mode = mode

    if mode=='no_smooth' or mode=='backoff':
      self.model = NGramModel(dataset,mode)
    elif mode=='fixed_order':
      self.model = FixedOrderModel(dataset)
    elif mode=='fastinf':
      # TODO: work out the suffix situation
      suffix = 'perfect'
      self.model = FastinfModel(dataset,suffix)
    else:
      raise RuntimeError("Unknown mode")

    self.dataset = dataset
    self.actions = actions
    self.t = 0
    self.bounds = bounds

    # zero stuff out
    self.reset_actions()

  def __repr__(self):
    return "BeliefState: \n%s\n%s"%(
      self.get_p_c(), zip(self.taken,self.observations))

  def get_p_c(self):
    return self.model.get_probabilities()

  def reset_actions(self):
    "Zero the 'taken' info of the actions and the observations."
    self.taken = np.zeros(len(self.actions))
    self.observations = np.zeros(len(self.actions))

  def update_with_score(self,action_ind,score):
    "Update the taken and observations lists, the model, and get the new marginals."
    self.taken[action_ind] = 1
    self.observations[action_ind] = score
    self.model.update_with_observations(self.taken,self.observations)

  def featurize(self):
    """
    Return featurized representation of the current belief state.
    """
    if self.mode in ngram_modes:
      features = self.p_c
    if self.mode=='fastinf':
      features = self.model.get_infogains()
    else:
      raise RuntimeError("Impossible")

    def H(x): return np.sum([-x_i*ut.log(x_i) -(1-x_i)*ut.log(1-x_i) for x_i in x])
    entropy = H(b['priors'].priors)
    #features += [entropy]

    time_to_start = 0
    if b.bounds:
      if b.bounds[0]>0:
        time_to_start = max(0, (b.bounds[0]-b['t'])/b.bounds[0])
      time_to_deadline = max(0, (b.bounds[1]-b['t'])/b.bounds[1])
    else:
      time_to_start = 0
      time_to_deadline = 1
    #features += [time_to_start,time_to_deadline]
    #features += [time_to_deadline]

    return np.array(features)