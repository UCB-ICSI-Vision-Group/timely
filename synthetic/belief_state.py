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

  def __init__(self,dataset,actions,mode='fixed_order',bounds=None,model=None,fastinf_model_name='perfect'):
    assert(mode in self.accepted_modes)
    self.mode = mode
    self.dataset = dataset
    self.actions = actions
    self.bounds = bounds
    self.fastinf_model_name = fastinf_model_name

    if mode=='no_smooth' or mode=='backoff':
      if model:
        assert(isinstance(model,NGramModel))
        self.model = model
      else:
        self.model = NGramModel(dataset,mode)
    elif mode=='fixed_order':
      if model:
        assert(isinstance(model,FixedOrderModel))
        self.model = model
      else:
        self.model = FixedOrderModel(dataset)
    elif mode=='fastinf':
      if model:
        assert(isinstance(model,FastinfModel))
        self.model = model
      else:
        num_actions = len(self.actions)
        self.model = FastinfModel(dataset,self.fastinf_model_name,num_actions)
    else:
      raise RuntimeError("Unknown mode")

    self.model.reset()
    self.t = 0
    self.reset_actions()

  def __repr__(self):
    return "BeliefState: \n%s\n%s"%(
      self.get_p_c(), zip(self.taken,self.observations))

  def get_p_c(self):
    return self.model.p_c

  def reset_actions(self):
    "Zero the 'taken' info of the actions and the observations."
    self.taken = np.zeros(len(self.actions))
    self.observations = np.zeros(len(self.actions))

  def update_with_score(self,action_ind,score):
    "Update the taken and observations lists, the model, and get the new marginals."
    self.taken[action_ind] = 1
    self.observations[action_ind] = score
    self.model.update_with_observations(self.taken,self.observations)
    self.full_feature = self.compute_full_feature()

  num_time_blocks = 1
  num_features = num_time_blocks * 4 # [P(C) P(not C) H(C) 1]
  def compute_full_feature(self):
    """
    Return featurized representation of the current belief state.
    The features are in action blocks, meaning that this method returns
    a vector of size self.num_features*len(self.actions).
    Return a matrix that can be zeroed out everywhere except the
    relevant action_ind, and then flattened.
    NOTE: Keep the class variable num_features synced with the behavior here.
    """
    p_c = self.get_p_c()
    p_not_c = 1 - p_c
    h_c = -p_c*ut.log2(p_c) + -p_not_c*ut.log2(p_not_c)
    h_c[h_c==-0]=0
    ones = np.ones(len(self.actions))

    # TODO: work out the time-blocks

    return np.vstack((p_c,p_not_c,h_c,ones))

  def block_out_action(self, full_feature, action_ind=-1):
    """
    Take a full_feature matrix and zero out all the values except those
    in the relevant action block.
    If action_ind < 0, returns the flat feature with nothing zeroed out.
    """
    if action_ind < 0:
      # flatten in column-major format (column index varies slowest)
      return full_feature.flatten('F')
    assert(action_ind<len(self.actions))
    feature = np.zeros(np.prod(full_feature.shape))
    start_ind = action_ind*self.num_features
    feature[start_ind:start_ind+self.num_features] = full_feature[:,action_ind]
    return feature
