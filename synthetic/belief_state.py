from common_mpi import *
from common_imports import *

import synthetic.config as config
from synthetic.class_priors import ClassPriors

class BeliefState:
	"Encapsulates stuff that we keep track of during policy execution."

  def __init__(self,dataset,actions,bounds=None):
  	self.dataset = dataset
  	self.actions = actions
    self.p_c = None # TODO
    self.t = 0
    self.bounds = bounds
    self.reset_actions()

  def reset_actions(self):
    "Zero the 'taken' info of the actions and the observations."
    self.taken = np.zeros(len(self.actions))
    self.observations = np.zeros(len(self.actions))

  def featurize(self):
    """
    Return featurized representation of the current belief state.
    """
    features = self.p_c

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

  def update_with_score(self,action,score):
  	action_ind = self.actions.index(action)
  	self.observations