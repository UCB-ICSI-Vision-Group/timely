from synthetic.common_mpi import *
from synthetic.common_imports import *
import synthetic.config as config

from synthetic.dataset import Dataset
from synthetic.dataset_policy import DatasetPolicy
from synthetic.belief_state import BeliefState

class TestBeliefState(object):
  def setup(self):
    d = Dataset('test_pascal_trainval',force=True)
    d2 = Dataset('test_pascal_test',force=True)
    config = {'detectors': ['csc_default']}
    self.dp = DatasetPolicy(d,d2,**config)
    self.bs = BeliefState(d,self.dp.actions)

  def test_featurization(self):
    ff = self.bs.compute_full_feature()
    np.set_printoptions(precision=2)
    print self.bs.block_out_action(ff,-1)
    print self.bs.block_out_action(ff,0)
    print self.bs.block_out_action(ff,3)
    # TODO: make asserts here
