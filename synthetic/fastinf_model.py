from common_imports import *
from common_mpi import *

from synthetic.ngram_model import InferenceModel

class FastinfModel(InferenceModel):
  def __init__(self,dataset,suffix):
    self.fname = config.get_fastinf_res_file(dataset,suffix)

  def update_with_observations(self, observations):
    # TODO: implement
    None