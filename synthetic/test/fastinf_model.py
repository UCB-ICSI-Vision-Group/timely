from synthetic.common_imports import *
from synthetic.common_mpi import *
import synthetic.config as config

from synthetic.dataset import Dataset
from synthetic.fastinf_model import FastinfModel

def test():
  dataset = Dataset('full_pascal_trainval')
  fm = FastinfModel(dataset,'perfect',20)
  # NOTE: just took values from a run of the thing
  correct = [float(x) for x in "0.050273  0.050599  0.070062  0.038771  0.050848  0.040276  0.1553\
  0.071435  0.11141   0.030083  0.050744  0.087569  0.058135  0.049606\
  0.40039   0.054032  0.020131  0.073353  0.054743  0.055462".split()]
  assert(np.all(fm.p_c == correct))
  observations = np.zeros(20)
  taken = np.zeros(20)
  fm.update_with_observations(taken,observations)
  assert(np.all(fm.p_c == correct))
  observations[5] = 1
  taken[5] = 1
  fm.update_with_observations(taken,observations)
  correct = [float(x) for x in  "0.035533   0.090826   0.037577   0.033263   0.01753    0.97085    0.53843\
    0.028521   0.016417   0.022835   0.0077634  0.034858   0.040242   0.078955\
    0.45147    0.014064   0.016861   0.015754   0.042481   0.014201".split()]
  assert(np.all(fm.p_c == correct))
  observations[15] = 0
  taken[15] = 1
  fm.update_with_observations(taken,observations)
  correct = [float(x) for x in "3.56180000e-02   9.13280000e-02   3.75970000e-02   3.33390000e-02\
   1.72460000e-02   9.71430000e-01   5.42560000e-01   2.82360000e-02\
   1.55320000e-02   2.28960000e-02   7.32410000e-03   3.47720000e-02\
   4.04000000e-02   7.95920000e-02   4.52520000e-01   4.49710000e-04\
   1.68850000e-02   1.50560000e-02   4.26550000e-02   1.38560000e-02".split()]
  assert(np.all(fm.p_c == correct))
  