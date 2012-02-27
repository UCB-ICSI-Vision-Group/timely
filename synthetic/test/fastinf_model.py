from synthetic.common_imports import *
from synthetic.common_mpi import *
import synthetic.config as config

from synthetic.dataset import Dataset
from synthetic.fastinf_model import FastinfModel

def test():
  dataset = Dataset('full_pascal_trainval')
  fm = FastinfModel(dataset,'perfect')
  # NOTE: just took values from a run of the thing
  correct = ['0.94973', '0.9494', '0.92994', '0.96123', '0.94915', '0.95972', '0.8447', '0.92857', '0.88859', '0.96992', '0.94926', '0.91243', '0.94186', '0.95039', '0.59961', '0.94597', '0.97987', '0.92665', '0.94526', '0.94454']
  assert(np.all(fm.p_c == correct))
  observations = np.zeros(20)
  taken = np.zeros(20)
  fm.update_with_observations(taken,observations)
  assert(np.all(fm.p_c == correct))
  observations[5] = 1
  taken[5] = 1
  fm.update_with_observations(taken,observations)
  correct = ['0.96447', '0.90917', '0.96242', '0.96674', '0.98247', '0.029152', '0.46157', '0.97148', '0.98358', '0.97717', '0.99224', '0.96514', '0.95976', '0.92104', '0.54853', '0.98594', '0.98314', '0.98425', '0.95752', '0.9858']
  assert(np.all(fm.p_c == correct))
  observations[15] = 0
  taken[15] = 1
  fm.update_with_observations(taken,observations)
  correct = ['0.96438', '0.90867', '0.9624', '0.96666', '0.98275', '0.02857', '0.45744', '0.97176', '0.98447', '0.9771', '0.99268', '0.96523', '0.9596', '0.92041', '0.54748', '0.99955', '0.98312', '0.98494', '0.95735', '0.98614']
  assert(np.all(fm.p_c == correct))
  