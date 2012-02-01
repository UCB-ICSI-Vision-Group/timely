#!/usr/bin/env python
"""
Wrapper script for running the final stages of our CVPR12 experiments.
"""
import shutil
import glob
import argparse
import matplotlib.pyplot as plt
import random
import string

from common_imports import *
from common_mpi import *

from synthetic.dataset import Dataset
from synthetic.dataset_policy import DatasetPolicy
from synthetic.sliding_windows import SlidingWindows 
from synthetic.evaluation import Evaluation
import synthetic.config as config

def load_configs(name):
  """
  Load the config in json format and return as list of experiments.
  Look for config_dir/#{name}.json
  """
  full_filename = opjoin(config.config_dir,name+'.json')
  print("Loading %s"%full_filename)
  assert(opexists(full_filename))
  with open(full_filename) as f:
    cf = json.load(f)
  # Gather multiple values of settings, if given
  num_conditions = 1
  if 'bounds' in cf:
    bounds_list = []
    if isinstance(cf['bounds'][0], list):
      bounds_list = cf['bounds']
    else:
      bounds_list = [cf['bounds']]
    num_conditions *= len(bounds_list)
  
  configs = []
  for i in range(0,num_conditions):
    configs.append(dict(cf))
    configs[i]['bounds'] = bounds_list[i%len(bounds_list)]
  return configs

def main():
  parser = argparse.ArgumentParser(
    description="Run experiments with the timely detection system.")

  # TODO: this should be specified in config file as well
  parser.add_argument('--test_dataset',
    choices=['val','test','train','trainval'],
    default='val',
    help="""Dataset to use for testing. Run on val until final runs.
    The training dataset is inferred (val->train; test->trainval).""")

  parser.add_argument('--first_n', type=int,
    help='only take the first N images in the datasets')

  parser.add_argument('--configs',
    help="""List of config files to run on.
    They have to be in results/configs/*.json, but just give the filename without extension.
    If the name passed in is a dirname, then will go through all the config files in that directory.""")

  parser.add_argument('--force', action='store_true', 
    default=False, help='force overwrite')

  parser.add_argument('--wholeset_prs', action='store_true', 
    default=False, help='evaluate in the final p-r regime')

  parser.add_argument('--no_policy', action='store_true', 
    default=False, help='do not use the policy when evaluating wholeset_pr')

  parser.add_argument('--no_apvst', action='store_true', 
    default=False, help='do NOT evaluate in the ap vs. time regime')

  parser.add_argument('--det_configs', action='store_true', 
    default=False, help='output detector statistics to det_configs')

  args = parser.parse_args()
  print(args)

  # If config file is not given, just run one experiment using default config
  if not args.configs:
    configs = [DatasetPolicy.default_config]
  else:
    configs = load_configs(args.configs)

  # Load the dataset
  dataset = Dataset('full_pascal_'+args.test_dataset)
  if args.first_n:
    dataset.images = dataset.images[:args.first_n]

  # Infer train_dataset
  if args.test_dataset=='test':
    train_dataset = Dataset('full_pascal_trainval')
  elif args.test_dataset=='val':
    train_dataset = Dataset('full_pascal_train')
  else:
    print("Impossible, setting train_dataset to dataset")
    train_dataset = dataset

  tables = []
  all_bounds = []
  dps = []

  for config_f in configs:
    sw = SlidingWindows(dataset,train_dataset)
    dp = DatasetPolicy(dataset, train_dataset, sw, **config_f)

    ev = Evaluation(dp)
    dps.append(dp)
    all_bounds.append(dp.bounds)

    # output the det configs first if asked
    if args.det_configs:
      dp.output_det_statistics()

    # evaluate in the AP vs. Time regime, unless told not to
    if not args.no_apvst:
      table = ev.evaluate_dets_vs_t_avg(None,force=args.force)
      if comm_rank==0:
        tables.append(table)
      #ev.evaluate_dets_vs_t_whole(all_dets,force=args.force)

    # optionally, evaluate in the standard PR regime
    if args.wholeset_prs:
      if args.no_policy:
        dets = dp.get_ext_dets()
        ev.evaluate_detections_whole(dets,force=args.force)
      else:
        ev.evaluate_detections_whole(None,force=args.force)

  # and plot the comparison if multiple config files were given
  if not args.no_apvst and len(configs)>1 and comm_rank==0:
    dirname = config.get_evals_dir(dataset.get_name())
    filename = '-'.join([dp.get_config_name() for dp in dps])
    # TODO: temp cause filename too long or something
    filename = ''.join(random.choice(string.ascii_uppercase + string.digits) for x in range(8))
    full_filename = os.path.join(dirname, '%s.png'%filename)
    full_filename_no_legend = os.path.join(dirname, '%s_no_legend.png'%filename)
    Evaluation.plot_ap_vs_t(tables, full_filename, all_bounds, with_legend=True)
    Evaluation.plot_ap_vs_t(tables, full_filename_no_legend, all_bounds, with_legend=False)
    
if __name__ == '__main__':
  main()

