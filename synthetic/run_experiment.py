#!/usr/bin/env python
"""
Wrapper script for running the final stages of our CVPR12 experiments.
"""
import os,time,sys,shutil,re,glob
import json
import argparse
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
import random
import string

from synthetic.dataset import Dataset
from synthetic.dataset_policy import DatasetPolicy
from synthetic.sliding_windows import SlidingWindows 
from synthetic.evaluation import Evaluation
from synthetic.config import Config
import synthetic.util as ut

from mpi4py import MPI
from synthetic.safebarrier import safebarrier
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

def load_configs(args_string):
  """
  Load the config, expected in json format.
  If filepath is a directory, loads all json files in it.
  Can also be list of json filenames without extensions, separated by commas.
  """
  def load_config(full_filename):
    print(full_filename)
    assert(os.path.exists(full_filename))
    with open(full_filename) as f:
      config = json.load(f)
    if 'bounds' in config:
      config['bounds'] = tuple(config['bounds']) # IMPORTANT!
    return config

  config_filenames = args_string.split(',')
  # check if we only have one file and if its a directory
  if len(config_filenames)==1:
    if os.path.isdir(Config.config_dir+'/'+config_filenames[0]):
      dirname = config_filenames[0]
      config_filenames = glob.glob(Config.config_dir+'/'+dirname+'/*.json')
      return [load_config(filename) for filename in config_filenames]
  # load the json files
  full_filename = lambda filename: os.path.join(Config.config_dir,filename+'.json')
  return [load_config(full_filename(filename)) for filename in config_filenames]

def main():
  parser = argparse.ArgumentParser(description='Execute different functions of our system')
  parser.add_argument('--test_dataset', choices=['val','test','train','trainval'],
      default='test',
      help="""Dataset to use for testing. Can only be set to one thing. Run on
      val until ready for final runs.
      the training dataset is automatically inferred (val->train and test->trainval).""")
  parser.add_argument('--first_n', type=int,
      help='only take the first N images in the datasets')
  parser.add_argument('--configs', help="""List of config files to run on.
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
  if not args.configs:
    configs = [DatasetPolicy.default_config]
  else:
    configs = load_configs(args.configs)
  print(args)

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
  for config in configs:
    sw = SlidingWindows(dataset,train_dataset)
    dp = DatasetPolicy(dataset, train_dataset, sw, **config)
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
    dirname = Config.get_evals_dir(dataset.get_name())
    filename = '-'.join([dp.get_config_name() for dp in dps])
    # TODO: temp cause filename too long or something
    filename = ''.join(random.choice(string.ascii_uppercase + string.digits) for x in range(8))
    full_filename = os.path.join(dirname, '%s.png'%filename)
    full_filename_no_legend = os.path.join(dirname, '%s_no_legend.png'%filename)
    Evaluation.plot_ap_vs_t(tables, full_filename, all_bounds, with_legend=True)
    Evaluation.plot_ap_vs_t(tables, full_filename_no_legend, all_bounds, with_legend=False)
    
if __name__ == '__main__':
  main()

