from os.path import join,exists
import numpy as np
import getpass

import synthetic.util as ut

class Config:
  VOCyear = '2007'

  ##################
  # CODE AND MISC DATA PATHS 
  ##################
  if exists('/home/tobibaum'):
    repo_dir = '/home/tobibaum/Documents/Vision/object_detection/'
    save_dir = '/home/tobibaum/Documents/Vision/data/'
  elif exists('/Users/sergeyk/'):
    repo_dir = '/Users/sergeyk/research/object_detection/'
    save_dir = '/Users/sergeyk/research/object_detection/synthetic/temp_data/'
  # ICSI:
  elif exists('/u/vis/'):
    user = getpass.getuser()
    if user=='tobibaum':
      repo_dir = '/u/tobibaum/object_detection/'
      save_dir = '/u/vis/x1/tobibaum/data/'
    if user=='sergeyk':
      repo_dir = '/u/sergeyk/research/object_detection/'
      save_dir = '/tscratch/tmp/sergeyk/object_detection/synthetic/'
  else:
    raise RuntimeError("Can't set paths correctly")
  script_dir = join(repo_dir, 'synthetic')

  ##################
  # INPUT DATA PATHS
  ##################
  test_support_dir = join(script_dir, 'test_support')
  data1 = join(test_support_dir,'data1.json')
  VOC_dir = join(repo_dir, 'VOCdevkit/%(year)s/VOC%(year)s/')%{'year':VOCyear}
  pascal_paths = {
      'test_pascal_train':    join(test_support_dir,'train.txt'),
      'test_pascal_val':      join(test_support_dir,'val.txt'),
      'full_pascal_train':    join(VOC_dir,'ImageSets/Main/train.txt'),
      'full_pascal_val':      join(VOC_dir,'ImageSets/Main/val.txt'),
      'full_pascal_trainval': join(VOC_dir,'ImageSets/Main/trainval.txt'),
      'full_pascal_test':     join(VOC_dir,'ImageSets/Main/test.txt')}
  pascal_classes = ['aeroplane','bicycle', 'bird','boat','bottle','bus','car',
                    'cat','chair','cow','diningtable','dog', 'horse',
                    'motorbike','person','pottedplant','sheep','sofa','train',
                    'tvmonitor']
  kernels = ['linear', 'rbf']

  ##################
  # OUTPUT DATA PATHS
  # Reproducible results should live in own repo.
  # We do not use the VOCyear in any of the paths currently. If in the future
  # we run on different years, we'll handle that with a top-level directory
  # split.
  ##################
  # ./results
  res_dir = join(script_dir, 'results/')
  if exists('/u/vis/') and getpass.getuser() == 'sergeyk':
    res_dir = '/u/vis/x1/sergeyk/object_detection/results/'
  ut.makedirs(res_dir)

  temp_res_dir = save_dir
  ut.makedirs(temp_res_dir)

  # ./results/sliding_windows_{dataset}
  @classmethod
  def get_sliding_windows_dir(cls, dataset_name):
    sliding_windows_dir = join(Config.res_dir, 'sliding_windows_%s'%dataset_name)
    ut.makedirs(sliding_windows_dir)
    return sliding_windows_dir

  # ./results/sliding_windows_{dataset}/metaparams
  @classmethod
  def get_sliding_windows_metaparams_dir(cls, dataset_name):
    dirname = join(Config.get_sliding_windows_dir(dataset_name), 'metaparams')
    ut.makedirs(dirname)
    return dirname 

  # ./results/sliding_windows_{dataset}/stats.pickle
  @classmethod
  def get_window_stats_results(cls, dataset_name):
    return join(Config.get_sliding_windows_dir(dataset_name), 'stats.pickle')

  # ./results/sliding_windows/{stat}/{cls}.png
  @classmethod
  def get_window_stats_plot(clas, dataset_name, stat, cls):
    window_stats_plot_dir = join(Config.get_sliding_windows_dir(dataset_name), stat)
    ut.makedirs(window_stats_plot_dir)
    return join(window_stats_plot_dir, '%s.png'%cls)

  # ./results/sliding_windows_{dataset}/params
  # NOTE: in temp_res_dir!
  @classmethod
  def get_sliding_windows_cached_dir(cls, dataset_name):
    sliding_windows_dir = join(Config.temp_res_dir, 'sliding_windows_%s'%dataset_name)
    sliding_windows_cached_dir = join(sliding_windows_dir, 'cached')
    ut.makedirs(sliding_windows_cached_dir)
    return sliding_windows_cached_dir

  # ./results/sliding_windows_{dataset}/params
  @classmethod
  def get_sliding_windows_params_dir(cls, dataset_name):
    sliding_windows_params_dir = join(Config.get_sliding_windows_dir(dataset_name), 'params')
    ut.makedirs(sliding_windows_params_dir)
    return sliding_windows_params_dir

  # ./results/jumping_windows_{dataset}/
  @classmethod
  def get_jumping_windows_dir(cls, dataset_name):
    dirname = join(Config.res_dir, 'jumping_windows_%s'%dataset_name)
    ut.makedirs(dirname)
    return dirname 

  @classmethod
  def get_windows_params_grid(cls, dataset_name):
    return join(Config.get_sliding_windows_params_dir(dataset_name), 'window_params_grid.csv')

  @classmethod
  def get_window_params_json(cls, dataset_name):
    return join(Config.get_sliding_windows_params_dir(dataset_name), '%s.txt')

  # ./results/evaluations
  evals_dir = join(res_dir, 'evals')
  ut.makedirs(evals_dir)

  # ./results/evaluations/{dataset_name}
  @classmethod
  def get_evals_dir(cls,dataset_name):
    dirname = join(Config.evals_dir,dataset_name)
    ut.makedirs(dirname)
    return dirname

  # ./results/evaluations/{dataset_name}/{dp_config_name}
  @classmethod
  def get_evals_dp_dir(cls,dataset_policy):
    dirname = Config.get_evals_dir(dataset_policy.dataset.get_name())
    subdirname = join(dirname, dataset_policy.get_config_name())
    ut.makedirs(subdirname)
    return subdirname

  # ./results/evaluations/{dataset_name}/{dp_config_name}/cached_dets.npy
  @classmethod
  def get_dp_detections_filename(cls,dataset_policy):
    dirname = Config.get_evals_dp_dir(dataset_policy)
    return join(dirname, 'cached_dets.npy')

  # ./results/evaluations/{dataset_name}/{dp_config_name}/weights/
  @classmethod
  def get_dp_weights_dirname(cls,dataset_policy):
    dirname = Config.get_evals_dp_dir(dataset_policy)
    subdirname = join(dirname,'weights')
    ut.makedirs(subdirname)
    return subdirname

  @classmethod
  def get_cached_dataset_filename(cls,name):
    assert(name in Config.pascal_paths)
    dirname = join(cls.script_dir,'cached_datasets')
    ut.makedirs(dirname)
    filename = join(dirname, str(Config.VOCyear)+'_'+name+'.pickle')
    return filename

  dpm_may25_dirname = '/tscratch/tmp/sergeyk/object_detection/dets_may25_DP/'

  config_dir = join(script_dir,'configs')

  dets_configs_dir = join(res_dir,'det_configs')
  ut.makedirs(dets_configs_dir)

  # ./res_dir/ext_dets/{dataset}_*.npy
  @classmethod
  def get_ext_dets_filename(cls, dataset, suffix):
    dataset_name = dataset.name # NOTE does not depend on # images
    dirname = join(Config.res_dir,'ext_dets')
    ut.makedirs(dirname)
    return join(dirname, '%s_%s.npy'%(dataset_name,suffix))
  
  # directory for gist features
  gist_dir = join(res_dir, 'gist_features/')
  ut.makedirs(gist_dir)
    
  @classmethod
  def get_gist_dict_filename(cls, dataset):
    return Config.gist_dir + dataset + '.npy'
  
  ut.makedirs(join(gist_dir,'svm'))
  
  @classmethod
  def get_gist_svm_filename(cls, for_cls):
    return join(Config.gist_dir,'svm',for_cls)
