#!/usr/bin/env python
"""
Wrapper script to execute different functions of our system.
"""

import shutil
import argparse
import matplotlib.pyplot as plt

from common_imports import *
from common_mpi import *

from synthetic.dataset import Dataset
from synthetic.dataset_policy import DatasetPolicy
from synthetic.evaluation import Evaluation
from synthetic.sliding_windows import SlidingWindows 
import synthetic.config as config
from synthetic.extractor import Extractor
from synthetic.training import *
from synthetic.jumping_windows import JumpingWindowsDetector,LookupTable,RootWindow
from synthetic.jumping_windows_with_grid import JumpingWindowsDetectorGrid,\
  LookupTable,RootWindow

def main():
  parser = argparse.ArgumentParser(description='Execute different functions of our system')
  parser.add_argument('mode',
    choices=[
      'detect','evaluate',
      'window_stats', 'evaluate_metaparams', 'evaluate_jw',
      'evaluate_get_pos_windows', 'train_svm',
      'extract_sift','extract_assignments','extract_codebook',
      'evaluate_jw_grid', 'final_metaparams',
      'assemble_dpm_dets','ctfdet','assemble_ctf_dets'
      ])
  parser.add_argument('--test_dataset', choices=['val','test','train'],
      default='test', help='dataset to use for testing. the training dataset \
      is automatically inferred (val->train and test->trainval).')
  parser.add_argument('--first_n', type=int,
      help='only take the first N images in the datasets')
  parser.add_argument('--bounds', type=str,
      help='the start_time and deadline_time for the ImagePolicy and corresponding evaluation. ex: (1,5)')
  parser.add_argument('--name', help='name for this run')
  parser.add_argument('--priors', default='random', help= \
      "list of choice for the policy for selecting the next action. choose from random, oracle,fixed_order, no_smooth, backoff. ex: --priors=random,oracle,no_smooth")
  parser.add_argument('--compare_evals', action='store_true', 
      default=False, help='plot all the priors modes given on same plot'),
  parser.add_argument('--detector', choices=['perfect','perfect_with_noise', 'dpm','ctf'],
      default='perfect', help='detector type')
  parser.add_argument('--force', action='store_true', 
      default=False, help='force overwrite')
  parser.add_argument('--gist', action='store_true', 
      default=False, help='use GIST as one of the actions')
  parser.add_argument('--clear_tmp', action='store_true', 
      default=False, help='clear the cached windows folder before running'),
  parser.add_argument('--feature_type', choices=['sift','dsift'], 
      default='dsift', help='use this feature type'),
  parser.add_argument('--kernel', choices=['chi2','rbf'], 
      default='chi2', help='kernel to train svm on'),
      
  args = parser.parse_args()
  if args.priors:
    args.priors = args.priors.split(',')
  if args.bounds:
    args.bounds = [float(x) for x in re.findall(r'\d+', args.bounds)]
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
  
  # Create window generator
  sw = SlidingWindows(dataset,train_dataset)

  if args.clear_tmp:
    dirname = config.get_sliding_windows_cached_dir(train_dataset.get_name())
    shutil.rmtree(dirname)
    dirname = config.get_sliding_windows_cached_dir(dataset.get_name())
    shutil.rmtree(dirname)

  if args.mode=='assemble_dpm_dets':
    policy = DatasetPolicy(dataset,train_dataset,sw)
    dets = policy.load_ext_detections(dataset,suffix='dpm_may25')

  if args.mode=='assemble_ctf_dets':
    policy = DatasetPolicy(dataset,train_dataset,sw)
    dets = policy.load_ext_detections(dataset,'ctf','ctf_default')
    dets = policy.load_ext_detections(dataset,'ctf','ctf_nohal')
    dets = policy.load_ext_detections(dataset,'ctf', 'ctf_halfsize')

  if args.mode=='evaluate_get_pos_windows':
    evaluate_get_pos_windows(train_dataset)
    return

  if args.mode=='window_stats':
    """
    Compute and plot the statistics of ground truth window parameters.
    """
    results = SlidingWindows.get_dataset_window_stats(train_dataset,plot=True)

  if args.mode=='detect' or args.mode=='evaluate':
    tables = []
    for prior in args.priors:
      dp = DatasetPolicy(dataset=dataset, train_dataset=train_dataset,
          sw=sw, detector=args.detector, class_priors_mode=prior,
          suffix=args.name, bounds=args.bounds, gist=args.gist)

      all_dets = None
      if args.mode=='detect':
        all_dets = dp.detect_in_dataset(force=args.force)
        ev.evaluate_detections_whole(all_dets,force=args.force)

      if args.mode=='evaluate':
        ev = Evaluation(dp)
        table = ev.evaluate_dets_vs_t_avg(all_dets,force=args.force)
        if comm_rank==0:
          tables.append(table)
        #ev.evaluate_dets_vs_t_whole(all_dets,force=args.force)

    # If asked to, and did more than one condition, plot comparison as well
    if args.compare_evals and len(args.priors)>1 and comm_rank==0:
      filename = os.path.join(config.evals_dir, '%s_%s_%s_%s.png'%(
            dataset.get_name(),
            '-'.join(args.priors),
            args.detector,
            args.name))
      Evaluation.plot_ap_vs_t(tables,filename,args.bounds)

  if args.mode=='ctfdet':
    """Run Pedersoli's detector on the dataset and assemble into one Table."""
    run_pedersoli(dataset)

  if args.mode=='evaluate_jw':
    """
    Evaluate the jumping window approach by producing plots of recall vs.
    #windows.
    """
    # TODO hack: both sw and jw should subclass something like WindowGenerator
    jw = JumpingWindowsDetector(use_scale=True)
    sw.jw = jw
    #classes = dataset.classes
    classes = ['car']
#    classes = ['bicycle' ,'car','horse', 'sofa',\
#               'bird',  'chair',     'motorbike', 'train',\
#               'boat',  'cow',       'person',    'tvmonitor',\
#               'bottle','diningtable',  'pottedplant',\
#               'bus','dog'     ,'sheep']
    for cls_idx in range(mpi_rank, len(classes), mpi_size):
    #for cls in dataset.classes:
      cls = classes[cls_idx]
      dirname = config.get_jumping_windows_dir(dataset.get_name())
      filename = os.path.join(dirname,'%s'%cls)
      sw.evaluate_recall(cls, filename, metaparams=None, mode='jw', plot=True)
  
  if args.mode=='evaluate_jw_grid':
    """
    Evaluate the jumping window approach by producing plots of recall vs.
    #windows.
    """
    sw = SlidingWindows(dataset,train_dataset)
    jw = JumpingWindowsDetectorGrid()
    sw.jw = jw
    for cls in dataset.classes:
      dirname = config.get_jumping_windows_dir(dataset.get_name())
      filename = os.path.join(dirname,'%s'%cls)
      if os.path.isfile(config.save_dir + 'JumpingWindows/'+cls):
        sw.evaluate_recall(cls, filename, metaparams=None, mode='jw', plot=True)

  if args.mode=='train_svm':
    randomize = not os.path.exists('/home/tobibaum')
    
    d = Dataset('full_pascal_train')
    dtest = Dataset('full_pascal_val')  
    e = Extractor()  
    classes = config.pascal_classes  
    num_words = 3000
    iters = 5
    feature_type = 'dsift'
    codebook_samples = 15
    num_pos = 'max'
    testsize = 'max'
    if args.first_n:
      num_pos = args.first_n
      testsize = 1.5*num_pos
     
    kernel = args.kernel
    
    if mpi_rank == 0:
      ut.makedirs(config.save_dir + 'features/' + feature_type + '/times/')
      ut.makedirs(config.save_dir + 'features/' + feature_type + '/codebooks/times/')
      ut.makedirs(config.save_dir + 'features/' + feature_type + '/svms/train_times/')
      
    for cls_idx in range(mpi_rank, len(classes), mpi_size): 
    #for cls in classes:
      cls = classes[cls_idx]
      codebook = e.get_codebook(d, feature_type)
      pos_arr = d.get_pos_windows(cls)
      
      neg_arr = d.get_neg_windows(pos_arr.shape[0], cls, max_overlap=0)
      
      if not num_pos == 'max':    
        if not randomize:
          pos_arr = pos_arr[:num_pos]
          neg_arr = pos_arr[:num_pos]
        else:
          rand = np.random.random_integers(0, pos_arr.shape[0] - 1, size=num_pos)
          pos_arr = pos_arr[rand]
          rand = np.random.random_integers(0, neg_arr.shape[0] - 1, size=num_pos)
          neg_arr = neg_arr[rand]     
      pos_table = ut.Table(pos_arr, ['x','y','w','h','img_ind'])
      neg_table = ut.Table(neg_arr, pos_table.cols)      
      train_with_hard_negatives(d, dtest,  num_words,codebook_samples,codebook,\
                                cls, pos_table, neg_table,feature_type, \
                                iterations=iters, kernel=kernel, L=2, \
                                testsize=testsize,randomize=randomize)

  if args.mode=='evaluate_metaparams':
    """
    Grid search over metaparams values for get_windows_new, with the AUC of
    recall vs. # windows evaluation.
    """
    sw.grid_search_over_metaparams()
    return

  if args.mode=='final_metaparams':
    dirname = config.get_sliding_windows_metaparams_dir(train_dataset.get_name())
    # currently these are the best auc/complexity params
    best_params_for_classes = [
        (62,15,12,'importance',0), #aeroplane
        (83,15,12,'importance',0), #bicycle
        (62,15,12,'importance',0), #bird
        (62,15,12,'importance',0), #boat
        (125,12,12,'importance',0), #bottle
        (83,12,9,'importance',0), #bus
        (125,15,9,'importance',0), #car
        (125,12,12,'linear',0), #cat
        (125,15,9,'importance',0), #chair
        (125,9,6,'importance',0), #cow
        (125,15,6,'linear',0), #diningtable
        (62,15,12,'importance',0), #dog
        (83,15,6,'importance',0), #horse
        (83,12,6,'importance',0), #motorbike
        (83,15,12,'importance',0), #person
        (83,15,6,'importance',0), #pottedplant
        (83,15,12,'importance',0), #sheep
        (83,9,6,'importance',0), #sofa
        (62,12,6,'importance',0), #train
        (62,12,12,'importance',0), #tvmonitor
        (125,9,12,'importance',0) #all
        ]
    # ACTUALLY THEY ARE ALL THE SAME!
    cheap_params = (62, 9, 6, 'importance', 0)
    for i in range(comm_rank,dataset.num_classes(),comm_size):
      cls = dataset.classes[i]
      best_params = best_params_for_classes[i]
      #samples,num_scales,num_ratios,mode,priority,cls = cheap_params

      metaparams = {
        'samples_per_500px': samples,
        'num_scales': num_scales,
        'num_ratios': num_ratios,
        'mode': mode,
        'priority': 0 }
      filename = '%s_%d_%d_%d_%s_%d'%(
          cls,
          metaparams['samples_per_500px'],
          metaparams['num_scales'],
          metaparams['num_ratios'],
          metaparams['mode'],
          metaparams['priority'])
      filename = os.path.join(dirname,filename)

      tables = sw.evaluate_recall(cls,filename,metaparams,'sw',plot=True,force=False)

      metaparams = {
        'samples_per_500px': samples,
        'num_scales': num_scales,
        'num_ratios': num_ratios,
        'mode': mode,
        'priority': 1 }
      filename = '%s_%d_%d_%d_%s_%d'%(
          cls,
          metaparams['samples_per_500px'],
          metaparams['num_scales'],
          metaparams['num_ratios'],
          metaparams['mode'],
          metaparams['priority'])
      filename = os.path.join(dirname,filename)

      tables = sw.evaluate_recall(cls,filename,metaparams,'sw',plot=True,force=False)
    return

  if args.mode=='extract_sift':
    e=Extractor()
    e.extract_all(['sift'], ['full_pascal_trainval','full_pascal_test'], 0, 0) 
    
  if args.mode=='extract_assignments':
    e=Extractor()
    feature_type = 'sift'
    for image_set in ['full_pascal_trainval','full_pascal_test']:
      d = Dataset(image_set)
      codebook = e.get_codebook(d, feature_type)  
      print 'codebook loaded'
      
      for img_ind in range(mpi_rank,len(d.images),mpi_size):
        img = d.images[img_ind]
      #for img in d.images:
        e.get_assignments(np.array([0,0,img.size[0],img.size[1]]), feature_type, \
                          codebook, img)

  if args.mode=='extract_codebook':
    d = Dataset('full_pascal_trainval')
    e = Extractor()
    codebook = e.get_codebook(d, args.feature_type)

#######
def evaluate_get_pos_windows(dataset):
  """Fetch some positive and negative windows."""
  #for cls in dataset.classes:
  for cls in ['aeroplane','bicycle','bird','person']:
    t = time.time()
    print(cls)
    print(dataset.get_pos_windows(cls).shape)
    print("time: %f"%(time.time()-t))
    t = time.time()
    print(dataset.get_neg_windows(5000,cls).shape)
    print("time: %f"%(time.time()-t))
    
if __name__ == '__main__':
  main()
