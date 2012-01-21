import time,copy,os,datetime
import scipy.io
import numpy as np
from pprint import pprint
import json

import re
import util as ut
from synthetic.config import Config 
from synthetic.class_priors import ClassPriors
from synthetic.dataset import Dataset
from synthetic.evaluation import Evaluation
from synthetic.gist_detector import GistPriors
from synthetic.detector import *
from synthetic.ext_detector import ExternalDetector
from synthetic.bounding_box import BoundingBox

from mpi4py import MPI
from synthetic.safebarrier import safebarrier
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

class ImageAction:
  def __init__(self, name, obj):
    self.name = name
    self.obj = obj

  def __repr__(self):
    return self.name

class DatasetPolicy:
  # run_experiment.py uses this and __init__ uses as default values
  default_config = {
    'suffix': 'nov19', # can use this to re-run on same param due to changed code
    'detector': 'ext', # perfect,perfect_with_noise,ext
    'class_priors_mode': 'random', # random,oracle,fixed_order,no_smooth,backoff
    'dets_suffixes': ['dpm_may25'], # further specifies which detector to use
    'bounds': None, # start and deadline times for the policy
    'gist': 0, # use the GIST action? 0/1
    'with_entropy': 0, # use the entropy feature in belief state featurization?
    'with_times': 0, # use the time-related features in belief state featurization?
    'learn_policy': 'manual' # can be manual,advanced,advanced_pair,greedy,lspi
  }

  def get_name(self):
    return "%s_%s"%(self.dataset.get_name(), self.get_config_name()) 

  def get_config_name(self):
    """All params except for dataset."""
    middle = ""
    if self.bounds:
      middle += '-'.join((str(x) for x in self.bounds))
    if self.gist:
      middle += "with_gist"
    if self.with_entropy:
      middle += "with_entropy"
    if self.with_times:
      middle += "with_times"
    dets_suffixes = '-'.join(self.dets_suffixes)
    name = '_'.join(
        (self.class_priors_mode,
        dets_suffixes,
        middle,
        self.learn_policy,
        self.suffix))
    return name

  @classmethod
  def get_cols(cls):
    return Detector.get_cols() + ['cls_ind','img_ind','time']

  def __init__(self, dataset, train_dataset, sw, **kwargs):
    config = copy.copy(DatasetPolicy.default_config)
    config.update(kwargs)

    # required stuff
    self.dataset = dataset
    self.train_dataset = train_dataset
    self.sw = sw

    # config stuff: we take a shortcut
    self.__dict__.update(config)

    delay_initialize = kwargs.pop('delay_initialize', False)
    if not delay_initialize:
      self.initialize_fields()

  def initialize_fields(self):
    """
    Allow outside callers to change the config and then call this to
    re-init.
    """
    print("Running with config:")
    pprint(self.__dict__)
    self.priors = ClassPriors(self.train_dataset,mode=self.class_priors_mode)
    self.ev = Evaluation(self)

    #####
    # Construct the actions list
    self.actions = []
    
    # GIST, if it is to be added, is the first action
    if self.gist:
      gist_obj = GistPriors(self.dataset.name)
      self.actions.append(ImageAction('gist', gist_obj))

    # synthetic perfect detector
    if self.detector=='perfect':
      for cls in self.dataset.classes:
        det = PerfectDetector(self.dataset, cls, self.sw)
        self.actions.append(ImageAction('perfect_%s'%cls,det))
    # synthetic perfect detector with noise in the detections
    elif self.detector=='perfect_with_noise':
      for cls in self.dataset.classes:
        det = PerfectDetectorWithNoise(self.dataset, cls, self.sw)
        self.actions.append(ImageAction('perfect_noise_%s'%cls,det))
    # real detectors, with pre-cached detections
    elif self.detector=='ext':
      for suffix in self.dets_suffixes:
        # load the dets from cache file
        # TODO: move this regexp matching into load_ext_detections
        if re.search('dpm',suffix):
          self.all_dets = self.load_ext_detections(self.dataset, 'dpm', suffix)
        elif re.search('ctf',suffix):
          self.all_dets = self.load_ext_detections(self.dataset, 'ctf', suffix)
        elif re.search('csc',suffix):
          self.all_dets = self.load_ext_detections(self.dataset, 'csc', suffix)
        else:
          print(suffix)
          raise RuntimeError('Unknown detector type in suffix')
        # now we have all_dets; parcel them out to classes
        for cls in self.dataset.classes:
          cls_ind = self.dataset.get_ind(cls)
          all_dets_for_cls = self.all_dets.filter_on_column('cls_ind',cls_ind,omit=True)
          det = ExternalDetector(self.dataset, cls, self.sw, all_dets_for_cls, suffix)
          self.actions.append(ImageAction('%s_%s'%(suffix,cls), det))
    else:
      raise RuntimeError("Unknown mode for detectors")

    # oh baby we're starting soon
    self.time_elapsed = 0

    # The default weights are just identity weights on the corresponding class
    # priors
    # TODO: load from something: filename constructed from config_name
    if self.learn_policy == 'manual':
      self.weights = np.zeros((len(self.actions),len(self.get_feature_vec())))
      # The gist action is first, so offset the weights
      if self.gist:
        np.fill_diagonal(self.weights[1:,:],1)
      else:
        naive_aps = [action.obj.config['naive_ap|present'] for action in self.actions]
        np.fill_diagonal(self.weights,naive_aps)
      self.write_out_weights()
    elif self.learn_policy == 'greedy':
      self.weights = np.zeros((len(self.actions),len(self.get_feature_vec())))
      # The gist action is first, so offset the weights
      if self.gist:
        np.fill_diagonal(self.weights[1:,:],1)
      else:
        naive_aps = [action.obj.config['naive_ap|present'] for action in self.actions]
        np.fill_diagonal(self.weights,naive_aps)
      self.learn_weights()
      self.write_out_weights()
    elif self.learn_policy == 'advanced' or self.learn_policy == 'advanced_pair' or self.learn_policy=='slope' or self.learn_policy=='slope_pair':
      None
    else:
      raise RuntimeError('Not yet implemented')
    
    # And now we're ready to start
    self.reset_actions()
  
  def get_ext_dets(self):
    """Return external detections straight from their cache."""
    return self.all_dets

  def detect_in_dataset(self,force=False):
    """
    Return list of detections for the whole dataset.
    Check for cached file first.
    """
    # check for cached detections
    filename = Config.get_dp_detections_filename(self)
    dets_table = None
    if not force and os.path.exists(filename):
      dets_table = np.load(filename)[()]
      print("DatasetPolicy: Loaded whole-set dets from cache.")
      return dets_table
    images = self.dataset.images
    results = []
    for i in range(comm_rank,len(images),comm_size):
      dets = self.detect_in_image(images[i])
      results.append(dets)
    all_results = None
    if comm_rank == 0:
      all_results = []
    safebarrier(comm)
    all_results = comm.reduce(results, op=MPI.SUM, root=0)
    if comm_rank==0:
      dets_table = ut.Table(cols=self.get_cols())
      dets_table.arr = np.vstack(all_results)
      np.save(filename,dets_table)
      print("Found %d dets"%dets_table.shape()[0])
    safebarrier(comm)
    dets_table = comm.bcast(dets_table,root=0)
    return dets_table

  def learn_weights(self):
    """
    Runs iterations of generating samples with current weights and training new
    weight vectors based on the collected samples.
    What it does depends on self.learn_policy setting.
    """
    # check for file containing the relevant statistics. if it does not exist,
    # collect samples and write it out.
    # NOTE: the filename depends only on the detector type
    None

  def output_det_statistics(self):
    # collect samples and display the statistics of times and naive and
    # actual_ap increases for each class 
    # TODO: increase sample size
    det_configs = {}
    all_samples = self.collect_samples(sample_size=-1)
    if all_samples:
      # construct ndarray of this stuff
      action_inds = reduce(lambda x,y: x+y,
          (samples['action_ind'] for samples in all_samples))
      times = reduce(lambda x,y: x+y,
          (samples['time'] for samples in all_samples))
      naive_aps = reduce(lambda x,y: x+y,
          (samples['naive_ap'] for samples in all_samples))
      actual_aps = reduce(lambda x,y: x+y,
          (samples['actual_ap'] for samples in all_samples))
      img_inds = reduce(lambda x,y: x+y,
          (samples['img_ind'] for samples in all_samples))
      sample_array = np.array((action_inds,times,naive_aps,actual_aps,img_inds)).T
      cols = ['action_ind','time','naive_ap','actual_ap','img_ind']
      table = ut.Table(sample_array,cols)
      # go through actions
      for ind,action in enumerate(self.actions):
        st = table.filter_on_column('action_ind', ind)
        means = np.mean(st.arr[:,1:],0)
        det_configs[self.actions[ind].name] = {}
        # TODO: should depend on image size
        det_configs[self.actions[ind].name]['avg_time'] = means[0]
        det_configs[self.actions[ind].name]['naive_ap'] = means[1]
        det_configs[self.actions[ind].name]['actual_ap'] = means[2]
        if isinstance(action.obj,Detector):
          img_inds = st.subset_arr('img_ind').astype(int)
          cls_ind = action.obj.cls_ind
          d = self.dataset
          presence_inds = np.array([d.images[img_ind].contains_cls_ind(cls_ind) for img_ind in img_inds])
          st_present = np.atleast_2d(st.arr[np.flatnonzero(presence_inds),:])
          if st_present.shape[0]>0:
            means = np.mean(st_present[:,2:],0)
            det_configs[self.actions[ind].name]['naive_ap|present'] = means[0]
            det_configs[self.actions[ind].name]['actual_ap|present'] = means[1]
          st_absent = np.atleast_2d(st.arr[np.flatnonzero(presence_inds==False),:])
          if st_absent.shape[0]>0:
            means = np.mean(st_absent[:,2:],0)
            det_configs[self.actions[ind].name]['naive_ap|absent'] = means[0]
            det_configs[self.actions[ind].name]['actual_ap|absent'] = means[1]
      # NOTE: probably only makes sense when running with one detector
      det_suffix = '-'.join(self.dets_suffixes)
      filename = os.path.join(Config.dets_configs_dir,det_suffix+'.txt')
      json.dumps(det_configs)
      with open(filename,'w') as f:
        json.dump(det_configs,f)
    safebarrier(comm)

  def collect_samples(self, sample_size=200):
    """
    Runs MPI-parallelized sample collection with the current policy on
    the train dataset.
    If sample_size==-1, uses whole dataset.
    """
    sample_images = self.dataset.images
    if not sample_size<0:
      images = self.dataset.images
      rand_ind = np.random.permutation(len(images))
      sample_images = np.take(images, rand_ind[:sample_size])
    all_samples = []
    for i in range(comm_rank,len(sample_images),comm_size):
      dets, samples = self.detect_in_image(sample_images[i],with_samples=True)
      all_samples.append(samples)
    final_samples = None
    if comm_rank==0:
      final_samples = []
    safebarrier(comm)
    final_samples = comm.reduce(all_samples,root=0)
    return final_samples

  def write_out_weights(self, name='default'):
    """Write self.weights out to canonical filename given the name."""
    filename = os.path.join(Config.get_dp_weights_dirname(self), name+'.txt')
    np.savetxt(filename, self.weights, fmt='%.2f') 

  ################
  # Image Policy stuff
  ################
  def update_actions(self):
    """Update the values of actions according to the current belief state."""
    if self.learn_policy=='advanced' or \
        self.learn_policy=='advanced_pair' or \
        self.learn_policy=='slope' or self.learn_policy=='slope_pair':
      for ind,action in enumerate(self.actions):
        if isinstance(action.obj, Detector):
          det = action.obj
          taken_other = 0
          if ind+20 < len(self.actions):
            det_other = self.actions[ind+20].obj
            taken_other = self.taken[ind+20]
          elif ind-20 >= 0:
            det_other = self.actions[ind-20].obj
            taken_other = self.taken[ind-20]
          #print det.cls_ind
          P = self.priors.priors[det.cls_ind]
          time_to_deadline = max(0,self.bounds[1]-self.time_elapsed)
          norm_constant = self.y_to_go * time_to_deadline
          if self.learn_policy=='slope' or self.learn_policy == 'slope_pair':
            val =  P * det.config['naive_ap|present']
            if self.learn_policy=='slope_pair':
              if not taken_other == 0:
                val -= P*det_other.config['naive_ap|present']
            val /= det.config['avg_time']
            val *= time_to_deadline
          else:
            # TODO: in the future, use norm_constant maybe
            #val = time_to_deadline * P * det.config['actual_ap|present']
            #val += time_to_deadline * (1-P) * det.config['actual_ap|absent']
            val = time_to_deadline * P * det.config['naive_ap|present']
            val -= P * det.config['naive_ap|present'] * det.config['avg_time']
            #val -= (1-P) * det.config['actual_ap|absent'] * det.config['avg_time']
            if self.learn_policy=='advanced_pair':
              if not taken_other == 0:
                assert(det_other.cls_ind == det.cls_ind)
                val -= P*det_other.config['actual_ap|present']*time_to_deadline
                val -= (1-P)*det_other.config['actual_ap|absent']*time_to_deadline
                val += P * det_other.config['actual_ap|present'] * det.config['avg_time']
                val += (1-P) * det_other.config['actual_ap|absent'] * det.config['avg_time']
                val += P * det_other.config['actual_ap|present'] * det.config['avg_time']
            #val /= norm_constant
          self.values[ind] = val
          self.ap_estimates[ind] = P*det.config['naive_ap|present']
          if self.learn_policy=='advanced_pair':
            if not taken_other == 0:
              self.ap_estimates[ind] -= P * det_other.config['actual_ap|present']
              self.ap_estimates[ind] -= (1-P) * det_other.config['actual_ap|absent']
          if self.learn_policy == 'slope_pair':
            if not taken_other == 0:
              self.ap_estimates[ind] -= P*det_other.config['naive_ap|present']
        else:
          self.values[ind] = np.random.rand() 
          self.ap_estimates[ind] = 0
    else:
      self.values = np.dot(self.weights,self.get_feature_vec())

  def reset_actions(self):
    self.taken = np.zeros(len(self.actions)) 
    self.values = np.zeros(len(self.actions)) 
    self.ap_estimates = np.zeros(len(self.actions))
    self.y_to_go = 1

  def pick_next_action_ind(self):
    """Return the index of the next action to take."""
    if np.all(self.taken):
      return -1
    untaken_inds = np.flatnonzero(self.taken==0)
    vals = self.values[untaken_inds]
    max_untaken_ind = vals.argmax()
    ap_estimate = self.ap_estimates[max_untaken_ind]
    self.y_to_go = max(0, self.y_to_go-ap_estimate)
    return untaken_inds[max_untaken_ind]

  def get_feature_vec(self):
    """
    Return featurized representation of self.
    Feature representation is [P(C_1), ..., P(C_K), 1].
    """
    features = self.priors.priors
    
    # TODO: add in the entropy feature
    def mylog(x): return np.log2(x) if not x==0 else 0
    def H(x): return np.sum([-x_i*mylog(x_i) -(1-x_i)*mylog(1-x_i) for x_i in x])
    entropy = H(self.priors.priors)
    #features += [entropy]

    # TODO: add in the time_to_go features
    time_to_start = 0
    if self.bounds:
      if self.bounds[0]>0:
        time_to_start = max(0, (self.bounds[0]-self.time_elapsed)/self.bounds[0])
      # TODO: should time_to_deadline be time between deadline and 0 or deadline
      # and start_time?
      time_to_deadline = max(0, (self.bounds[1]-self.time_elapsed)/self.bounds[1])
    else:
      time_to_start = 0
      time_to_deadline = 1
    #features += [time_to_start,time_to_deadline]
    #features += [time_to_deadline]

    return np.array(features)

  def detect_in_image(self, image, with_samples=False):
    """
    Return list of detections in the image, with each row as self.dets_cols.
    This is what runs the policy.
    If with_samples=True, also return <s,a,r,s',r_n,time> samples.
    """
    gt = image.get_ground_truth(include_diff=True)
    actual_time_start = time.time()
    img_ind = self.dataset.get_img_ind(image)
    
    # reset the state
    # TODO: separate NGramModel from ClassPriors, to maintain the benefit of
    # caching answers
    self.priors = ClassPriors(self.train_dataset,mode=self.class_priors_mode)
    self.time_elapsed = 0
    self.reset_actions()
    self.update_actions()

    action_sequence = []
    all_detections = []

    samples = {} 
    # a sample collects
    samples['state'] = []           # the belief state feature vector
    samples['action_ind'] = []      # action index
    samples['naive_ap'] = []        # resulting naive AP increase      
    samples['actual_ap'] = []       # resulting non-naive AP increase
    samples['time'] = []            # time taken by the detector
    samples['next_state'] = []      # next belief state feature vector
    samples['next_action_ind'] = [] # next action index
    samples['img_ind'] = []         # sort of stupid, but I need this and this
    # is easiest right now
    
    # If gist mode is on, the first action is gist :-)
    if self.gist:
      next_action_ind = 0
    else:
      next_action_ind = self.pick_next_action_ind()

    prev_ap = None
    while True:
      action = self.actions[next_action_ind]
      action_sequence.append(action.name)
      self.taken[next_action_ind] = 1

      samples['state'].append(self.get_feature_vec())
      samples['action_ind'].append(next_action_ind)

      if isinstance(action.obj, Detector):
        # detector actions
        det = action.obj
        cls = det.cls
        cls_ind = self.dataset.classes.index(cls)
        detections, time_e = det.detect(image)
        self.time_elapsed += time_e
        samples['time'].append(time_e)

        if detections.shape[0]>0:
          c_vector = np.tile(cls_ind,(np.shape(detections)[0],1)) 
          i_vector = np.tile(img_ind,(np.shape(detections)[0],1))
          detections = np.hstack((detections, c_vector, i_vector))          
        else:
          detections = np.array([])
        dets_table = ut.Table(detections,det.get_cols()+['cls_ind','img_ind'])

        # compute the greedy AP increase
        ap,rec,prec = self.ev.compute_pr(dets_table,gt)
        samples['naive_ap'].append(ap)

        all_detections.append(detections)
        nonempty_dets = [dets for dets in all_detections if dets.shape[0]>0]
        all_dets_table = ut.Table(np.array([]),dets_table.cols)
        if len(nonempty_dets)>0:
          all_dets_table = ut.Table(np.concatenate(nonempty_dets,0),dets_table.cols)
        # and the actual AP increase
        ap,rec,prec = self.ev.compute_pr(all_dets_table,gt)
        ap_diff = ap
        if prev_ap:
          ap_diff = ap-prev_ap
        prev_ap = ap
        samples['actual_ap'].append(ap_diff)

        # compute the posterior given these detections, and update the class priors
        posterior = det.compute_posterior(image, dets_table, oracle=False)
        self.priors.update_with_posterior(cls_ind, posterior)
      else: 
        # gist scene context action
        gist_obj = action.obj
        gist_priors = gist_obj.get_priors_lam(image, self.priors.priors)
        self.priors.update_with_gist(gist_priors)
        time_gist = 1
        self.time_elapsed += time_gist
        samples['time'].append(time_gist)
        # does not lead to any detections
        samples['naive_ap'].append(0)
        samples['actual_ap'].append(0)

      # pick the next action, having updated the priors
      self.update_actions()
      next_action_ind = self.pick_next_action_ind()
      samples['next_state'].append(self.get_feature_vec())
      samples['next_action_ind'].append(next_action_ind)
      samples['img_ind'].append(img_ind)

      #print("vals: %s"%self.values)
      #print("ap_estimates: %s"%self.ap_estimates)
      #print("y_to_go: %s"%self.y_to_go)
      #print("time_elapsed: %s"%self.time_elapsed)

      # check for stopping conditions
      if next_action_ind < 0:
        break
      if self.bounds and not self.class_priors_mode=='oracle':
        if self.time_elapsed > self.bounds[1]:
          break

    # assert that we got the sample number of samples for all dimensions
    lens = [len(samples[key]) for key in samples.keys()]
    assert(len(np.unique(lens))==1)

    # in case of 'oracle' mode, re-sort the detections and times in order of AP
    # contributions
    times = samples['time']
    if self.class_priors_mode=='oracle':
      naive_aps = np.array(samples['naive_ap'])
      sorted_inds = np.argsort(-naive_aps)
      all_detections = np.take(all_detections, sorted_inds)
      times = np.take(times, sorted_inds)

    # now construct the final return array, with correct times
    assert(len(all_detections)==len(times))
    cum_times = np.cumsum(times)
    all_times = []
    all_nonempty_detections = []
    for i,dets in enumerate(all_detections):
      num_dets = dets.shape[0]
      if num_dets > 0:
        all_nonempty_detections.append(dets)
        t_vector = np.tile(cum_times[i],(num_dets,1)) 
        all_times.append(t_vector)
    if len(all_nonempty_detections)>0:
      all_detections = np.concatenate(all_nonempty_detections,0)
      all_times = np.concatenate(all_times,0)
      # appending 'time' column at end, as promised
      all_detections = np.hstack((all_detections,all_times))
      # we probably went over deadline with the oracle mode, so trim it down
      if self.bounds:
        if np.max(all_times)>self.bounds[1]:
          first_overdeadline_ind = np.flatnonzero(all_times>self.bounds[1])[0]
          all_detections = all_detections[:first_overdeadline_ind,:]
    else:
      all_detections = np.array([])

    print("Done with image, took %.3f s"%(time.time()-actual_time_start))
    if with_samples:
      return (all_detections, samples)
    return all_detections

  ###############
  # External detections stuff
  ###############
  def load_ext_detections(self,dataset,mode,suffix,force=False):
    """
    Loads multi-image, multi-class array of detections for all images in the
    given dataset.
    Loads from canonical cache location.
    """
    t = time.time()
    filename = Config.get_ext_dets_filename(dataset, suffix)
    # check for cached full file
    if os.path.exists(filename) and not force:
      all_dets_table = np.load(filename)[()]
    else:
      # TODO also return times, or process them to add to dets?
      all_dets = []
      for i in range(comm_rank,len(dataset.images),comm_size):
        image = dataset.images[i]
        if mode=='dpm':
          # NOTE: not actually using the given suffix in the call below
          dets = self.load_dpm_dets_for_image(image)
          ind_vector = np.ones((np.shape(dets)[0],1)) * i
          dets = np.hstack((dets,ind_vector))
          cols = ['x','y','w','h','dummy','dummy','dummy','dummy','score','time','cls_ind','img_ind']
          good_ind = [0,1,2,3,8,9,10,11]
          dets = dets[:,good_ind]
        elif mode=='csc':
          # NOTE: not actually using the given suffix in the call below
            dets = self.load_csc_dpm_dets_for_image(image)
            ind_vector = np.ones((np.shape(dets)[0],1)) * i
            dets = np.hstack((dets,ind_vector))
        elif mode=='ctf':
          # Split the suffix into ctf and the main part
          actual_suffix = suffix.split('_')[1]
          dets = self.load_ctf_dets_for_image(image, actual_suffix)
          ind_vector = np.ones((np.shape(dets)[0],1)) * i
          dets = np.hstack((dets,ind_vector))
        else:
          raise RuntimeError('Unknown detections mode')
        all_dets.append(dets)
      final_dets = None
      if comm_rank==0:
        final_dets = []
      safebarrier(comm)
      final_dets = comm.reduce(all_dets,op=MPI.SUM,root=0)
      all_dets_table = None
      if comm_rank == 0:
        all_dets_table = ut.Table()
        all_dets_table.name = suffix
        all_dets_table.cols = ['x', 'y', 'w', 'h', 'score', 'time', 'cls_ind', 'img_ind']
        all_dets_table.arr = np.vstack(final_dets)
        np.save(filename,all_dets_table)
        print("Found %d dets"%all_dets_table.shape()[0])
      safebarrier(comm)
      all_dets_table = comm.bcast(all_dets_table,root=0)
    time_elapsed = time.time()-t
    print("DatasetPolicy.load_ext_detections took %.3f"%time_elapsed)
    return all_dets_table

  def load_ctf_dets_for_image(self, image, suffix='default'):
    """Load multi-class array of detections for this image."""
    t = time.time()
    dirname = '/u/vis/x1/sergeyk/object_detection/ctfdets/%s/'%suffix
    time_elapsed = time.time()-t
    filename = os.path.join(dirname, image.name+'.npy')
    dets_table = np.load(filename)[()]
    print("On image %s, took %.3f s"%(image.name, time_elapsed))
    return dets_table.arr

  def load_csc_dpm_dets_for_image(self, image):
    """
    Loads HOS's cascaded dets.
    """
    t = time.time()
    name = os.path.splitext(image.name)[0]
    # if test dataset, use HOS's detections. if not, need to output my own
    if re.search('test', self.dataset.name):
      dirname = '/u/vis/song/rl_detection/datadir/voc-release4/2007/tmp/dets_test_original_cascade_wholeset/'
      filename = dirname+'%s_dets_all_test_original_cascade_wholeset.mat'%name
    else:
      dirname = '/u/vis/x1/sergeyk/rl_detection/voc-release4/2007/tmp/dets_nov19/'
      filename = dirname+'%s_dets_all_nov19.mat'%name
    if not os.path.exists(filename):
      print("File does not exist!")
      return None
    mat = scipy.io.loadmat(filename)
    dets = mat['dets_mc']
    times = mat['times_mc']
    feat_time = times[0,0]
    dets_seq = []
    cols = ['x1','y1','x2','y2','dummy','dummy','dummy','dummy','dummy','dummy','score'] 
    for cls_ind,cls in enumerate(self.dataset.classes):
      cls_dets = dets[cls_ind][0]
      if cls_dets.shape[0]>0:
        good_ind = [0,1,2,3,10]
        cls_dets = cls_dets[:,good_ind]
        det_time = times[cls_ind,1]
        # all detections get the final time
        cls_dets = ut.append_index_column(cls_dets, det_time)
        cls_dets = ut.append_index_column(cls_dets, cls_ind)
        # convert from corners!
        cls_dets[:,:4] = BoundingBox.convert_arr_from_corners(cls_dets[:,:4])
        cls_dets[:,:4] = BoundingBox.clipboxes_arr(cls_dets[:,:4], (0,0,image.size[0],image.size[1]))
        dets_seq.append(cls_dets)
    cols = ['x','y','w','h','score','time','cls_ind'] 
    dets_mc = ut.collect(dets_seq, Detector.nms_detections, cols)
    time_elapsed = time.time()-t
    print("On image %s, took %.3f s"%(image.name, time_elapsed))
    return dets_mc

  def load_dpm_dets_for_image(self, image, suffix='dets_all_may25_DP'):
    """
    Loads multi-class array of detections for an image from .mat format.
    self.suffix supercedes given suffix if present.
    """
    t = time.time()
    name = os.path.splitext(image.name)[0]
    # TODO: figure out how to deal with different types of detections
    filename = os.path.join('/u/vis/x1/sergeyk/rl_detection/voc-release4/2007/tmp/dets_may25_DP/%(name)s_dets_all_may25_DP.mat'%{'name': name})
    if not os.path.exists(filename):
      filename = os.path.join('/u/vis/x1/sergeyk/rl_detection/voc-release4/2007/tmp/dets_jun1_DP_trainval/%(name)s_dets_all_jun1_DP_trainval.mat'%{'name': name})
      if not os.path.exists(filename):
        filename = os.path.join(Config.test_support_dir,'dets/%s_dets_all_may25_DP.mat'%name)
        if not os.path.exists(filename):
          print("File does not exist!")
          return None
    mat = scipy.io.loadmat(filename)
    dets = mat['dets_mc']
    times = mat['times_mc']
    feat_time = times[0,0]
    dets_seq = []
    cols = ['x1','y1','x2','y2','dummy','dummy','dummy','dummy','score','time'] 
    for cls_ind,cls in enumerate(Config.pascal_classes):
      cls_dets = dets[cls_ind][0]
      if cls_dets.shape[0]>0:
        det_time = times[cls_ind,1]
        # all detections get the final time
        cls_dets = ut.append_index_column(cls_dets, det_time)
        cls_dets = ut.append_index_column(cls_dets, cls_ind)
        # convert from corners!
        cls_dets[:,0:4] = BoundingBox.convert_arr_from_corners(cls_dets[:,0:4])
        cls_dets[:,:4] = BoundingBox.clipboxes_arr(cls_dets[:,:4],(0,0,image.size[0],image.size[1]))
        dets_seq.append(cls_dets)
    cols = ['x','y','w','h','dummy','dummy','dummy','dummy','score','time','cls_ind'] 
    dets_mc = ut.collect(dets_seq, Detector.nms_detections, cols)
    #dets_mc[:,:4] = BoundingBox.clipboxes_arr(dets_mc[:,:4],(0,0,image.size[0],image.size[1]))
    time_elapsed = time.time()-t
    print("On image %s, took %.3f s"%(image.name, time_elapsed))
    return dets_mc

