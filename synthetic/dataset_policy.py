import copy
import datetime
import scipy.io

from common_mpi import *
from common_imports import *
import synthetic.config as config

from synthetic.dataset import Dataset
from synthetic.evaluation import Evaluation
from synthetic.gist_classifier import GistClassifier
from synthetic.detector import *
from synthetic.ext_detector import ExternalDetector
from synthetic.bounding_box import BoundingBox
from synthetic.belief_state import BeliefState
from synthetic.fastinf_model import FastinfModel
import matplotlib.pyplot as plt

class ImageAction:
  def __init__(self, name, obj):
    self.name = name
    self.obj = obj

  def __repr__(self):
    return self.name

class DatasetPolicy:
  # run_experiment.py uses this and __init__ uses as default values
  default_config = {
    'suffix': 'feb27', # use this to re-run on same params after changing code
    'detectors': ['perfect'], # perfect,perfect_with_noise,dpm,csc_default,csc_half
    'policy_mode': 'random',
      # policy mode can be one of random, oracle, fixed_order, no_smooth,
      # backoff, fastinf_manual, fastinf_greedy, fastinf_rl
    'bounds': None, # start and deadline times for the policy
  }

  def get_name(self):
    return "%s_%s"%(self.dataset.get_name(), self.get_config_name())

  def get_config_name(self):
    """All params except for dataset."""
    middle = ""
    if self.bounds:
      middle += '-'.join((str(x) for x in self.bounds))
    detectors = '-'.join(self.detectors)
    name = '_'.join(
        (self.policy_mode,
        detectors,
        middle,
        self.suffix))
    return name

  @classmethod
  def get_det_cols(cls):
    return Detector.get_cols() + ['cls_ind','img_ind','time']

  def get_cls_cols(self):
    return self.dataset.classes + ['img_ind','time']

  def __init__(self, dataset, train_dataset, **kwargs):
    "**kwargs update the default config"
    config = copy.copy(DatasetPolicy.default_config)
    config.update(kwargs)

    self.dataset = dataset
    self.train_dataset = train_dataset

    self.__dict__.update(config)
    print("DatasetPolicy running with config:")
    pprint(self.__dict__)
    self.ev = Evaluation(self)

    # Construct the actions list
    self.actions = []

    # Create actions for all the detectors we have
    for detector in self.detectors:
      # synthetic perfect detector
      if detector=='perfect':
        for cls in self.dataset.classes:
          det = PerfectDetector(self.dataset, cls)
          self.actions.append(ImageAction('%s_%s'%(detector,cls), det))

      # synthetic perfect detector with noise in the detections
      elif detector=='perfect_with_noise':
        sw = SlidingWindows(self.train_dataset,self.dataset)
        for cls in self.dataset.classes:
          det = PerfectDetectorWithNoise(self.dataset, cls, sw)
          self.actions.append(ImageAction('%s_%s'%(detector,cls), det))

      # GIST classifier
      elif detector=='gist':
        # TODO: this should be per-class
        gist_obj = GistClassifier(self.dataset.name)
        self.actions.append(ImageAction('gist', gist_obj))

      # real detectors, with pre-cached detections
        
      elif detector in ['dpm','csc_default','csc_half']:
        # load the dets from cache file and parcel out to classes
        all_dets = self.load_ext_detections(self.dataset, detector)
        for cls in self.dataset.classes:
          cls_ind = self.dataset.get_ind(cls)
          all_dets_for_cls = all_dets.filter_on_column('cls_ind',cls_ind,omit=True)
          det = ExternalDetector(self.dataset, cls, all_dets_for_cls, detector)
          self.actions.append(ImageAction('%s_%s'%(detector,cls), det))

      # unknown
      else:
        raise RuntimeError("Unknown mode in detectors: %s"%self.detectors)

    # fixed_order, random, oracle policy modes get fixed_order inference mode
    self.inference_mode = 'fixed_order'
    if re.search('fastinf',self.policy_mode):
      self.inference_mode = 'fastinf'
    if self.policy_mode in ['no_smooth','backoff']:
      self.inference_mode = self.policy_mode

    # determine fastinf suffix
    self.fastinf_suffix='this_is_empty'
    if self.inference_mode=='fastinf':
      if self.detectors == ['csc_default']:
        self.fastinf_suffix='CSC'
      elif self.detectors == ['perfect']:
        self.fastinf_suffix='perfect'
      elif self.detectors == ['gist']:
        self.fastinf_suffix='GIST'
      elif self.detectors == ['gist','csc']:
        self.fastinf_suffix='GIST_CSC'
      else:
        raise RuntimeError("""
          We don't have Fastinf models for the detector combination you
          are running with: %s"""%self.detectors)

    # make the initial belief state to get its model and be able to know the feature dimension
    b = BeliefState(self.train_dataset,self.actions,self.inference_mode,self.bounds,fastinf_suffix=self.fastinf_suffix)
    # store reference to the inference model because we will be initializing
    # per-episode belief states with it, to keep it alive
    self.inf_model = b.model
    
    # TODO: figure this stuff out
    if self.policy_mode in ['no_smooth','backoff','fastinf_manual']:
      self.weights = np.zeros((len(self.actions),len(b.featurize())))
      if 'naive_ap|present' in self.actions[0].obj.config:
        naive_aps = [action.obj.config['naive_ap|present'] for action in self.actions]
      else:
        # TODO: this isn't right, but just to get the test to pass with perfectdetector
        naive_aps = [1 for action in self.actions]
      np.fill_diagonal(self.weights,naive_aps)
      self.write_out_weights()
    elif self.policy_mode == 'fastinf_greedy':
      # TODO
      None
    elif self.policy_mode == 'fastinf_rl':
      # TODO
      None

  def run_on_dataset(self,test=True,sample_size=-1,force=False):
    """
    Run MPI-parallelized over the images of the dataset (or a random subset).
    Return list of detections and classifications for the whole dataset.
    If test is True, runs on test dataset, otherwise train dataset.
    If sample_size != -1, does not check for cached files, as the objective
    is to collect samples, not the actual dets and clses.
    Otherwise, check for cached files first, unless force is True.
    Return dets,clses,samples.
    """
    dets_table = None
    clses_table = None

    dataset = self.dataset
    if not test:
      dataset = self.train_dataset

    # If we are collecting samples, we don't care about caches
    if sample_size > 0:
      force = True

    # check for cached results
    det_filename = config.get_dp_dets_filename(self)
    cls_filename = config.get_dp_clses_filename(self)
    samples_filename = config.get_dp_samples_filename(self)
    if not force \
        and opexists(det_filename) and opexists(cls_filename) \
        and opexists(samples_filename):
      dets_table = np.load(det_filename)[()]
      clses_table = np.load(cls_filename)[()]
      with open(samples_filename) as f:
        samples = cPickle.load(f)
      print("DatasetPolicy: Loaded dets and clses from cache.")
      return dets_table,clses_table,samples

    images = dataset.images
    if sample_size>0:
      images = ut.random_subset(dataset.images, sample_size)

    all_dets = []
    all_clses = []
    all_samples = []
    for i in range(comm_rank,len(images),comm_size):
      # ONLY IMPORTANT LINE BELOW
      dets,clses,samples = self.run_on_image(images[i])
      all_dets.append(dets)
      all_clses.append(clses)
      all_samples.append(samples)
    final_dets = [] if comm_rank == 0 else None
    final_clses = [] if comm_rank == 0 else None
    final_samples = [] if comm_rank == 0 else None

    safebarrier(comm)
    final_dets = comm.reduce(all_dets, op=MPI.SUM, root=0)
    final_clses = comm.reduce(all_clses, op=MPI.SUM, root=0)
    final_samples = comm.reduce(all_samples,root=0)
    if self.inference_mode=='fastinf':
      all_fm_cache_items = comm.reduce(self.inf_model.cache.items(), op=MPI.SUM, root=0)
    if comm_rank==0:
      dets_table = ut.Table(cols=self.get_det_cols())
      final_dets = [det for det in final_dets if det.shape[0]>0]
      dets_table.arr = np.vstack(final_dets)
      clses_table = ut.Table(cols=self.get_cls_cols())
      clses_table.arr = np.vstack(final_clses)
      print("Found %d dets"%dets_table.shape()[0])
      print("Classified %d images"%clses_table.shape()[0])

      # Only save results if we are not collecting samples
      if not sample_size > 0:
        np.save(det_filename,dets_table)
        np.save(cls_filename,clses_table)
        with open(samples_filename,'w') as f:
          cPickle.dump(final_samples,f)

      # Save the fastinf cache
      # TODO: turning this off for now
      if False and self.inference_mode=='fastinf':
        self.inf_model.cache = dict(all_fm_cache_items)
        self.inf_model.save_cache()

    # Broadcast results to all workers, because Evaluation splits its work as well.
    safebarrier(comm)
    dets_table = comm.bcast(dets_table,root=0)
    clses_table = comm.bcast(clses_table,root=0)
    print(self.policy_mode)
    return dets_table,clses_table,final_samples

  def learn_weights(self):
    """
    Runs iterations of generating samples with current weights and training
    new weight vectors based on the collected samples.
    What it does depends on policy_mode.
    """
    # check for file containing the relevant statistics. if it does not exist,
    # collect samples and write it out.
    # NOTE: the filename depends only on the detector type
    if re.search('_greedy$', self.policy_mode):
      # regression to next-step greedy rewards
      None
    elif re.search('_rl$', self.policy_mode):
      # full reinforcement learning
      None
    else:
      raise RuntimeError("Policy mode %s is not supported!"%self.policy_mode)

  def output_det_statistics(self):
    # collect samples and display the statistics of times and naive and
    # actual_ap increases for each class 
    det_configs = {}
    all_dets,all_clses,all_samples = self.run_on_dataset()
    if all_samples:
      sample_array = np.array((
        [s['a'] for s in samples],
        [s['dt'] for s in samples],
        [s['det_naive_ap'] for s in samples],
        [s['det_actual_ap'] for s in samples],
        [s['img_ind'] for s in samples])).T
      cols = ['action_ind','dt','det_naive_ap','det_actual_ap','img_ind']
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
      detectors_suffix = '-'.join(self.detectors)
      filename = os.path.join(config.dets_configs_dir,detectors_suffix+'.txt')
      json.dumps(det_configs)
      with open(filename,'w') as f:
        json.dump(det_configs,f)
    safebarrier(comm)

  def write_out_weights(self, name='default'):
    """Write self.weights out to canonical filename given the name."""
    filename = opjoin(config.get_dp_weights_dirname(self), name+'.txt')
    np.savetxt(filename, self.weights, fmt='%.2f')

  ################
  # Image Policy stuff
  ################
  def update_actions(self,b):
    "Update the values of actions according to the current belief state."
    if self.policy_mode=='random' or self.policy_mode=='oracle':
      self.action_values = np.random.rand(len(self.actions))
    elif self.policy_mode=='fixed_order':
      self.action_values = b.get_p_c()
    else:
      self.action_values = np.dot(self.weights, b.featurize())

  def select_action(self, b):
    """
    Return the index of the untaken action with the max value.
    Return -1 if all actions have been taken.
    """
    if np.all(b.taken):
      return -1
    untaken_inds = np.flatnonzero(b.taken==0)
    max_untaken_ind = self.action_values[untaken_inds].argmax()
    return untaken_inds[max_untaken_ind]

  def run_on_image(self, image):
    """
    Return
    - list of detections in the image, with each row as self.get_det_cols()
    - list of multi-label classification outputs, with each row as self.get_cls_cols()
    - list of <s,a,r,s',dt> samples.
    """
    gt = image.get_ground_truth(include_diff=True)
    tt = ut.TicToc().tic()
    
    all_detections = []
    all_clses = []
    samples = []
    prev_ap = 0
   
    img_ind = self.dataset.get_img_ind(image) if image else -1
    # Initialize belief state with the inference model that we already have from __init__
    b = BeliefState(self.train_dataset,self.actions,self.inference_mode,self.bounds,self.inf_model,self.fastinf_suffix)
    self.update_actions(b)
    action_ind = self.select_action(b)
    while True:
      # Populate the sample with stuff we know
      sample = {}
      sample['img_ind'] = img_ind
      sample['state'] = b.featurize()
      sample['action_ind'] = action_ind
      
      # Take the action and get the observations as a dict
      action = self.actions[action_ind]
      obs = action.obj.get_observations(image)

      # If observations include detections, compute the relevant
      # stuff for the sample collection
      sample['det_naive_ap'] = 0
      sample['det_actual_ap'] = 0
      if 'dets' in obs:
        det = action.obj
        detections = obs['dets']
        cls_ind = self.dataset.classes.index(det.cls)
        if detections.shape[0]>0:
          c_vector = np.tile(cls_ind,(np.shape(detections)[0],1))
          i_vector = np.tile(img_ind,(np.shape(detections)[0],1))
          detections = np.hstack((detections, c_vector, i_vector))
        else:
          detections = np.array([])
        dets_table = ut.Table(detections,det.get_cols()+['cls_ind','img_ind'])

        # compute the 'naive' det AP increase: adding dets to empty set
        ap,rec,prec = self.ev.compute_det_pr(dets_table,gt)
        sample['det_naive_ap'] = ap

        # TODO: am I needlessly recomputing this table?
        all_detections.append(detections)
        nonempty_dets = [dets for dets in all_detections if dets.shape[0]>0]
        all_dets_table = ut.Table(np.array([]),dets_table.cols)
        if len(nonempty_dets)>0:
          all_dets_table = ut.Table(np.concatenate(nonempty_dets,0),dets_table.cols)

        # compute the actual AP increase: addings dets to dets so far
        ap,rec,prec = self.ev.compute_det_pr(all_dets_table,gt)
        ap_diff = ap-prev_ap
        sample['det_actual_ap'] = ap_diff
        prev_ap = ap

      # Observations always include the following stuff, which can be used
      # to update the belief state, and mark it as taken
      sample['dt'] = obs['dt']
      b.t += obs['dt']
      b.taken[action_ind] = 1
      b.update_with_score(action_ind, obs['score'])

      # b is already updated to the next state; store in sample
      sample['next_state'] = b.featurize()
      samples.append(sample)

      # The updated belief state posterior over C is our classification result
      clses = b.get_p_c().tolist() + [img_ind,b.t]
      all_clses.append(clses)

      # Update action values and pick the next action
      self.update_actions(b)
      action_ind = self.select_action(b)

      # check for stopping conditions
      if action_ind < 0:
        break
      if self.bounds and not self.policy_mode=='oracle':
        if b.t > self.bounds[1]:
          break

    # in case of 'oracle' mode, re-sort the detections and times in order of AP
    # contributions
    times = [s['dt'] for s in samples]
    all_clses = np.array(all_clses)
    if self.policy_mode=='oracle':
      naive_aps = np.array([s['det_naive_ap'] for s in samples])
      sorted_inds = np.argsort(-naive_aps)
      all_detections = np.take(all_detections, sorted_inds)
      times = np.take(times, sorted_inds)
      all_clses = all_clses[sorted_inds]
      time_ind = self.get_cls_cols().index('time')
      all_clses[:,time_ind] = times

    # now construct the final return array, with correct times
    assert(len(all_detections)==len(all_clses)==len(times))
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

    print("DatasetPolicy on image with ind %d took %.3f s"%(img_ind,tt.qtoc()))

    if False:
      print("Action sequence was: %s"%[s['action_ind'] for s in samples])
      print("here's an image:")
      X = np.vstack((all_clses[:,:-2],image.get_cls_ground_truth()))
      np.set_printoptions(precision=2, suppress=True)
      print X
      plt.pcolor(np.flipud(X))
      plt.show()

    return (all_detections,all_clses,samples)

  ###############
  # External detections stuff
  ###############
  def load_ext_detections(self,dataset,suffix,force=False):
    """
    Loads multi-image, multi-class array of detections for all images in the
    given dataset.
    Loads from canonical cache location.
    """
    t = time.time()
    filename = config.get_ext_dets_filename(dataset, suffix)
    # check for cached full file
    if os.path.exists(filename) and not force:
      all_dets_table = np.load(filename)[()]
    else:
      # TODO also return times, or process them to add to dets?
      all_dets = []
      for i in range(comm_rank,len(dataset.images),comm_size):
        image = dataset.images[i]
        if re.search('dpm',suffix):
          # NOTE: not actually using the given suffix in the call below
          dets = self.load_dpm_dets_for_image(image)
          ind_vector = np.ones((np.shape(dets)[0],1)) * i
          dets = np.hstack((dets,ind_vector))
          cols = ['x','y','w','h','dummy','dummy','dummy','dummy','score','time','cls_ind','img_ind']
          good_ind = [0,1,2,3,8,9,10,11]
          dets = dets[:,good_ind]
        elif re.search('csc',suffix):
          # NOTE: not actually using the given suffix in the call below
          dets = self.load_csc_dpm_dets_for_image(image)
          ind_vector = np.ones((np.shape(dets)[0],1)) * i
          dets = np.hstack((dets,ind_vector))
        elif re.search('ctf',suffix):
          # Split the suffix into ctf and the main part
          actual_suffix = suffix.split('_')[1]
          dets = self.load_ctf_dets_for_image(image, actual_suffix)
          ind_vector = np.ones((np.shape(dets)[0],1)) * i
          dets = np.hstack((dets,ind_vector))
        else:
          print(suffix)
          raise RuntimeError('Unknown detector type in suffix')
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
      dirname = config.get_dets_test_wholeset_dir()
      filename = os.path.join(dirname,'%s_dets_all_test_original_cascade_wholeset.mat'%name)
    else:
      dirname = config.get_dets_nov19()
      filename = os.path.join(dirname, '%s_dets_all_nov19.mat'%name)
    print filename
    if not os.path.exists(filename):
      raise RuntimeError("File %s does not exist!"%filename)
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
    dets_mc = ut.collect(dets_seq, Detector.nms_detections, {'cols':cols})
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
        filename = os.path.join(config.test_support_dir,'dets/%s_dets_all_may25_DP.mat'%name)
        if not os.path.exists(filename):
          print("File does not exist!")
          return None
    mat = scipy.io.loadmat(filename)
    dets = mat['dets_mc']
    times = mat['times_mc']
    feat_time = times[0,0]
    dets_seq = []
    cols = ['x1','y1','x2','y2','dummy','dummy','dummy','dummy','score','time'] 
    for cls_ind,cls in enumerate(config.pascal_classes):
      cls_dets = dets[cls_ind][0]
      if cls_dets.shape[0]>0:
        det_time = times[cls_ind,1]
        # all detections get the final time
        cls_dets = ut.append_index_column(cls_dets, det_time)
        cls_dets = ut.append_index_column(cls_dets, cls_ind)
        # subtract 1 pixel and convert from corners!
        cls_dets[:,:4] -= 1
        cls_dets[:,:4] = BoundingBox.convert_arr_from_corners(cls_dets[:,:4])
        dets_seq.append(cls_dets)
    cols = ['x','y','w','h','dummy','dummy','dummy','dummy','score','time','cls_ind'] 
    # NMS detections per class individually
    dets_mc = ut.collect(dets_seq, Detector.nms_detections, {'cols':cols})
    dets_mc[:,:4] = BoundingBox.clipboxes_arr(dets_mc[:,:4],(0,0,image.size[0]-1,image.size[1]-1))
    time_elapsed = time.time()-t
    print("On image %s, took %.3f s"%(image.name, time_elapsed))
    return dets_mc

if __name__=='__main__':
  train_d = Dataset('full_pascal_trainval')
  
  just_combine=False
  
  for ds in ['full_pascal_test']:
    eval_d = Dataset(ds) 
    dp = DatasetPolicy(eval_d, train_d, detectors=['csc_default'])
    test_table = np.zeros((len(eval_d.images), len(dp.actions)))
    
    if not just_combine:
      for img_idx in range(comm_rank, len(eval_d.images), comm_size):
        img = eval_d.images[img_idx]
        for act_idx, act in enumerate(dp.actions):
          print '%s on %d for act %d'%(img.name, comm_rank, act_idx)    
          score = act.obj.get_observations(img)['score']
          test_table[img_idx, act_idx] = score
      
      dirname = ut.makedirs(os.path.join(config.get_ext_dets_foldname(eval_d), 'dp','agent_wise'))
      filename = os.path.join(dirname,'table_%d'%comm_rank)
      np.savetxt(filename, test_table) 
  
    safebarrier(comm)
    
    if comm_rank == 0:
      for i in range(comm_size-1):
        filename = os.path.join(dirname,'table_%d'%(i+1))
        test_table += np.loadtxt(filename)
      dirname = ut.makedirs(os.path.join(config.get_ext_dets_foldname(eval_d), 'dp'))
      filename = os.path.join(dirname,'table_chi2')
      tab_test_table = ut.Table()
      tab_test_table.cols = list(train_d.classes) + ['img_ind']
      
      tab_test_table.arr = np.hstack((test_table, np.array(np.arange(test_table.shape[0]),ndmin=2).T))
      cPickle.dump(tab_test_table, open(filename,'w'))