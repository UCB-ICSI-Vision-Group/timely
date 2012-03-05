import datetime
import scipy.io
import sklearn

from common_mpi import *
from common_imports import *
import synthetic.config as config

from synthetic.dataset import Dataset
from synthetic.sample import Sample
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
    'suffix': 'default', # use this to re-run on same params after changing code
    'detectors': ['perfect'], # perfect,perfect_with_noise,dpm,csc_default,csc_half
    'policy_mode': 'random',
      # policy mode can be one of random, oracle, fixed_order,
      # no_smooth, backoff, fastinf
    'bounds': [0,20], # start and deadline times for the policy
    'weights_mode': 'manual_1',
    # manual_1, manual_2, manual_3, greedy, rl_regression, rl_lspi
    'rewards_mode': 'det_actual_ap'
    # det_actual_ap, entropy, auc_ap, auc_entropy
  }

  def get_config_name(self):
    "All params except for dataset."
    bounds = ''
    if self.bounds:
      bounds = '-'.join((str(x) for x in self.bounds))
    detectors = '-'.join(self.detectors)
    name = '_'.join(
        [self.policy_mode,
        detectors,
        bounds,
        self.weights_mode,
        self.rewards_mode,
        self.suffix])
    return name

  @classmethod
  def get_det_cols(cls):
    return Detector.get_cols() + ['cls_ind','img_ind','time']

  def get_cls_cols(self):
    return self.dataset.classes + ['img_ind','time']

  def __init__(self, test_dataset, train_dataset, weights_dataset_name=None, **kwargs):
    """
    Initialize the DatasetPolicy, getting it ready to run on the whole dataset
    or a single image.
    - test, train, weights datasets should be:
      - val,  train,    val:    for training the weights
      - test, trainval, val:    for final run
    - **kwargs update the default config
    """
    config = copy.copy(DatasetPolicy.default_config)
    config.update(kwargs)

    self.dataset = test_dataset
    self.train_dataset = train_dataset
    if not weights_dataset_name:
      weights_dataset_name = self.dataset.name
    self.weights_dataset_name = weights_dataset_name

    self.__dict__.update(config)
    print("DatasetPolicy running with:")
    pprint(self.__dict__)
    self.ev = Evaluation(self)
    self.tt = ut.TicToc()

    # Determine inference mode:
    # fixed_order, random, oracle policy modes get fixed_order inference mode
    if re.search('fastinf',self.policy_mode):
      self.inference_mode = 'fastinf'
    elif self.policy_mode == 'random':
      self.inference_mode = 'random'
    elif self.policy_mode in ['no_smooth','backoff']:
      self.inference_mode = self.policy_mode
    else:
      self.inference_mode = 'fixed_order'

    # Determine fastinf model name
    self.fastinf_model_name='this_is_empty'
    if self.inference_mode=='fastinf':
      if self.detectors == ['csc_default']:
        self.fastinf_model_name='CSC'
      elif self.detectors == ['perfect']:
        self.fastinf_model_name='perfect'
      elif self.detectors == ['gist']:
        self.fastinf_model_name='GIST'
      elif self.detectors == ['gist','csc_default']:
        self.fastinf_model_name='GIST_CSC'
      else:
        raise RuntimeError("""
          We don't have Fastinf models for the detector combination you
          are running with: %s"""%self.detectors)

    # load the actions and the corresponding weights and we're ready to go
    self.test_actions = self.init_actions()
    self.actions = self.test_actions
    self.load_weights()

  def init_actions(self,train=False):
    """
    Return list of actions, which involves initializing the detectors.
    If train==True, the external detectors will load trainset detections.
    """
    dataset = self.dataset
    if train:
      dataset = self.train_dataset

    actions = []
    for detector in self.detectors:
      # synthetic perfect detector
      if detector=='perfect':
        for cls in dataset.classes:
          det = PerfectDetector(dataset, self.train_dataset, cls)
          actions.append(ImageAction('%s_%s'%(detector,cls), det))

      # synthetic perfect detector with noise in the detections
      elif detector=='perfect_with_noise':
        sw = SlidingWindows(self.train_dataset,dataset)
        for cls in dataset.classes:
          det = PerfectDetectorWithNoise(dataset, self.train_dataset, cls, sw)
          actions.append(ImageAction('%s_%s'%(detector,cls), det))

      # GIST classifier
      elif detector=='gist':
        for cls in self.dataset.classes:
          gist_table = np.load(config.get_gist_dict_filename(self.dataset))
          det = GistClassifier(cls, self.dataset.name,gist_table)
          self.actions.append(ImageAction('%s_%s'%(detector,cls), det))

      # real detectors, with pre-cached detections
      elif detector in ['dpm','csc_default','csc_half']:
        # load the dets from cache file and parcel out to classes
        all_dets = self.load_ext_detections(dataset, detector)
        for cls in dataset.classes:
          cls_ind = dataset.get_ind(cls)
          all_dets_for_cls = all_dets.filter_on_column('cls_ind',cls_ind,omit=True)
          det = ExternalDetector(
            dataset, self.train_dataset, cls, all_dets_for_cls, detector)
          actions.append(ImageAction('%s_%s'%(detector,cls), det))

      else:
        raise RuntimeError("Unknown mode in detectors: %s"%self.detectors)
      return actions

  def load_weights(self,weights_mode=None,force=False):
    """
    Set self.weights to weights loaded from file or constructed from scratch
    according to weights_mode and self.weights_dataset_name.
    """
    if not weights_mode:
      weights_mode = self.weights_mode

    filename = config.get_dp_weights_filename(self)
    if not force and opexists(filename):
      self.weights = np.loadtxt(filename)
      return self.weights

    # Construct the weight vector by concatenating per-action vectors
    # The features are [P(C) P(not C) H(C) 1]
    weights = np.zeros((len(self.actions), BeliefState.num_features))

    # OPTION 1: the corresponding manual weights are [1 0 0 0]
    if weights_mode == 'manual_1':
      weights[:,0] = 1
      weights = weights.flatten()

    elif weights_mode in ['manual_2','manual_3']:
      # Figure out the statistics if they aren't there
      if 'naive_ap|present' not in self.actions[0].obj.config or \
         'actual_ap|present' not in self.actions[0].obj.config or \
         'actual_ap|absent' not in self.actions[0].obj.config:
        det_configs = self.output_det_statistics()
        self.test_actions = self.init_actions()
        self.actions = self.test_actions

      # OPTION 2: the manual weights are [naive_ap|present 0 0 0]
      if weights_mode == 'manual_2':
        for action in self.actions:
          print action.obj.config
        weights[:,0] = [action.obj.config['naive_ap|present'] for action in self.actions]

      # OPTION 3: the manual weights are [actual_ap|present actual_ap|absent 0 0]
      elif weights_mode == 'manual_3':
        weights[:,0] = [action.obj.config['actual_ap|present'] for action in self.actions]
        # TODO: get rid of this
      #  weights[:,1] = [action.obj.config['actual_ap|absent'] for action in self.actions]

      else:
        None # impossible
      weights = weights.flatten()

    elif weights_mode in ['greedy','rl_regression','rl_lspi']:
      weights = self.learn_weights(weights_mode)

    else:
      raise ValueError("unsupported weights_mode %s"%weights_mode)
    self.weights = weights
    return weights

  def get_reshaped_weights(self):
    "Return weights vector reshaped to (num_actions, num_features)."
    return self.weights.reshape((len(self.actions),BeliefState.num_features))

  def write_weights_image(self,filename):
    "Output plot of current weights to filename."
    self.write_feature_image(self.get_reshaped_weights(),filename)

  def write_feature_image(self,feature,filename):
    "Output plot of feature to filename."
    plt.clf()
    p = plt.pcolor(feature)
    p.axes.invert_yaxis()
    plt.colorbar()
    plt.savefig(filename)

  def run_on_dataset(self,train=False,sample_size=-1,force=False):
    """
    Run MPI-parallelized over the images of the dataset (or a random subset).
    Return list of detections and classifications for the whole dataset.
    If test is True, runs on test dataset, otherwise train dataset.
    If sample_size != -1, does not check for cached files, as the objective
    is to collect samples, not the actual dets and clses.
    Otherwise, check for cached files first, unless force is True.
    Return dets,clses,samples.
    """
    self.tt.tic('run_on_dataset')
    print("Beginning run on dataset, with train=%s, sample_size=%s"%
      (train,sample_size))

    dets_table = None
    clses_table = None

    dataset = self.dataset
    self.actions = self.test_actions
    if train:
      dataset = self.train_dataset
      if not hasattr(self,'train_actions'):
        self.train_actions = self.init_actions(train)
      self.actions = self.train_actions

    # If we are collecting samples, we don't care about caches
    if sample_size > 0:
      force = True

    # Check for cached results
    det_filename = config.get_dp_dets_filename(self,train)
    cls_filename = config.get_dp_clses_filename(self,train)
    if not force \
        and opexists(det_filename) and opexists(cls_filename):
      dets_table = np.load(det_filename)[()]
      clses_table = np.load(cls_filename)[()]
      samples = None
      print("DatasetPolicy: Loaded dets and clses from cache.")
      return dets_table,clses_table,samples

    images = dataset.images
    if sample_size>0:
      images = ut.random_subset(dataset.images, sample_size)

    # Collect the results distributedly
    all_dets = []
    all_clses = []
    all_samples = []
    for i in range(comm_rank,len(images),comm_size):
      dets,clses,samples = self.run_on_image(images[i],dataset)
      all_dets.append(dets)
      all_clses.append(clses)
      all_samples += samples
    safebarrier(comm)

    # Aggregate the results
    final_dets = [] if comm_rank == 0 else None
    final_clses = [] if comm_rank == 0 else None
    final_samples = [] if comm_rank == 0 else None
    final_dets = comm.reduce(all_dets, op=MPI.SUM, root=0)
    final_clses = comm.reduce(all_clses, op=MPI.SUM, root=0)
    final_samples = comm.reduce(all_samples, op=MPI.SUM, root=0)
    #if self.inference_mode=='fastinf':
      # all_fm_cache_items = comm.reduce(self.inf_model.cache.items(), op=MPI.SUM, root=0)

    # Save if root
    if comm_rank==0:
      dets_table = ut.Table(cols=self.get_det_cols())
      final_dets = [det for det in final_dets if det.shape[0]>0]
      dets_table.arr = np.vstack(final_dets)
      clses_table = ut.Table(cols=self.get_cls_cols())
      clses_table.arr = np.vstack(final_clses)
      print("Found %d dets"%dets_table.shape()[0])

      # Only save results if we are not collecting samples
      if not sample_size > 0:
        np.save(det_filename,dets_table)
        np.save(cls_filename,clses_table)

      # Save the fastinf cache
      # TODO: turning this off for now
      if False and self.inference_mode=='fastinf':
        self.inf_model.cache = dict(all_fm_cache_items)
        self.inf_model.save_cache()
    else:
      print("comm_rank %d reached last safebarrier in run_on_dataset"%comm_rank)
    safebarrier(comm)

    # Broadcast results to all workers, because Evaluation splits its work as well.
    dets_table = comm.bcast(dets_table,root=0)
    clses_table = comm.bcast(clses_table,root=0)
    final_samples = comm.bcast(final_samples,root=0)
    if comm_rank==0:
      print("Running the %s policy on %d images took %.3f s"%(
        self.policy_mode, len(images), self.tt.qtoc('run_on_dataset')))
    return dets_table,clses_table,final_samples

  def learn_weights(self,mode='greedy'):
    """
    Runs iterations of generating samples with current weights and training
    new weight vectors based on the collected samples.
    - mode can be ['greedy','rl_regression','rl_lspi']
    """
    print("Beginning to learn regression weights")
    self.tt.tic('learn_weights:all')

    # Need to have weights set here to collect samples, so let's set
    # to manual_1 to get a reasonable execution trace.
    self.weights = self.load_weights(weights_mode='manual_1') 
    if comm_rank==0:
      print("Initial weights:")
      np.set_printoptions(precision=2,suppress=True,linewidth=160)
      print self.get_reshaped_weights()

    # Collect samples (parallelized)
    num_samples = 200 # actually this refers to images
    dets,clses,all_samples = self.run_on_dataset(False,num_samples)
    
    # Loop until max_iterations or the error is below threshold
    error = threshold = 0.001
    max_iterations = 8
    for i in range(0,max_iterations):
      # do regression with cross-validated parameters (parallelized)
      weights = None
      if comm_rank==0:
        weights, error = self.regress(all_samples, mode)
        self.weights = weights

        try:
          # write image of the weights
          img_filename = opjoin(
            config.get_dp_weights_images_dirname(self),'iter_%d.png'%i)
          self.write_weights_image(img_filename)

          # write image of the average feature
          all_features = self.construct_X_from_samples(all_samples)
          avg_feature = np.mean(all_features,0).reshape(
            len(self.actions),BeliefState.num_features)
          img_filename = opjoin(
            config.get_dp_features_images_dirname(self),'iter_%d.png'%i)
          self.write_feature_image(avg_feature, img_filename)
        except:
          print("Couldn't plot, no big deal.")

        print("""
  After iteration %d, we've trained on %d samples and
  the weights and error are:"""%(i,len(all_samples)))
        np.set_printoptions(precision=2,suppress=True,linewidth=160)
        print self.get_reshaped_weights()
        print error

        # after the first iteration, check if the error is small
        if i>0 and error<threshold:
          break

        print("Now collecting more samples with the updated weights...")

      # collect more samples (parallelized)
      safebarrier(comm)
      weights = comm.bcast(weights,root=0)
      self.weights = weights
      new_dets,new_clses,new_samples = self.run_on_dataset(False,num_samples)

      if comm_rank==0:
        # We either collect only unique samples or all samples
        only_unique_samples = False
        if only_unique_samples:
          self.tt.tic('learn_weights:unique_samples')
          for sample in new_samples:
            if not (sample in all_samples):
              all_samples.append(sample)
          print("Only adding unique samples to all_samples took %.3f s"%self.tt.qtoc('learn_weights:unique_samples'))
        else:
          all_samples += new_samples

    if comm_rank==0:
      print("Done training regression weights! Took %.3f s total"%
        self.tt.qtoc('learn_weights:all'))
      # Save the weights
      filename = config.get_dp_weights_filename(self)
      np.savetxt(filename, self.weights, fmt='%.6f')

    safebarrier(comm)
    weights = comm.bcast(weights,root=0)
    self.weights = weights
    return weights

  def construct_X_from_samples(self,samples):
    "Return array of (len(samples), BeliefState.num_features*len(self.actions)."
    b = self.get_b()
    feats = []
    for sample in samples:
      feats.append(b.block_out_action(sample.state,sample.action_ind))
    return np.array(feats)

  def compute_reward_from_samples(self, samples, mode='greedy',
    discount=0.9, attr=None):
    """
    Return vector of rewards for the given samples.
    - mode=='greedy' just uses the actual_ap of the taken action
    - mode=='rl_regression' or 'rl_lspi' uses discounted sum
    of actual_aps to the end of the episode
    - attr can be ['det_actual_ap', 'entropy', 'auc_ap', 'auc_entropy']
    """ 
    if not attr:
      attr = self.rewards_mode
    y = np.array([getattr(sample,attr) for sample in samples])
    if mode=='greedy':
      return y
    elif mode=='rl_regression':
      y = np.zeros(len(samples))
      for sample_idx in range(len(samples)):
        sample = samples[sample_idx]
        reward = getattr(sample,attr)
        i = 1
        while sample_idx + i < len(samples):
          next_samp = samples[sample_idx + i]
          if next_samp.step_ind == 0:
            # New episode begins here
            break
          reward += discount**i*getattr(next_samp,attr)
          i += 1
        y[sample_idx] = reward
      return y    
    return None

  def regress(self,samples,mode,warm_start=False):
    """
    Take list of samples and regress from the features to the rewards.
    If warm_start, start with the current values of self.weights.
    Return tuple of weights and error.
    """
    X = self.construct_X_from_samples(samples)
    y = self.compute_reward_from_samples(samples,mode)
    assert(X.shape[0]==y.shape[0])
    print("Number of non-zero rewards: %d/%d"%(
      np.sum(y!=0),y.shape[0]))

    from sklearn.cross_validation import KFold
    folds = KFold(X.shape[0], 4)
    alpha_errors = []
    alphas = [0.0001, 0.001, 0.01, 0.1, 1]
    for alpha in alphas:
      clf = sklearn.linear_model.Lasso(alpha=alpha,max_iter=2000)
      errors = []
      # TODO parallelize this? seems fast enough
      for train_ind,val_ind in folds:
        clf.fit(X[train_ind,:],y[train_ind])
        errors.append(ut.mean_squared_error(clf.predict(X[val_ind,:]),y[val_ind]))
      alpha_errors.append(np.mean(errors))
    best_ind = np.argmin(alpha_errors)
    best_alpha = alphas[best_ind]
    clf = sklearn.linear_model.Lasso(alpha=best_alpha,max_iter=200)
    clf.fit(X,y)
    print("Best lambda was %.3f"%best_alpha)
    weights = clf.coef_
    error = ut.mean_squared_error(clf.predict(X),y)
    return (weights, error)

  def get_b(self):
    "Fetch a belief state, and if we don't have one, initialize one."
    if not hasattr(self,'b'):
      self.run_on_image(self.dataset.images[0],self.dataset)
    return self.b

  def output_det_statistics(self):
    """
    Collect samples and display the statistics of times and naive and
    actual_ap increases for each class, on the train dataset.
    """
    det_configs = {}
    all_dets,all_clses,all_samples = self.run_on_dataset(train=True)
    if comm_rank==0:
      cols = ['action_ind','dt','det_naive_ap','det_actual_ap','img_ind']
      sample_array = np.array([[getattr(s,col) for s in all_samples] for col in cols]).T
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
          d = self.train_dataset
          presence_inds = np.array([d.images[img_ind].contains_cls_ind(cls_ind) for img_ind in img_inds])
          st_present = np.atleast_2d(st.arr[np.flatnonzero(presence_inds),:])
          if st_present.shape[0]>0:
            means = np.mean(st_present[:,2:],0)
            det_configs[self.actions[ind].name]['naive_ap|present'] = means[0]
            det_configs[self.actions[ind].name]['actual_ap|present'] = means[1]
          else:
            # this is just for testing, where there may not be enough dets
            det_configs[self.actions[ind].name]['naive_ap|present'] = 0
            det_configs[self.actions[ind].name]['actual_ap|present'] = 0
          st_absent = np.atleast_2d(st.arr[np.flatnonzero(presence_inds==False),:])
          if st_absent.shape[0]>0:
            means = np.mean(st_absent[:,2:],0)
            # naive_ap|absent is always 0
            det_configs[self.actions[ind].name]['actual_ap|absent'] = means[1]
          else:
            det_configs[self.actions[ind].name]['actual_ap|absent'] = 0
      detectors = '-'.join(self.detectors)
      filename = opjoin(
        config.get_dets_configs_dir(self.train_dataset), detectors+'.txt')
      with open(filename,'w') as f:
        json.dump(det_configs,f)
    safebarrier(comm)
    det_configs = comm.bcast(det_configs,root=0)
    return det_configs

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
      ff = b.compute_full_feature()
      self.action_values = np.array([\
        np.dot(self.weights, b.block_out_action(ff,action_ind)) for action_ind in range(len(self.actions))])

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

  def run_on_image(self, image, dataset, verbose=False):
    """
    Return
    - list of detections in the image, with each row as self.get_det_cols()
    - list of multi-label classification outputs, with each row as self.get_cls_cols()
    - list of <s,a,r,s',dt> samples.
    """
    gt = image.get_ground_truth(include_diff=True)
    self.tt.tic('run_on_image')

    all_detections = []
    all_clses = []
    samples = []
    prev_ap = 0
    img_ind = dataset.get_img_ind(image)

    # If we have previously run_on_image(), then we already have a reference to an inf_model
    # Otherwise, we make a new one and store a reference to it, to keep it alive
    if hasattr(self,'inf_model'):
      b = BeliefState(self.train_dataset, self.actions, self.inference_mode,
        self.bounds, self.inf_model, self.fastinf_model_name)
    else:
      b = BeliefState(self.train_dataset, self.actions, self.inference_mode,
        self.bounds, fastinf_model_name=self.fastinf_model_name)
      self.b = b
      self.inf_model = b.model

    self.update_actions(b)
    action_ind = self.select_action(b)
    step_ind = 0
    initial_clses = np.array(b.get_p_c().tolist() + [img_ind,0])
    entropy_prev = np.mean(b.get_entropies())
    while True:
      # Populate the sample with stuff we know
      sample = Sample()
      sample.step_ind = step_ind
      sample.img_ind = img_ind
      sample.state = b.compute_full_feature()
      sample.action_ind = action_ind
      sample.t = b.t

      # prepare for AUC reward stuff
      time_to_deadline = 0
      if self.bounds:
        time_to_deadline = max(0,self.bounds[1]-b.t)
      sample.auc_ap = 0

      # Take the action and get the observations as a dict
      action = self.actions[action_ind]
      obs = action.obj.get_observations(image)
      dt = obs['dt']

      # If observations include detections, compute the relevant
      # stuff for the sample collection
      sample.det_naive_ap = 0
      sample.det_actual_ap = 0
      if 'dets' in obs:
        det = action.obj
        detections = obs['dets']
        cls_ind = dataset.classes.index(det.cls)
        if detections.shape[0]>0:
          c_vector = np.tile(cls_ind,(np.shape(detections)[0],1))
          i_vector = np.tile(img_ind,(np.shape(detections)[0],1))
          detections = np.hstack((detections, c_vector, i_vector))
        else:
          detections = np.array([])
        dets_table = ut.Table(detections,det.get_cols()+['cls_ind','img_ind'])

        # compute the 'naive' det AP increase: adding dets to empty set
        ap,rec,prec = self.ev.compute_det_pr(dets_table,gt)
        sample.det_naive_ap = ap

        # TODO: am I needlessly recomputing this table?
        all_detections.append(detections)
        nonempty_dets = [dets for dets in all_detections if dets.shape[0]>0]
        all_dets_table = ut.Table(np.array([]),dets_table.cols)
        if len(nonempty_dets)>0:
          all_dets_table = ut.Table(np.concatenate(nonempty_dets,0),dets_table.cols)

        # compute the actual AP increase: addings dets to dets so far
        ap,rec,prec = self.ev.compute_det_pr(all_dets_table,gt)
        ap_diff = ap-prev_ap
        sample.det_actual_ap = ap_diff

        # detector AUC reward: this was tricky to get right :)
        auc_ap = time_to_deadline * ap_diff - ap_diff * dt / 2
        divisor = (time_to_deadline*(1-prev_ap))
        if divisor == 0:
          auc_ap = 1
        else:
          auc_ap /= divisor
        if dt > time_to_deadline:
          auc_ap = 0
        assert(auc_ap>=-1 and auc_ap<=1)
        sample.auc_ap = auc_ap
        prev_ap = ap

      b.update_with_score(action_ind, obs['score'])

      # mean entropy
      entropy = np.mean(b.get_entropies())
      dh = entropy_prev-entropy # this is actually -dh :)
      sample.entropy = dh
      entropy_prev = entropy

      auc_entropy = time_to_deadline * dh - dh * dt / 2
      divisor = (time_to_deadline * entropy)
      if divisor == 0:
        auc_entropy = 1
      else:
        auc_entropy /= divisor
      if dt > time_to_deadline:
        auc_entropy = 0
      assert(auc_entropy>=-1 and auc_entropy<=1)
      sample.auc_entropy = auc_entropy

      b.t += dt
      sample.dt = dt
      samples.append(sample)
      step_ind += 1

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
    # contributions, and actually re-gather p_c's for clses.
    action_inds = [s.action_ind for s in samples]
    if self.policy_mode=='oracle':
      naive_aps = np.array([s.det_naive_ap for s in samples])
      sorted_inds = np.argsort(-naive_aps,kind='merge') # order-preserving
      all_detections = np.take(all_detections, sorted_inds)
      sorted_action_inds = np.take(action_inds, sorted_inds)

      # actually go through the whole thing again, getting new p_c's
      b.reset()
      all_clses = []
      for action_ind in sorted_action_inds:
        action = self.actions[action_ind]
        obs = action.obj.get_observations(image)
        b.t += obs['dt']
        b.update_with_score(action_ind, obs['score'])
        clses = b.get_p_c().tolist() + [img_ind,b.t]
        all_clses.append(clses)

    # now construct the final dets array, with correct times
    times = [s.dt for s in samples]
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
    all_clses = np.array(all_clses)

    if verbose:
      print("DatasetPolicy on image with ind %d took %.3f s"%(
        img_ind,self.tt.qtoc('run_on_image')))

    # TODO: temp debug thing
    if False:
      print("Action sequence was: %s"%[s.action_ind for s in samples])
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
  @classmethod
  def load_ext_detections(cls,dataset,suffix,force=False):
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
          dets = cls.load_dpm_dets_for_image(image,dataset)
          ind_vector = np.ones((np.shape(dets)[0],1)) * i
          dets = np.hstack((dets,ind_vector))
          cols = ['x','y','w','h','dummy','dummy','dummy','dummy','score','time','cls_ind','img_ind']
          good_ind = [0,1,2,3,8,9,10,11]
          dets = dets[:,good_ind]
        elif re.search('csc',suffix):
          # NOTE: not actually using the given suffix in the call below
          dets = cls.load_csc_dpm_dets_for_image(image,dataset)
          ind_vector = np.ones((np.shape(dets)[0],1)) * i
          dets = np.hstack((dets,ind_vector))
        elif re.search('ctf',suffix):
          # Split the suffix into ctf and the main part
          actual_suffix = suffix.split('_')[1]
          dets = cls.load_ctf_dets_for_image(image, dataset, actual_suffix)
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

  @classmethod
  def load_ctf_dets_for_image(cls, image, dataset, suffix='default'):
    """Load multi-class array of detections for this image."""
    t = time.time()
    dirname = '/u/vis/x1/sergeyk/object_detection/ctfdets/%s/'%suffix
    time_elapsed = time.time()-t
    filename = os.path.join(dirname, image.name+'.npy')
    dets_table = np.load(filename)[()]
    print("On image %s, took %.3f s"%(image.name, time_elapsed))
    return dets_table.arr

  @classmethod
  def load_csc_dpm_dets_for_image(cls, image, dataset):
    """
    Loads HOS's cascaded dets.
    """
    t = time.time()
    name = os.path.splitext(image.name)[0]
    # if uest dataset, use HOS's detections. if not, need to output my own
    if re.search('test', dataset.name):
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
    for cls_ind,cls in enumerate(dataset.classes):
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

  @classmethod
  def load_dpm_dets_for_image(cls, image, dataset, suffix='dets_all_may25_DP'):
    """
    Loads multi-class array of detections for an image from .mat format.
    """
    t = time.time()
    name = os.path.splitext(image.name)[0]
    # TODO: figure out how to deal with different types of detections
    dets_dir = '/u/vis/x1/sergeyk/rl_detection/voc-release4/2007/tmp/dets_may25_DP'
    filename = opjoin(dets_dir, '%s_dets_all_may25_DP.mat'%name)
    if not opexists(filename):
      dets_dir = '/u/vis/x1/sergeyk/rl_detection/voc-release4/2007/tmp/dets_jun1_DP_trainval'
      filename = opjoin(dets_dir, '%s_dets_all_jun1_DP_trainval.mat'%name)
      if not opexists(filename):
        filename = opjoin(config.test_support_dir,'dets/%s_dets_all_may25_DP.mat'%name)
        if not opexists(filename):
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
    
  for ds in ['full_pascal_trainval']: # 'full_pascal_test'
    eval_d = Dataset(ds) 
    dp = DatasetPolicy(eval_d, train_d, detectors=['csc_default'])
    test_table = np.zeros((len(eval_d.images), len(dp.actions)))
    
    if not just_combine:
      for img_idx in range(comm_rank, len(eval_d.images), comm_size):
        img = eval_d.images[img_idx]
        for act_idx, act in enumerate(dp.actions):
          print '%s on %d for act %d'%(img.name, comm_rank, act_idx)    
          score = act.obj.get_observations(img)['score']
          print '%s, score: %f'%(img.name, score)
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
      filename = os.path.join(dirname,'table_linear_20')
      tab_test_table = ut.Table()
      tab_test_table.cols = list(eval_d.classes) + ['img_ind']
      
      tab_test_table.arr = np.hstack((test_table, np.array(np.arange(test_table.shape[0]),ndmin=2).T))
      cPickle.dump(tab_test_table, open(filename,'w'))
