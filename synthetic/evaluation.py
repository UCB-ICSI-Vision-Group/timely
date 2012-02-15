import operator
import matplotlib.pyplot as plt
from mako.template import Template

from common_mpi import *
from common_imports import *

from synthetic.dataset import Dataset
from synthetic.image import BoundingBox
import synthetic.config as config
from synthetic.safebarrier import safebarrier

class Evaluation:
  """
  Class to output evaluations of detections.
  Has to be initialized with a DatasetPolicy to correctly set the paths.
  """

  MIN_OVERLAP = 0.5
  TIME_INTERVALS = 12

  def __init__(self,dataset_policy=None,dataset=None,name='default'):
    """
    Must have either dataset_policy or dataset and name.
    If dataset_policy is given, dataset and name are ignored.
    """
    assert(dataset_policy or (dataset and name))

    if dataset_policy:
      self.dp = dataset_policy
      self.dataset = dataset_policy.dataset
      self.name = dataset_policy.get_config_name()
    else:
      self.dataset = dataset
      self.name = name

    self.time_intervals = Evaluation.TIME_INTERVALS
    self.min_overlap = Evaluation.MIN_OVERLAP

    # Determine filenames and create directories
    self.results_path = config.get_evals_dp_dir(self.dp)

    # wholeset evaluations
    self.dashboard_filename = opjoin(self.results_path, 'dashboard_%s.html')
    self.wholeset_aps_filename = opjoin(self.results_path, 'aps_whole.txt')

    # evals/{name}/wholeset_detailed
    dirname = opjoin(self.results_path, 'wholeset_detailed')
    ut.makedirs(dirname)
    self.pr_whole_png_filename = opjoin(dirname, 'pr_whole_%s.png')
    self.pr_whole_txt_filename = opjoin(dirname, 'pr_whole_%s.txt')
    self.apvst_whole_png_filename = opjoin(dirname, 'apvst_wholes.png')
    self.apvst_whole_txt_filename = opjoin(dirname, 'apvst_whole_%s.txt')
    self.apvst_whole_data_filename = opjoin(dirname, 'apvst_whole_table.npy')
    self.apvst_whole_png_filename = opjoin(dirname, 'apvst_whole.png')

    # avg-image evaluations
    self.apvst_avg_data_filename = opjoin(self.results_path, 'apvst_avg_table.npy')
    self.apvst_avg_png_filename = opjoin(self.results_path, 'apvst_avg.png')

    self.pr_avg_png_filename = opjoin(self.results_path, 'pr_avg_%s.png')
    self.pr_avg_txt_filename = opjoin(self.results_path, 'pr_avg_%s.txt')
    self.apvst_data_filename = opjoin(self.results_path, 'apvst_table.npy')
    self.apvst_png_filename = opjoin(self.results_path, 'apvst.png')

  ##############
  # Avg- and Whole-set AP vs. Time
  ##############
  def evaluate_dets_vs_t(self,dets=None,plot=True,force=False):
    """
    Evaluate detections in the AP vs Time regime and write out plots to
    canonical places.
    We evaluate on the whole dataset instead of averaging per-image,
    and so get rid of error bars.
    """
    bounds = self.dp.bounds if self.dp and self.dp.bounds else None

    table = None
    filename = self.apvst_avg_data_filename
    if os.path.exists(filename) and not force:
      if comm_rank==0:
        table = np.load(filename)[()]
    else:
      if not dets:
        dets = self.dp.detect_in_dataset()

      # determine time sampling points
      all_times = dets.subset_arr('time')
      num_points = self.time_intervals
      points = ut.importance_sample(all_times,num_points)
      # make sure bounds are included in the sampling points if given
      if bounds:
        points = np.sort(np.array(points.tolist() + list(bounds)))
        num_points += 2

      # do this now to save time in the inner loop later
      gt_for_image_list = []
      img_dets_list = []
      gt = self.dataset.get_ground_truth(include_diff=True)
      for img_ind,image in enumerate(self.dataset.images):
        gt_for_image_list.append(gt.filter_on_column('img_ind',img_ind))
        img_dets_list.append(dets.filter_on_column('img_ind',img_ind))

      ap_means = np.zeros(num_points)
      ap_stds = np.zeros(num_points)
      arr = np.zeros((num_points,3))
      for i in range(comm_rank,num_points,comm_size):
        tt = ut.TicToc().tic()
        point = points[i]
        aps = []
        num_dets = 0
        for img_ind,image in enumerate(self.dataset.images):
          gt_for_image = gt_for_image_list[img_ind]
          img_dets = img_dets_list[img_ind]
          dets_to_this_point = img_dets.filter_on_column('time',point,operator.le)
          num_dets += dets_to_this_point.shape()[0]
          ap,rec,prec = self.compute_pr(dets_to_this_point,gt_for_image)
          aps.append(ap)
        arr[i,:] = [point,np.mean(aps),np.std(aps)]
        print("Calculating avg-image AP (%.3f) of the %d detections up to %.3fs took %.3fs"%(
          np.mean(aps),num_dets,point,tt.qtoc()))
      arr_all = None
      if comm_rank == 0:
        arr_all = np.zeros((num_points,3))
      safebarrier(comm)
      comm.Reduce(arr,arr_all)
      if comm_rank==0:
        table = ut.Table(arr_all, ['time','ap_mean','ap_std'], self.name)
        np.save(filename,table)
    # Plot the table
    if plot and comm_rank==0:
      try:
        Evaluation.plot_ap_vs_t([table],self.apvst_avg_png_filename, bounds)
      except:
        print("Could not plot")
    return table

  @classmethod
  def compute_auc(cls,times,vals,bounds=None):
    """
    Return the area under the curve of vals vs. times, within given bounds.
    """
    if bounds:
      valid_ind = np.flatnonzero((times>=bounds[0]) & (times<=bounds[1]))
      times = times[valid_ind]
      vals = vals[valid_ind]
    auc = np.trapz(vals,times)
    return auc

  @classmethod
  def plot_ap_vs_t(cls, tables, filename, all_bounds=None, with_legend=True):
    """
    Take list of Tables containing AP vs. Time information.
    Bounds are given as a list of the same length as tables, or not at all.
    If table.cols contains 'ap_mean' and 'ap_std' and only one table is given,
    plots error area around the curve.
    If bounds are not given, uses the min and max values for each curve.
    If only one bounds is given, uses that for all tables.
    The legend is assembled from the .name fields in the Tables.
    Save plot to given filename.
    Does not return anything.
    """
    print("here1")
    plt.clf()
    colors = ['black','orange','#4084ff','purple']
    styles = ['-','--','-..','-.']
    prod = [x for x in itertools.product(colors,styles)]
    print(all_bounds)
    if not all_bounds:
      all_bounds = [None for table in tables]
    elif not isinstance(all_bounds[0], types.ListType):
      all_bounds = [all_bounds for table in tables]
    else:
      assert(len(all_bounds)==len(tables))
    
    for i,table in enumerate(tables):
      print("Plotting %s"%table.name)
      bounds = all_bounds[i]
      print 'i:', i
      style = prod[i][1]
      color = prod[i][0]
      times = table.subset_arr('time')
      if 'ap_mean' in table.cols and 'ap_std' in table.cols:
        vals = table.subset_arr('ap_mean')
        stdevs = table.subset_arr('ap_std')
        if len(tables)==1:
          plt.fill_between(times,vals-stdevs,vals+stdevs,color='#4084ff',alpha=0.3)
      else:
        vals = table.subset_arr('ap_mean')
      auc = Evaluation.compute_auc(times,vals,bounds)

      high_bound_val = vals[-1]
      if bounds:
        high_bound_val = vals[times.tolist().index(bounds[1])]

      label = "(%.2f, %.2f) %s"%(auc,high_bound_val,table.name)
      print(label)
      plt.plot(times, vals, style,
          linewidth=2,color=color,label=label)

      # draw vertical lines at bounds, if given
      if bounds:
        low_bound_val = vals[times.tolist().index(bounds[0])]
        plt.vlines(bounds[0],0,low_bound_val,alpha=0.8)
        plt.vlines(bounds[1],0,high_bound_val,alpha=0.8)
    if with_legend:
      plt.legend(loc='upper left')
      leg = plt.gca().get_legend()
      ltext = leg.get_texts()
      plt.setp(ltext, fontsize='small')
    plt.xlabel('Time',size=14)
    plt.ylabel('AP',size=14)
    plt.ylim(0,1)
    plt.grid(True)
    plt.savefig(filename)

  def evaluate_detections_whole(self,dets=None,force=False):
    """
    Output evaluations over the whole dataset in all formats:
    - multi-class (one PR plot)
    - per-class PR plots (only detections of that class are considered)
    """
    if not dets:
      assert(self.dp != None)
      dets = self.dp.detect_in_dataset()

    # Per-Class
    num_classes = len(self.dataset.classes)
    dist_aps = np.zeros(num_classes)
    for cls_ind in range(comm_rank, num_classes, comm_size):
      cls = self.dataset.classes[cls_ind] 
      cls_dets = dets.filter_on_column('cls_ind',cls_ind)
      cls_gt = self.dataset.get_ground_truth_for_class(cls,include_diff=True)
      dist_aps[cls_ind] = self.compute_and_plot_pr(cls_dets, cls_gt, cls, force)
    aps = None
    if comm_rank==0:
      aps = np.zeros(num_classes)
    safebarrier(comm)
    comm.Reduce(dist_aps,aps)

    # the rest is done by rank==0
    if comm_rank == 0:
      # Multi-class
      gt = self.dataset.get_ground_truth(include_diff=True)
      filename = opjoin(self.results_path, 'pr_whole_multiclass')
      if force or not os.path.exists(filename):
        t = time.time()
        print("Evaluating %d dets in the multiclass setting..."%dets.shape()[0])
        ap_mc = self.compute_and_plot_pr(dets, gt, 'multiclass')
        time_elapsed = time.time()-t
        print("...took %.3fs"%time_elapsed)

      # Write out the information to a single overview file
      with open(self.wholeset_aps_filename, 'w') as f:
        f.write("Multiclass: %.3f\n"%ap_mc)
        f.write(','.join(self.dataset.classes)+'\n')
        f.write(','.join(['%.3f'%ap for ap in aps])+'\n')

      # Assemble everything in one HTML file, the Dashboard
      if False:
        filename = self.pr_whole_png_filename%'all'
        names = list(self.dataset.classes)
        names.append('multiclass')
        aps.append(ap_mc)
        recs.append(rec_mc)
        precs.append(prec_mc)
        self.plot_pr_grid(aps,recs,precs,names,filename)

        eval_type = 'whole'
        template = Template(filename=config.eval_template_filename)
        filename = self.dashboard_filename%eval_type
        names = list(self.dataset.classes)
        names.append('avg')
        aps.append(np.mean(aps))
        with open(filename, 'w') as f:
          f.write(template.render(
            title=eval_type,
            names=names, aps=aps,
            all_pr_filename=os.path.basename(self.pr_whole_png_filename%'all')))
    safebarrier(comm)

  def compute_and_plot_pr(self, dets, gt, name, force=False):
    """
    Helper function. Compute the precision-recall curves from the detections
    and ground truth and output them to files.
    Return ap.
    """
    filename = self.pr_whole_txt_filename%name
    if force or not os.path.exists(filename):
      [ap,rec,prec] = self.compute_pr(dets, gt)
      try:
        self.plot_pr(ap,rec,prec,name,self.pr_whole_png_filename%name)
      except:
        None
      with open(filename, 'w') as f:
        f.write("%f\n"%ap)
        for i in range(np.shape(rec)[0]):
          f.write("%f %f\n"%(rec[i], prec[i]))
    else:
      with open(filename) as f:
        ap = float(f.readline())
    return ap

  def plot_pr(self, ap, rec, prec, name, filename, force=False):
    """Plot the Precision-Recall curve, saving png to filename."""
    if opexists(filename) and not force:
      print("plot_pr: not doing anything as file exists")
      return
    label = "%s: %.3f"%(name,ap)
    plt.clf()
    plt.plot(rec,prec,label=label,color='black',linewidth=5)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.legend(loc='lower left')
    plt.xlabel('Recall',size=16)
    plt.ylabel('Precision',size=16)
    plt.grid(True)
    plt.savefig(filename)

  def plot_pr_grid(self, aps, recs, precs, names, filename, force=False):
    """
    Plot P-R curves for all the data passed in in a grid of minimal size.
    Save png to filename.
    """
    if opexists(filename) and not force:
      print("plot_pr_grid: not doing anything as file exists")
      return
    h = int(np.round(np.sqrt(len(names))))
    w = int(np.ceil(np.sqrt(len(names))))
    # TODO: increase figsize or resolution here or something, plots are too big
    # for the figure size right now
    fig, axes = plt.subplots(h, w, sharex=True, sharey=True)
    left = 0.0625
    bottom = 0.05
    width = 1-2*left
    height = 1-2*bottom
    ax = fig.add_axes([left,bottom,width,height],frameon=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('Recall',size=16)
    ax.set_ylabel('Precision',size=16)
    for i in range(len(names)):
      ax = axes.flat[i]
      ap = aps[i]
      label = "%s: %.3f"%(names[i],ap)
      ax.plot(recs[i], precs[i], label=label, linewidth=5)
      ax.set_xlim(0,1)
      ax.set_ylim(0,1)
      ax.legend(loc='lower left')
      ax.set_title(names[i])
      ax.grid(True)
    plt.setp([a.get_xticklabels() for a in axes[0,:]], visible=False)
    plt.setp([a.get_yticklabels() for a in axes[:,1]], visible=False)
    plt.savefig(filename)
  
  ##############################
  # Computation of Precision-Recall and Average Precision
  ##############################
  def compute_pr(self, dets, gt):
    pr_and_hard_neg = self.compute_pr_and_hard_neg(dets, gt)
    return pr_and_hard_neg[:3]
  
  def compute_hard_neg(self, dets, gt):
    pr_and_hard_neg = self.compute_pr_and_hard_neg(dets, gt)
    return pr_and_hard_neg[4]
    
  def compute_pr_and_hard_neg(self, dets, gt):
    """
    Take Table of detections and Table of ground truth.
    Ground truth can be for a single image or a whole dataset.
    and can contain either all classes or just one class (but the cls_ind col
    must be present in either case).
    Depending on these decisions, the meaning of the PR evaluation is
    different.
    In particular, if gt is for a single class but dets are for multiple
    classes, there will be a lot of false positives!
    NOTE: modifies dets in-place (sorts by score)
    Return ap, recall, and precision vectors as tuple.
    """
    # if dets or gt are empty, return 0's
    nd = dets.arr.shape[0]
    if nd < 1 or gt.shape()[0] < 1:
      ap = 0
      rec = np.array([0])
      prec = np.array([0])
      return (ap,rec,prec)
    tt = ut.TicToc().tic()

    # augment gt with a column keeping track of matches
    cols = list(gt.cols) + ['matched']
    arr = np.zeros((gt.arr.shape[0],gt.arr.shape[1]+1))
    arr[:,:-1] = gt.arr
    gt = ut.Table(arr,cols)

    # sort detections by confidence
    dets.sort_by_column('score',descending=True)

    # match detections to ground truth objects
    npos = gt.filter_on_column('diff',0).shape()[0]
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    hard_neg = np.zeros(nd)
    for d in range(nd):
      if tt.qtoc() > 15:
        print("... on %d/%d dets"%(d,nd))
        tt.tic()

      det = dets.arr[d,:]

      # find ground truth for this image
      if 'img_ind' in gt.cols:
        img_ind = det[dets.ind('img_ind')]
        inds = gt.arr[:,gt.ind('img_ind')] == img_ind
        gt_for_image = gt.arr[inds,:]
      else:
        gt_for_image = gt.arr
      
      if gt_for_image.shape[0] < 1:
        # this can happen if we're passing ground truth for a class
        # false positive due to detection in image that does not contain the class
        fp[d] = 1 
        hard_neg[d] = 1
        continue

      # find the maximally overlapping ground truth element for this detection
      overlaps = BoundingBox.get_overlap(gt_for_image[:,:4],det[:4])
      jmax = overlaps.argmax()
      ovmax = overlaps[jmax]

      # assign detection as true positive/don't care/false positive
      if ovmax >= self.MIN_OVERLAP:
        if not gt_for_image[jmax,gt.ind('diff')]:
          is_matched = gt_for_image[jmax,gt.ind('matched')]
          if is_matched == 0:
            if gt_for_image[jmax,gt.ind('cls_ind')] == det[dets.ind('cls_ind')]:
              # true positive
              tp[d] = 1
              gt_for_image[jmax,gt.ind('matched')] = 1
            else:
              # false positive due to wrong class
              fp[d] = 1
              hard_neg[d] = 1
          else:
            # false positive due to multiple detection
            # this is still a correct answer, so not a hard negative
            fp[d] = 1
        else:
          None
          # NOT a false positive because object is difficult!
      else:
        # false positive due to not matching any ground truth object
        fp[d] = 1
        hard_neg[d] = 1
      # NOTE: this is very important: otherwise, gt.arr does not get the
      # changes we make to gt_for_image
      if 'img_ind' in gt.cols:
        gt.arr[inds,:] = gt_for_image
    fp=np.cumsum(fp)
    tp=np.cumsum(tp)
    rec=1.*tp/npos
    prec=1.*tp/(fp+tp)

    ap = self.compute_ap(rec,prec)
    return (ap,rec,prec,hard_neg)

  def compute_ap(self,rec,prec):
    """
    Takes recall and precision vectors and computes piecewise area under the
    curve.
    """
    mprec = np.hstack((0,prec,0))
    mrec = np.hstack((0,rec,1))
    # make sure prec is monotonically decreasing
    for i in range(len(mprec)-1,0,-1):
      mprec[i-1]=max(mprec[i-1],mprec[i])
    # find piecewise area under the curve
    i = np.add(np.nonzero(mrec[1:] != mrec[0:-1]),1)
    ap = np.sum((mrec[i]-mrec[np.subtract(i,1)])*mprec[i])
    return ap

