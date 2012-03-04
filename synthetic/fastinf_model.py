import pexpect

from common_imports import *
from common_mpi import *
import synthetic.config as config

from synthetic.ngram_model import InferenceModel
from synthetic.fastInf import FastinfDiscretizer

class FastinfModel(InferenceModel):
  def __init__(self,dataset,model_name,num_actions,m='0',r2='1'):
    # TODO: experiment with different values of fastinf

    self.dataset = dataset
    self.fd = FastinfDiscretizer(self.dataset, model_name)
    self.res_fname = config.get_fastinf_res_file(dataset,model_name,m,r2)

    # TODO: experiment with different amounts of smoothing
    # amount of smoothing is correlated with fastinf slowness, values [0,1)
    self.smoothing = 0
    self.cache_fname = config.get_fastinf_cache_file(dataset,model_name,m,r2,self.smoothing)

    if opexists(self.cache_fname):
      with open(self.cache_fname) as f:
        print("Loading fastinf cache from file")
        self.cache = cPickle.load(f)
    else:
      self.cache = {}
    self.cmd = config.fastinf_bin+" -i %s -m 0 -Is %f"%(self.res_fname, self.smoothing)
    self.num_actions = num_actions
    self.tt = ut.TicToc().tic()
    self.process = pexpect.spawn(self.cmd)
    marginals = self.get_marginals()
    #print("FastinfModel: Computed initial marginals in %.3f sec"%self.tt.qtoc())

  def save_cache(self):
    "Write cache out to file with cPickle."
    print("Writing cache out to file with cPickle")
    with open(self.cache_fname,'w') as f:
      cPickle.dump(self.cache,f)

  def update_with_observations(self, taken, observations):
    self.tt.tic()
    evidence = self.dataset.num_classes()*['?']
    for i in np.flatnonzero(taken):
      evidence[i] = str(self.fd.discretize_value(observations[i],i))
    evidence = "(%s %s )"%(self.dataset.num_classes()*' ?', ' '.join(evidence))
    print evidence
    try:
      marginals = self.get_marginals(evidence)
    except Exception as e:
      print("comm_rank %d: something went wrong in fastinf:get_marginals!!!"%
        comm_rank)
      print e
      print str(self.process)
      # restart process
      try:
        self.process.close(force=True)
      except Exception as e2:
        print("comm_rank %d: can't close process!"%comm_rank)
      self.process = pexpect.spawn(self.cmd)
      self.get_marginals()
    #print("FastinfModel: Computed marginals given evidence in %.3f sec"%self.tt.qtoc())

  def reset(self):
    """
    Get back to the initial state, erasing the effects of any evidence
    that has been applied.
    Sends totally uninformative evidence to get back to the priors.
    Is actually instantaneous due to caching.
    """
    observations = taken = np.zeros(self.num_actions)
    self.update_with_observations(taken,observations)

  def get_marginals(self,evidence=None):
    """
    Parse marginals out of printed output of infer_timely.
    If evidence is given, first sends it to stdin of the process.
    Also update self.p_c with the marginals.
    """
    if evidence:
      if evidence in self.cache:
        print "Fetching cached marginals"
        marginals = self.cache[evidence]
        self.p_c = np.array([m[1] for m in marginals[:20]])
        return marginals
      self.process.sendline(evidence)
    self.process.expect('Enter your evidence')
    output = self.process.before
    marginals = FastinfModel.extract_marginals(output)
    # TODO: not caching for fear of ulimit
    #self.cache[evidence] = marginals
    self.p_c = np.array([m[1] for m in marginals[:20]])
    return marginals

  @classmethod
  def extract_marginals(cls, output):
    """
    Parse the output of the infer_timely binary for the variable marginals.
    Return a list of lists, where the index in the outer list is the index
    of the variable, and the index of an inner list is the index of that
    variable's value.
    """
    lines = output.split('\r\n')
    try:
      ind = lines.index('# belief marginals / exact marginals / KL Divergence') 
    except ValueError:
      print("ERROR: cannot find marginals in output")
    marginals = []
    for line in lines[ind+1:]:
      if re.search('Partition',line) or re.search('^\w*$',line):
        break
      vals = line.split()[1:]
      vals = [float(v) for v in vals]
      marginals.append(vals)
    return marginals
