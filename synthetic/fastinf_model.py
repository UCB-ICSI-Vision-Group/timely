import pexpect

from common_imports import *
from common_mpi import *
import synthetic.config as config

from synthetic.ngram_model import InferenceModel
from synthetic.fastInf import FastinfDiscretizer

class FastinfModel(InferenceModel):
  def __init__(self,dataset,suffix,num_actions,m='0',r2=''):
    self.dataset = dataset
    self.suffix = suffix
    self.fd = FastinfDiscretizer(self.dataset, self.suffix)
    self.num_actions = num_actions
    self.res_fname = config.get_fastinf_res_file(dataset,suffix,m,r2)
    self.cache_fname = config.get_fastinf_cache_file(dataset,suffix)
    if opexists(self.cache_fname):
      with open(self.cache_fname) as f:
        self.cache = cPickle.load(f)
    else:
      self.cache = {}
    self.cmd = config.fastinf_bin+" -i %s -m 0 -Is 0"%self.res_fname
    self.tt = ut.TicToc().tic()
    self.process = pexpect.spawn(self.cmd)
    print "FastinfModel: Process started"
    marginals = self.get_marginals()
    print(self.p_c)
    print("FastinfModel: Computed initial marginals in %.3f sec"%self.tt.qtoc())

  def save_cache(self):
    "Write cache out to file with cPickle."
    with open(self.cache_fname,'w') as f:
      cPickle.dump(self.cache,f)

  def update_with_observations(self, taken, observations):
    self.tt.tic()
    evidence = self.dataset.num_classes()*['?']
    for i in np.flatnonzero(taken):
      evidence[i] = str(self.fd.discretize_value(observations[i]))
    evidence = "(%s %s )"%(self.dataset.num_classes()*' ?', ' '.join(evidence))
    marginals = self.get_marginals(evidence)
    print("FastinfModel: Computed marginals given evidence in %.3f sec"%self.tt.qtoc())

  def reinit_marginals(self):
    "Sends totally uninformative evidence to get back to the priors."
    observations = taken = np.zeros(self.num_actions)
    self.update_with_observations(taken,observations)

  def get_marginals(self,evidence=None):
    """
    Parse marginals out of printed output of infer_timely.
    If evidence is given, first sends it to stdin of the process.
    Also update self.p_c with the marginals.
    """
    if evidence:
      print "Evidence:"
      print evidence
      if evidence in self.cache:
        print "Fetching cached marginals"
        marginals = self.cache[evidence]
        self.p_c = np.array([m[1] for m in marginals[:20]])
        return marginals
      self.process.sendline(evidence)
    self.process.expect('Enter your evidence')
    output = self.process.before
    marginals = FastinfModel.extract_marginals(output)
    self.cache[evidence] = marginals
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
