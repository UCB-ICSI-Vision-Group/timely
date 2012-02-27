import pexpect

from common_imports import *
from common_mpi import *
import synthetic.config as config

from synthetic.ngram_model import InferenceModel

class FastinfModel(InferenceModel):
  def __init__(self,dataset,suffix):
    self.dataset = dataset
    self.fname = config.get_fastinf_res_file(dataset,suffix)
    self.cmd = config.fastinf_bin+" -i %s -m 0 -Is 0"%self.fname
    self.tt = ut.TicToc().tic()
    self.process = pexpect.spawn(self.cmd)
    print "FastinfModel: Process started"
    marginals = self.get_marginals()
    print(self.p_c)
    print("FastinfModel: Computed initial marginals in %.3f sec"%self.tt.qtoc())

  def update_with_observations(self, taken, observations):
    # TODO: discretize in the same way as the trained thing here
    self.tt.tic()
    evidence = self.dataset.num_classes()*['?']
    for i in np.flatnonzero(taken):
      evidence[i] = str(observations[i])
    evidence = "(%s %s )"%(self.dataset.num_classes()*' ?', ' '.join(evidence))
    marginals = self.get_marginals(evidence)
    print(self.p_c)
    print("FastinfModel: Computed marginals given evidence in %.3f sec"%self.tt.qtoc())

  def get_marginals(self,evidence=None):
    """
    Parse marginals out of printed output of infer_timely.
    If evidence is given, first sends it to stdin of the process.
    Also update self.p_c with the marginals.
    """
    if evidence:
      print "Evidence:"
      print evidence
      self.process.sendline(evidence)
    self.process.expect('Enter your evidence'  )
    output = self.process.before
    marginals = FastinfModel.extract_marginals(output)
    self.p_c = [m[0] for m in marginals[:20]]
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
      if re.search('Partition',line):
        break
      vals = line.split()[1:]
      marginals.append(vals)
    return marginals
