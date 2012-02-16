"""
CS281a Class Project
MRF for class co-occurrences in PASCAL

Author: Tobias Baumgartner
Contact: tobi.baum@gmail.com
"""

import numpy as np
import matplotlib.pyplot as plt
import os as os 
import time
import synthetic.config as config
import synthetic.util as ut
import cPickle
from string import atof

class PriorsJT():
  def __init__(self, dataset):
    dataset_name = dataset.name
    self.cls_counts = self.get_cls_counts(dataset, dataset_name)
    self.create_co_occ(dataset)
    
  def fold_to_tri(self, mat):
    if not mat.shape[0] == mat.shape[1]:
      raise RuntimeError('Non-square matrix cannot be folded')
    n = mat.shape[0]
    for i in range(n):
      for j in range(i):
        mat[j, i] += mat[i, j]
        mat[i, j] = 0    
    return mat
  
  def create_co_occ(self, d):
    classes = d.classes
    images = d.images
    num_cls = len(classes)
    co_occ = np.zeros((num_cls, num_cls))
    for idx in range(len(images)):
      gt = d.get_ground_truth_for_img_inds([idx])
      objs = gt.arr[:, gt.cols.index('cls_ind')]
      objs = np.unique(objs).tolist()
      for obj_ind, obj in enumerate(objs):
        oth_objs = objs[:obj_ind] + objs[obj_ind + 1:]
        for oth_obj in oth_objs:
          co_occ[obj, oth_obj] += 1
#    norm = sum(co_occ, 0)
    self.cooc = co_occ
    #co_occ = np.divide(co_occ,np.transpose(np.tile(norm, (co_occ.shape[0],1))))
    
    for i in range(20):
      co_occ[i, i] = 0
      for j in range(i):
        co_occ[i, j] = 0
    return co_occ  
    #return fold_to_tri(co_occ)
  
  def test_fold(self):
    mat = np.matrix('[1 0 2; 2 1 0; 1 1 1]')
    mat_corr = np.matrix('[1 2 3; 0 1 1; 0 0 1]')
    self.fold_to_tri(mat)
    assert(mat.all() == mat_corr.all())
    
  def convert_mat_fg(self, filename, evidence):
    cooc = self.cooc
    cls_counts = self.cls_counts
    #print cls_counts
    of = open(filename, 'w')
    num_objs = np.sum(cooc)  
    for i in range(cooc.shape[0]):
      cooc[i, i] = 0;
    
    combs = np.transpose(np.matrix(np.where(cooc > 0)))
    of.write('%d\n\n' %combs.shape[0])
    for i in range(len(cls_counts)):
      if i in evidence:
        p0 = 1 - evidence[i]
        p1 = evidence[i] 
        of.write('1\n')
        of.write('%d\n' % i)
        of.write('2\n2\n\n')
        of.write('0     %f\n' % p0)
        of.write('1     %f\n\n' % p1)    
      else:
        p0 = num_objs - cls_counts[i]
        p1 = cls_counts[i]
        
    
    for comb in combs:    
      V0 = comb[0, 0]
      V1 = comb[0, 1]
      num_cooc = cooc[comb[0, 0], comb[0, 1]]
      
      p1 = cls_counts[comb[0, 1]] - num_cooc
      p2 = cls_counts[comb[0, 0]] - num_cooc
      p3 = num_cooc
      p0 = (num_objs - p3 - p2 - p1)/5.
      #print p0, p1, p2, p3
  
      of.write('2\n')
      of.write('%d %d\n' % (V0, V1))
      of.write('2 2\n4\n')
      of.write('0     %f\n' % p0)
      of.write('1     %f\n' % p1)
      of.write('2     %f\n' % p2)
      of.write('3     %f\n\n' % p3)
      
      
  def get_cls_counts(self, d, dataset):
    count_dir = os.path.join(config.data_dir, 'cls_counts/')
    ut.makedirs(count_dir)
    count_file = os.path.join(count_dir, dataset)
    if not os.path.isfile(count_file):
      classes = d.classes
      cls_counts = []
      for cls in classes:
        cls_gt = d.get_ground_truth_for_class(cls)      
        cls_counts.append(np.unique(cls_gt.arr[:, cls_gt.cols.index('img_ind')]).size)
      cPickle.dump(cls_counts, open(count_file, 'w'))
    else:
      cls_counts = cPickle.load(open(count_file, 'r'))
    self.cls_counts = cls_counts
    return cls_counts 

  def get_probabilities(self, observed_inds=[], observed_vals=[]):
    filename = 'mine.fg'
    evidence = {}
    for idx, ind in enumerate(observed_inds):
      evidence[ind] = observed_vals[idx]
        
    #cooc = np.random.random((20,20))
    self.convert_mat_fg(filename, evidence)
    os.system('./justJT ' + filename + ' > output')    
    result = open('output', 'r').readlines()
    os.remove('output')
    margins = []
    for line in result:
      margins.append(atof(line.split()[2][:-2]))    
    #for ev in evidence:
      #margins[ev] = 0
    #print margins.index(max(margins)), classes[margins.index(max(margins))]

    return margins
  
import synthetic.dataset
if __name__ == '__main__':
  dataset = 'full_pascal_train'
  d = synthetic.dataset.Dataset(dataset)
  jt = PriorsJT(d)
  
#  cls_pri = ClassPriors(d,mode='backoff')
  classes = d.classes
  gt = d.get_ground_truth()
  avg_obj_per_img = gt.shape()[0] / float(len(d.images))
#  print 'objects per image:', avg_obj_per_img
  t = time.time()
  cooc = jt.create_co_occ(d)
  cls_counts = jt.get_cls_counts(d, dataset)
  
  # prism
  plt.matshow(cooc, fignum=2, cmap='gray')
  plt.savefig('fig')
  filename = 'mine.fg'
  evidence = {}
  num_imgs = len(d.images)
  
  #cooc = np.random.random((20,20))
  jt.convert_mat_fg( filename, evidence)
  print filename
  os.system('./justJT ' + filename + ' > output')
  
  result = open('output', 'r').readlines()
  #os.remove('output')
  margins = []
  
  for line in result:
    print line
    margins.append(atof(line.split()[2][:-2]))
  
  
  #for ev in evidence:
  #  margins[ev] = 0
  
  print margins.index(max(margins)), classes[margins.index(max(margins))]
  print 'margs', len(margins)  
  
  print 'time:', time.time() - t
  
  
  
