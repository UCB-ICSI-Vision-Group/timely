from synthetic.common_imports import *
from synthetic.common_mpi import *

from synthetic.fastInf import *
from synthetic.gist_classifier import GistClassifier, cls_for_dataset
from synthetic.dataset import Dataset

def create_gist_model_for_dataset(d):
  dataset = d.name  

  table = cls_for_dataset(dataset)    
  return table  

def discretize_table(table, num_bins, asInt=True):
  new_table = np.zeros(table.shape)
  print table[0,:]
  for coldex in range(table.shape[1]):
    col = table[:, coldex]
    print type(col)
    print sum(abs(col))
    print col
    bounds = ut.importance_sample(col, num_bins+1)
    
    # determine which bin these fall in
    col_bin = np.zeros((table.shape[0],1))
    bin_values = np.zeros(bounds.shape)
    last_val = 0.
    for bidx, b in enumerate(bounds):
      bin_values[bidx] = (last_val + b)/2.
      last_val = b
      col_bin += np.matrix(col < b, dtype=int).T
    bin_values = bin_values[1:]    
    col_bin[col_bin == 0] = 1  
    if asInt:
      a = num_bins - col_bin
      new_table[:, coldex] = a[:,0] 
    else:    
      for rowdex in range(table.shape[0]):
        new_table[rowdex, coldex] = bin_values[int(col_bin[rowdex]-1)]
  if asInt:    
    return new_table.astype(int)
  else:
    return new_table
  
def create_tables():
  datasets = ['full_pascal_trainval','full_pascal_test','full_pascal_train','full_pascal_val']
  for dataset in datasets:
    table = create_gist_model_for_dataset(dataset)
    savefile = config.get_fastinf_data_file(dataset)
    cPickle.dump(table, open(savefile, 'w'))
    print table
    print table.shape

#def gist_probabilities_for_images(gist, images, num_classes):
#  table = np.zeros((len(images), num_classes))
#  for img_idx, img in enumerate(images):
#    table[img_idx, :] = gist.get_priors(img)
#  return table

if __name__=='__main__':
  #create_tables()
  num_bins = 5
  dataset = 'full_pascal_trainval'
  
  d = Dataset(dataset)
  table_gt = d.get_cls_ground_truth().arr.astype(int)

  # replace this with a method to get the probs for each image
  # ---------------->8-----------------
  table = create_gist_model_for_dataset(d)
  #table = plausible_assignments(table_gt)
  # ----------------8<-----------------
  print table
  
  discr_table = discretize_table(table, num_bins)  
  data = np.hstack((table_gt, discr_table))
  
  print discr_table
  suffix = 'gist'
  filename = config.get_fastinf_mrf_file(dataset,suffix)
  data_filename = config.get_fastinf_data_file(dataset,suffix)
  filename_out = config.get_fastinf_res_file(dataset,suffix)
  
  if comm_rank == 0:
    write_out_mrf(data, num_bins, filename, data_filename)  
    result = execute_lbp(filename, data_filename, filename_out)

  #print data
  
  #print table
  
  
  
    