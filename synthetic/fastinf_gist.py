from synthetic.common_imports import *
from synthetic.common_mpi import *

from synthetic.gist_classifier import GistClassifier, cls_for_dataset
from synthetic.dataset import Dataset

def create_gist_model_for_dataset(d):
  dataset = d.name  

  table = cls_for_dataset(dataset)    
  return table  
  
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
  from synthetic.fastInf import *
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
    write_out_mrf(data[:100,:], num_bins, filename, data_filename)  
    result = execute_lbp(filename, data_filename, filename_out)

  #print data
  
  #print table
  
  
  
    