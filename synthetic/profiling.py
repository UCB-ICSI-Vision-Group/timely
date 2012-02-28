from synthetic.run_experiment import load_configs
from synthetic.dataset import Dataset
from synthetic.dataset_policy import DatasetPolicy
dataset = Dataset('full_pascal_test')
first_n = 40
dataset.images = dataset.images[:first_n]
train_dataset = Dataset('full_pascal_trainval')
config = load_configs('feb27')[0]
dp = DatasetPolicy(dataset, train_dataset, **config)
%prun dets,clses,samples = dp.run_on_dataset(force=True)