from synthetic.common_imports import *
from synthetic.dataset import Dataset

d = Dataset('synthetic') 
f = d.plot_coocurrence()
f = d.plot_coocurrence(second_order=True)

d = Dataset('full_pascal_train')
f = d.plot_coocurrence()
f = d.plot_coocurrence(second_order=True)

d = Dataset('full_pascal_val')
f = d.plot_coocurrence()
f = d.plot_coocurrence(second_order=True)

d = Dataset('full_pascal_test')
f = d.plot_coocurrence()
f = d.plot_coocurrence(second_order=True)
