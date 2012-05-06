"""
Runner script used in development while switchign to pandas.DataFrame.
"""

from synthetic.common_imports import *
from synthetic.dataset import Dataset

d = Dataset('test_data1')
print(d)