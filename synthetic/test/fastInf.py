import synthetic.common_imports
from synthetic.fastInf import *
import matplotlib.pyplot as plt

class TestFastInf:
  def __init__(self):
    self.d = Dataset('test_pascal_train_tobi')
    
  def test_discretization_sane(self):
    d = self.d
    num_bins = 15
    table = np.random.randn(2501,1)
    #table += np.min(table)
    print table
    bounds, discr_table = discretize_table(table, num_bins)
    
    print np.min(bounds[:,0]), np.max(bounds[:,0])
    print np.min(table[:,0]), np.max(table[:,0])
    print discr_table[:,0]
    print bounds.shape
    
    x = np.hstack(discr_table.T.tolist())
    a,_,_ = plt.hist(x, num_bins)
    plt.show()
    print a
  
  def test_importance_sample(self):
    # Not a very good test... wrote it to have a look at it.
    values = np.array([0,4.5,5.5,6.125,6.375,6.625,6.875,7.5,8.5,13]).astype(float)
    bounds = ut.importance_sample(values, 6)
    bounds_gt = np.array([0,4,6,7,9,13])
    comp = np.absolute(bounds-bounds_gt).astype(int)
    np.testing.assert_equal(comp, np.zeros(comp.shape))  
    
  def test_importance_sample_two(self):
    num_bins = 20
    values = range(num_bins)
    values += [0]*8
    values += [1]*10    
    values += [2]*7
    values += [3]*3
    
    table = np.asmatrix(values).T
    print table.shape
    plt.hist(values, num_bins)
    _, discr_table = discretize_table(table, num_bins)
    x = np.hstack(discr_table.T.tolist())
    plt.hist(x, num_bins)
    plt.show()
    
        
  def test_determine_bin(self):
    values = np.array([0, 0.05,0.073,0.0234,0.1,0.13423,0.123534,0.1253,0.212,0.2252,0.43,0.3]).astype(float)
    bounds = np.array([0,0.1,0.2,0.3,np.max(values)])
      
    bins = determine_bin(values, bounds, 4)
    bins_gt = np.array([0,0,0,0,1,1,1,1,2,2,3,3])
    
    np.testing.assert_equal(bins, bins_gt)
    
  def test_discretize_value_perfect(self):
    val = 0.3
    d = Dataset('full_pascal_trainval')
    suffix = 'perfect'
    discr = discretize_value(val, d, suffix)
    np.testing.assert_equal(discr, np.zeros(discr.shape))
    
    val = 1
    discr = discretize_value(val, d, suffix)
    np.testing.assert_equal(discr, np.ones(discr.shape))

    
                          
if __name__=='__main__':
  tester = TestFastInf()
  #tester.test_determine_bin()
  #tester.test_discretization_sane()
  tester.test_importance_sample_two()  
  #tester.test_discretize_value_perfect()                        
  