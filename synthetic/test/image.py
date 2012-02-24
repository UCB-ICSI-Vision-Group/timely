from common_imports import *
from common_mpi import *

from synthetic.image import *
from synthetic.sliding_windows import *

class TestImage:
  def test_get_windows(self):
    image = Image(size=(3,2))
    window_params = WindowParams(
        min_width=2,stride=1,scales=[1,0.5],aspect_ratios=[0.5])
    windows = image.get_windows(window_params)
    correct = np.array(
        [[ 0., 0., 2., 2.],
         [ 0., 0., 3., 2.],
         [ 1., 0., 3., 2.],
         [ 2., 0., 2., 2.],
         [ 0., 1., 2., 2.],
         [ 0., 1., 3., 2.],
         [ 1., 1., 3., 2.],
         [ 2., 1., 2., 2.],
         [ 0., 0., 3., 3.],
         [ 0., 0., 4., 3.]])
    assert(windows.shape[0] > 0)
    print(windows)
    assert(np.all(correct == windows))

  def test_get_random_windows(self):
    image = Image(size=(3,2))
    window_params = WindowParams(
        min_width=2,stride=1,scales=[1,0.5],aspect_ratios=[0.5])
    windows = image.get_random_windows(window_params,2)
    assert(windows.shape[0] == 2)
    windows = image.get_random_windows(window_params,3)
    assert(windows.shape[0] == 3)

  def test_get_windows_lots(self):
    t = time.time()
    image = Image(size=(640,480))
    window_params = WindowParams()
    window_params.min_width=10
    window_params.stride=8
    window_params.aspect_ratios=[0.5,1,1.5]
    window_params.scales=1./2**np.array([0,0.5,1,1.5,2])
    print(window_params)
    windows = image.get_windows(window_params)
    time_passed = time.time()-t
    print("Generating windows took %.3f seconds"%time_passed)
    print(np.shape(windows))
    print(windows[:10,:])
    rand_ind = np.random.permutation(np.shape(windows)[0])[:10]
    print(windows[rand_ind,:])
