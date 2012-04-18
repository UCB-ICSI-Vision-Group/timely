from synthetic.common_imports import *
from synthetic.common_mpi import *
import synthetic.config as config

from synthetic.image import *
from synthetic.sliding_windows import *

def test_constructor():
  image = Image(20,10,['A','B','C'],'test_image')
  assert(image.width == 20 and image.height == 10)
  assert(image.classes == ['A','B','C'])
  assert(image.name == 'test_image')

def test_load_json_data():
  data = {
      "name": "test_image",
      "size": [640,480],
      "objects": [
        {"class":"A", "bbox": [0,0,0,0], "diff": 0, "trun": 0},
        {"class":"B", "bbox": [1,1,1,1], "diff": 0, "trun": 0},
        {"class":"C", "bbox": [2,2,2,2], "diff": 0, "trun": 0}
      ]
  }
  classes = ['A','B','C']
  image = Image.load_from_json_data(data,classes)
  assert(image.width == 640 and image.height == 480)
  assert(image.classes == ['A','B','C'])
  assert(image.name == 'test_image')
  objects_df = DataFrame(np.array([
    [0,0,0,0,0,0,0],
    [1,1,1,1,1,0,0],
    [2,2,2,2,2,0,0]]), columns=BoundingBox.columns+['cls_ind','diff','trun'])
  assert(image.objects_df == objects_df)

def test_ground_truth_methods():
  None

def test_get_random_windows():
  image = Image(width=3,height=2,classes=[],name='test')
  window_params = WindowParams(
      min_width=2,stride=1,scales=[1,0.5],aspect_ratios=[0.5])
  windows = image.get_random_windows(window_params,2)
  assert(windows.shape[0] == 2)
  windows = image.get_random_windows(window_params,3)
  assert(windows.shape[0] == 3)

def test_get_windows_lots():
  t = time.time()
  image = Image(width=640,height=480,classes=[],name='test')
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
