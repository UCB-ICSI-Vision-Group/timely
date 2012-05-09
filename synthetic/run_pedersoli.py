#!/usr/bin/env python

import time,os,sys
import argparse
import numpy as np

from ctfdet import util2
from ctfdet import pyrHOG2
from ctfdet import pyrHOG2RL

from synthetic.dataset import Dataset
from synthetic.bounding_box import BoundingBox
import synthetic.util as ut

from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

def main():
  parser = argparse.ArgumentParser(description='Execute different functions of our system')
  parser.add_argument('--first_n', type=int,
      help='only take the first N images in the datasets')
  parser.add_argument('--name', help='name for this run',
      default='default',choices=['default','nolateral','nohal','halfsize'])
  parser.add_argument('--force', action='store_true', 
      default=False, help='force overwrite')

  args = parser.parse_args()
  print(args)

  #configuration class
  class config(object):
    pass
  cfg=config()
  cfg.testname="../ctfdet/data/finalRL/%s2_test"  #object model
  cfg.bottomup=False                      #use complete search
  cfg.resize=1.0                          #resize the input image
  cfg.hallucinate=True                    #use HOGs up to 4 pixels
  cfg.initr=1                             #initial radious of the CtF search
  cfg.ratio=1                             #radious at the next levels
  cfg.deform=True                         #use deformation
  cfg.usemrf=True                         #use lateral constraints

  if args.name=='default':
    cfg
    # sticking with the default params
  elif args.name == 'nolateral':
    cfg.usemrf = False
  elif args.name == 'nohal':
    cfg.hallucinate = False
  elif args.name == 'halfsize':
    cfg.resize = 0.5

  # fuck it, do both
  test_datasets = ['val','test','train']
  for test_dataset in test_datasets:
    # Load the dataset
    dataset = Dataset('full_pascal_'+test_dataset)
    if args.first_n:
      dataset.images = dataset.images[:args.first_n]
    
    # create directory for storing cached detections
    dirname = './temp_data'
    if os.path.exists('/u/sergeyk'):
      dirname = '/u/vis/x1/sergeyk/object_detection'
    dirname = dirname+'/ctfdets/%s'%(args.name)
    ut.makedirs(dirname)

    num_images = len(dataset.images)
    for img_ind in range(comm_rank, num_images, comm_size):
      # check for existing det
      image = dataset.images[img_ind]
      filename = os.path.join(dirname,image.name+'.npy')
      if os.path.exists(filename) and not args.force:
        #table = np.load(filename)[()]
        continue

      #read the image
      imname = dataset.get_image_filename(img_ind) 
      img=util2.myimread(imname,resize=cfg.resize)    
      #compute the hog pyramid
      f=pyrHOG2.pyrHOG(img,interv=10,savedir="",notsave=True,notload=True,hallucinate=cfg.hallucinate,cformat=True)

      #for each class
      all_dets = []
      for ccls in dataset.classes:
        t=time.time()
        cls_ind = dataset.get_ind(ccls)
        print "%s Img %d/%d Class: %s"%(test_dataset, img_ind+1,num_images,ccls)
        #load the class model
        m=util2.load("%s%d.model"%(cfg.testname%ccls,7))
        res=[]
        t1=time.time()
        #for each aspect
        for clm,m in enumerate(m):
          #scan the image with left and right models
          res.append(pyrHOG2RL.detectflip(f,m,None,hallucinate=cfg.hallucinate,initr=cfg.initr,ratio=cfg.ratio,deform=cfg.deform,bottomup=cfg.bottomup,usemrf=cfg.usemrf,small=False,cl=clm))
        fuse=[]
        numhog=0
        #fuse the detections
        for mix in res:
            tr=mix[0]
            fuse+=mix[1]
            numhog+=mix[3]
        rfuse=tr.rank(fuse,maxnum=300)
        nfuse=tr.cluster(rfuse,ovr=0.3,inclusion=False)
        #print "Number of computed HOGs:",numhog
        time_elapsed = time.time()-t
        print "Elapsed time: %.3f s"%time_elapsed

        bboxes = [nf['bbox'] for nf in nfuse]
        scores = [nf['scr'] for nf in nfuse]
        assert(len(bboxes)==len(scores))
        if len(bboxes)>0:
          arr = np.zeros((len(bboxes),7))
          arr[:,:4] = BoundingBox.convert_arr_from_corners(np.array(bboxes))
          arr[:,4] = scores
          arr[:,5] = time_elapsed
          arr[:,6] = cls_ind
          all_dets.append(arr)
      cols = ['x','y','w','h','score','time','cls_ind']
      if len(all_dets)>0:
        all_dets = np.concatenate(all_dets,0)
      else:
        all_dets = np.array([])
      table = Table(all_dets,cols)
      np.save(filename,table)

if __name__ == '__main__':
  main()
