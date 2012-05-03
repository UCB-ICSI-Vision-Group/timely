'''
Created on Feb 6, 2012

@author: tobibaum
'''

import numpy as np
import random
import math
from IPython import embed

def MeanShiftCluster(dataPts, bandWidth):
  dataPts = dataPts.T
  [numDim, numPts] = dataPts.shape
  numClust        = 0
  bandSq          = bandWidth**2
  initPtInds      = range(numPts)
  maxPos          = np.max(dataPts,1)
  minPos          = np.min(dataPts,1)
  boundBox        = maxPos-minPos
  sizeSpace       = np.linalg.norm(boundBox, ord=2)
  stopThresh      = 1e-3*bandWidth
  X2_clust       = []
  beenVisitedFlag = np.zeros((1,numPts))
  numInitPts      = numPts
  clusterVotes    = np.zeros((1,numPts))
  
  while numInitPts:
    rand = random.random()
    #rand = 0
    tempInd         = int(math.floor( (numInitPts-1e-6)*rand))
    stInd           = initPtInds[tempInd]                  
    myMean          = dataPts[:,stInd]                     
    myMembers       = []                                                             
    thisClusterVotes    = np.zeros((1,numPts))
    
    while 1:

      x_scale = dataPts[2,:] - dataPts[0,:] 
      y_scale = dataPts[3,:] - dataPts[1,:]
        
      datascales = np.vstack((x_scale, y_scale, x_scale, y_scale))
      datascales[datascales == 0] = 1
  
      m_scale_x = myMean[2] - myMean[0]
      m_scale_y = myMean[3] - myMean[1]
      meanscales = np.vstack((m_scale_x, m_scale_y, m_scale_x, m_scale_y))

      bsq = 1./np.tile(meanscales,(1,numPts))/datascales;
      sqDistToAll = sum(np.multiply(np.square((np.tile(myMean,(numPts,1)).T - dataPts)), bsq))

      #embed()
      inInds = (np.where(sqDistToAll < bandSq))[0]         
      inInds = inInds.tolist()     
      thisClusterVotes[:,inInds] = thisClusterVotes[:,inInds]+1      
      myOldMean   = myMean         
      myMean      = np.mean(dataPts[:,inInds],1)          
      
      myMembers   = np.hstack((myMembers,inInds))

      beenVisitedFlag[0,myMembers.tolist()] = 1
                             
      if np.linalg.norm(myMean-myOldMean, ord=2) < stopThresh:
                
        mergeWith = 0;
        for cN in range(numClust):
          x_scale = dataPts[2,cN] - dataPts[0,cN] 
          y_scale = dataPts[3,cN] - dataPts[1,cN]            
          datascales = np.vstack((x_scale, y_scale, x_scale, y_scale))
          
          #datascales = [(X2_clust(3,cN) - X2_clust(1,cN));(X2_clust(4,cN) - X2_clust(2,cN));(X2_clust(3,cN) - X2_clust(1,cN));(X2_clust(4,cN) - X2_clust(2,cN))];
          x_mean = myMean[2] - myMean[0]
          y_mean = myMean[3] - myMean[1]
          meanscales = np.vstack((x_mean, y_mean, x_mean, y_mean))
          
          bsq = 1/meanscales/datascales;
          
          distToOther = math.sqrt(sum(np.multiply(np.power((myMean - X2_clust[cN]),2), bsq)))
          if distToOther < bandWidth/2:
            mergeWith = cN;
            break

        if mergeWith > 0:
          X2_clust[mergeWith] = 0.5*(myMean+X2_clust[mergeWith])
          clusterVotes[mergeWith,:] = clusterVotes[mergeWith,:] + thisClusterVotes
        else:
          numClust = numClust+1;                   
          X2_clust.append(myMean)
          clusterVotes = np.vstack((clusterVotes,thisClusterVotes))        
        break     
    
    initPtInds = np.where(beenVisitedFlag == 0)[1]
    numInitPts = initPtInds.shape[0]
  return np.hstack(X2_clust).T

  

if __name__=='__main__':
  
  X = np.matrix([[-323.5, -402.5,   29.5,   95.5],
 [-327.5, -394.5,   25.5,  103.5],
 [-327.5, -402.5,   25.5,   95.5],
 [-331.5, -394.5,   21.5,  103.5],
 [-331.5, -402.5,  21.5 ,  95.5],
 [-335.5, -402.5,   17.5,   95.5]])
  X_clust = MeanShiftCluster(X,.25)
  X_true = np.matrix([[-329.5, -400 + 1/6., 23.5, 98 + 1/6.]])
  #assert(X_true.any() == X_clust.any())
  np.testing.assert_equal(np.asarray(X_clust), np.asarray(X_true))
  
  
  X2 = np.matrix([[ -327.5000, -378.5000,   25.5000,  119.5000],
 [ -331.5000, -382.5000,   21.5000,  115.5000],
 [ -331.5000, -390.5000,   21.5000,  107.5000],
 [ -327.5000, -478.5000,   25.5000,   19.5000]])
  
  X2_true = np.matrix([[-330 - 1/6. ,-384 + 1/6.,   23 - 1/6.,  114 + 1/6.],
 [-327.5      ,  -478.5  ,        25.5       ,   19.5       ]])
  X2_clust = MeanShiftCluster(X2,.25)
  
  X2_clust = np.sort(X2_clust, 0)
  X2_true = np.sort(X2_true, 0)
  np.testing.assert_equal(np.asarray(X2_clust), np.asarray(X2_true))
  