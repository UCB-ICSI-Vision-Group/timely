Synthetic Approach README
===

Our efforts to be more intelligent about the object detection window proposal system are floundering on the PASCAL VOC dataset.
A synthetic dataset allows us to control properties of the data, so that:
* we can ensure that our solution to the problem is able to derive the data-generating statistics to inform the policy
  - we will experiment with different data-generating statistics and ensure that the algorithm is able to match a variety of them
* we can describe the scaling performance of our solution as the size of the dataset or the complexity of the problem is increased.

Dataset: Ground Truth
---
We want this module to be easily replaced by actual data.
Since our evaluation goal is the PASCAL VOC detection task, we stick to their ground truth format.
For each image, we have its dimensions and a list of bounding boxes of objects in it.
Each bounding box has a class label and difficulty and occlusion flags.

[ ] Provide function that generates a dataset of ground truth of given size
  - start with all images of the same dimension, don't use the flags, and have easily understandable cooccurence statistics.

Detections
---
This module replaces the actual detector.
Given an image and its ground truth, it generates a list of detections matching the format of our detector.
We can control the behavior and amount of noise in this system.
At a high level, we can describe the performance of a detector by it's Precision-Recall behavior, and the single AP number.

[ ] Provide function that generates a list of detections given a ground truth image object.
  - start with this module simply outputting the ground truth (perfect precision and recall; AP=1)
  - then provide a variation that introduces noise, for a desired AP of, say, 0.5.

[ ] Introduce the concept of time into detections, and make the less powerful detector take less time than the more powerful detector.

Policy
---
This module decides which action to take.
In the first implementation, actions consist of detectors to run.

[ ] Provide a hard-coded policy that is simply an ordering of detectors to run and evaluate it.

Evaluation
---
This module plots the performance of 
1) individual detectors
2) policies

[ ] Provide a function that evaluates the performance of an individual detector in the Precision-Recall regime.

[ ] Provide a function that evaluates the performance of a policy in the Precision-Recall-Time regime (AP-Time).
