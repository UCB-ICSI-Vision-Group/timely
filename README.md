Timely Object Detection
===

This repository contains the working code for the NIPS 2012 submission on Timely Object Detection.

This is a private repository, and is superseded by the [public code release](https://github.com/sergeyk/timely_object_recognition) (with supporting `skvisutils` and `skpyutils` repos) for the paper.
However, this repo contains some code that was not published, and should be kept for archival purposes.

Files
---
- data/ is not tracked in the repository but should contain
  - results/: external repo for tracking generated results
  - VOC2007/: the 2007 VOC data
- synthetic/ is the code directory
- fastInf/ is external inference code

Overview
---
The project focuses on a multi-class detection policy.
The overall motivation is the performance vs. time evaluation.

A. Single-class detector: window proposals

The evaluation method is plotting recall-#windows curves, keeping track of how long feature extraction and generating the windows take

- A.1. Jumping window proposals
  - mostly following Vijay's implementation
    - storing scale-invariant offsets instead of actual pixel values
  - ranked either by feature discriminativess or window parameter object likelihood (or combination)
- A.2. Sliding window proposals
  - parametrized in terms of min_width, stride, aspect_ratios, and scales.
  - these parameters are set from data in different ways
  - ranked either in fixed-order (scanning) or according to their object likelihood

B. Single-class detector: classification

  - Chi-square (with Subhransu's trick) vs. RBF
  - generate performance vs. time curves by timing how long it takes to process N window proposals from the first stage

C. Multi-class policy

Goal is to be able to learn a closed-loop policy

  - Given a detector for each class, what is the most efficient way to search through the classes?
  - What if there are multiple detectors per class (use the DPM detector as one of them, and ours as another)
