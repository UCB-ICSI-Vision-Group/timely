Timely Object Detection
===

Repository for upcoming NIPS 2012 submission on Timely Object Detection.

Files
---
- data/ is not tracked in the repository but should contain
  - results/: external repo for tracking generated results
  - VOC2007/: the 2007 VOC data
- timely/ is the code directory
- writeups/ has latex papers, posters, and slides.

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

Ideas
---
- using a saliency map:
  - generate sliding window proposals by combining a saliency map with the object likelihood map
  - score jumping window proposals with a saliency map
- plot recall vs. #windows for a class given other class presence or scene prior--to show that single-class detector efficiency improves with additional data received--which reduces its expected time
- general philosophy of particle filtering/coarse-to-fine vs. cascade: instead of rejecting regions a priori, look only in a few regions and let that guide the next places you look
- evaluate the "valley" of detectors: what's the minimal overlap required to successfully get the detection?
  : in terms of pascal overlap, for example
