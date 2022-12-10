# CS7643-final-project





# Update 



Dec. 10, 2022

Jie: working on revising the backbones of Panoptic-FPN. I am thinking about creating deeper Resnets based on the current architecture. 

* Implemented a revised Resnet backbone (added one more bottleneck layer) and trained the model from scratch (no pre-trained weights).
  * Training iteration: 270,000
  * Base learning rate: 0.0025 
  * Batch size: 2 
  * Training time: around 24 hours 
* Implemented a VGG19-based backbone and trained the model from scratch (no pre-trained weights). 
  * Training iteration: 90,000
  * Base learning rate: 0.01 
  * Batch size: 8 
  * Training time: around 24 hours (much slower than the Resnet-based model)
* Takeaway: reproduced the implementation of various fundamental panoptic-FPN (feature pymaid network) architectures and understood the implementation details of panoptic segmentation.  

