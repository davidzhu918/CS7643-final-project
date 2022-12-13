# CS7643-final-project



In this project, we present using various deep-learning approaches to perform image panoptic segmentation on MS-COCO dataset. Specifically, we explored both a bottom-up approach (Panoptic DeepLab) and a top-down approach (Panoptic FPN) for image panoptic segmentation tasks. We experimented with various modifications to the existing state-of-the-art models and discovered promising results. 

## Implementation of a modified Resnet backbone 

Relevant code can be in the following directory  

  ./experiments  
  --panoptic_fpn
    --panoptic_fpn_R_50_3x_one_more_bottle_neck  
      --resnet.py
    
## Impletement of a VGG backbone 

Relevant code can be found in the following directory  

  ./experiments  
  --panoptic_fpn  
    --panoptic_fpn_VGG_backbone  
      --fpn.py  
      --vgg19.py
