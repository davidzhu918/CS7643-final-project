# CS7643-final-project

- Anurag Sen (asen48@gatech.edu)
- Jie Song (jsong407@gatech.edu)
- Zhiling Zhou (zhiling_zhou@gatech.edu)
- Zixiang Zhu (zzhu72@gatech.edu)

In this project, we present using various deep-learning approaches to perform image panoptic segmentation on MS-COCO dataset. Specifically, we explored both a bottom-up approach (Panoptic DeepLab) and a top-down approach (Panoptic FPN) for image panoptic segmentation tasks. We experimented with various modifications to the existing state-of-the-art models and discovered promising results.

## Implementation of a custom Panoptic DeepLab with single ASPP head:

Relevant code:

  ./panoptic_seg_single_aspp.py

## Implementation of a modified Resnet backbone 

Relevant code can be in the following directory  

  ./experiments  
  --panoptic_fpn    
    --panoptic_fpn_R_50_3x_one_more_bottle_neck  
      --resnet.py
    
## Implementation of a VGG backbone 

Relevant code can be found in the following directory  

  ./experiments  
  --panoptic_fpn  
    --panoptic_fpn_VGG_backbone  
      --fpn.py  
      --vgg19.py

## Implementation of a Res2Net backbone:

Relevant code can be found in the following directories:

  ./experiments
  --panoptic_fpn
    --panoptic_fpn_r_101_3x

Entire mask-rcnn folder in root directory can be ignored for results.
It was largely using a tutorial in order to understand the implementation initially.

## Evaluation of DETR

Relevant code can be found in the following directories:

  ./DETR_panoptic_eval.ipynb
