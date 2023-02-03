# Steel-defect-detector_segmentation

![header](images/header.png)
Source from Kaggle: [Severstal: Steel Defect Detection](https://www.kaggle.com/c/severstal-steel-defect-detection)

## Requirements

1) keras_applications
2) image-classifiers
3) efficientnet
4) segmentation_models
- Source from: https://github.com/qubvel/segmentation_models

```bash
pip install -r requirements.txt
```

## Introduction

- Four defects type in dataset. The example of each defect type is shown as below.

  - Defect 1.  
  <img src="images/defect1-1.png" align="center" width="50%"/>

  - Defect 2.  
  <img src="images/defect2-1.png" align="center" width="50%"/>

  - Defect 3.  
  <img src="images/defect3-1.png" align="center" width="50%"/>

  - Defect 4.  
  <img src="images/defect4-1.png" align="center" width="50%"/>

- The percentage of each defect are shown as below. The percentage defect 3 is more than half of dataset. 
<img src="images/data_static.png" align="center" width="60%"/>

- Analyze mask area sizes  
<img src="images/defect_pixel_jistogram1.png" align="center" width="50%"/>

- Plot images with large mask areas(>200000 pixel) picked by random index.  Defect 3 masks seem to contain a lot of empty space without any defects.  
<img src="images/image with large mask_defect3.png" align="center" width="50%"/>

## Define trainng model
- ResUnet (Residual U-Net)
<img src="images/unet model.png" align="center" width="80%"/>

- Loss function
  - Add class_weights for each class to balance dataset.
  - ``Catagorical_cross_entropy`` + ``Dice loss``
  - ``Catagorical_cross_entropy`` 
- Optimizer
  - ``Adam``
   
## Evaluation
 - Create confusion matrix for each pixel. It is shown as below. 
 - The precision of model using CCE is better than using CCE+Dice in this case.
 - 
|CCE|CCE+Dice|
|:--:|:--:|
|<img src="images/cm_cce_class-weight.png" align="center" width="70%"/>|<img src="images/cm_cce_dice_class-weight.png" align="center" width="70%"/>|

## License
[MIT](https://choosealicense.com/licenses/mit/)
