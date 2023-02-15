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

## Dataset

- Four defects type in dataset. The example of each defect type is shown as below.

  - Defect 1.  
  <img src="images/defect1-1.png" align="center" width="50%"/>

  - Defect 2.  
  <img src="images/defect2-1.png" align="center" width="50%"/>

  - Defect 3.  
  <img src="images/defect3-1.png" align="center" width="50%"/>

  - Defect 4.  
  <img src="images/defect4-1.png" align="center" width="50%"/>

- The count of each defect are shown as below. This dataset is imbalanced.  
  <img src="images/data_static.jpg" align="center" width="40%"/>

## Summary
- Trying two model:

|[Unet](https://arxiv.org/abs/1505.04597)|[Feature Pyramid Network (FPN)](http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf)|
|:--:|:--:|
|<img src="images/unet.png" align="center" width="70%"/>|<img src="images/fpn.png" align="center" width="70%"/>|

- Using ``EfficientNet`` for encoder to extract feature.
- Activation function: ``Sigmoid``
- Add class_weights for each class to balance dataset.
- Loss function: ``binary_cross_entropy`` + ``Dice loss``  
- Optimizer: ``Adam``

## Evaluation
 - Create confusion matrix for each pixel. It is shown as below. 
 - The precision of model using CCE is better than using CCE+Dice in this case.
 - 
|CCE|CCE+Dice|
|:--:|:--:|
|<img src="images/cm_cce_class-weight.png" align="center" width="70%"/>|<img src="images/cm_cce_dice_class-weight.png" align="center" width="70%"/>|

## License
[MIT](https://choosealicense.com/licenses/mit/)
