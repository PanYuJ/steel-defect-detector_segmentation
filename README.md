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

- Four defects type in dataset. The image of each defect type is shown as below.

  - Defect 1.  
<img src="images/defect1-1.png" align="center" width="50%"/><img src="images/defect1-2.png" align="center" width="50%"/>
  - Defect 2.  
<img src="images/defect2-1.png" align="center" width="50%"/><img src="images/defect2-2.png" align="center" width="50%"/>
  - Defect 3.  
<img src="images/defect3-1.png" align="center" width="50%"/><img src="images/defect3-2.png" align="center" width="50%"/>
  - Defect 4.  
<img src="images/defect4-1.png" align="center" width="50%"/><img src="images/defect4-2.png" align="center" width="50%"/>

- The percentage of each defect are shown as below. The percentage defect 3 is more than half of dataset. 
<img src="images/data_statistics.png" align="center" width="60%"/>

## Model
- ResUnet
<img src="images/defect1-2.png" align="center" width="50%"/>

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
