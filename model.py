## 'segmentation_models' is refered from resource: https://github.com/qubvel/segmentation_models
import tensorflow as tf
import segmentation_models
from segmentation_models import Unet
from segmentation_models import FPN
from utils.metrics import F1_score
from utils.loss import BCEDiceLoss
from utils.loss import CCEDiceLoss

segmentation_models.set_framework('tf.keras')
segmentation_models.framework()

def model_seg(BACKBONE='efficientnetb2', input_shape=(256,1600,1) , classes=4, weight_path=''):
  
  """
  Arg:
    BACKBONE: 'resnet18', 'resnet34', 'resnet50'.
              'efficientnetb0', 'efficientnetb1', 'efficientnetb2',
              'efficientnetb3', 'efficientnetb4', 'efficientnetb5'
              
    input_shape: The shape of image.
    classes: The count of defect type (if including background, classes equal to 5).
    weight_path: Model weight file path.
  """
  
  # Define metrics
  F1_score = F1_score()
  
  # Define loss functions
  bce_dice_loss = BCEDiceLoss( class_nums=4, BCE_weights=0.5, Dice_weights=0.5, per_image_all=True ,threshold=0.5, soft_mode=False)
  cce_dice_loss = CCEDiceLoss( class_nums=5, CCE_weights=0.75, Dice_weights=0.25)
  
  # Define optimizers
  sgd = tf.keras.optimizers.SGD(lr=1E-3, momentum=0.9, nesterov=True)
  adam = tf.keras.optimizers.Adam(learning_rate=0.003)
  RMSprop = tf.keras.optimizers.RMSprop(learning_rate=0.005)
  
  # LOAD UNET WITH PRETRAINING FROM IMAGENET
  backbone = BACKBONE
  model = FPN(BACKBONE, input_shape=input_shape, classes=classes, activation='sigmoid', encoder_weights=None)
  model.compile(optimizer=adam, loss=bce_dice_loss, metrics=[F1_score()])
  
  if weight_path!='':
    model.load_weights(weight_path)
    
  return model
