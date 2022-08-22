## 'segmentation_models' is refered from resource: https://github.com/qubvel/segmentation_models
import tensorflow as tf
import segmentation_models
from segmentation_models import Unet

segmentation_models.set_framework('tf.keras')
segmentation_models.framework()

def model_unet(BACKBONE='resnet34', input_shape=(256,1600,1) , classes=5):
  
  """
  Arg:
    BACKBONE: 'resnet18', 'resnet34', 'resnet50'.
    input_shape: The shape of image.
    classes: The count of defect type (including background)
  """
  
  ## Loss function:
  class_weight = [10,10,1,6,1]

  # Define metrics
  iou_score = segmentation_models.metrics.IOUScore(per_image=True)

  # Define loss functions
  dice_loss = segmentation_models.losses.DiceLoss(per_image=True)
  cce = segmentation_models.losses.CategoricalCELoss(class_weights=class_weight) # CategoricalCELoss with class_weights
  cce_dice = dice_loss + cce # Categorical_cross_entropy + Dice_loss
  focal_cce = segmentation_models.losses.CategoricalFocalLoss()

  # Define optimizers
  sgd = tf.keras.optimizers.SGD(lr=1E-3, momentum=0.9, nesterov=True)
  adam = tf.keras.optimizers.Adam(learning_rate=0.005)
  RMSprop = tf.keras.optimizers.RMSprop(learning_rate=0.005)
  
  # LOAD UNET WITH PRETRAINING FROM IMAGENET

  backbone = BACKBONE
  model = Unet(BACKBONE, input_shape=input_shape, classes=classes, activation='softmax', encoder_weights=None)
  model.compile(optimizer=adam, loss=cce_dice, metrics=[iou_score])
  
  return model
