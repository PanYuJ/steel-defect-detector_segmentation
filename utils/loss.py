from keras import backend

'''
Create loss function: 
BCE, CCE, Dice_loss

combined loss function:
CCEDiceLoss, BCEDiceLoss 
'''

def average(x, per_image=False, class_weights=None, **kwargs):
  
  if per_image:
      x = backend.mean(x, axis=0)
  if class_weights is not None:
      x = x * class_weights
  return backend.mean(x)

def _gather_channels(x, indexes, **kwargs):
    
  if backend.image_data_format() == 'channels_last':
      x = backend.permute_dimensions(x, (3, 0, 1, 2))
      x = backend.gather(x, indexes)
      x = backend.permute_dimensions(x, (1, 2, 3, 0))
  else:
      x = backend.permute_dimensions(x, (1, 0, 2, 3))
      x = backend.gather(x, indexes)
      x = backend.permute_dimensions(x, (1, 0, 2, 3))
  return x


def get_reduce_axes(per_image, **kwargs):
    
  axes = [1, 2] if backend.image_data_format() == 'channels_last' else [2, 3]
  if not per_image:
      axes.insert(0, 0)
  return axes


def gather_channels(*xs, indexes=None, **kwargs):   
  if indexes is None:
      return xs
  elif isinstance(indexes, (int)):
      indexes = [indexes]
  xs = [_gather_channels(x, indexes=indexes, **kwargs) for x in xs]
  return xs

def round_if_needed(x, threshold, soft_mode=True, class_nums=5, **kwargs):
  if soft_mode:
    x = tf.math.argmax(x, axis=-1)
    x = tf.one_hot(x, class_nums)
    return x   
  else:  
    if threshold is not None:
        x = backend.greater(x, threshold)
        x = backend.cast(x, backend.floatx())
    return x

def f_score(gt, pr, beta=1, class_weights=1, class_indexes=None, smooth=K.epsilon(), per_image=False, threshold=None, soft_mode=True, class_nums=4, per_image_all=True, **kwargs):  
# Args:
#     gt: ground truth 4D keras tensor (B, H, W, C) or (B, C, H, W)
#     pr: prediction 4D keras tensor (B, H, W, C) or (B, C, H, W)
#     class_weights: 1. or list of class weights, len(weights) = C
#     class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.
#     beta: f-score coefficient
#     smooth: value to avoid division by zero
#     per_image: if ``True``, metric is calculated as mean over images in batch (B),
#         else over whole batch
#     threshold: value to round predictions (use ``>`` comparison), if ``None`` prediction will not be round
# Returns:
#     F-score in range [0, 1]

  gt, pr = gather_channels(gt, pr, indexes=class_indexes, **kwargs)
  pr = round_if_needed(pr, threshold, soft_mode=soft_mode, class_nums=class_nums, **kwargs) 
  axes = get_reduce_axes(per_image, **kwargs)

  # calculate score
  tp = K.sum(tf.multiply(gt, pr), axis=axes) 
  fp = K.sum(pr, axis=axes) - tp
  fn = K.sum(gt, axis=axes) - tp

  score = ((1 + beta ** 2) * tp + smooth) \
          / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
  if per_image_all:
    score = average(score, per_image, class_weights, **kwargs)
  return score

class DiceLoss(tf.keras.losses.Loss):
    """
    Args:
        beta: Float or integer coefficient for precision and recall balance.
        class_weights: Array (``np.array``) of class weights (``len(weights) = num_classes``).
        class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.
        per_image: If ``True`` loss is calculated for each image in batch and then averaged,
        else loss is calculated for the whole batch.
        smooth: Value to avoid division by zero.
    Returns:
        A callable ``dice_loss`` instance. Can be used in ``model.compile(...)`` function`
        or combined with other losses.
    """

    def __init__(self, beta=1, class_weights=None, class_indexes=None, per_image=False, per_image_all=True, smooth=K.epsilon(), soft_mode=True, class_nums=5, threshold=None):
        super().__init__(name='dice_loss')
        self.beta = beta
        self.class_weights = class_weights if class_weights is not None else 1
        self.class_indexes = class_indexes
        self.per_image = per_image
        self.smooth = smooth
        self.soft_mode=soft_mode
        self.class_nums = class_nums
        self.threshold = threshold
        self.per_image_all = per_image_all

    def __call__(self, gt, pr, **kwargs):
        return 1 - f_score(
            gt,
            pr,
            beta=self.beta,
            class_weights=self.class_weights,
            class_indexes=self.class_indexes,
            smooth=self.smooth,
            per_image=self.per_image,
            soft_mode = self.soft_mode,
            class_nums = self.class_nums,
            threshold = self.threshold,
            per_image_all = self.per_image_all           
            )
      
 def categorical_crossentropy(gt, pr, class_weights=1., class_indexes=None, **kwargs):    
    
    gt, pr = gather_channels(gt, pr, indexes=class_indexes, **kwargs)

    # scale predictions so that the class probas of each sample sum to 1
    axis = 3 if backend.image_data_format() == 'channels_last' else 1
    pr /= backend.sum(pr, axis=axis, keepdims=True)

    # clip to prevent NaN's and Inf's
    pr = backend.clip(pr, backend.epsilon(), 1 - backend.epsilon())

    # calculate loss
    output = gt * backend.log(pr) * class_weights
    return - backend.mean(output)

class CategoricalCELoss(tf.keras.losses.Loss):
    """Creates a criterion that measures the Categorical Cross Entropy between the
    ground truth (gt) and the prediction (pr).
    .. math:: L(gt, pr) = - gt \cdot \log(pr)
    Args:
        class_weights: Array (``np.array``) of class weights (``len(weights) = num_classes``).
        class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.
    Returns:
        A callable ``categorical_crossentropy`` instance. Can be used in ``model.compile(...)`` function
        or combined with other losses.
    """

    def __init__(self, class_weights=None, class_indexes=None):
        super().__init__(name='categorical_crossentropy')
        self.class_weights = class_weights if class_weights is not None else 1
        self.class_indexes = class_indexes

    def __call__(self, gt, pr, **kwargs):
        return categorical_crossentropy(
            gt,
            pr,
            class_weights=self.class_weights,
            class_indexes=self.class_indexes,
            
        )
class CCEDiceLoss(tf.keras.losses.Loss):
  
    def __init__(self, beta=1, class_weights=None, class_indexes=None, per_image=False, smooth=K.epsilon(), threshold=None, soft_mode=True, per_image_all=True, class_nums=5, CCE_weights=0.5, Dice_weights=0.5):
        super().__init__(name='CCE_dice_loss')
        self.beta = beta
        self.class_weights = class_weights if class_weights is not None else 1
        self.class_indexes = class_indexes
        self.per_image = per_image
        self.smooth = smooth
        self.soft_mode = soft_mode
        self.class_nums = class_nums
        self.CCE_weights = CCE_weights
        self.Dice_weights = Dice_weights
        self.threshold = threshold
        self.per_image_all = per_image_all

        if self.CCE_weights !=0:
          self.cce_loss = CategoricalCELoss(class_weights=self.class_weights,
                             class_indexes=self.class_indexes)

        if self.Dice_weights !=0:
          self.dice_loss = DiceLoss(beta = self.beta,
                        class_weights = None,
                        class_indexes = self.class_indexes,
                        smooth = self.smooth,
                        per_image = self.per_image,
                        threshold = self.threshold,
                        per_image_all = self.per_image_all
                        )
             
    def __call__(self, gt, pr, **kwargs):

        if self.CCE_weights ==0:
          return self.cce_loss(gt, pr) * self.CCE_weights

        if self.Dice_weights ==0:
          return self.dice_loss(gt, pr) * self.Dice_weights

        cce = self.cce_loss(gt, pr)
        dice = self.dice_loss(gt, pr)
        loss = self.CCE_weights*cce + self.Dice_weights*dice
        return loss
        
def binary_crossentropy(gt, pr, class_weights=None, **kwargs):
    if class_weights is None:
      return K.mean(K.binary_crossentropy(gt, pr))
    else:
      return K.mean(K.binary_crossentropy(gt, pr)*class_weights)

class BinaryCELoss(tf.keras.losses.Loss):
    """Creates a criterion that measures the Binary Cross Entropy between the
    ground truth (gt) and the prediction (pr).
    .. math:: L(gt, pr) = - gt \cdot \log(pr) - (1 - gt) \cdot \log(1 - pr)
    Returns:
        A callable ``binary_crossentropy`` instance. Can be used in ``model.compile(...)`` function
        or combined with other losses.
    """

    def __init__(self, class_weights=None):
        super().__init__(name='binary_crossentropy')
        self.class_weights = class_weights

    def __call__(self, gt, pr):
        return binary_crossentropy(gt, pr, self.class_weights)

class BCEDiceLoss(tf.keras.losses.Loss):
  
    def __init__(self, beta=1, class_weights=None, class_indexes=None, per_image=False, smooth=K.epsilon(), threshold=0.5, soft_mode=False, per_image_all=True, class_nums=4, BCE_weights=0.5, Dice_weights=0.5):
        super().__init__(name='BCE_dice_loss')
        self.beta = beta
        self.class_weights = class_weights if class_weights is not None else 1
        self.class_indexes = class_indexes
        self.per_image = per_image
        self.smooth = smooth
        self.soft_mode = soft_mode
        self.class_nums = class_nums
        self.BCE_weights = BCE_weights
        self.Dice_weights = Dice_weights
        self.threshold = threshold
        self.per_image_all = per_image_all

        if self.BCE_weights !=0:
          self.bce_loss = BinaryCELoss(self.class_weights)

        if self.Dice_weights !=0:
          self.dice_loss = DiceLoss(beta = self.beta,
                        class_weights = None,
                        class_indexes = self.class_indexes,
                        smooth = self.smooth,
                        per_image = self.per_image,
                        threshold = self.threshold,
                        per_image_all = self.per_image_all,
                        soft_mode= self.soft_mode,
                        class_nums = self.class_nums

                        )
             
    def __call__(self, gt, pr, **kwargs):

        if self.BCE_weights ==0:
          return self.bce_loss(gt, pr) * self.BCE_weights

        if self.Dice_weights ==0:
          return self.dice_loss(gt, pr) * self.Dice_weights

        bce = self.bce_loss(gt, pr)
        dice = self.dice_loss(gt, pr)
        loss = self.BCE_weights*bce + self.Dice_weights*dice
        return loss
