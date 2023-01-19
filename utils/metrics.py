import itertools
import matplotlib.pyplot as plt
import numpy as np

# Define metrics
class F1_score(tf.keras.metrics.Metric):

  def __init__(self, name='F1_score', **kwargs):
    super().__init__(name=name, **kwargs)
    self.true_positives = self.add_weight(name='tp', initializer='zeros')
    self.false_positives = self.add_weight(name='fp', initializer='zeros')
    self.false_negatives = self.add_weight(name='fn', initializer='zeros')

  def update_state(self, y_true, y_pred, threshold=0.5, sample_weight=None):
    y_true = tf.cast(y_true, 'float')
    y_pred = tf.cast(tf.greater(tf.cast(y_pred, 'float'), threshold), 'float')

    self.true_positives.assign_add(tf.reduce_sum(tf.multiply(y_true,y_pred)))
    self.false_positives.assign_add(tf.reduce_sum((1 - y_true) * y_pred))
    self.false_negatives.assign_add(tf.reduce_sum(y_true * (1 - y_pred)))

  def result(self):
    p = self.true_positives / (self.true_positives + self.false_positives + K.epsilon())
    r = self.true_positives / (self.true_positives + self.false_negatives + K.epsilon())

    f1 = 2*( p * r ) / (p + r + K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

  def reset_state(self):
    self.true_positives.assign(0)
    self.false_positives.assign(0)
    self.false_negatives.assign(0)

# Plot confusion_matrix
def plot_confusion_matrix(cm, target_names, title_name=None, cmap=None, normalize=True, result_name='result.png'):
  
  accuracy = np.trace(cm) / float(np.sum(cm))
  misclass = 1 - accuracy

  if cmap is None:
      cmap = plt.get_cmap('Blues')

  plt.figure(figsize=(8, 6))
  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title_name)
  plt.colorbar()

  if target_names is not None:
      tick_marks = np.arange(len(target_names))
      plt.xticks(tick_marks, target_names, rotation=45)
      plt.yticks(tick_marks, target_names)

  if normalize:
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


  thresh = cm.max() / 1.5 if normalize else cm.max() / 2
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      if normalize:
          plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
      else:
          plt.text(j, i, "{:,}".format(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")


  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
  plt.savefig(result_name) 
  plt.show()


  


  
