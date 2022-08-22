from utils.metrics import plot_confusion_matrix
import tensorflow as tf
from model import model_unet as model
from Dataloader import loader as loader
from Data_generator import DataGenerator as DataGenerator
from utils.preprocess import preprocess as preprocess
from sklearn.metrics import confusion_matrix
import numpy as np
from keras import backend as K

preprocess = preprocess
BACKBONE = 'resnet34'
input_shape = (256, 1600, 1)
classes = 5 # 4+1 (including background)
model = model(BACKBONE, input_shape, classes)
model.load_weights('./weights/Unet_cce_class_weight_aug2_preprocess255_rmsprop-e2-10_2.h5')

# Create dataframe for evaluating .
file_path = './kaggle'
train_df = loader(file_path)

# set the evaluation data to batch to avoid OOM.
# 100 image per batch.
blockLength = 100
idx = int(0.9*len(train_df))
cm_sum = np.zeros(shape=(5,5))
class_wise_iou_mean = []
class_wise_dice_score_mean = []

# Predict.
for begin in np.arange(idx,len(train_df),blockLength):
  test_batch = DataGenerator(train_df[begin:begin+blockLength:1], img_path=file_path, batch_size=1, shuffle=False, preprocess=preprocess, aug_mode=False)
  predict_mask = model.predict(test_batch, verbose=1)
  
  for i in range(len(predict_mask)):
    g_true = np.zeros(shape=(256,1600))
    test_batch_msk = np.squeeze(test_batch[i][1], axis=0)
    results = np.argmax(predict_mask[i,], axis=2)
    
    for t in range(5):
      g_true[test_batch_msk[:,:,t]==1] = t

    pred_f = K.flatten(results)
    g_true_f = K.flatten(g_true)
    
    # Calculate confusion matrix for each image per pixel.
    cm = confusion_matrix(g_true_f, pred_f, labels=[0,1,2,3,4]) 
    cm_sum = cm_sum + cm # Sum each image 
    
# Convert confusion matrix values to percentage
np.set_printoptions(suppress=True, precision=2)
cm_sum_mean = np.round((cm_sum.astype('float') / cm_sum.sum(axis=1).T[:,np.newaxis]),2)

# Plot confusion matrix
target_names = ['Defect1', 'Defect2', 'Defect3', 'Defect4', 'Background']
plot_confusion_matrix(cm_sum_mean, target_names=target_names, title_name='Confusion matrix_CCE_class-weight', cmap=None, normalize=True)
