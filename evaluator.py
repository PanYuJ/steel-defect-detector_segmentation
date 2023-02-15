
import argparse
import os
import tensorflow as tf
import numpy as np
from keras import backend as K

from model import model_unet as model
from Dataloader import loader as loader
from Data_generator import DataGenerator as DataGenerator
from utils.metrics import plot_confusion_matrix
from utils.preprocess import preprocess as preprocess
from sklearn.metrics import confusion_matrix

def evaluator(opt):
  
  weights, img_size, backbone, class_num= opt.weights, opt.img_size, opt.backbone, opt.class_num
  
 # Create repository to store evaluating results.
  if not os.path.exists(save_dir):
    os.mkdir(opts.save_path)
    
  model = model(backbone, img_shape=img_size, classes=class_num, weights=weights)

  # Create dataframe for evaluating .
  file_path = './kaggle/train_images'
  _, val_df = loader(file_path, crop=False)

  # set the evaluation data to batch to avoid OOM.
  # 100 image per batch.
  blockLength = 100
  
  # Find threshold to get best f1_score
  thresholds = [np.round(x, 1) for x in np.arange(0, 1, 0.1)]
  result = pd.DataFrame(columns=['threshold', 'true_positives', 'false_positives', 'false_negatives', 'precision', 'recall', 'f1_score'])
  
  result['threshold'] = thresholds
  for i in range(len(result)):
    result['true_positives'].iloc[i] = [0,0,0,0]
    result['false_positives'].iloc[i] = [0,0,0,0]
    result['false_negatives'].iloc[i] = [0,0,0,0]
    result['precision'].iloc[i] = [0,0,0,0]
    result['recall'].iloc[i] = [0,0,0,0]
    result['f1_score'].iloc[i] = [0,0,0,0]
  
  # Predict.
  for begin in np.arange(0,len(val_df),blockLength):
    test_batch = DataGenerator(val_df[begin:begin+blockLength:1], 
                               img_path=file_path, 
                               subset='val', 
                               batch_size=1, 
                               shuffle=False, 
                               preprocess=preprocess, 
                               label_sparse=False, 
                               label_mode='binary'
                              )

    predict_mask = model.predict(test_batch, verbose=1)

    for i, true_batch in enumerate(test_batch):
      pred = predict_result[idx]
    true_batch_squ = np.squeeze(true_batch[1], axis=0)

    pred_f = pred.reshape(-1,4)
    true_f = true_batch_squ.reshape(-1,4)
   
    for idx, threshold in enumerate(thresholds):
      for i in range(4):
        pred = (pred_f[:,i] >threshold).astype('int')

        result['true_positives'].iloc[idx][i] += np.sum(pred * true_f[:,i], axis=0)
        result['false_positives'].iloc[idx][i] += np.sum((1 - true_f[:,i]) * pred, axis=0)
        result['false_negatives'].iloc[idx][i] += np.sum(true_f[:,i] * (1-pred), axis=0)

  for i in range(len(result)):
    for j in range(4):
      result['precision'][i][j] = np.round(result['true_positives'][i][j] / (result['true_positives'][i][j] + result['false_positives'][i][j] + K.epsilon()), 3)
      result['recall'][i][j] = np.round(result['true_positives'][i][j] / (result['true_positives'][i][j] + result['false_negatives'][i][j] + K.epsilon()), 3)
      result['f1_score'][i][j] = np.round(2*(result['precision'][i][j] * result['recall'][i][j]) / (result['precision'][i][j] + result['recall'][i][j] + K.epsilon()),3)

  result_list = result['f1_score'].values.tolist()
  result_list = np.array(result_list)
  for i in range(4):
    idx = np.argmax(result_list[:,i])
    print('Defect{}: best threshold:{}, F1_score:{}'.format(i+1, result['threshold'][idx], result_list[:,i][idx]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='./result', help='save path for training weight and log.csv')
    parser.add_argument('--weights', type=str, default='', help='training weights path')
    parser.add_argument('--img_size', nargs='+', type=int, default=[256, 1600, 1], help='[height, width, channel] image sizes')
    parser.add_argument('--backbone', type=str, default='efficientnetb2', help='model backbone. efficientnetb0-5, resnet34')
    parser.add_argument('--class_num', type=int, default=4)
    opt = parser.parse_args()
