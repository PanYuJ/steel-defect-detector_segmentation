from utils.metrics import plot_confusion_matrix
import tensorflow as tf
from model import model_unet as model
from Dataloader import loader as loader
from Data_generator import DataGenerator as DataGenerator
from utils.preprocess import preprocess as preprocess
from sklearn.metrics import confusion_matrix
import numpy as np
from keras import backend as K

def evaluator(opt):
  
  result_save,  weights, img_size, backbone, class_num= \
        opt.result_save, opt.weights, opt.img_size, opt.backbone, opt.class_num,
  
 # Create repository to store training weight
  mdir = result_save / 'matrix'
  mdir.mkdir(parents=True, exist_ok=True)  # make dir
  result = mdir / 'confusion_matrix.png'
  
  model = model(backbone, img_shape=img_size, classes=class_num, weights=weights)

  # Create dataframe for evaluating .
  file_path = './kaggle'
  _, val_df = loader(file_path, crop=True)

  # set the evaluation data to batch to avoid OOM.
  # 100 image per batch.
  blockLength = 100
  cm_sum = np.zeros(shape=(5,5))
  
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

    for i, predict_mask_batch in enumerate(predict_mask):
      test_batch_msk = np.squeeze(test_batch[i][1], axis=0)   
      g_true = np.zeros(shape=(img_size[0],img_size[1]))

      for t in range(class_num):
        g_true[test_batch_msk[:,:,t]==1] = t+1
        thresholded = np.zeros_like(predict_mask_batch[:,:,t])
        thresholded[predict_mask_batch[:,:,t]>thresholds[t]] = 1
        predict_mask_thred[thresholded==1] = t+1
        
      pred_f = K.flatten(results)
      g_true_f = K.flatten(g_true)

      # Calculate confusion matrix for each image per pixel.
      cm = confusion_matrix(g_true_f, pred_f, labels=[i for i in range(class_num)]) 
      cm_sum = cm_sum + cm # Sum each image 
     
  # Convert confusion matrix values to percentage
  np.set_printoptions(suppress=True, precision=2)
  cm_sum_mean = np.round((cm_sum.astype('float') / cm_sum.sum(axis=1).T[:,np.newaxis]),2)

  # Plot confusion matrix
  target_names = ['Background', 'Defect1', 'Defect2', 'Defect3', 'Defect4']
  plot_confusion_matrix(cm_sum_mean, target_names=target_names, title_name='Confusion matrix', cmap=None, normalize=True, result_name=result_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='./result', help='save path for training weight and log.csv')
    parser.add_argument('--weights', type=str, default='', help='training weights path')
    parser.add_argument('--img_size', nargs='+', type=int, default=[256, 1600, 1], help='[height, width, channel] image sizes')
    parser.add_argument('--backbone', type=str, default='efficientnetb2', help='model backbone. efficientnetb0-5, resnet34')
    parser.add_argument('--class_num', type=int, default=4)
    parser.add_argument('--result_save', type=str, default='./result)
    opt = parser.parse_args()
