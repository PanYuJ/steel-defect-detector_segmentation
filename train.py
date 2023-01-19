import argparse
from model import model_seg
from Dataloader import loader as loader
from Data_generator import DataGenerator
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
import os
from utils.preprocess import preprocess as preprocess
import tensorflow as tf

def train(opt):
  save_dir, epochs, batch_size, weights, img_size, backbone, class_num= \
        opt.save_dir, opt.epochs, opt.batch_size, opt.weights, opt.img_size, opt.backbone, opt.class_num
  
  # Create repository to store training weight
  wdir = save_dir / 'weights'
  wdir.mkdir(parents=True, exist_ok=True)  # make dir
  best = wdir / 'best.pt'
  
  # Create repository to store model training log
  logdir = save_dir / 'log'
  logdir.mkdir(parents=True, exist_ok=True)  # make dir
  log = wdir / 'log.csv'
  
  # Define callback function
  reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                factor=0.5,
                                patience=5,
                                verbose=True,
                                mode='min',
                                cooldown=5,
                                min_lr=1e-09,
                                eps=1e-09 
                                )

  CP_callback = ModelCheckpoint(filepath=best,
                                save_weights_only=True,
                                monitor='val_F1_score',
                                mode='max',
                                save_best_only=True
                               )

  log_csv = CSVLogger(log, separator=',', append=False)

  es = EarlyStopping(monitor='val_loss', patience= 10 , mode = 'min')
  
  # Create model for segmentation
  input_shape = img_size
  # classes = 5 # 4+1 (including background)

  weight_path = weights
  model = model_seg(backbone, input_shape=img_size, classes=class_num, weight=weights)

  file_path = './kaggle'
  
  # Create dataframe
  train_df, val_df = loader(file_path, crop=True)

  # Define custom callback functions
  callback_list = [reduce_lr, CP_callback, log_csv, es]

  # Define preprocess for data
  preprocess_input = preprocess
  # Prepare training data and validation data.

  augmentations = A.Compose([ A.HorizontalFlip(p=0.5),
                              A.VerticalFlip(p=0.5),
                              ], p=0.75)

  train_batches = DataGenerator(df=train_df, 
                                img_path=file_path, 
                                subset='train', 
                                shuffle=True,  
                                preprocess=preprocess_input, 
                                augment_transform=augmentations, 
                                label_sparse=False, 
                                label_mode='binary'
                               )

  val_batches = DataGenerator(df=val_df, 
                              img_path=file_path, 
                              subset='val', 
                              shuffle=False, 
                              preprocess=preprocess_input, 
                              label_sparse=False, 
                              label_mode='binary'
                             )

  history = model.fit(train_batches, validation_data = val_batches, epochs=50 ,verbose=1, callbacks=callback_list)
opt.save_dir, opt.epochs, opt.batch_size, opt.weights, opt.img_size, opt.backbone_name, opt.class_num
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='./result', help='save path for training weight and log.csv')
    parser.add_argument('--epochs', type=int, default=50,)
    parser.add_argument('--batch_size', type=int, default=15, help='total batch size for all GPUs')
    parser.add_argument('--weights', type=str, default='', help='hyperparameters path')
    parser.add_argument('--img_size', nargs='+', type=int, default=[256, 1600, 1], help='[height, width, channel] image sizes')
    parser.add_argument('--backbone_name', type=str, default='efficientnetb2', help='model backbone. efficientnetb0-5, resnet34')
    parser.add_argument('--class_num', type=int, default=4)
    opt = parser.parse_args()
