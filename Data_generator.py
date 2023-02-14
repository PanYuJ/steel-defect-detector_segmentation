import tensorflow as tf
import albumentations as A
import cv2
from utils.rle_parse import rle2mask
import numpy as np
import os 

#Image agumentation 
augmentations = A.Compose([ A.HorizontalFlip(p=0.5),
                            A.VerticalFlip(p=0.5),
                            ])

class DataGenerator(tf.keras.utils.Sequence):
  def __init__(self, df, batch_size = 8, subset="train", data_path='', class_num=4, img_size=(256,1600),
               shuffle=False, preprocess=None, info={}, 
               augment_transform=None, plot_mode=False, label_sparse=False, mask_label_mode='multi'):
    
    super().__init__()
    self.df = df
    self.shuffle = shuffle
    self.subset = subset
    self.batch_size = batch_size
    self.preprocess = preprocess
    self.info = info  
    self.augment_transform = augment_transform
    self.plot_mode = plot_mode
    self.label_sparse = label_sparse
    self.mask_label_mode = mask_label_mode
    self.data_path = data_path
    self.img_size = img_size
    self.class_num = class_num

    self.on_epoch_end()
    
  def __len__(self):
    return int(np.floor(len(self.df) / self.batch_size))
  
  def on_epoch_end(self):
    self.indexes = np.arange(len(self.df))
    if self.shuffle == True:
      np.random.shuffle(self.indexes)
  
  def __getitem__(self, index):
    
    if self.plot_mode:
      # store image and label in RGB 
      X = np.empty((self.batch_size,self.img_size[0], self.img_size[1],3),dtype=np.float32)
    else:
      # store image and label in grayscale mode
      X = np.empty((self.batch_size,self.img_size[0], self.img_size[1],1),dtype=np.float32)

    # Convert rle to mask
    if self.mask_label_mode =='multi':
      if self.label_sparse:
        y = np.zeros((self.batch_size, self.img_size[0], self.img_size[1] ,1),dtype=np.float32)
      else:
        y = np.zeros((self.batch_size, self.img_size[0], self.img_size[1] ,self.class_num+1),dtype=np.float32)

    if self.mask_label_mode =='binary':
        y = np.zeros((self.batch_size, self.img_size[0], self.img_size[1] ,self.class_num),dtype=np.float32)

    indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
    for i,f in enumerate(self.df['ImageId'].iloc[indexes]):
      self.info[index*self.batch_size+i]=f

      if self.plot_mode:
        # load image in RGB
        image = cv2.imread(self.data_path + f, cv2.IMREAD_COLOR)
        image = image.astype(np.float32)
        image = cv2.resize(image,(self.img_size[1],self.img_size[0]),interpolation=cv2.INTER_NEAREST)
        X[i,] = image
        
      else:
        # load image in grayscale
        image = cv2.imread(self.data_path + f, cv2.IMREAD_GRAYSCALE)
        image = image.astype(np.float32)
        image = cv2.resize(image,(self.img_size[1],self.img_size[0]),interpolation=cv2.INTER_NEAREST)
        image = image[:,:,np.newaxis]
        X[i,] = image

      if self.subset == 'train' or self.subset == 'val':
        # Convert rle to mask
        if self.mask_label_mode =='multi':
          if self.label_sparse: 
            for j in range(self.class_num):
              rle_msk = rle2mask(self.df['e'+str(j+1)].iloc[indexes[i]])*(j+1)
              rle_msk = cv2.resize(rle_msk, (self.img_size[1],self.img_size[0]),interpolation=cv2.INTER_NEAREST)     
              y[i,rle_msk==1,0] = j+1

          else:
            for j in range(self.class_num):
              rle_msk = rle2mask(self.df['e'+str(j+1)].iloc[indexes[i]])
              rle_msk = rle_msk.astype(np.float32)
              rle_msk = cv2.resize(rle_msk, (self.img_size[1],self.img_size[0]),interpolation=cv2.INTER_NEAREST)
              y[i,:,:,j+1] = rle_msk
            y[i,:,:,0] = 1-y[i,:,:,1:self.class_num+1].sum(-1)

        if self.mask_label_mode =='binary':
          for j in range(self.class_num):
            rle_msk = rle2mask(self.df['e'+str(j+1)].iloc[indexes[i]])
            rle_msk = rle_msk.astype(np.float32)
            rle_msk = cv2.resize(rle_msk, (self.img_size[1],self.img_size[0]),interpolation=cv2.INTER_NEAREST)
            y[i,:,:,j] = rle_msk
              
        if self.preprocess!=None:
          X[i,] = self.preprocess(X[i,])       
        
        if self.subset == 'train': 
          # Agumentation
          if self.augment_transform != None:
            transformed = self.augment_transform(image=X[i,], mask=y[i,])
            X[i,] = transformed['image']
            y[i,] = transformed['mask'] 

    if self.subset == 'train' or self.subset == 'val':  
      return X, y

    else: return X
