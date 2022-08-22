import tensorflow as tf
import imgaug.augmenters as iaa
import cv2
from utils.rle_parse import rle2mask
import numpy as np
import os 

#Image agumentation 
aug2 = iaa.Fliplr(0.5)
aug3 = iaa.Flipud(0.5)
aug4 = iaa.Emboss(alpha=(1), strength=1)
aug5 = iaa.DirectedEdgeDetect(alpha=(0.8), direction=(1.0))
aug6 = iaa.BilateralBlur(d=(3, 10), sigma_color=(10, 250), sigma_space=(10, 250))

class DataGenerator(tf.keras.utils.Sequence):
  def __init__(self, df, img_path, batch_size = 8, subset="train", shuffle=False, preprocess=None, info={}, aug_mode=False):
    """
    Arg:
      df: training dataframe.
      img_path: The path of raw image repository.
      preprocess: The data preprocessing function.
    """
    super().__init__()
    self.df = df
    self.shuffle = shuffle
    self.subset = subset
    self.batch_size = batch_size
    self.preprocess = preprocess
    self.info = info
    self.aug_mode = aug_mode
    self.img_path = img_path
    
    if self.subset == "train":
      self.data_path = os.path.join(img_path,'train_images/')
    elif self.subset == "test":
      self.data_path = os.path.join(img_path,'test_images/')
    self.on_epoch_end()

  def __len__(self):
    return int(np.floor(len(self.df) / self.batch_size))
  
  def on_epoch_end(self):
    self.indexes = np.arange(len(self.df))
    if self.shuffle == True:
      np.random.shuffle(self.indexes)
  
  def __getitem__(self, index):
    # store image and label in grayscale mode 
    X = np.zeros((self.batch_size,256,1600,1),dtype=np.float32)
    y = np.zeros((self.batch_size,256,1600,5),dtype=np.float32)
    
    indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
    for i,f in enumerate(self.df['ImageId'].iloc[indexes]):
      self.info[index*self.batch_size+i]=f          
      image = cv2.imread(self.data_path + f, cv2.IMREAD_GRAYSCALE)
      image = cv2.resize(image,(1600,256))
      image = image[:,:,np.newaxis]
      X[i,] = image

      if self.subset == 'train': 
        for j in range(4):
          y[i,:,:,j] = rle2mask(self.df['e'+str(j+1)].iloc[indexes[i]])
        y[i,:,:,4] = 1-y[i,:,:,:4].sum(-1)

        if self.aug_mode == True:       
          a = np.random.uniform()
          if a<0.3:
            X[i,] = aug2.augment_image(X[i,])
            y[i,] = aug2.augment_image(y[i,])
          elif a<0.6:
            X[i,] = aug3.augment_image(X[i,])
            y[i,] = aug3.augment_image(y[i,])

     
    if self.preprocess!=None:
      X = self.preprocess(X)

    if self.subset == 'train':
      return X, y

    else: return X
