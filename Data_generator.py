import tensorflow as tf
import albumentations as A
import cv2
from utils.rle_parse import rle2mask
import numpy as np
import os 

#Image agumentation 
augmentations = A.Compose([ A.HorizontalFlip(p=0.5),
                            A.VerticalFlip(p=0.5),
                            ], p=0.75)

class DataGenerator(tf.keras.utils.Sequence):
  def __init__(self, df, 
               batch_size = 8, 
               subset="train",
               img_size=(height, width),
               img_path = None
               shuffle=False, 
               preprocess=None, 
               auto_sample_weights=False, 
               class_weights=[], 
               info={}, 
               augment_transform=None, 
               plot_mode=False, 
               label_sparse=False, 
               mask_label_mode='multi'):
    """
    Arg:
      df: training dataframe.
      img_path: The path of raw image repository.
      preprocess: The data preprocessing function.
    """
    super().__init__()
    self.df = df
    self.img_path = img_path
    self.shuffle = shuffle
    self.subset = subset
    self.batch_size = batch_size
    self.preprocess = preprocess
    self.info = info  
    self.augment_transform = augment_transform
    self.plot_mode = plot_mode
    self.label_sparse = label_sparse
    self.auto_sample_weights = auto_sample_weights
    self.class_weights = class_weights
    self.mask_label_mode = mask_label_mode
    
    if self.subset == "train":
      self.data_path = self.img_path + 'train_images/'
    elif self.subset == "val":
      self.data_path = self.img_path + 'train_images/'
    elif self.subset == "test":
      self.data_path = self.img_path + 'test_images/'
      
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
      X = np.empty((self.batch_size,img_size[0],img_size[1],3),dtype=np.float32)
    else:
      # store image and label in grayscale mode
      X = np.empty((self.batch_size,img_size[0],img_size[1],1),dtype=np.float32)

    # Convert rle to mask
    if self.mask_label_mode =='multi':
      if self.label_sparse:
        y = np.zeros((self.batch_size,img_size[0],img_size[1],1),dtype=np.float32)
      else:
        y = np.zeros((self.batch_size,img_size[0],img_size[1],5),dtype=np.float32)

    if self.mask_label_mode =='binary':
        y = np.zeros((self.batch_size,img_size[0],img_size[1],4),dtype=np.float32)

    indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
    for i,f in enumerate(self.df['ImageId'].iloc[indexes]):
      self.info[index*self.batch_size+i]=f

      if self.plot_mode:
        # load image in RGB
        image = cv2.imread(self.data_path + f, cv2.IMREAD_COLOR)
        image = image.astype(np.float32)
        image = cv2.resize(image,(img_size[1],img_size[0]),interpolation=cv2.INTER_NEAREST)
        X[i,] = image
        
      else:
        # load image in grayscale
        image = cv2.imread(self.data_path + f, cv2.IMREAD_GRAYSCALE)
        image = image.astype(np.float32)
        image = cv2.resize(image,(img_size[1],img_size[0]),interpolation=cv2.INTER_NEAREST)
        image = image[:,:,np.newaxis]
        X[i,] = image

      if self.subset == 'train':
        # Convert rle to mask
        if self.mask_label_mode =='multi':
          if self.label_sparse: 
            for j in range(4):
              rle_msk = rle2mask(self.df['e'+str(j+1)].iloc[indexes[i]])*(j+1)
              rle_msk = cv2.resize(rle_msk, (img_size[1],img_size[0]),interpolation=cv2.INTER_NEAREST)     
              y[i,rle_msk==1,0] = j+1

          else:
            for j in range(4):
              rle_msk = rle2mask(self.df['e'+str(j+1)].iloc[indexes[i]])
              rle_msk = rle_msk.astype(np.float32)
              rle_msk = cv2.resize(rle_msk, (img_size[1],img_size[0]),interpolation=cv2.INTER_NEAREST)
              y[i,:,:,j+1] = rle_msk
            y[i,:,:,0] = 1-y[i,:,:,1:5].sum(-1)

        if self.mask_label_mode =='binary':
          for j in range(4):
            rle_msk = rle2mask(self.df['e'+str(j+1)].iloc[indexes[i]])
            rle_msk = rle_msk.astype(np.float32)
            rle_msk = cv2.resize(rle_msk, (img_size[1],img_size[0]),interpolation=cv2.INTER_NEAREST)
            y[i,:,:,j] = rle_msk
              
        if self.preprocess!=None:
          X[i,] = self.preprocess(X[i,])       
        
        # Store sample weights array
        if self.auto_sample_weights:
          sample_weights = np.empty((self.batch_size,img_size[0],img_size[1],1),dtype=np.float32)

        # Agumentation
        if self.augment_transform != None:
          transformed = self.augment_transform(image=X[i,], mask=y[i,])
          X[i,] = transformed['image']
          y[i,] = transformed['mask'] 
          

        if self.subset == 'train':
          # Define sample weights
          if self.auto_sample_weights:
            assert self.label_sparse==True, 'Using Sample_weights need y data in sparse format'
            y_squ = np.squeeze(y[i,].reshape((-1,1)))        
            weight = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_squ), y=y_squ)        
            weight = np.array(weight)

            if len(self.class_weights)!=0:           
              sample_weight = weight * self.class_weights[np.unique(y[i,]).astype(int)] # Five class, class_weight = [0.2, 1.94, 10.47, 0.26, 1.71]
            else:
              sample_weight = weight  
            
            sample_weight = np.array(sample_weight)       
            np.set_printoptions(precision=5)

            for k in range(len(np.unique(y[i,]))):     
              sample_weights[i,y[i,]==np.unique(y[i,])[k]] = sample_weight[k]

    if self.subset == 'train':
      if self.auto_sample_weights: 
        return X, y, sample_weights
        
      else: return X, y

    elif self.subset == 'val': 
      return X, y
        
    elif: return X
