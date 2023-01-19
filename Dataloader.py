import os
import zipfile
import pandas as pd
import sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from utils import crop_image

# https://www.kaggle.com/competitions/severstal-steel-defect-detection

def split_df(df, trsize):
  train_df, valid_df = train_test_split(df, train_size=trsize, shuffle=True, random_state=123)
  
  print('train_df length: ', len(train_df), 'valid_df length: ', len(valid_df)) 
  return train_df, valid_df    

def loader(file_path):
 
    """
    Arg:
      file_path: The path of repository that storing Raw datas.
      zip_path: The path of zip file.
    """
  # Extract zip file including image and label after download from Kaggle  
    if not os.path.isdir(file_path):
        os.makedirs(file_path)
    
    # Bulid data list in pandas Dataframe
    ## Colume of list: ImageId_ClassId, ClassId, EncodedPixels
    csv_path = os.path.join(file_path, 'train.csv')
    df = pd.read_csv(csv_path)
    
    df['ClassId'] = df['ClassId'].astype(int)
    df.sort_values(by=['ImageId'])
    df_tmp = df.pivot(index='ImageId', columns='ClassId', values='EncodedPixels')
    
    # Change colume to ImageId, e1, e2, e3, e4, count
    train_df2 = pd.DataFrame(df_tmp.index)
    train_df2['e1'] = df_tmp[:].iloc[:,0].values
    train_df2['e2'] = df_tmp[:].iloc[:,1].values
    train_df2['e3'] = df_tmp[:].iloc[:,2].values
    train_df2['e4'] = df_tmp[:].iloc[:,3].values
    train_df2.reset_index(inplace=True,drop=True)
    train_df2.fillna('',inplace=True); 
    train_df2['count'] = np.sum(train_df2.iloc[:,1:]!='',axis=1).values
    
    # Shuffle list
    train_df2= sklearn.utils.shuffle(train_df2, random_state=2000)
    train_df2.reset_index(inplace=True, drop=True)
    
    # Remove data without defect
    train_df2_drop = train_df2.drop(train_df2[train_df2['count']==0].index)
    train_df2_shuffle = sklearn.utils.shuffle(train_df2_drop, random_state=2000)
    train_df2_shuffle.reset_index(inplace=True,drop=True)
    
    # Split dataset to train dataset and vaildation dataset
    train_df, valid_df  = split_df(train_df2_shuffle, 0.9)
    
    if crop:
      crop_df_train = crop_image(train_df, subset='train', save_path='/content/kaggle', load_path='/content/kaggle/train_images', crop_w_ratio=0.25, crop_h_ratio=1.0)
      crop_df_val = crop_image(valid_df, subset='val', save_path='/content/kaggle', load_path='/content/kaggle/train_images', crop_w_ratio=0.25, crop_h_ratio=1.0)
      return crop_df_train, crop_df_val
    
    else:
      return train_df, valid_df


