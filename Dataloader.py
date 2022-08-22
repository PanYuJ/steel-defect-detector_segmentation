import os
import zipfile
import pandas as pd
import sklearn
import numpy as np

# https://www.kaggle.com/competitions/severstal-steel-defect-detection

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
    train_df2_shuffle = sklearn.utils.shuffle(train_df2, random_state=2000)
    train_df2_shuffle.reset_index(inplace=True, drop=True)
    
    return train_df2_shuffle


