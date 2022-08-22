import numpy as np
import pandas as pd

def rle2mask(rle):
    # CONVERT RLE TO MASK 
    if (pd.isnull(rle))|(rle==''): 
        return np.zeros((256,1600) ,dtype=np.uint8)
    
    height= 256
    width = 1600
    mask= np.zeros( width*height ,dtype=np.uint8)

    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]-1
    lengths = array[1::2]    
    for index, start in enumerate(starts):
        mask[int(start):int(start+lengths[index])] = 1
    return mask.reshape( (height,width), order='F' )[::1,::1]

def mask2rle(mask):
    startEnd = np.diff(np.concatenate(([0],mask.T.flatten(),[0])))
    starts   = np.where(startEnd== 1)[0]
    if len(starts) == 0:
        return ''
    ends     = np.where(startEnd==-1)[0]
    length   = ends - starts
    starts  += 1    # it seems the data set pixel index starts at 1
    return ' '.join(['{} {}'.format(s,l) for s,l in zip(starts,length)])
