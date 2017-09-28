# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 10:08:11 2017

@author: 黄逸杰
"""

import numpy as np
import matplotlib.pyplot as pl
from PIL import Image

numI=269
numS=48
root="E:/OneDrive/Projects/TianChiChallenge/data/val_CRF_Output/"
gt=np.zeros([numI,1,numS,numS,numS])
for i in range(numI):
    for j in range(numS):
        label=Image.open(root+str(i)+"/label"+str(i)+"_"+str(j)+".ppm")
        label = np.asarray(label)
        gt[i,0,j,:,:]=label[:,:,0]/255.0
gt=gt.astype(np.int8)
#for i in range(numI):
#    pl.imshow(gt[i,0,23,:,:])
#    pl.show()

np.save("E:/OneDrive/Projects/TianChiChallenge/data/val_CRF_label.npy",gt)