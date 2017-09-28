# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 15:29:51 2017

@author: 黄逸杰
"""
import numpy as np
import matplotlib.pyplot as pl
from PIL import Image
im=np.load("E:/OneDrive/Projects/TianChiChallenge/data/val_patch_48.npy")
gt = np.load("E:/OneDrive/Projects/TianChiChallenge/data/val_mask_48.npy")*255
for iid in range(im.shape[0]):
    out=np.concatenate((im[iid,23],gt[iid,23]),axis=1)
    out=Image.fromarray(out)
    out.save("E:/OneDrive/Projects/TianChiChallenge/data/train_for_CRF/compare/"+str(iid)+"_"+'23'+".gif")  
    for i in range(48):
        img=Image.fromarray(im[iid,i])
        gti=Image.fromarray(gt[iid,i])
        #out.save("E:/OneDrive/Projects/TianChiChallenge/data/train_for_CRF/"+str(iid)+"_"+'23'+".gif")
        gti.save("E:/OneDrive/Projects/TianChiChallenge/data/val_for_CRF/gt"+str(iid)+"_"+str(i)+".gif")
        img.save("E:/OneDrive/Projects/TianChiChallenge/data/val_for_CRF/patch"+str(iid)+"_"+str(i)+".gif")
    
