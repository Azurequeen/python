# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 15:29:51 2017

@author: 黄逸杰
"""
import numpy as np
import matplotlib.pyplot as pl
from PIL import Image
im=np.load("G:/LUNG/Patch48/patch_48.npy")
gt = np.load("G:/LUNG/Patch48/mask_48.npy")*255
for iid in range(975):
    #for i in range(48):
        out=np.concatenate((im[iid,23],gt[iid,23]),axis=1)
        out=Image.fromarray(out)    
        #img=Image.fromarray(im[iid,23])
        #gti=Image.fromarray(gt[iid,23])
        out.save("E:/forCRF_2/patch"+str(iid)+"_"+'23'+".gif")
        #gti.save("E:/forCRF_2/gt"+str(iid)+"_"+str(23)+".gif")
    
