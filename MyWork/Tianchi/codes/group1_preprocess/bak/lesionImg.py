# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 15:29:51 2017

@author: 黄逸杰
"""
import numpy as np
import matplotlib.pyplot as pl
from PIL import Image

for iid in range(1186):
    im = np.load("E:/OneDrive/Projects/densecrf/examples/lung/resampledData3D_48_normalized/patch"+str(iid)+".npy")
    gt = np.load("E:/OneDrive/Projects/densecrf/examples/lung/resampledData3D_48_mask_normalized/mask"+str(iid)+".npy")*255
#    for i in range(48):
#        img=Image.fromarray(im[i])
#        gti=Image.fromarray(gt[i])
#        img.save("E:/forCRF/patch"+str(iid)+"_"+str(i)+".gif")
#        gti.save("E:/forCRF/gt"+str(iid)+"_"+str(i)+".gif")
    out=np.concatenate((im[23],gt[23]),axis=1)
    out=Image.fromarray(out)
    out.save("E:/forCRF_2/patch"+str(iid)+"_"+'23'+".gif")
    
    
