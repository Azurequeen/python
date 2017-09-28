# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 19:31:25 2017

@author: 
"""

import numpy as np
import os
import matplotlib.pyplot as pl
import csv
import random
from PIL import Image
imgs_dir="H:/LUNA 2016/resampledData3D_normalized_uint/"
csv_dir="H:/LUNA 2016/annotations_resampled.csv"
#for path, subdirs, files in os.walk(imgs_dir): #list all files, directories in the path
#        for i in range(len(files)):            
#            img=np.load("H:/LUNA 2016/resampledData3D_normalized/"+files[i])
#            img=img.astype(np.uint8)
#            np.save("H:/LUNA 2016/resampledData3D_normalized_uint/"+files[i],img)
#for path, subdirs, files in os.walk(imgs_dir): #list all files, directories in the path
#    for i in range(len(files)):            
#        img=np.load(imgs_dir+files[i])
csvfile = file(csv_dir, 'rb')
reader = csv.reader(csvfile)

window=64
N_lesions=1186
N_Background=20
gt=np.zeros([N_lesions*N_Background,1,window,window,window],dtype=np.uint8)
np.save("H:/LUNA 2016/resampledData3D_normalized_mask/gt_background.npy",gt)

patch=np.zeros([N_lesions*N_Background,1,window,window,window],dtype=np.uint8)
center=np.zeros(3)
center=center.astype(np.int)
i=-1
#label=np.load("E:/CRFoutput/label.npy")

#gt=gt.astype(np.uint8)

for line in reader:  
    if i==-1:
        i=i+1
        continue
    else:
        img=np.load("H:/LUNA 2016/resampledData3D_normalized_uint/"+line[0]+'.npy')
        seg=np.load("H:/LUNA 2016/resampledData3D_normalized_uint_segmented/"+line[0]+'.npy')
        for t in range(N_Background):
            while(1):            
                center=[random.randint(window/2,img.shape[2]-window/2),random.randint(window/2,img.shape[1]-window/2),random.randint(window/2,img.shape[0]-window/2)]
                if (center[0]-int(line[1]))**2+(center[1]-int(line[2]))**2+(center[2]-int(line[3]))**2>(window/2)**2 and seg[center[2],center[1],center[0]]>0:
                    break
            patch[i*N_Background+t,0,:,:,:]=img[center[2]-window/2:center[2]+window/2,center[1]-window/2:center[1]+window/2,center[0]-window/2:center[0]+window/2]
        i=i+1
        print i
csvfile.close() 

np.save("H:/LUNA 2016/resampledData3D_normalized_mask/patch_background.npy",patch)
#gt=np.zeros([N_lesions*N_Background,1,window,window,window],dtype=np.uint8)
#np.save("H:/LUNA 2016/resampledData3D_normalized_mask/gt_background.npy",gt)

#label=np.load("H:/LUNA 2016/proj0306/midData3D/label.npy")
#label=label.astype(np.uint8)