# -*- coding: utf-8 -*-
"""
Created on Thu May 04 16:12:57 2017

@author: 黄逸杰
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 13:53:54 2017

@author: 黄逸杰
"""


import numpy as np
import os
import matplotlib.pyplot as pl
import csv
import random
from PIL import Image
imgs_dir="F:/TianChiChallengeNPY_Data/resampledData/trainData/"
csv_dir="E:/OneDrive/Projects/TianChiChallenge/data/csv/annotations_vol.csv"
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

window=48
N_lesions=975
#N_Object=10000

#spatch=np.zeros([N_lesions*N_sampling,1,window,window,window],dtype=np.uint8)
#patch=patch.astype(np.uint8)
center=np.zeros(3)
center=center.astype(np.int)
i=-1
label_o=np.load("E:/OneDrive/Projects/TianChiChallenge/data/train_CRF_label.npy")
w=label_o.shape[2]
label=np.zeros([N_lesions,1,window,window,window],dtype=np.uint8)
label[:,:,window/2-w/2:window/2+w/2,window/2-w/2:window/2+w/2,window/2-w/2:window/2+w/2]=label_o

#gt=gt.astype(np.uint8)

bak_filename=0
for line in reader:  
    if i==-1:
        i=i+1
        continue
    else:
        filename=line[0]
        if bak_filename!=filename:
            if bak_filename!=0:
                np.save('F:/TianChiChallengeNPY_Data/Full_GT/'+str(bak_filename)+'.npy',gt)            
            img=np.load(imgs_dir+line[0]+'.npy')
            gt=np.zeros(img.shape,dtype=np.uint8)
        
        center=[int(line[3]),int(line[2]),int(line[1])]
        for j in range(3):
            if center[j]-window/2<0:
                center[j]=window/2
            elif center[j]+window/2>img.shape[j]:
                center[j]=img.shape[j]-window/2
        gt[center[0]-window/2:center[0]+window/2,center[1]-window/2:center[1]+window/2,center[2]-window/2:center[2]+window/2]=gt[center[0]-window/2:center[0]+window/2,center[1]-window/2:center[1]+window/2,center[2]-window/2:center[2]+window/2]+label[i,0]
        mask=gt>0
        mx=np.ma.array(gt,mask=mask)
        gt=mx.mask.astype(np.uint8)
        assert(np.max(gt)<=1)
        i1=Image.fromarray(np.concatenate((img[center[0]],gt[center[0]]*255),axis=1))
        i1.save("E:/OneDrive/Projects/TianChiChallenge/data/WholeGTcompare/"+str(i)+".gif")
        print i
        i=i+1
        bak_filename=filename
        
np.save('F:/TianChiChallengeNPY_Data/Full_GT/'+str(bak_filename)+'.npy',gt)  
csvfile.close() 


#label=np.load("H:/LUNA 2016/proj0306/midData3D/label.npy")
#label=label.astype(np.uint8)