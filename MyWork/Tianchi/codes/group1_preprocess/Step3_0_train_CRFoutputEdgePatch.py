# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 10:08:11 2017

@author: 黄逸杰
"""

import numpy as np
import matplotlib.pyplot as pl
from PIL import Image


root="E:/OneDrive/Projects/TianChiChallenge/data/train_CRF_Output"
#gt=np.zeros([numI,1,numS,numS,numS])
Edgeid=[]
f=open('E:/OneDrive/Projects/TianChiChallenge/codes/group1_preprocess/EdgePatchID.txt')
line=f.readline()
while line:
    #print line
    Edgeid.append(line[0:len(line)])
    line=f.readline()
for i in range(len(Edgeid)):
    for j in range(48):
        label=Image.open('E:/OneDrive/Projects/TianChiChallenge/data/train_for_CRF/gt'+Edgeid[i]+'_'+str(j)+'.gif')
        label=np.asarray(label)
        label=label*255
        l=np.zeros([48,48,3],dtype=np.uint8)
        l[:,:,0]=label
        l[:,:,1]=label
        l[:,:,2]=label
        l=Image.fromarray(l)
        l.save('E:/OneDrive/Projects/TianChiChallenge/data/train_CRF_Output/'+Edgeid[i]+'/label'+Edgeid[i]+'_'+str(j)+'.ppm')
        
#root="E:/OneDrive/Projects/TianChiChallenge/data/train_CRF_Output/"
#
#numI=len(Edgeid)
#numS=48
#gt=np.zeros([numI,1,numS,numS,numS])
#EdgePatch=np.zeros([numI,1,numS,numS,numS])
#for i in range(len(Edgeid)):
#    for j in range(numS):
#        patch=Image.open(root+Edgeid[i]+"/compare"+Edgeid[i]+"_"+str(j)+".ppm")
#        patch=np.asarray(patch)
#        patch=patch[:,0:numS]
#        EdgePatch[i,0,j]=patch[:,:,0]
#        label=Image.open(root+Edgeid[i]+"/label"+Edgeid[i]+"_"+str(j)+".ppm")
#        label = np.asarray(label)
#        gt[i,0,j]=label[:,:,0]/255.0
#    compare=np.concatenate([EdgePatch[i,0,23]/255.0,gt[i,0,23]],axis=1)
#    compare=Image.fromArray(compare)
#    compare.save('E:/OneDrive/Projects/TianChiChallenge/data/EdgePatchCompare/'+str(i)+'.ppm')
#gt=gt.astype(np.int8)
##for i in range(numI):
##    pl.imshow(gt[i,0,23,:,:])
##    pl.show()
#
#np.save("E:/OneDrive/Projects/TianChiChallenge/data/train_CRF_label_Edge.npy",gt)       
#for i in range(numI):
#    for j in range(numS):
#        label=Image.open(root+str(i)+"/label"+str(i)+"_"+str(j)+".ppm")
#        label = np.asarray(label)
#        gt[i,0,j,:,:]=label[:,:,0]/255.0
#gt=gt.astype(np.int8)
##for i in range(numI):
##    pl.imshow(gt[i,0,23,:,:])
##    pl.show()
#
#np.save("E:/OneDrive/Projects/TianChiChallenge/data/train_CRF_label.npy",gt)