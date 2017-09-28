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

Edgeid=[]
f=open('E:/OneDrive/Projects/TianChiChallenge/codes/group1_preprocess/EdgePatchID.txt')
line=f.readline()
while line:
    #print line
    Edgeid.append(line[0:len(line)])
    line=f.readline()

EdgeidI=np.zeros([len(Edgeid)],dtype=np.int)

for i in range(len(Edgeid)):
    EdgeidI[i]=int(Edgeid[i])
    
Edgeid=EdgeidI


csvfile = file(csv_dir, 'rb')
reader = csv.reader(csvfile)

window=48
out_w=44
N_lesions=Edgeid.shape[0]
#N_Object=10000
N_sampling=20
N_Background=100
patch=np.zeros([N_lesions*N_sampling,1,window,window,window],dtype=np.uint8)
#patch=patch.astype(np.uint8)
center=np.zeros(3)
center=center.astype(np.int)
i=-1
#label_o=np.load("E:/OneDrive/Projects/TianChiChallenge/data/train_CRF_label.npy")
#w=label_o.shape[2]
#label=np.zeros([975,1,window,window,window],dtype=np.uint8)
#label[:,:,window/2-w/2:window/2+w/2,window/2-w/2:window/2+w/2,window/2-w/2:window/2+w/2]=label_o
gt=np.zeros([N_lesions*N_sampling,1,window,window,window],dtype=np.uint8)
#gt=gt.astype(np.uint8)
Eid=0
def d2r(d):
    return float(d)/180.0*np.pi
for line in reader:  
    if i==-1:
        i=i+1
        continue
    elif np.where(Edgeid==i)[0].shape[0]!=0:
        img=np.load(imgs_dir+line[0]+'.npy')
        gt_img=np.load("F:/TianChiChallengeNPY_Data/Full_GT/"+line[0]+'.npy')
        for t in range(N_sampling):
            radius=int(line[4])
            ra=out_w/2-radius/2
            move=[random.randint(-ra,ra),random.randint(-ra,ra),random.randint(-ra,ra)]
            #move=[0,0,0]            
            center[0]=int(line[1])
            center[1]=int(line[2])
            center[2]=int(line[3])
            #gt[i,0,:,:,:]=label[i,0,48-realmove]
            #pl.imshow(img[zcenter,ycenter-window/2:ycenter+window/2,xcenter-window/2:xcenter+window/2],cmap='gray')
            #pl.show()

            #patch[i*N_sampling+t,0,:,:,:]=img[center[2]-window/2:center[2]+window/2,center[1]-window/2:center[1]+window/2,center[0]-window/2:center[0]+window/2]
            r=random.randint(0,359)
            #r=0
            r_center=center.astype(np.float)
            temp_x=center[0]-img.shape[2]/2
            temp_y=center[1]-img.shape[1]/2            
            r_center[0]=temp_x*np.cos(d2r(r))-temp_y*np.sin(d2r(r))+img.shape[2]/2
            r_center[1]=temp_x*np.sin(d2r(r))+temp_y*np.cos(d2r(r))+img.shape[1]/2
            r_center=r_center.astype(np.int)
            r_center+=move
            for j in range(3):
                if r_center[j]<=window/2:
                    r_center[j]=window/2
                if r_center[j]>=img.shape[2-j]-window/2:
                    r_center[j]=img.shape[2-j]-window/2            
            for pid in range(window):

                pa=img[r_center[2]-window/2+pid]
                pa=Image.fromarray(pa).rotate(-r,Image.BILINEAR)
                pa=np.asarray(pa)        
                pb=pa[r_center[1]-window/2:r_center[1]+window/2,r_center[0]-window/2:r_center[0]+window/2]                
                patch[Eid*N_sampling+t,0,pid]=np.asarray(pb)
                
                gr=gt_img[r_center[2]-window/2+pid]
                gr=Image.fromarray(gr).rotate(-r,Image.NEAREST)
                gr=np.asarray(gr)
                gy=gr[r_center[1]-window/2:r_center[1]+window/2,r_center[0]-window/2:r_center[0]+window/2]
                gt[Eid*N_sampling+t,0,pid]=np.asarray(gy)
                
            compare=np.zeros([window,window*2])    
            compare[:,0:window]=patch[Eid*N_sampling+t,0,window/2-1-move[2],:,:]
            compare[:,window:window*2]=gt[Eid*N_sampling+t,0,window/2-1-move[2],:,:]*255
            compare=Image.fromarray(compare)
            compare.save("F:/TianChiChallengeNPY_Data/compareEdge/"+str(i)+"_"+str(t)+".gif")
            pl.imshow(compare,cmap='gray')
        
        print i
        i=i+1
        Eid=Eid+1
    else: 
        i=i+1

csvfile.close() 

np.save("E:/OneDrive/Projects/TianChiChallenge/data/final_train_Edgepatch.npy",patch)
np.save("E:/OneDrive/Projects/TianChiChallenge/data/final_train_Edgegt.npy",gt)

