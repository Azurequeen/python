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
from skimage import data,util,transform
imgs_dir="F:/TianChiChallengeNPY_Data/resampledData/trainData/"
csv_dir="F:/TianChiChallengeNPY_Data/data/csv/annotations_vol.csv"
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
out_w=44
N_lesions=975
#N_Object=10000
N_sampling=20
N_Background=100
patch=np.zeros([N_lesions*N_sampling,1,window,window,window],dtype=np.uint8)
patch1=np.zeros([0,1,window,window,window],dtype=np.uint8)#([N_lesions*N_sampling,1,window,window,window],dtype=np.uint8)
patch2=np.zeros([0,1,window,window,window],dtype=np.uint8)
#patch=patch.astype(np.uint8)
center=np.zeros(3)
center=center.astype(np.int)
i=-1
label_o=np.load("F:/TianChiChallengeNPY_Data/data/train_CRF_label.npy")
w=label_o.shape[2]
label=np.zeros([N_lesions,1,window,window,window],dtype=np.uint8)
label[:,:,window/2-w/2:window/2+w/2,window/2-w/2:window/2+w/2,window/2-w/2:window/2+w/2]=label_o
gt=np.zeros([N_lesions*N_sampling,1,window,window,window],dtype=np.uint8)
#gt=gt.astype(np.uint8)
def d2r(d):
    return float(d)/180.0*np.pi
for line in reader:  
    if i==-1:
        i=i+1
        continue
    else:
        img=np.load(imgs_dir+line[0]+'.npy')
        gt_img=np.load("F:/TianChiChallengeNPY_Data/Full_GT/"+line[0]+'.npy')
        for t in range(N_sampling):
            radius=int(line[4])
            ra=out_w/2-radius/2
            ra=0
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
                patch[i*N_sampling+t,0,pid]=np.asarray(pb)
                
                gr=gt_img[r_center[2]-window/2+pid]
                gr=Image.fromarray(gr).rotate(-r,Image.NEAREST)
                gr=np.asarray(gr)
                gy=gr[r_center[1]-window/2:r_center[1]+window/2,r_center[0]-window/2:r_center[0]+window/2]
                gt[i*N_sampling+t,0,pid]=np.asarray(gy)
            
#            if radius<=6:
#                scale=random.uniform(3.0,4.0)
#                CentR=int(scale*radius)
#            elif radius<=12 and radius>6:
#                scale=random.uniform(1.5,2.5)
#                CentR=int(scale*radius)
#            elif radius>12 and radius<24:
#                scale=random.uniform(1.5,2)
#                CentR=int(scale*radius)
#            elif radius>24 and radius<30
#                scale=random.uniform(1.0,1.5)
#                CentR=int(scale*radius)
#            else:
#                CentR=48
            if radius<=16:
                #scale=random.uniform(2.0,3.0)
                CentR=3*radius#int(scale*radius)
#            elif radius>30:
#                CentR=48
            else:
                CentR=random.randint(40,48)
                
                
            
            patch_temp=patch[i*N_sampling+t,0,w/2-CentR/2:w/2+CentR/2,w/2-CentR/2:w/2+CentR/2,w/2-CentR/2:w/2+CentR/2]
            if CentR!=48:
                temp=transform.resize(patch_temp,(window,window,window),mode='nearest')#[1:49,1:49,1:49]
                #patch[i*N_sampling+t,0]
                temp=(temp*255).astype(np.uint8)  
                temp=np.reshape(temp,[1,1,48,48,48])
                patch[i*N_sampling+t,0]=temp
                if radius<=9:
                    patch1=np.concatenate((patch1,temp),axis=0)
                if radius>=6:
                    patch2=np.concatenate((patch2,temp),axis=0)
            else:
                temp=patch_temp.reshape([1,1,48,48,48])#np.reshape([1,1,48,48,48],patch_temp)
                if radius<=9:
                    patch1=np.concatenate((patch1,temp),axis=0)   
                if radius>=6:
                    patch2=np.concatenate((patch2,temp),axis=0)
                patch[i*N_sampling+t,0]=(patch_temp).astype(np.uint8)  
            compare=np.zeros([window,window])    
            compare[:,0:window]=patch[i*N_sampling+t,0,window/2-1-move[2],:,:]
            #compare[:,window:window*2]=gt[i*N_sampling+t,0,window/2-1-move[2],:,:]*255
            compare=Image.fromarray(compare)
            compare.save("F:/TianChiChallengeNPY_Data/compare_class/"+str(i)+"_"+str(t)+".gif")
            pl.imshow(compare,cmap='gray')
        i=i+1
        print i
csvfile.close() 

np.save("F:/TianChiChallengeNPY_Data/data/Resized_Train_Patch_1.npy",patch1)
np.save("F:/TianChiChallengeNPY_Data/data/Resized_Train_Patch_2.npy",patch2)
np.save("F:/TianChiChallengeNPY_Data/data/Resized_Train_Patch.npy",patch)
#np.save("F:/TianChiChallengeNPY_Data/data/final_train_gt.npy",gt)

#label=np.load("H:/LUNA 2016/proj0306/midData3D/label.npy")
#label=label.astype(np.uint8)
