# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 10:42:03 2017

@author: 黄逸杰
"""


import numpy as np
from numpy import array
import os
import matplotlib.pyplot as pl
import csv
import random
from PIL import Image
from skimage import transform
from scipy.interpolate import RegularGridInterpolator as rgi
#from scipy import interpolate
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
out_w=48
N_lesions=975
#N_Object=10000
N_sampling=40
N_Background=100

patch=np.zeros([N_lesions*N_sampling,1,window,window,window],dtype=np.uint8)
#patch=patch.astype(np.uint8)
center=np.zeros(3)
center=center.astype(np.int)
i=-1
#label_o=np.load("F:/TianChiChallengeNPY_Data/data/train_CRF_label.npy")
#w=label_o.shape[2]
#label=np.zeros([N_lesions,1,window,window,window],dtype=np.uint8)
#label[:,:,window/2-w/2:window/2+w/2,window/2-w/2:window/2+w/2,window/2-w/2:window/2+w/2]=label_o
#gt=np.zeros([N_lesions*N_sampling,1,window,window,window],dtype=np.uint8)
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
            #ra=out_w/2-radius/2
            ra=1
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
            pc=np.zeros([window,window,window],dtype=np.uint8)
            for pid in range(window):
                pa=img[r_center[2]-window/2+pid]
                pa=Image.fromarray(pa).rotate(-r,Image.BILINEAR)
                pa=np.asarray(pa)        
                pb=pa[r_center[1]-window/2:r_center[1]+window/2,r_center[0]-window/2:r_center[0]+window/2]                
                pc[pid]=np.asarray(pb)
                
            crop=random.randint(0,6)
            pc=pc[crop:window-crop,crop:window-crop,crop:window-crop]
            #my_interpolating_function = rgi((window-2*crop,window-2*crop,window-2*crop), pc)
            #pc = my_interpolating_function(array([window,window,window]).T)
            pd=transform.resize(pc,(window+2,window+2,window+2),mode='nearest')*255
            patch[i*N_sampling+t,0]=pd[1:window+1,1:window+1,1:window+1].astype(np.uint8)
            pl.imshow(pc[(window-2*crop)/2],cmap='gray')
            pl.show()
            pl.imshow(pd[window/2],cmap='gray')
            pl.show()
            #patch[i*N_sampling+t,0]=pc.astype(np.uint8)                  
                #patch[i*N_sampling+t,0,pid]=np.asarray(pb)
                
                #gr=gt_img[r_center[2]-window/2+pid]
                #gr=Image.fromarray(gr).rotate(-r,Image.NEAREST)
                #gr=np.asarray(gr)
                #gy=gr[r_center[1]-window/2:r_center[1]+window/2,r_center[0]-window/2:r_center[0]+window/2]
                #gt[i*N_sampling+t,0,pid]=np.asarray(gy)
                
            compare=np.zeros([window,window],dtype=np.uint8)    
            compare=patch[i*N_sampling+t,0,window/2-1-move[2],:,:]
            #compare[:,window:window*2]=gt[i*N_sampling+t,0,window/2-1-move[2],:,:]*255
            compare=Image.fromarray(compare)
            compare.save("F:/TianChiChallengeNPY_Data/compare_class/"+str(i)+"_"+str(t)+".gif")
            pl.imshow(compare,cmap='gray')
        i=i+1
        print i
csvfile.close() 

np.save("F:/TianChiChallengeNPY_Data/data/pos_train_patch_48.npy",patch)
#np.save("F:/TianChiChallengeNPY_Data/data/final_train_gt.npy",gt)