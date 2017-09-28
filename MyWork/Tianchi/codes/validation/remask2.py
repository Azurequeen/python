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
imgs_dir="G:/LUNG/resampledData3D_normalized_uint/"
csv_dir="G:/LUNG/annotations_resampled.csv"
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
N_lesions=1186
#N_Object=10000
N_sampling=20
N_Background=100
patch=np.zeros([N_lesions*N_sampling,1,window,window,window],dtype=np.uint8)
#patch=patch.astype(np.uint8)
center=np.zeros(3)
center=center.astype(np.int)
i=-1
label_o=np.load("E:/CRFoutput_Edge/label.npy")
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
        img=np.load("G:/LUNG/resampledData3D_normalized_uint/"+line[0]+'.npy')
        gt_img=np.load("G:/LUNG/resampledData3D_mask_uint/"+line[0]+'.npy')
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
                patch[i*N_sampling+t,0,pid]=np.asarray(pb)
                
                gr=gt_img[r_center[2]-window/2+pid]
                gr=Image.fromarray(gr).rotate(-r,Image.NEAREST)
                gr=np.asarray(gr)
                gy=gr[r_center[1]-window/2:r_center[1]+window/2,r_center[0]-window/2:r_center[0]+window/2]
                gt[i*N_sampling+t,0,pid]=np.asarray(gy)
                
            compare=np.zeros([window,window*2])    
            compare[:,0:window]=patch[i*N_sampling+t,0,window/2-1-move[2],:,:]
            compare[:,window:window*2]=gt[i*N_sampling+t,0,window/2-1-move[2],:,:]*255
            compare=Image.fromarray(compare)
            compare.save("G:/LUNG/compare2/"+str(i)+"_"+str(t)+".gif")
            pl.imshow(compare,cmap='gray')
        i=i+1
        print i
csvfile.close() 

np.save("G:/LUNG/patch.npy",patch)
np.save("G:/LUNG/gt.npy",gt)

#label=np.load("H:/LUNA 2016/proj0306/midData3D/label.npy")
#label=label.astype(np.uint8)
