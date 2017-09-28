#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 14:37:14 2017

@author: customer
"""

import numpy as np
import random
import os
import SimpleITK as sitk
import csv
import matplotlib.pyplot as pl
from PIL import Image
import sys


from skimage import data, util,transform
from skimage.measure import label,regionprops

sys.path.append("..")
from mhd_util import *


out_w=10
w=48
ProjName='train'
img_root='/media/customer/Disk1/LUNG/challenge/'+ProjName+'/img/'
pred_root='/media/customer/Disk1/LUNG/challenge/'+ProjName+'/output/'
seg_root='/media/customer/Disk1/LUNG/challenge/'+ProjName+'/reseg/'
val_root='/home/customer/document/lung-nodule-detection/challenge/ClassTrain/Data/'+ProjName


def dis(center1,center2):
    d=(center1[0]-center2[0])**2+(center1[1]-center2[1])**2+(center1[2]-center2[2])**2
    d=d**0.5
    return d
if __name__ == '__main__':
    csvfile = file('/home/customer/document/lung-nodule-detection/challenge/'+ProjName+'_annotations_vol.csv', 'rb')
    #csvfile = file('/home/customer/document/lung-nodule-detection/challenge/'+ProjName+'_annotations_vol_Expanded.csv', 'rb')
    reader = csv.reader(csvfile)
    p=-1
    Recall=0
    #pos_patch=np.zeros([1,1,w,w,w],dtype=np.uint8)
    #neg_patch=np.zeros([1,1,w,w,w],dtype=np.uint8)
    bak_filename='LKDS-00375.npy'#'LKDS-00539.npy'#'1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860.npy'
    center=np.zeros([1,3],dtype=np.int)
    #ngcnt=1
    readcnt=0
    thresh=50
    pnum=0
    nnum=0
    #csvf=file('/home/customer/document/lung-nodule-detection/challenge/'+ProjName+'_Submission'+str(thresh)+'.csv','w')
    
    #fieldnames=['seriesuid','coordX','coordY','coordZ','probability']
    for line in reader:
        if p<0:
            p=p+1
            continue
        #elif p>300:
        #    break
        else:
            pos_patch=np.zeros([0,1,w,w,w],dtype=np.uint8)
            neg_patch=np.zeros([0,1,w,w,w],dtype=np.uint8)
            #pos_mean=np.zeros([0],dtype=np.uint8)
            #neg_mean=np.zeros([0],dtype=np.uint8)
            
            pos_patch2=np.zeros([0,1,w,w,w],dtype=np.uint8)
            neg_patch2=np.zeros([0,1,w,w,w],dtype=np.uint8)
            
            #pos_mean_temp=np.zeros([1],dtype=np.uint8)
            #neg_mean_temp=np.zeros([1],dtype=np.uint8)
            filename=line[0]+'.npy'
            if filename==bak_filename:
                #initialize
                if readcnt==0:
                    img = sitk.ReadImage(img_root+filename+'.mhd')  
                    img = sitk.GetArrayFromImage(img)
                    #m=np.mean(img)
                    
                    pred_image=sitk.ReadImage(pred_root+filename+'.mhd')
                    pred = sitk.GetArrayFromImage(pred_image)
                    #pred2_image=sitk.ReadImage(pred_root+filename+'2.mhd')
                    #pred2 = sitk.GetArrayFromImage(pred_image)
                    
                    seg = sitk.ReadImage(seg_root+filename+'.mhd')  
                    seg = sitk.GetArrayFromImage(seg)           
                    #maskout
                    mask=seg>0
                    mx=np.ma.array(seg,mask=mask)
                    seg=mx.mask.astype(np.uint8)
                    pred=pred*seg
                    
                    mask=pred>thresh
                    mx=np.ma.array(pred,mask=mask)
                    pred_temp=mx.mask.astype(np.uint8)*255
                    
                    connect_map=label(pred_temp, connectivity= 2)
                    props = regionprops(connect_map)
#                   write_mhd_file('/media/customer/新加卷/LUNG/temp.mhd', pred, [seg.shape[2],seg.shape[1],seg.shape[0]]
                    center=np.zeros([1,3],dtype=np.int)
                    Diam=np.zeros([1],dtype=np.int)
                    center[0,0]=int(line[3])
                    center[0,1]=int(line[2])
                    center[0,2]=int(line[1])
                    Diam[0]=int(line[4])
                    readcnt=1
                else:
                    center2=np.zeros([1,3],dtype=np.int)
                    Diam2=np.zeros([1],dtype=np.int)
                    center2[0,0]=int(line[3])
                    center2[0,1]=int(line[2])
                    center2[0,2]=int(line[1])
                    Diam2[0]=int(line[4])
                    center=np.concatenate((center,center2),axis=0)
                    Diam=np.concatenate((Diam,Diam2),axis=0)
            else:
                #pos patch filling

                for c in range(center.shape[0]):

                    for t in range(3):
                        if center[c,t]<w/2:
                            center[c,t]=w/2 
                        elif center[c,t]>pred.shape[t]-w/2:
                            center[c,t]=pred.shape[t]-w/2
                    if Diam[c]*3<w:
                        D=Diam[c]*3
                    else:
                        D=48
                    patch=img[center[c,0]-D/2:center[c,0]+D/2,center[c,1]-D/2:center[c,1]+D/2,center[c,2]-D/2:center[c,2]+D/2]
                    if D!=48:
                        patch=transform.resize(patch,[w,w,w],mode='nearest')
                        patch=(patch*255).astype(np.uint8)
                    else:
                        patch=patch.astype(np.uint8)
                    compare=patch[23]
                    compare=Image.fromarray(compare)
                    compare.save('/home/customer/document/lung-nodule-detection/challenge/ClassTrain/Data/'+ProjName+'/pos_compare2/'+str(p-1)+'_'+str(c)+'.gif')

                    #if Diam[c]<=9:
                    pos_patch=np.concatenate((pos_patch,np.reshape(patch,[1,1,w,w,w])),axis=0)
                    #if Diam[c]>=6:
                        #pos_patch2=np.concatenate((pos_patch2,np.reshape(patch,[1,1,w,w,w])),axis=0)
                    #pos_mean_temp[0]=np.average(seg*img,weights=seg).astype(np.uint8)
                    #pos_mean=np.concatenate((pos_mean,pos_mean_temp),axis=0)
                #neg patch filling
                #neg_mean_temp[0]=np.average(seg*img,weights=seg).astype(np.uint8)
                for i in range(len(props)):
                    location=props[i]['bbox']
                    startz=location[0]
                    starty=location[1]
                    startx=location[2]
                    endz=location[3]
                    endy=location[4]
                    endx=location[5]
                    Diam=np.max([endz-startz,endy-starty,endx-startx])
                    pcenter=np.zeros([3])
                    #pcenter[0]=int((endz+startz)/2)
                    #pcenter[1]=int((endz+startz)/2)
                    #pcenter[2]=int((endz+startz)/2)
                            
                    pred_patch=pred[startz:endz,starty:endy,startx:endx]
                    patch_center=np.where(pred_patch>thresh)                                        
                    l_center=np.zeros([3])
                    for j in range(3):
                        l_center[j]=np.average(patch_center[j],weights=pred_patch[patch_center])

                    pcenter[0]=startz+l_center[0]
                    pcenter[1]=starty+l_center[1]
                    pcenter[2]=startx+l_center[2]    
                    pcenter=pcenter.astype(np.int)
                    if Diam*3<w:
                        D=Diam*3
                    else:
                        D=48
                    for t in range(3):
                        if pcenter[t]-D/2<0:
                            D=pcenter[t]*2
                        elif pcenter[t]+D/2>pred.shape[t]:
                            D=(pred.shape[t]-pcenter[t])*2
                    
                    neg=1
                    for j in range(center.shape[0]):
                        if dis(pcenter,center[j])<24:
                            neg=0
                            
                    if neg==1 and Diam>=3 and Diam<40:# and filename!='LKDS-00778.npy' and p!=32 and p!=33:# and p!=32 and p!=33 and p!=34 and p!=35:
                        
                        patch=img[pcenter[0]-D/2:pcenter[0]+D/2,pcenter[1]-D/2:pcenter[1]+D/2,pcenter[2]-D/2:pcenter[2]+D/2]
                        
                        if D!=48:
                            patch=transform.resize(patch,[w,w,w],mode='nearest')                            
                            patch=(patch*255).astype(np.uint8)
                        else:
                            patch=patch.astype(np.uint8)
                        compare=patch[23]
                        compare=Image.fromarray(compare)
                        compare.save('/home/customer/document/lung-nodule-detection/challenge/ClassTrain/Data/'+ProjName+'/neg_compare2/'+bak_filename+'_'+str(pcenter)+'_'+str(Diam)+'.gif')#str(p-1)+'_'+str(i)+'.gif')
                        #if Diam<=9:
                        neg_patch=np.concatenate((neg_patch,np.reshape(patch,[1,1,w,w,w])),axis=0)
                        #if Diam>=6:
                            #neg_patch2=np.concatenate((neg_patch2,np.reshape(patch,[1,1,w,w,w])),axis=0)
                        
                        #neg_mean=np.concatenate((neg_mean,neg_mean_temp),axis=0)
                        

                print bak_filename
                print "neg_count="+str(neg_patch.shape[0]+neg_patch2.shape[0])
                print "pos_count="+str(center.shape[0])
                print "connect_count="+str(len(props))
                print str(p-1)
                #ngcnt=neg_patch.shape[0]
                #reinitialize
                readcnt=0
                
                if readcnt==0:
                    img = sitk.ReadImage(img_root+filename+'.mhd')  
                    img = sitk.GetArrayFromImage(img)
                    pred_image=sitk.ReadImage(pred_root+filename+'.mhd')
                    pred = sitk.GetArrayFromImage(pred_image)
                    #pred2_image=sitk.ReadImage(pred_root+filename+'2.mhd')
                    #pred2 = sitk.GetArrayFromImage(pred_image)
                    seg = sitk.ReadImage(seg_root+filename+'.mhd')  
                    seg = sitk.GetArrayFromImage(seg)           
                    #maskout
                    mask=seg>0
                    mx=np.ma.array(seg,mask=mask)
                    seg=mx.mask.astype(np.uint8)
                    pred=pred*seg
                    mask=pred>thresh
                    mx=np.ma.array(pred,mask=mask)
                    pred_temp=mx.mask.astype(np.uint8)*255
                    
                    connect_map=label(pred_temp, connectivity= 2)
                    props = regionprops(connect_map)
                    center=np.zeros([1,3],dtype=np.int)
                    Diam=np.zeros([1],dtype=np.int)
                    center[0,0]=int(line[3])
                    center[0,1]=int(line[2])
                    center[0,2]=int(line[1])
                    Diam[0]=int(line[4])
                    readcnt=1
                bak_filename=filename
            p=p+1
            #pos_patch=pos_patch[1:pos_patch.shape[0]]
            #neg_patch=neg_patch[1:neg_patch.shape[0]]        
            #np.save(val_root+"pos.npy",pos_patch)
            if neg_patch.shape[0]!=0 and filename!='LKDS-00052.npy' and filename!='LKDS-00053.npy' and filename!='LKDS-00238.npy'and filename!='LKDS-00413.npy'and filename!='LKDS-00778.npy':#and p!=32 and p!=33 and p!=34 and p!=35:
                np.save(val_root+"/neg/"+filename,neg_patch)#str(nnum),neg_patch) 
            #if neg_patch2.shape[0]!=0 and filename!='LKDS-00052.npy' and filename!='LKDS-00053.npy' and filename!='LKDS-00238.npy'and filename!='LKDS-00413.npy'and filename!='LKDS-00778.npy':
                #np.save(val_root+"/neg2/"+filename,neg_patch2)
                #np.save(val_root+"/neg/"+str(nnum)+"neg_mean.npy",neg_mean) 
                #print(neg_patch.shape[0]==neg_mean.shape[0])
                #nnum=nnum+1
            if pos_patch.shape[0]!=0:# and filename!='LKDS-00778.npy':
                np.save(val_root+"/pos/"+filename,pos_patch) 
            #if pos_patch2.shape[0]!=0:
                #np.save(val_root+"/pos2/"+filename,pos_patch2) 
                #np.save(val_root+"/pos/"+str(pnum)+"pos_mean.npy",pos_mean) 
                #print(pos_patch.shape[0]==pos_mean.shape[0])
                #pnum=pnum+1
#                #patch=img[startz-5:endz+5,starty-5:endy+5,startx-5:endx+5]
#                #pd=pred[startz-5:endz+5,starty-5:endy+5,startx-5:endx+5]
            