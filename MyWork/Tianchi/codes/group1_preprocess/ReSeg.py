#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 08:30:28 2017

@author: customer
"""

from skimage import data, util
from skimage.measure import label,regionprops
import numpy as np
import random
import os
import SimpleITK as sitk
import csv
import matplotlib.pyplot as pl
from PIL import Image
from mhd_util import *
import skimage.morphology as sm
ProjName='new_test'
out_w=10
if __name__ == '__main__':
    csvfile = file('/media/customer/Disk1/LUNG/challenge/csv/'+ProjName+'_annotations_vol.csv', 'rb')
    reader = csv.reader(csvfile)
    p=-1
    Recall=0
    #results=np.zeros([975,2])
    gaussian=np.zeros([out_w,out_w,out_w],dtype=np.float32)
    d=5.0
    for z in range(out_w):
        for y in range(out_w):
            for x in range(out_w):
                gaussian[z,y,x]=np.exp(-((z-out_w/2.0)**2+(y-out_w/2.0)**2+(x-out_w/2.0)**2)/(2.0*d*d))
    d=int(d)
    for line in reader:
        if p<0:# or p>1000:
            p=p+1
            continue
        else:
            filename=line[0]+'.npy'
            #img = sitk.ReadImage('/media/customer/新加卷/LUNG/img/img'+filename+'.mhd')  
            #img = sitk.GetArrayFromImage(img)
            #pred_image=sitk.ReadImage('/media/customer/新加卷/LUNG/output-Unary/'+filename+'.mhd')
            seg = sitk.ReadImage('/media/customer/Disk1/LUNG/challenge/'+ProjName+'/seg/'+filename+'.mhd')
            seg = sitk.GetArrayFromImage(seg) 
#            for z in range(seg.shape[0]-out_w):
#                for y in range(seg.shape[1]-out_w):
#                    for x in range(seg.shape[2]-out_w):
#                        if seg[z,y,x]>0:
#                            seg[z,y,x]=255
            mask=seg>0
            mx=np.ma.array(seg,mask=mask)
            seg=mx.mask.astype(np.uint8)*255
            print np.max(seg)
            #write_mhd_file('/media/customer/新加卷/LUNG/seg/'+'temp'+'.mhd', seg, [seg.shape[2],seg.shape[1],seg.shape[0]])
#            seg2=seg
#            for i in range(seg.shape[0]):
#                seg2[i]=sm.closing(seg[i],sm.disk(9))
            
            seg = sitk.GetImageFromArray(seg)#sitk.ReadImage('/media/customer/新加卷/LUNG/seg/temp.mhd')
            flt=sitk.DiscreteGaussianImageFilter()
            flt.SetMaximumKernelWidth(10)
            flt.SetVariance(5)
            seg2 = flt.Execute(seg)
            seg = sitk.GetArrayFromImage(seg)
            seg2 = sitk.GetArrayFromImage(seg2)

            mask=seg2>0
            mx=np.ma.array(seg2,mask=mask)
            seg2=mx.mask.astype(np.uint8)*255
            pl.imshow(seg[seg.shape[0]/2],cmap='gray')
            pl.show()
            pl.imshow(seg2[seg.shape[0]/2],cmap='gray')
            pl.show()
            
            img = sitk.ReadImage('/media/customer/Disk1/LUNG/challenge/'+ProjName+'/img/'+filename+'.mhd')  
            img = sitk.GetArrayFromImage(img)
            
            reseg=img*(seg2/255)
            
            write_mhd_file('/media/customer/Disk1/LUNG/challenge/'+ProjName+'/reseg/'+filename+'.mhd', reseg, [reseg.shape[2],reseg.shape[1],reseg.shape[0]])
            print(filename)
            