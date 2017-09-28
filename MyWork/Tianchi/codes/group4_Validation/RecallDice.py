# -*- coding: utf-8 -*-
"""
Created on Wed May 03 17:00:21 2017

@author: 黄逸杰
"""

import numpy as np
import random
import os
import SimpleITK as sitk
import csv
import matplotlib.pyplot as pl
from PIL import Image
working_path = "/home/customer/document/lung-nodule-detection/data/"
weights_path = "/home/customer/document/lung-nodule-detection/data/weights/"
output_path = "/home/customer/document/lung-nodule-detection/data/output/"
window=48
out_w=44
p=-1
data_form=np.float32

thresh_d=0.01

if __name__ == '__main__':
    csvfile = file('G:/LUNG/annotations_resampled.csv', 'rb')
    reader = csv.reader(csvfile)
    correct=0
    G_Dice=0
    for line in reader:
        if p<1000:
            p=p+1
            continue
        else:
            filename=line[0]+'.npy'         
            pred_image=sitk.ReadImage('G:/LUNG/output-Unary/'+filename+'.mhd')            
            pred = sitk.GetArrayFromImage(pred_image)
            pred=pred.astype(np.float32)/255.0
            pred=pred[int(line[3])-window/2:int(line[3])+window/2,int(line[2])-window/2:int(line[2])+window/2,int(line[1])-window/2:int(line[1])+window/2]
            label=np.zeros([48,48,48],dtype=data_form)
            for i in range(48):
                l=Image.open('E:/CRFoutput/'+str(p)+'/label'+str(p)+'_'+str(i)+'.ppm')
                l=np.asarray(l)[:,:,0].astype(data_form)/255.0
                label[i]=l
#            for z in range(48):
#                for y in range(48):
#                    for x in range(48):
#                        if pred[z,y,x]>thresh:
#                            pred[z,y,x]=1.0
#                        else:
#                            pred[z,y,x]=0
            OR=np.sum(pred*label)
            Dice=2*OR/(np.sum(pred)+np.sum(label))
            G_Dice+=Dice
            if Dice>thresh_d:
                correct+=1
            print(Dice)
            print(correct)
        p=p+1
        print(p)
    G_Dice/=186