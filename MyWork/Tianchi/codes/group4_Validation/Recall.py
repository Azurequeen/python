# -*- coding: utf-8 -*-
"""
Created on Wed May 03 15:03:50 2017

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
if __name__ == '__main__':
    csvfile = file('G:/LUNG/annotations_resampled.csv', 'rb')
    reader = csv.reader(csvfile)
    for line in reader:
        if p<999:
            p=p+1
            continue
        else:
            filename=line[0]+'.npy'         
            img = np.load('G:/LUNG/orig/'+filename).astype(data_form)/255.0
            pred_image=sitk.ReadImage('G:/LUNG/output-Unary/'+filename+'.mhd')            
            pred = sitk.GetArrayFromImage(pred_image)
            pred=pred.astype(np.float32)/255.0
            for i in range(-4,5,1):
                label=Image.open('E:/CRFoutput/'+str(p)+'/label'+str(p)+'_'+str(24+i)+'.ppm')
                label=np.asarray(label)[:,:,0].astype(data_form)/255.0
                patch=img[int(line[3])+i,int(line[2])-window/2:int(line[2])+window/2,int(line[1])-window/2:int(line[1])+window/2]
                pred2=pred[int(line[3])+i,int(line[2])-window/2:int(line[2])+window/2,int(line[1])-window/2:int(line[1])+window/2]
                out=np.concatenate((patch,label,pred2),axis=1)
                out=(out*255).astype(np.uint8)
                #pl.imshow(out,cmap='gray')
                #pl.show()
                im=Image.fromarray(out)
                im.save('G:/LUNG/compare3/'+filename+'_'+str(i+4)+'.gif')
        p=p+1
        print(p)
        