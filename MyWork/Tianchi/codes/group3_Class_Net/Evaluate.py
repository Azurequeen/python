#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 16:20:51 2017

@author: customer
"""

from __future__ import print_function

import numpy as np
from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, Activation, Reshape, core,normalization,concatenate,add,Cropping3D
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.utils.vis_utils import plot_model
from PIL import Image
import random
import os
from config import config
from mhd_util import *
from NetModel_conc import *
import matplotlib.pyplot as pl
working_path = "/home/customer/document/lung-nodule-detection/data/"
weights_path = "/home/customer/document/lung-nodule-detection/data/weights/"
output_path = "/home/customer/document/lung-nodule-detection/data/output/"

window=48
out_w=44
height = window
width = window
depth = window
batch_size=1
K.set_image_dim_ordering('th')  # Theano dimension ordering in this code
crop_size=2
data_form=np.float32
shape=0

ClassRoot='/home/customer/document/lung-nodule-detection/challenge/ClassTrain/Data/'
if __name__ == '__main__':
    model=ClassNet_FlexWindow()
    #model36=ClassNet_36()
    #model24=ClassNet_24()
    #plot(model,to_file='/home/customer/document/lung-nodule-detection/huangyj/ClassModel.png',show_shapes=True)
    
    #plot_model(model,to_file='/home/customer/document/lung-nodule-detection/challenge/ClassTrain/ClassModel48.png',show_shapes=True)
    #plot_model(model36,to_file='/home/customer/document/lung-nodule-detection/challenge/ClassTrain/ClassModel36.png',show_shapes=True)
    #plot_model(model24,to_file='/home/customer/document/lung-nodule-detection/challenge/ClassTrain/ClassModel24.png',show_shapes=True)
    use_existing=True
    WeightName='classnet_Flex_7_7.hdf5'
    ProjName='train'
    if use_existing:
        model.load_weights(weights_path + WeightName)

    patch1=np.load(ClassRoot+ProjName+'/'+'Resized_Train_Patch_pos.npy')
    #patch1=np.load(ClassRoot+ProjName+'/'+'Class'+'_'+ProjName+'_pos2_resized.npy')
    patch2=np.load(ClassRoot+ProjName+'/'+'Class'+'_'+ProjName+'_neg_resized.npy')
    gt1=np.zeros([patch1.shape[0],1,2])
    gt2=np.zeros([patch2.shape[0],1,2])
    gt1[:,:,0]=1
    gt2[:,:,1]=1
    #sample_weight1=np.ones([patch1.shape[0]],dtype=np.float32)
    #sample_weight1=sample_weight1*1.0
    #sample_weight2=np.ones([patch2.shape[0]],dtype=np.float32)
    #sample_weight2=sample_weight2*1.0
    
    #patch1=np.repeat(patch1,20,axis=0)
    #gt1=np.repeat(gt1,20,axis=0)    
    #sample_weight1=np.repeat(sample_weight1,20,axis=0)
    
    patch=np.concatenate((patch1,patch2),axis=0)
    gt=np.concatenate((gt1,gt2),axis=0)
    patch=patch.astype(np.float32)/255
    gt=gt.astype(np.float32)    
    #sample_weight=np.concatenate((sample_weight1,sample_weight2),axis=0)
    
    loss=np.zeros([patch.shape[0],2])
    for i in range(patch.shape[0]):
        loss[i]=model.evaluate(x=patch[i:i+1],
                              y=gt[i:i+1],
                              batch_size=1,#batch_size, 
                              verbose=1,            
                              )
        if loss[i,0]>0.3:
            compare=Image.fromarray((patch[i,0,23]*255).astype(np.uint8))
            compare.save('/home/customer/document/lung-nodule-detection/challenge/ClassTrain/Data/train/hard/'+str(i)+'.gif')   
            
    np.save(ClassRoot+ProjName+'/'+'weight.npy',loss)