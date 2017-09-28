#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 14:44:18 2017

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
import random
import os
from config import config
from mhd_util import *
from NetModel import *
import matplotlib.pyplot as pl
working_path = "/home/customer/document/lung-nodule-detection/data/"
weights_path = "/home/customer/document/lung-nodule-detection/data/weights/"
output_path = "/home/customer/document/lung-nodule-detection/data/output/"

window=48
out_w=44
height = window
width = window
depth = window
batch_size=8
K.set_image_dim_ordering('th')  # Theano dimension ordering in this code
crop_size=2
data_form=np.float32
shape=0

def Active_Generate(Xin,Yin,Win):#,mean):    
    while 1:
        #img_window=Xin.shape[2]
        img_n=Xin.shape[0]
        for s in range(img_n/batch_size):
            #X=np.zeros([batch_size,1,img_window,img_window,img_window],dtype=data_form)
            #Y=np.zeros([batch_size,1,2],dtype=data_form)
            #W=np.zeros([batch_size],dtype=data_form)
            i=[]
            for t in range(batch_size):
                #i=random.randint(0,img_n-1)   
                i.append(random.randint(0,img_n-1))
#                X[t]=Xin[i].astype(data_form)/255.0
#                Y[t]=Yin[i].astype(data_form)
#                W[t]=Win[i].astype(data_form)
            X=Xin[i].astype(data_form)/255.0
            Y=Yin[i].astype(data_form)
            W=Win[i].astype(data_form)
            #m=mean[i]
                #r=random.randint(1,4)
#                if test==False:
#                    if r==1:
#                        X[t]=X[t]
#                    elif r==2:
#                        X[t]=X[t].transpose([0,1,3,2])
#                    elif r==3:
#                        X[t]=X[t,:,:,:,::-1]
#                    elif r==4:
#                        X[t]=X[t,:,:,::-1,:]
            #for t in range(batch_size):        
            #    X[t]=X[t]-m[t]
            #X[:]/=std

            yield (X, Y, W)

ClassRoot='/home/customer/document/lung-nodule-detection/challenge/ClassTrain/Data/'
if __name__ == '__main__':
    #model48=ClassNet_48()
    #model36=ClassNet_36()
    #model24=ClassNet_24()
#    #plot(model,to_file='/home/customer/document/lung-nodule-detection/huangyj/ClassModel.png',show_shapes=True)
#    
    #plot_model(model48,to_file='/home/customer/document/lung-nodule-detection/challenge/ClassTrain/ClassModel48.png',show_shapes=True)
    #plot_model(model36,to_file='/home/customer/document/lung-nodule-detection/challenge/ClassTrain/ClassModel36.png',show_shapes=True)
    #plot_model(model24,to_file='/home/customer/document/lung-nodule-detection/challenge/ClassTrain/ClassModel24.png',show_shapes=True)
    aug=20
    use_existing=False
    model=ClassNet_48()#ClassNet_MultiScale()
    WeightName_load='classnet_Flex_7_10.hdf5'#_7_6_2.hdf5'
    WeightName_save='classnet_Flex_7_10.hdf5'
    plot_model(model,to_file='/home/customer/document/lung-nodule-detection/challenge/ClassTrain/'+WeightName_save+'.png',show_shapes=True)
    if use_existing:
        model.load_weights(weights_path + WeightName_load)
    #plot_model(model24,to_file='/home/customer/document/lung-nodule-detection/challenge/ClassTrain/ClassModel_conc6_19.png',show_shapes=True)
    #patch1=np.load("/media/customer/新加卷/LUNG/pos.npy")
    patch1=np.load(ClassRoot+'train/'+"Resized_Train_Patch_pos.npy")#"Class_Train_Pos_64.npy")#[:,:,8:56,8:56,8:56]#[:,:,:,:,::-1]#.transpose([0,1,2,4,3])#.astype(np.float32)/255.0    
    #patch1=patch1[0:patch1.shape[0]]#-aug]
#    patch1=np.zeros([patch1_temp.shape[0]/2,1,48,48,48],dtype=np.uint8)
#    for i in range(patch1.shape[0]/20):
#        patch1[i*10:i*10+10]=patch1_temp[i*20:i*20+10]        
    #patch1=patch1[0:20000]2
    patch2=np.load(ClassRoot+'train/'+"Class_train_neg_resized.npy")#[:,:,:,:,::-1]#.transpose([0,1,2,4,3])
    #patch3=np.load(ClassRoot+'train/'+"Class_Train_BakNeg.npy")
    #patch2=np.concatenate((patch2,patch3),axis=0)
    sample_weight1=np.ones([patch1.shape[0]],dtype=np.float32)
    sample_weight1=sample_weight1*1.0
    sample_weight2=np.ones([patch2.shape[0]],dtype=np.float32)
    sample_weight2=sample_weight2*2.0
    gt1=np.zeros([patch1.shape[0],1,2])
    gt2=np.zeros([patch2.shape[0],1,2])
    gt1[:,:,0]=1
    gt2[:,:,1]=1

#    val_patch1=patch1[patch1.shape[0]*0.9:patch1.shape[0]]
#    val_patch2=patch2[patch2.shape[0]*0.9:patch2.shape[0]]
#    val_gt1=gt1[patch1.shape[0]*0.9:patch1.shape[0]]
#    val_gt2=gt2[patch2.shape[0]*0.9:patch2.shape[0]]
#    
#    patch1=patch1[0:patch1.shape[0]*0.9]
#    patch2=patch2[0:patch2.shape[0]*0.9]
#    gt1=gt1[0:patch1.shape[0]*0.9]
#    gt2=gt2[0:patch2.shape[0]*0.9]
    
    #patch1=np.repeat(patch1,aug,axis=0)
    #gt1=np.repeat(gt1,aug,axis=0)
    patch=np.concatenate((patch1,patch2),axis=0)
    gt=np.concatenate((gt1,gt2),axis=0)
    sample_weight=np.concatenate((sample_weight1,sample_weight2),axis=0)
    #sample_weight+=np.load(ClassRoot+'train/weight.npy')[:,0]
    #sample_weight[patch1.shape[0]:sample_weight.shape[0]]=0
    #sample_weight+=1.0
    #mean=np.zeros([window,window,window])
#    for z in range(window):
#        for y in range(window):
#            for x in range(window):
#                mean[z,y,x]=np.mean(patch[:,:,z,y,x])    
    print("train loading completed")

    #train_mean1_temp=np.load(ClassRoot+'train/'+'Class_train_pos_50_mean.npy')
    #train_mean1=np.zeros([train_mean1_temp.shape[0]*20],dtype=np.uint8)
    #for i in range(train_mean1_temp.shape[0]):
    #    train_mean1[aug*i:aug*i+aug]=train_mean1_temp[i]
        
    #train_mean2=np.load(ClassRoot+'train/'+'Class_train_Neg_50_mean.npy')
    #train_mean=np.concatenate([train_mean1,train_mean2],axis=0)
        #std=np.load(ClassRoot+'train/'+'std.npy')
    #train_mean=train_mean/255.0
    
    #std/=255.0
    #print("mean computing completed")
    #patch-=mean
    #patch/=std
#    patch=patch.astype(np.float32)/255
#    gt=gt.astype(np.int)
    
    val_patch1=np.load(ClassRoot+'val/'+"Class_val_pos_resized.npy")
    val_patch2=np.load(ClassRoot+'val/'+"Class_val_neg_resized.npy")
    val_gt1=np.zeros([val_patch1.shape[0],1,2])
    val_gt2=np.zeros([val_patch2.shape[0],1,2])
    val_gt1[:,:,0]=1
    val_gt2[:,:,1]=1
    val_sample_weight1=np.ones([val_patch1.shape[0]],dtype=np.float32)
    val_sample_weight1=val_sample_weight1*2.0
    val_sample_weight2=np.ones([val_patch2.shape[0]],dtype=np.float32)
    val_sample_weight2=val_sample_weight2*1.0
    #val_mean1=np.load(ClassRoot+'val/'+'Class_val_pos_50_mean.npy')
    #val_mean2=np.load(ClassRoot+'val/'+'Class_val_Neg_50_mean.npy')
    
    #val_patch1=np.repeat(val_patch1,aug,axis=0)
    #val_gt1=np.repeat(val_gt1,aug,axis=0)    
    #val_sample_weight1=np.repeat(val_sample_weight1,aug,axis=0)
    #val_mean1=np.repeat(val_mean1,aug,axis=0)

    
    #val_mean=np.concatenate((val_mean1,val_mean2),axis=0)
    #val_mean=val_mean/255.0
    val_patch=np.concatenate((val_patch1,val_patch2),axis=0)
    val_gt=np.concatenate((val_gt1,val_gt2),axis=0)
        
    #val_patch=val_patch.astype(np.float32)/255
    #val_gt=val_gt.astype(np.float32)    
    val_sample_weight=np.concatenate((val_sample_weight1,val_sample_weight2),axis=0)
    print("val loading completed")
    #val_patch-=mean
    #val_patch/=std
    gen=Active_Generate(patch,gt,sample_weight)#,train_mean)
    test_gen=Active_Generate(val_patch,val_gt,val_sample_weight)#,val_mean)
    model_checkpoint = ModelCheckpoint(weights_path + WeightName_save, 
                               verbose=1, 
                               monitor='val_loss',
                               mode='auto', 
                               save_best_only=True) 
    
    #gt=np.repeat(gt,3,axis=2)
    #val_gt=np.repeat(val_gt,3,axis=2)
    
#    model.fit(x=patch,
#              y=gt,
#              batch_size=batch_size, 
#              epochs=30,
#              validation_data=(val_patch,val_gt),#,val_sample_weight),
#              verbose=1,
#              shuffle=1,
#              callbacks=[model_checkpoint]#,
#              #sample_weight=sample_weight
#              )
    model.fit_generator(gen,#generate_arrays_from_file(train_imgs,train_masks,patch_height,patch_width,batch_size,N_subimgs,N_imgs), 
                    epochs=30, 
                    steps_per_epoch=(patch.shape[0])/batch_size,
                    #samples_per_epoch=N_sample+N_back,
                    verbose=1, 
                    callbacks=[model_checkpoint],
                    validation_data=test_gen,
                    validation_steps=(val_patch.shape[0])/batch_size)
    
#    model2=GaussianTest()
#    patch2=model2.predict(patch1[0:1])
#    pl.imshow(patch2[0,0,23],cmap='gray')
#    pl.show()
#    pl.imshow(patch2[0,0,23],cmap='gray')
#    pl.show()
#    
#    model3=CroppingTest()
#    patch3=model3.predict(patch1[0:1])
#    
#    pl.imshow(patch3[0,0,23],cmap='gray')
#    pl.show()
#    pl.imshow(patch3[0,1,23],cmap='gray')
#    pl.show()
#    pl.imshow(patch3[0,2,23],cmap='gray')
#    pl.show()
#    
    