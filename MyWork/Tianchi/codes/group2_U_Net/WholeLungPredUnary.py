#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 13:49:56 2017

@author: customer
"""

from __future__ import print_function

import numpy as np
from keras.models import Model
from keras.layers import Input, merge, Conv3D, MaxPooling3D, UpSampling3D, Activation, Reshape, core,normalization,concatenate
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
#from keras.utils.visualize_util import plot
from keras import backend as K
import matplotlib.pyplot as pl
from mhd_util import *
import os
import sys
from config import config
import SimpleITK as sitk
import time

import csv

working_path = "/home/customer/document/lung-nodule-detection/data/"
weights_path = "/home/customer/document/lung-nodule-detection/data/weights/"
output_path = "/media/customer/Disk1/LUNG/challenge/"
ProjName='val'
height = 48
width = 48
depth = 48
step=11
out_w=44
K.set_image_dim_ordering('th')  # Theano dimension ordering in this code

data_form=np.float32

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + config["smooth"]) / (K.sum(y_true_f) + K.sum(y_pred_f) + config["smooth"])


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def unet_model_MultiScale():
    inputs = Input(config["input_shape"])
    conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv3D(32, (3, 3, 3), padding='same')(conv1)
    conv1 = normalization.BatchNormalization(epsilon=2e-05, axis=1, momentum=0.9, weights=None, beta_initializer='zero', gamma_initializer='one')(conv1)
    conv1 = core.Activation('relu')(conv1)
    pool1 = MaxPooling3D(pool_size=config["pool_size"])(conv1)

    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(64, (3, 3, 3), padding='same')(conv2)
    conv2 = normalization.BatchNormalization(epsilon=2e-05, axis=1, momentum=0.9, weights=None, beta_initializer='zero', gamma_initializer='one')(conv2)
    conv2 = core.Activation('relu')(conv2)
    
    pool2_1 = MaxPooling3D(pool_size=config["pool_size"])(conv2)
    conv3_1 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool2_1)
    conv3_1 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv3_1)
    
    pool2_2 = MaxPooling3D(pool_size=(4,4,4))(conv2)
    conv3_2 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool2_2)
    conv3_2 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv3_2)
    
    fuse = concatenate([UpSampling3D(size=config["pool_size"])(conv3_2), conv3_1], axis=1)
    conv3_f = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(fuse)

    up4 = concatenate([UpSampling3D(size=config["pool_size"])(conv3_f), conv2], axis=1)
    conv4 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up4)
    conv4 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv4)

    up5 = concatenate([UpSampling3D(size=config["pool_size"])(conv4), conv1], axis=1)
    conv5 = Conv3D(32, (3, 3, 3), activation='relu', padding='valid')(up5)
    conv5 = Conv3D(32, (3, 3, 3), activation='relu', padding='valid')(conv5)

    conv6 = Conv3D(config["n_labels"], (1, 1, 1))(conv5)
    conv6 = core.Reshape((1,out_w*out_w*out_w))(conv6)
    conv6 = core.Permute((2,1))(conv6)
    #conv6 = 
    act = Activation('sigmoid')(conv6)
    model = Model(inputs=inputs, outputs=act)

    #model.compile(optimizer=Adam(lr=config["initial_learning_rate"]), loss='categorical_crossentropy',metrics=['fbeta_score'])
    model.compile(optimizer=Adam(lr=config["initial_learning_rate"]), loss=dice_coef_loss,metrics=[dice_coef])
    return model
if __name__ == '__main__':
    print('-' * 30)
    print('Loading and preprocessing train data...')
    print('-' * 30)
    filename='LKDS-00539.npy'

    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)
    model = unet_model_MultiScale()
    # Saving weights to unet.hdf5 at checkpoints

    # loading best weights from training session
    print('-' * 30)
    print('Loading saved weights...')
    print('-' * 30)
    model.load_weights(weights_path + 'unet_unary_challenge.hdf5')
    #plot(model,to_file='/media/customer/新加卷/LUNG/model.png',show_shapes=True)
    print('-' * 30)
    print('Predicting masks on test data...')
    print('-' * 30)
    
    gaussian_weight=np.zeros([out_w,out_w,out_w],dtype=data_form)
    d=18.0
    for z in range(out_w):
        for y in range(out_w):
            for x in range(out_w):
                gaussian_weight[z,y,x]=np.exp(-((z-out_w/2.0)**2+(y-out_w/2.0)**2+(x-out_w/2.0)**2)/(2.0*d*d))
    #write_mhd_file(output_path+'gaussian'+'.mhd', gaussian_weight , [gaussian_weight.shape[2],gaussian_weight.shape[1],gaussian_weight.shape[0]])
    #pl.imshow(gaussian_weight[24],cmap='gray')
    unary_weight=np.zeros([out_w,out_w,out_w],dtype=data_form)+1.0
    
    csvfile = file('/media/customer/Disk1/LUNG/challenge/csv/'+ProjName+'_annotations_vol.csv', 'rb')
    reader = csv.reader(csvfile)
    p=-1
    filename_bak='bak'
    for line in reader:
        if p<0:
            p=p+1
            continue
        else:
            filename=line[0]+'.npy'
            if filename==filename_bak:
                p=p+1
                print(p)
                continue
            else:
                filename_bak=filename
                img = np.load('/media/customer/Disk1/LUNG/challenge/ResampledData/'+ProjName+'Data/'+filename).astype(data_form)/255.0
                #segment is a binary mask
                seg = np.load('/media/customer/Disk1/LUNG/challenge/segmentedData/'+ProjName+'Data/'+filename)
                mask=np.zeros(img.shape,dtype=data_form)
                weight_mask=np.zeros(img.shape,dtype=data_form)+0.01
                total=(img.shape[0]-out_w)*(img.shape[1]-out_w)*(img.shape[2]-out_w)/(step**3)
                i=0
                time1=time.time()
                weight=unary_weight
                for z in range(0,img.shape[0]-48,step):
                    for y in range(0,img.shape[1]-48,step):
                        for x in range(0,img.shape[2]-48,step):
                            tile=img[z:z+depth,y:y+height,x:x+width].reshape([1,1,48,48,48])
                            tile_mask = model.predict(tile, verbose=0)
                            tile_mask=tile_mask[0,:,0]
                            tile_mask=np.reshape(tile_mask,[out_w,out_w,out_w])
                            mask[z+2:z+2+out_w,y+2:y+2+out_w,x+2:x+2+out_w]=mask[z+2:z+2+out_w,y+2:y+2+out_w,x+2:x+2+out_w]+tile_mask*weight#*gaussian_weight
                            weight_mask[z+2:z+2+out_w,y+2:y+2+out_w,x+2:x+2+out_w]=weight_mask[z+2:z+2+out_w,y+2:y+2+out_w,x+2:x+2+out_w]+weight
                            i=i+1
                            if i%100==0:
                                sys.stdout.write(' ' * 4 + '\r'+str(i).zfill(4)+'/'+str(total))
                                sys.stdout.flush()
                mask=mask/weight_mask 
#                for z in range(img.shape[0]):
#                    for y in range(img.shape[1]):
#                        for x in range(img.shape[2]):
#                            if seg[z,y,x]==0:
#                                mask[z,y,x]=0
                time2=time.time()
                print('\n')
                print(filename)
                print('\n')
                print(time2-time1)
                mask=mask*255
                mask=mask.astype(np.uint8)
                img=img*255
                img=img.astype(np.uint8)
                seg=seg.astype(np.uint8)
                write_mhd_file(output_path+ProjName+'/output/'+filename+'.mhd', mask, [mask.shape[2],mask.shape[1],mask.shape[0]])
                write_mhd_file(output_path+ProjName+'/img/'+filename+'.mhd', img, [mask.shape[2],mask.shape[1],mask.shape[0]])
                write_mhd_file(output_path+ProjName+'/seg/'+filename+'.mhd', seg, [mask.shape[2],mask.shape[1],mask.shape[0]])
                #write_mhd_file(output_path+'weight'+'.mhd', weight_mask, [mask.shape[2],mask.shape[1],mask.shape[0]])
            
            p=p+1
            print(p)
    