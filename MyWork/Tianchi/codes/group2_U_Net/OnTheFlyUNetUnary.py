#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 18:47:36 2017

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
from keras import backend as K
import random
import os
from config import config
from mhd_util import *
working_path = "/media/customer/Disk1/LUNG/challenge/"
weights_path = "/home/customer/document/lung-nodule-detection/data/weights/"
output_path = "/home/customer/document/lung-nodule-detection/data/output/"

window=48
out_w=44
height = window
width = window
depth = window
batch_size=8
N_sample=19500#1940#20000
#N_back=0#10000
N_val=269#1860*2
#N_val_back=0#1860
K.set_image_dim_ordering('th')  # Theano dimension ordering in this code
crop_size=2
data_form=np.float32
shape=0
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    #p=y_true.eval()
    #print(p)
    if K.max(y_true_f)==0:
        y_true_ff=1-y_true_f
        y_pred_ff=1-y_pred_f
        intersection = K.sum(y_true_ff * y_pred_ff)
        return (2. * intersection + config["smooth"]) / (K.sum(y_true_ff) + K.sum(y_pred_ff) + config["smooth"])
    else:    
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + config["smooth"]) / (K.sum(y_true_f) + K.sum(y_pred_f) + config["smooth"])


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

#def unet_model_3d():
#    inputs = Input(config["input_shape"])
#    conv1 = Conv3D(32, 3, 3, 3, activation='relu', border_mode='same')(inputs)
#    conv1 = Conv3D(32, 3, 3, 3, border_mode='same')(conv1)
#    conv1 = normalization.BatchNormalization(epsilon=1e-06, mode=1, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one')(conv1)
#    conv1 = core.Activation('relu')(conv1)
#    pool1 = MaxPooling3D(pool_size=config["pool_size"])(conv1)
#
#    conv2 = Conv3D(64, 3, 3, 3, activation='relu', border_mode='same')(pool1)
#    conv2 = Conv3D(64, 3, 3, 3, border_mode='same')(conv2)
#    conv2 = normalization.BatchNormalization(epsilon=1e-06, mode=1, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one')(conv2)
#    conv2 = core.Activation('relu')(conv2)
#    pool2 = MaxPooling3D(pool_size=config["pool_size"])(conv2)
#
#    conv3 = Conv3D(128, 3, 3, 3, activation='relu', border_mode='same')(pool2)
#    conv3 = Conv3D(128, 3, 3, 3, activation='relu', border_mode='same')(conv3)
#
#    up4 = merge([UpSampling3D(size=config["pool_size"])(conv3), conv2], mode='concat', concat_axis=1)
#    conv4 = Conv3D(64, 3, 3, 3, activation='relu', border_mode='same')(up4)
#    conv4 = Conv3D(256, 3, 3, 3, activation='relu', border_mode='same')(conv4)
#
#    up5 = merge([UpSampling3D(size=config["pool_size"])(conv4), conv1], mode='concat', concat_axis=1)
#    conv5 = Conv3D(32, 3, 3, 3, activation='relu', border_mode='valid')(up5)
#    conv5 = Conv3D(32, 3, 3, 3, activation='relu', border_mode='valid')(conv5)
#
#    conv6 = Conv3D(config["n_labels"], 1, 1, 1)(conv5)
#    conv6 = core.Reshape((1,out_w*out_w*out_w))(conv6)
#    conv6 = core.Permute((2,1))(conv6)
#    #conv6 = 
#    act = Activation('sigmoid')(conv6)
#    model = Model(input=inputs, output=act)
#
#    #model.compile(optimizer=Adam(lr=config["initial_learning_rate"]), loss='categorical_crossentropy',metrics=['fbeta_score'])
#    model.compile(optimizer=Adam(lr=config["initial_learning_rate"]), loss=dice_coef_loss,metrics=[dice_coef])
#    return model
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

def Active_Generate(train_imgs,train_masks,test):
    
    while 1:
        img_window=train_imgs.shape[2]
        img_n=train_imgs.shape[0]
        for s in range(img_n/batch_size):
            X=np.zeros([batch_size,1,img_window,img_window,img_window],dtype=data_form)
            Y=np.zeros([batch_size,1,img_window-2*crop_size,img_window-2*crop_size,img_window-2*crop_size],dtype=data_form)
            for t in range(batch_size):
                i=random.randint(0,img_n-1)
                if test==False:
                    #if i<N_sample:
                        #j=random.randint(0,N_sample-1)
                    X[t]=train_imgs[i].astype(data_form)/255.0
                    Y[t,0]=train_masks[i,0,crop_size:window-crop_size,crop_size:window-crop_size,crop_size:window-crop_size].astype(data_form)
                        #Y[t,1]=1-train_masks[i,0,crop_size:window-crop_size,crop_size:window-crop_size,crop_size:window-crop_size].astype(data_form)
                    #else:
                    #    X[t]=train_imgs[i].astype(data_form)/255.0
                    #    Y[t,0]=0
                        #Y[t,1]=1
                else:
                    #if i<N_val:
                    X[t]=train_imgs[i].astype(data_form)/255.0
                    Y[t,0]=train_masks[i,0,crop_size:window-crop_size,crop_size:window-crop_size,crop_size:window-crop_size].astype(data_form)
                        #Y[t,1]=1-train_masks[i,0,crop_size:window-crop_size,crop_size:window-crop_size,crop_size:window-crop_size].astype(data_form)
                    #else:
                    #    X[t]=train_imgs[i].astype(data_form)/255.0
                    #    Y[t,0]=0
                        #Y[t,1]=1
            Y=np.reshape(Y,[Y.shape[0],Y.shape[1],Y.shape[2]*Y.shape[3]*Y.shape[4]]).transpose([0,2,1])
            yield (X, Y)

if __name__ == '__main__':
    print('-' * 30)
    print('Loading and preprocessing train data...')
    print('-' * 30)
    FromFile=False

    imgs_train = np.load(working_path + "final_train_patch.npy")
    mask_train = np.load(working_path + "final_train_gt.npy")
    
    
    #imgs_test=imgs_train[N_sample-N_val:N_sample]
    #mask_test=mask_train[N_sample-N_val:N_sample]
    #imgs_train=imgs_train[0:N_sample-N_val]
    #mask_train=mask_train[0:N_sample-N_val]
    imgs_test=np.load(working_path + "val_patch_48.npy")#imgs_train[N_sample:N_sample+1860]
    imgs_test=np.reshape(imgs_test,[imgs_test.shape[0],1,imgs_test.shape[1],imgs_test.shape[2],imgs_test.shape[3]])
    mask_test=np.load(working_path + "val_CRF_label.npy")#np.load(working_path + "val_mask_48.npy")#np.load(working_path + "val_CRF_label.npy")#mask_train[N_sample:N_sample+1860]
    mask_test=np.reshape(mask_test,[imgs_test.shape[0],1,imgs_test.shape[2],imgs_test.shape[3],imgs_test.shape[4]])
    #N_sample=imgs_train.shape[0]
    #N_val=imgs_test.shape[0]
    #imgs_back = np.load(working_path + "patch_background.npy")#[0:N_back]
    #test_back = imgs_back[20000:20000+N_val_back]
    #imgs_back = imgs_back[0:N_back]
    #mask_back = np.load(working_path + "gt_background.npy")[0:N_back]
    
    #imgs_train=imgs_train[0:N_sample]
    #mask_train=mask_train[0:N_sample]
    
    #imgs_train=np.concatenate((imgs_train,imgs_back),axis=0)
    #imgs_test=np.concatenate((imgs_test,test_back),axis=0)
    
    #test_back_mask=np.zeros([N_val_back,1,window,window,window],dtype=np.uint8)
    #mask_test=np.concatenate((mask_test,test_back_mask),axis=0)
    #mask_train=np.concatenate((mask_train,mask_back),axis=0)
    gen=Active_Generate(imgs_train,mask_train,False)
    gen_test=Active_Generate(imgs_test,mask_test,True)
        
    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)
    model = unet_model_MultiScale()
    # Saving weights to unet.hdf5 at checkpoints
    #model_checkpoint = ModelCheckpoint(weights_path + 'unet.hdf5', monitor='loss', save_best_only=True)
    model_checkpoint = ModelCheckpoint(weights_path + 'unet_unary_challenge.hdf5', 
                               verbose=1, 
                               monitor='val_loss',
                               mode='auto', 
                               save_best_only=True) 
    #
    # Should we load existing weights?
    # Set argument for call to train_and_predict to true at end of script
    use_existing=True
    if use_existing:
        model.load_weights(weights_path + 'unet_unary_challenge.hdf5')
    
    #
    # The final results for this tutorial were produced using a multi-GPU
    # machine using TitanX's.
    # For a home GPU computation benchmark, on my home set up with a GTX970
    # I was able to run 20 epochs with a training set size of 320 and
    # batch size of 2 in about an hour. I started getting reseasonable masks
    # after about 3 hours of training.
    #
    print('-' * 30)
    print('Fitting model...')
    print('-' * 30)

    (X,Y)=next(gen)
    #for i in range(8):
    #    m=np.reshape(Y[i,:,0],[out_w,out_w,out_w])
    #    write_mhd_file(output_path+str(i)+'y.mhd', m, [m.shape[2],m.shape[1],m.shape[0]])
    #model.fit(imgs_train, mask_train, batch_size=batch_size, nb_epoch=50, verbose=1, shuffle=True,validation_data=[imgs_test,mask_test],
    #          callbacks=[model_checkpoint])
    model.fit_generator(gen,#generate_arrays_from_file(train_imgs,train_masks,patch_height,patch_width,batch_size,N_subimgs,N_imgs), 
                    epochs=30, 
                    steps_per_epoch=(N_sample)/batch_size,
                    #samples_per_epoch=N_sample+N_back,
                    verbose=1, 
                    callbacks=[model_checkpoint],
                    validation_data=gen_test,
                    validation_steps=(N_val)/batch_size)
    # loading best weights from training session
    print('-' * 30)
    print('Loading saved weights...')
    print('-' * 30)
    #model.load_weights(weights_path + 'unet.hdf5')

    print('-' * 30)
    print('Predicting masks on test data...')
    print('-' * 30)
    
    #watch -n 0.5 nvidia-smi