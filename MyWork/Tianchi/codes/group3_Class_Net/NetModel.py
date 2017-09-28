#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 22:41:22 2017

@author: customer
"""

from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, Activation, Reshape, core,normalization,concatenate,add,Cropping3D,Dropout,GaussianNoise,Dense,Flatten
from keras.optimizers import Adam,Adagrad
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.utils.vis_utils import plot_model
import random
import os
import numpy as np
from config import config
from mhd_util import *
#working_path = "/home/customer/document/lung-nodule-detection/data/"
#weights_path = "/home/customer/document/lung-nodule-detection/data/weights/"
#output_path = "/home/customer/document/lung-nodule-detection/data/output/"

#window=48
#out_w=44
#height = window
#width = window
#depth = window
#batch_size=4
K.set_image_dim_ordering('th')  # Theano dimension ordering in this code
#crop_size=2
data_form=np.float32
#shape=0

#def spacial_red(inputs,channels):
#    ch1 = Conv3D(channels/4, (1, 1, 1), activation='relu', padding='same')(inputs)
#    #ch1 = Conv3D(channels*5/16, (3, 3, 3), activation='relu',padding='same')(ch1)
#    ch1 = Conv3D(channels*5/16, (3, 3, 3), padding='same')(ch1)
#    ch1 = normalization.BatchNormalization(epsilon=1e-06, axis=1, momentum=0.9, weights=None, beta_initializer='zero', gamma_initializer='one')(ch1)
#    ch1 = core.Activation('relu')(ch1)
#    ch1 = Conv3D(channels*7/16, (3, 3, 3), strides=(2,2,2), padding='same')(ch1)
#    ch1 = normalization.BatchNormalization(epsilon=1e-06, axis=1, momentum=0.9, weights=None, beta_initializer='zero', gamma_initializer='one')(ch1)
#    ch1 = core.Activation('relu')(ch1)
#    #ch1 = Conv3D(channels*7/16, 3, 3, 3, subsample=(2,2,2), activation='relu',padding='same')(ch1)
#    
#    ch2 = Conv3D(channels/4, (1, 1, 1), activation='relu', padding='same')(inputs)
#    ch2 = Conv3D(channels*5/16, (3, 3, 3), strides=(2,2,2), padding='same')(ch2)
#    ch2 = normalization.BatchNormalization(epsilon=1e-06, axis=1, momentum=0.9, weights=None, beta_initializer='zero', gamma_initializer='one')(ch2)
#    ch2 = core.Activation('relu')(ch2)
#    #ch2 = Conv3D(channels*5/16, 1, 1, 1, subsample=(2,2,2), activation='relu', padding='same')(ch2)
#    
#    ch3 = Conv3D(channels/4, (3, 3, 3), strides=(2,2,2), activation='relu', padding='same')(inputs)
#    
#    ch4 = MaxPooling3D(pool_size=(2,2,2))(inputs)
#    
#    out = concatenate([ch1,ch2,ch3,ch4], axis=1)
#    
#    return out
#
#def res(inputs,channels,n_channels):
#    ch1 = Conv3D(n_channels, (1, 1, 1), activation='relu', padding='same')(inputs)
#    #ch1 = Conv3D(n_channels, 3, 3, 3, activation='relu',padding='same')(ch1)
#    ch1 = Conv3D(n_channels, (3, 3, 3), padding='same')(ch1)
#    ch1 = normalization.BatchNormalization(epsilon=2e-05, axis=1, momentum=0.9, weights=None, beta_initializer='zero', gamma_initializer='one')(ch1)
#    ch1 = core.Activation('relu')(ch1)    
#    ch1 = Conv3D(n_channels, (3, 3, 3), padding='same')(ch1)
#    ch1 = normalization.BatchNormalization(epsilon=2e-05, axis=1, momentum=0.9, weights=None, beta_initializer='zero', gamma_initializer='one')(ch1)
#    ch1 = core.Activation('relu')(ch1)
#    
#    ch2 = Conv3D(n_channels, (1, 1, 1), activation='relu', padding='same')(inputs)
#    ch2 = Conv3D(n_channels, (3, 3, 3), padding='same')(ch2)
#    ch2 = normalization.BatchNormalization(epsilon=2e-05,  axis=1, momentum=0.9, weights=None, beta_initializer='zero', gamma_initializer='one')(ch2)
#    ch2 = core.Activation('relu')(ch2)
#    #ch2 = Conv3D(n_channels, (3, 3, 3), activation='relu', padding='same')(ch2)
#    
#    ch3 = Conv3D(n_channels, (3, 3, 3), activation='relu', padding='same')(inputs)
#    
#    out = concatenate([ch1,ch2,ch3],  axis=1)
#    out = Conv3D(channels, (1, 1, 1), activation='relu', padding='same')(out)
#    
#    out= add([inputs,out])
#    out= core.Activation('relu')(out)
#    
#    return out
#
#def f_red(inputs,channels):
#    out = Conv3D(channels/2, (1, 1, 1), activation='relu', padding='same')(inputs)
#    
#    return out
#
#def ClassNet():
#    inputs = Input(config["input_shape"])
#    conv1=Conv3D(64, (3, 3, 3), activation='relu', padding='same')(inputs)
#    
#    l=spacial_red(conv1,64)
#    #l=res(l,128,128)
#    
#    l=spacial_red(l,128)
#    #l=res(l,256,256)
#    
#    l=spacial_red(l,256)
#    #l=res(l,512,512)
#    
#    l=f_red(l,512)
#    #l=res(l,256,256)
#    
#    l=f_red(l,256)
#    
#    l=Conv3D(128,(6,6,6),activation='relu', padding='valid')(l)
#    l=Conv3D(2,(1,1,1))(l)
#    l=core.Reshape((2,1))(l)
#    l=core.Permute((2,1))(l)
#    act = Activation('softmax')(l)
#    model = Model(inputs=inputs, outputs=act)
#    model.compile(optimizer=Adam(lr=config["initial_learning_rate"]), loss='mean_squared_error',metrics=['accuracy'])
#    return model


def ConvNet48(ch):
    #conv1 = Conv3D(64, (5, 5, 5), padding='valid',activation='relu')(ch)
       
    conv2 = Conv3D(64, (3, 3, 3), padding='valid',activation='relu')(ch)
    conv2 = Conv3D(64, (3, 3, 3), padding='valid')(conv2)
    conv2 = normalization.BatchNormalization(epsilon=2e-05, axis=1, momentum=0.9, weights=None, beta_initializer='zero', gamma_initializer='one')(conv2)
    conv2 = core.Activation('relu')(conv2)
    pool1 = MaxPooling3D(pool_size=(2,2,2))(conv2)
    
    #conv3 = Conv3D(64, (3, 3, 3), padding='valid',activation='relu')(pool1)
    
    conv4 = Conv3D(128, (3, 3, 3), padding='valid',activation='relu')(pool1)   
    conv4 = Conv3D(128, (3, 3, 3), padding='valid',activation='relu')(conv4)
    conv4 = normalization.BatchNormalization(epsilon=2e-05, axis=1, momentum=0.9, weights=None, beta_initializer='zero', gamma_initializer='one')(conv4)
    conv4 = core.Activation('relu')(conv4)
    #conv4 = normalization.BatchNormalization(epsilon=2e-05, axis=1, momentum=0.9, weights=None, beta_initializer='zero', gamma_initializer='one')(conv4)
    #conv4 = core.Activation('relu')(conv4)
    #conv4 = Dropout(0.2)(conv4)
    pool2 = MaxPooling3D(pool_size=(2,2,2))(conv4)
    
    conv5 = Conv3D(256, (3, 3, 3), padding='valid',activation='relu')(pool2)    
    conv6 = Conv3D(256, (3, 3, 3), padding='valid',activation='relu')(conv5)
    conv6 = normalization.BatchNormalization(epsilon=2e-05, axis=1, momentum=0.9, weights=None, beta_initializer='zero', gamma_initializer='one')(conv6)
    conv6 = core.Activation('relu')(conv6)
    conv6 = Conv3D(256, (3, 3, 3), padding='valid',activation='relu')(conv6)
    conv6 = normalization.BatchNormalization(epsilon=2e-05, axis=1, momentum=0.9, weights=None, beta_initializer='zero', gamma_initializer='one')(conv6)
    conv6 = core.Activation('relu')(conv6)
    #pool3 = MaxPooling3D(pool_size=(2,2,2))(conv6)
    #conv6 = normalization.BatchNormalization(epsilon=2e-05, axis=1, momentum=0.9, weights=None, beta_initializer='zero', gamma_initializer='one')(conv6)
    #conv6 = core.Activation('relu')(conv6)
    #conv6 = Dropout(0.2)(conv6)
    #conv7 = Conv3D(256, (3, 3, 3), padding='valid',activation='relu')(pool3)
    conv6 = Flatten()(conv6)
    
    
    conv7 = Dense(256)(conv6)#Conv3D(256, (5, 5, 5), padding='valid',activation='relu')(conv6)
    #conv7 = Dropout(0.2)(conv7)
    #conv8 = Conv3D(2,(1,1,1), padding='same',activation='relu')(conv7)
    #act = Dense(2)(conv7)#Conv3D(2,(1,1,1), padding='same',activation='relu')(conv9)
    #act = Activation('softmax')(act)
    return conv7
    
def ConvNet32(ch): 
    #conv3 = Conv3D(64, (3, 3, 3), padding='valid',activation='relu')(ch)
    
    conv4 = Conv3D(64, (5, 5, 5), padding='valid')(ch)
    conv4 = normalization.BatchNormalization(epsilon=2e-05, axis=1, momentum=0.9, weights=None, beta_initializer='zero', gamma_initializer='one')(conv4)
    conv4 = core.Activation('relu')(conv4)
    #conv4 = Dropout(0.4)(conv4)
    
    pool2 = MaxPooling3D(pool_size=(2,2,2))(conv4)
    
    #conv5 = Conv3D(64, (3, 3, 3), padding='valid',activation='relu')(pool2)
    
    conv6 = Conv3D(64, (5, 5, 5), padding='valid')(pool2)
    conv6 = normalization.BatchNormalization(epsilon=2e-05, axis=1, momentum=0.9, weights=None, beta_initializer='zero', gamma_initializer='one')(conv6)
    conv6 = core.Activation('relu')(conv6)
    #conv6 = Dropout(0.4)(conv6)
    
    pool3 = MaxPooling3D(pool_size=(2,2,2))(conv6)
    #conv7 = Conv3D(64, (3, 3, 3), padding='valid',activation='relu')(conv6)
    #conv7 = Dropout(0.2)(conv6)
    conv8 = Conv3D(64, (3, 3, 3), padding='valid')(pool3)
    conv8 = normalization.BatchNormalization(epsilon=2e-05, axis=1, momentum=0.9, weights=None, beta_initializer='zero', gamma_initializer='one')(conv8)
    conv8 = core.Activation('relu')(conv8)
    #conv8 = Dropout(0.4)(conv8)
    conv8 = Flatten()(conv8)
    
    
    conv9 = Dense(256)(conv8)#Conv3D(256, (6, 6, 6), padding='valid',activation='relu')(conv8)
    #conv9 = Dropout(0.4)(conv9)
    
    #act = Dense(2)(conv9)#Conv3D(2,(1,1,1), padding='same',activation='relu')(conv9)
    #act = Activation('softmax')(act)
    return conv9

def ConvNet16(ch):   
    #conv5 = Conv3D(64, (3, 3, 3), padding='valid',activation='relu')(ch)
    
    conv6 = Conv3D(64, (3, 3, 3), padding='valid')(ch)
    conv6 = normalization.BatchNormalization(epsilon=2e-05, axis=1, momentum=0.9, weights=None, beta_initializer='zero', gamma_initializer='one')(conv6)
    conv6 = core.Activation('relu')(conv6)
    #conv6 = Dropout(0.2)(conv6)
    
    pool3 = MaxPooling3D(pool_size=(2,2,2))(conv6)
    
    #conv7 = Conv3D(64, (3, 3, 3), padding='valid',activation='relu')(conv6)
    
    conv7 = Conv3D(64, (3, 3, 3), padding='valid')(pool3)
    conv7 = normalization.BatchNormalization(epsilon=2e-05, axis=1, momentum=0.9, weights=None, beta_initializer='zero', gamma_initializer='one')(conv7)
    conv7 = core.Activation('relu')(conv7)
    #conv7 = Dropout(0.2)(conv7)
    
    
    conv8 = Conv3D(64, (3, 3, 3), padding='valid')(conv7)
    conv8 = normalization.BatchNormalization(epsilon=2e-05, axis=1, momentum=0.9, weights=None, beta_initializer='zero', gamma_initializer='one')(conv8)
    conv8 = core.Activation('relu')(conv8)
    #conv8 = Dropout(0.2)(conv8)
    
#    conv8 = Conv3D(64, (3, 3, 3), padding='valid')(conv7)
#    #conv9 = normalization.BatchNormalization(epsilon=2e-05, axis=1, momentum=0.9, weights=None, beta_initializer='zero', gamma_initializer='one')(conv9)
#    conv8 = core.Activation('relu')(conv8)
    #conv8 = Dropout(0.2)(conv8)
    
    conv9 = Flatten()(conv8)
    #conv9=Conv3D(64, (3, 3, 3), padding='valid',activation='relu')(conv8)
    
    
    conv10 = Dense(256)(conv9)#Conv3D(256, (4, 4, 4), padding='valid',activation='relu')(conv8)
    #conv10 = Dropout(0.2)(conv10)
    #act = Dense(2)(conv10)#Conv3D(2,(1,1,1), padding='same',activation='relu')(conv9)
    #act = Activation('softmax')(act)
    return conv10
def ClassNet_MultiScale():
    inputs = Input((1,48,48,48))
    #noise=GaussianNoise(stddev=0.01,input_shape=(1,48,48,48))(inputs)
    ch1=inputs
    #ch1=inputs#add([inputs,noise])
    ch2=Cropping3D(((8,8),(8,8),(8,8)))(inputs)
    ch3=Cropping3D(((16,16),(16,16),(16,16)))(inputs)
    
    #ch2=UpSampling3D(size=(2,2,2))(ch2)
    #ch3=UpSampling3D(size=(4,4,4))(ch3)
    
    ch1=ConvNet48(ch1)
    ch2=ConvNet32(ch2)
    ch3=ConvNet16(ch3)
    #ch2=ConvNet32(ch2)    
    #ch3=ConvNet12(ch3)
    
    #fusion=add([ch1,ch2,ch3])
    fusion=concatenate([ch1,ch2,ch3],axis=1)
    fusion=Dense(2)(fusion)#Conv3D(2,(1,1,1), padding='same',activation='relu')(fusion)
    fusion=core.Reshape((2,1))(fusion)
    #a=core.Reshape((6,1))(fusion)
    a=core.Permute((2,1))(fusion)
    act=Activation('softmax')(a)
    model = Model(inputs=inputs, outputs=act)
    model.compile(optimizer=Adam(lr=config["initial_learning_rate"]), loss='mean_squared_error',metrics=['accuracy'])
    return model
def ClassNet_48():
    inputs = Input((1,48,48,48))
    ch1=ConvNet48(inputs)
    ch1=Dense(256)(ch1)
    ch1=Dense(2)(ch1)
    a=core.Reshape((2,1))(ch1)
    a=core.Permute((2,1))(a)
    act=Activation('softmax')(a)
    model = Model(inputs=inputs, outputs=act)
    model.compile(optimizer=Adam(lr=config["initial_learning_rate"]), loss='categorical_crossentropy',metrics=['categorical_accuracy'])
    return model


def GaussianTest():
    inputs = Input((1,48,48,48))
    noise = GaussianNoise(stddev=0.5,input_shape=(1,48,48,48))(inputs)
    model = Model(inputs=inputs, outputs=noise)
    return model
def CroppingTest():
    inputs = Input((1,48,48,48))
    #noise=GaussianNoise(stddev=0.01,input_shape=(1,48,48,48))(inputs)
    ch1=inputs
    ch2=Cropping3D(((12,12),(12,12),(12,12)))(inputs)#(inputs)
    ch3=Cropping3D(((18,18),(18,18),(18,18)))(inputs)#(inputs)
    
    ch2=UpSampling3D(size=(2,2,2))(ch2)
    ch3=UpSampling3D(size=(4,4,4))(ch3)
    fusion=concatenate([ch1,ch2,ch3],axis=1)
    
    model = Model(inputs=inputs, outputs=fusion)
    return model
    
    