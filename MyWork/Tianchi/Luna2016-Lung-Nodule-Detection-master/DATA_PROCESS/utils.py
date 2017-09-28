# -*- coding: utf-8 -*-
import SimpleITK as sitk
import numpy as np
import csv
from glob import glob
import pandas as pd
import os
from tqdm import tqdm
from joblib import Parallel, delayed
import argparse
import sys
import theano
import theano.tensor as T
import copy
import matplotlib.pyplot as plt
import cv2

import keras
from keras.models import Model,load_model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.layers import Dropout
from keras.callbacks import *
from keras import initializers
from keras.layers import BatchNormalization
K.set_image_dim_ordering('th')  # Theano dimension ordering in this code

from sklearn.externals import joblib
from sklearn.cluster import KMeans
from skimage.transform import resize
from skimage import morphology
from skimage import measure
import warnings
warnings.filterwarnings("ignore")

img_rows = 512
img_cols = 512

def make_mask(center,diam,z,width,height,spacing,origin):
    mask = np.zeros([height,width]) # 0's everywhere except nodule swapping x,y to match img
    v_center = np.absolute((center-origin))/spacing
    v_diam = int(diam/(2*spacing[0])+1)
    v_xmin = np.max([0,int(v_center[0]-v_diam)])
    v_xmax = np.min([width-1,int(v_center[0]+v_diam)])
    v_ymin = np.max([0,int(v_center[1]-v_diam)]) 
    v_ymax = np.min([height-1,int(v_center[1]+v_diam)])
    v_xrange = range(v_xmin,v_xmax)
    v_yrange = range(v_ymin,v_ymax)
    x_data = [x*spacing[0]+origin[0] for x in range(width)]
    y_data = [x*spacing[1]+origin[1] for x in range(height)]
    for v_x in v_xrange:
        for v_y in v_yrange:
            p_x = spacing[0]*v_x + origin[0]
            p_y = spacing[1]*v_y + origin[1]
            if np.linalg.norm(center-np.array([p_x,p_y,z]))<=diam:
                mask[int(np.absolute((p_y-origin[1]))/spacing[1]),int(np.absolute((p_x-origin[0]))/spacing[0])] = 1.0
    return(mask)

def matrix2int16(matrix):
    m_min= np.min(matrix)
    m_max= np.max(matrix)
    matrix = matrix-m_min
    return(np.array(np.rint( (matrix-m_min)/float(m_max-m_min) * 65535.0),dtype=np.uint16))

def get_filename(file_list, case):
    for f in file_list:
        if case in f:
            return(f)

def load_train(data_path):
    folders = [x for x in os.listdir(data_path) if 'subset' in x]
    os.chdir(data_path)
    patients = []    
    for i in folders:
        os.chdir(data_path + i)
        patient_ids = [x for x in os.listdir(data_path + i) if '.mhd' in x]
        for id in patient_ids:
            j = '{}/{}'.format(i, id)
            patients.append(j)
    return patients

def cal_dist(pred_csv,anno_csv):
    dist = []
    ratio = []
    for index_csv,row_csv in pred_csv.iterrows():
        mini_ann = anno_csv[anno_csv['seriesuid']==row_csv['seriesuid']] 
        dist_ = []
        ratio_ = []
        for index_ann,row_ann in mini_ann.iterrows():        
            vec1 = np.array([row_csv['coordX'],row_csv['coordY'],row_csv['coordZ']])
            vec2 = np.array([row_ann['coordX'],row_ann['coordY'],row_ann['coordZ']])
            ratio_.append(numpy.linalg.norm(vec1 - vec2)/row_ann['diameter_mm'])
            dist_.append(numpy.linalg.norm(vec1 - vec2))
        ratio.append(np.min(ratio_))
        dist.append(np.min(dist_))  
    pred_csv['ratio'] = ratio
    pred_csv['dist'] = dist
    pred_csv = pred_csv.sort_values('dist')
    return pred_csv


def dice_coef(y_true,y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    smooth = 0.
    intersection = K.sum(y_true*y_pred)     
    return (2. * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)

def dice_coef_np(y_true,y_pred):
    smooth = 0.
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return ((2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth))


def gaussian_init(shape, name=None, dim_ordering=None):
    return initializations.normal(shape, scale=0.001, name=name, dim_ordering=dim_ordering)

def get_unet_small(options):
    inputs = Input((1, 512, 512))
    conv1 = Convolution2D(32, options['filter_width'], options['stride'], activation='elu',border_mode='same')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Convolution2D(32, options['filter_width'], options['stride'], activation='elu',border_mode='same', name='conv_1')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), name='pool_1')(conv1)
    pool1 = BatchNormalization()(pool1)

    conv2 = Convolution2D(64, options['filter_width'], options['stride'], activation='elu',border_mode='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Convolution2D(64, options['filter_width'], options['stride'], activation='elu',border_mode='same', name='conv_2')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), name='pool_2')(conv2)
    pool2 = BatchNormalization()(pool2)

    conv3 = Convolution2D(128, options['filter_width'], options['stride'], activation='elu',border_mode='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Convolution2D(128, options['filter_width'], options['stride'], activation='elu',border_mode='same', name='conv_3')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), name='pool_3')(conv3)
    pool3 = BatchNormalization()(pool3)

    conv4 = Convolution2D(256, options['filter_width'], options['stride'], activation='elu',border_mode='same')(pool3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Convolution2D(256, options['filter_width'], options['stride'], activation='elu',border_mode='same', name='conv_4')(conv4)
    conv4 = BatchNormalization()(conv4)
    # pool4 = MaxPooling2D(pool_size=(2, 2), name='pool_4')(conv4)

    # conv5 = Convolution2D(512, options['filter_width'], options['stride'], activation='elu',border_mode='same')(pool4)
    # conv5 = Dropout(0.2)(conv5)
    # conv5 = Convolution2D(512, options['filter_width'], options['stride'], activation='elu',border_mode='same', name='conv_5')(conv5)

    # up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    # conv6 = Convolution2D(256, options['filter_width'], options['stride'], activation='elu',border_mode='same')(up6)
    # conv6 = Dropout(0.2)(conv6)
    # conv6 = Convolution2D(256, options['filter_width'], options['stride'], activation='elu',border_mode='same', name='conv_6')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv4), conv3], mode='concat', concat_axis=1)

    conv7 = Convolution2D(128, options['filter_width'], options['stride'], activation='elu',border_mode='same')(up7)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Convolution2D(128, options['filter_width'], options['stride'], activation='elu',border_mode='same', name='conv_7')(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, options['filter_width'], options['stride'], activation='elu',border_mode='same')(up8)
    conv8 = Dropout(0.2)(conv8)
    conv8 = Convolution2D(64, options['filter_width'], options['stride'], activation='elu',border_mode='same', name='conv_8')(conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, options['filter_width'], options['stride'], activation='elu',border_mode='same')(up9)
    conv9 = Dropout(0.2)(conv9)
    conv9 = Convolution2D(32, options['filter_width'], options['stride'], activation='elu',border_mode='same', name='conv_9')(conv9)
    conv9 = BatchNormalization()(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid', name='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)
    model.summary()
    model.compile(optimizer=Adam(lr=options['lr'], clipvalue=1., clipnorm=1.), loss=dice_coef_loss, metrics=[dice_coef])

    return model

def train(use_existing, options, name, check_name = None):
    imgs_train = np.load(options['out_dir']+"trainImages.npy").astype(np.float32)
    imgs_mask_train = np.load(options['out_dir']+"trainMasks.npy").astype(np.float32)
    imgs_mask_train[imgs_mask_train > 0.] = 1.0
    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)    
    callbacks = [EarlyStopping(monitor='val_loss', 
                               patience = 15, 
                               verbose = 1),
                               ModelCheckpoint('/Volumes/solo/ali/Data/model/{}.h5'.format(name), 
                               monitor='val_loss', 
                               verbose = 0, save_best_only = True)]
    
    if check_name is not None:
        check_model = '/Volumes/solo/ali/Data/model/{}.h5'.format(check_name)
        model = load_model(check_model, 
                           custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
    else:
        model = get_unet_small(options)
    
    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    model.fit(x=imgs_train, y=imgs_mask_train, batch_size=options['batch_size']
              , epochs=options['epochs'], verbose=1, shuffle=True
              , validation_split=0.2, callbacks = callbacks,)
    return model

def dice_coef(y_true,y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    smooth = 0.
    intersection = K.sum(y_true*y_pred)     
    return (2. * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)

def world_2_voxel(world_coordinates, origin, spacing):
    stretched_voxel_coordinates = world_coordinates - origin
    voxel_coordinates = stretched_voxel_coordinates / spacing
    return voxel_coordinates

def voxel_2_world(voxel_coordinates, origin, spacing):
    stretched_voxel_coordinates = voxel_coordinates * spacing
    world_coordinates = stretched_voxel_coordinates + origin
    return world_coordinates

def get_lung_mask(imgs_to_process):
    for i in range(len(imgs_to_process)):
        img = imgs_to_process[i]
        mean = np.mean(img)
        std = np.std(img)
        img = img - mean
        img = img / std
        middle = img[100:400, 100:400]
        mean = np.mean(middle)
        max_im = np.max(img)
        min_im = np.min(img)

        img[img == max_im] = mean
        img[img == min_im] = mean

        kmeans = KMeans(n_clusters=2).fit(np.reshape(middle, [np.prod(middle.shape), 1]))
        centers = sorted(kmeans.cluster_centers_.flatten())
        threshold = np.mean(centers)
        thresh_img = np.where(img < threshold, 1.0, 0.0)  # threshold the image

        eroded = morphology.erosion(thresh_img, np.ones([4, 4]))
        dilation = morphology.dilation(eroded, np.ones([10, 10]))

        labels = measure.label(dilation)
        label_vals = np.unique(labels)
        regions = measure.regionprops(labels)
        good_labels = []
        for prop in regions:
            B = prop.bbox
            if B[2] - B[0] < 475 and B[3] - B[1] < 475 and B[0] > 40 and B[2] < 472:
                good_labels.append(prop.label)
        mask = np.ndarray([img.shape[0], img.shape[1]], dtype=np.int8)
        mask[:] = 0
        for N in good_labels:
            mask = mask + np.where(labels == N, 1, 0)
        mask = morphology.dilation(mask, np.ones([10, 10]))  # one last dilation
        imgs_to_process[i] = mask
    return imgs_to_process

def get_unet():
    inputs = Input((1,img_rows, img_cols))
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1.0e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model