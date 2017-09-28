import SimpleITK  # conda install -c https://conda.anaconda.org/simpleitk SimpleITK
import SimpleITK as sitk
import numpy as np
import pandas as pd
import ntpath
import shutil
import random
import math
import multiprocessing
import os
import glob
import h5py
from joblib import Parallel, delayed
from tqdm import tqdm
import time
import random
from random import randint
import pickle as pickle
import itertools
import sys
from scipy.ndimage.interpolation import zoom


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

def get_filename(file_list, case):
    for f in file_list:
        if case in f:
            return(f)

def resample(imgs, spacing, new_spacing,order=2):
    if len(imgs.shape)==3:
        new_shape = np.round(imgs.shape * spacing / new_spacing)
        true_spacing = spacing * imgs.shape / new_shape
        resize_factor = new_shape / imgs.shape
        imgs = zoom(imgs, resize_factor, mode = 'nearest',order=order)
        return imgs, true_spacing
    elif len(imgs.shape)==4:
        n = imgs.shape[-1]
        newimg = []
        for i in range(n):
            slice = imgs[:,:,:,i]
            newslice,true_spacing = resample(slice,spacing,new_spacing)
            newimg.append(newslice)
        newimg=np.transpose(np.array(newimg),[1,2,3,0])
        return newimg,true_spacing
    else:
        raise ValueError('wrong shape')
        
def worldToVoxelCoord(worldCoord, origin, spacing):
     
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord

def load_itk_image(filename):
    with open(filename) as f:
        contents = f.readlines()
        line = [k for k in contents if k.startswith('TransformMatrix')][0]
        transformM = np.array(line.split(' = ')[1].split(' ')).astype('float')
        transformM = np.round(transformM)
        if np.any( transformM!=np.array([1,0,0, 0, 1, 0, 0, 0, 1])):
            isflip = True
        else:
            isflip = False

    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
     
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
     
    return numpyImage, numpyOrigin, numpySpacing,isflip

def lumTrans(img):
    lungwin = np.array([-1200.,600.])
    newimg = (img-lungwin[0])/(lungwin[1]-lungwin[0])
    newimg[newimg<0]=0
    newimg[newimg>1]=1
    newimg = (newimg*255).astype('uint8')
    return newimg

def savenpy(data_path,df_node,img_file,pic_path):
    mini_df = df_node[df_node["file"]==img_file]
    resolution = np.array([1,1,1])
#     resolution = np.array([2,2,2])
    name = img_file.split('/')[-1][:-4]
    Mask,origin,spacing,isflip = load_itk_image(data_path + img_file)
    sliceim = np.copy(Mask)
    num_z, height, width = sliceim.shape
    if isflip:
        Mask = Mask[:,::-1,::-1]
        print(name,'is flip!')
    newshape = np.round(np.array(Mask.shape)*spacing/resolution).astype('int')
    m1 = Mask==3
    m2 = Mask==4
    Mask = m1+m2
    
    xx,yy,zz= np.where(Mask)
    box = np.array([[np.min(xx),np.max(xx)],[np.min(yy),np.max(yy)],[np.min(zz),np.max(zz)]])
    box = box*np.expand_dims(spacing,1)/np.expand_dims(resolution,1)
    box = np.floor(box).astype('int')
    margin = 5
    extendbox = np.vstack([np.max([[0,0,0],box[:,0]-margin],0),np.min([newshape,box[:,1]+2*margin],axis=0).T]).T
    
#     convex_mask = m1
#     dm1 = process_mask(m1)
#     dm2 = process_mask(m2)
#     dilatedMask = dm1+dm2
#     Mask = m1+m2
#     extramask = dilatedMask ^ Mask
#     bone_thresh = 210
#     pad_value = 170
    
    sliceim = lumTrans(sliceim)
#     sliceim = sliceim*dilatedMask+pad_value*(1-dilatedMask).astype('uint8')
#     bones = (sliceim*extramask)>bone_thresh
#     sliceim[bones] = pad_value
    sliceim1,_ = resample(sliceim,spacing,resolution,order=1)
    sliceim2 = sliceim1[extendbox[0,0]:extendbox[0,1],
                    extendbox[1,0]:extendbox[1,1],
                    extendbox[2,0]:extendbox[2,1]]
    sliceim = sliceim2[np.newaxis,...]    
    np.save(os.path.join(pic_path,name+'_clean.npy'),sliceim)
    

    islabel = True
    if islabel:

        this_annos = np.copy(mini_df)
        label = []
        if len(this_annos)>0:
            
            for c in this_annos:
                pos = worldToVoxelCoord(c[1:4][::-1],origin=origin,spacing=spacing)
                if isflip:
                    pos[1:] = Mask.shape[1:3]-pos[1:]
                label.append(np.concatenate([pos,[c[4]/spacing[1]]]))
            
        label = np.array(label)
        if len(label)==0:
            label2 = np.array([[0,0,0,0]])
        else:
            label2 = np.copy(label).T
            label2[:3] = label2[:3]*np.expand_dims(spacing,1)/np.expand_dims(resolution,1)
            label2[3] = label2[3]*spacing[1]/resolution[1]
            label2[:3] = label2[:3]-np.expand_dims(extendbox[:,0],1)
            label2 = label2[:4].T
        np.save(os.path.join(pic_path,name+'_label.npy'),label2)     
    

    print(name)

