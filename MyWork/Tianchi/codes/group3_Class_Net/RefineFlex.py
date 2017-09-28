#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May 21 14:03:38 2017

@author: customer
"""

from __future__ import print_function

import numpy as np
from keras.models import Model
from keras.layers import Input, merge, Conv3D, MaxPooling3D, UpSampling3D, Activation, Reshape, core,normalization,concatenate,add
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
#from keras.utils.visualize_util import plot
import random
import os
from config import config
from mhd_util import *
from skimage import data, util,transform
from skimage.measure import label,regionprops
import SimpleITK as sitk
import csv
import matplotlib.pyplot as pl
from PIL import Image
from mhd_util import *
from NetModel_conc import *#_conc import *
out_w=10
w=48

working_path = "/home/customer/document/lung-nodule-detection/data/"
weights_path = "/home/customer/document/lung-nodule-detection/data/weights/"
output_path = "/home/customer/document/lung-nodule-detection/data/output/"

window=48
#out_w=44
height = window
width = window
depth = window
batch_size=8
#N_sample=4680-740#20000#4680-740#20000
#N_back=0#10000
#N_val=740#1860*2#740#1860*2
#N_val_back=0#1860
K.set_image_dim_ordering('th')  # Theano dimension ordering in this code
crop_size=2
data_form=np.float32
shape=0


def dis(center1,center2):
    d=(center1[0]-center2[0])**2+(center1[1]-center2[1])**2+(center1[2]-center2[2])**2
    d=d**0.5
    return d
if __name__ == '__main__':
    os.system("sh /home/customer/document/lung-nodule-detection/challenge/ClassTrain/Data/train/clean_val.sh")
#    model48=ClassNet_48()
#    model36=ClassNet_36()
#    model24=ClassNet_24()
#
#    model48.load_weights(weights_path + 'classnet_48.hdf5')
#    model36.load_weights(weights_path + 'classnet_36.hdf5')
#    model24.load_weights(weights_path + 'classnet_24.hdf5')
#    model=model24
    model=ClassNet_FlexWindow()#FlexWindow()
    model2=ClassNet_FlexWindow()
    #model3=ClassNet_FlexWindow()
    #plot(model,to_file='/home/customer/document/lung-nodule-detection/huangyj/ClassModel.png',show_shapes=True)
    ProjName='val'
    Display=True
    thresh=50
    thresh_seg=50
    #plot(model,to_file='/media/customer/新加卷2/LUNG/ClassModel.png',show_shapes=True)

    model.load_weights(weights_path + 'classnet_Flex_6_27.hdf5')#_6_29.hdf5')#conc_6_22_best.hdf5')
    model2.load_weights(weights_path + 'classnet_Flex_7_7.hdf5')
    #model3.load_weights(weights_path + 'classnet_Flex_6_27.hdf5')
    #model2=ClassNet_FlexWindow()
    #model2.load_weights(weights_path + 'classnet_Flex_6_27.hdf5')#conc_6_22_best.hdf5')
    
    csvfile = file('/media/customer/Disk1/LUNG/challenge/csv/'+ProjName+'_annotations_vol.csv', 'rb')
    reader = csv.reader(csvfile)
    p=-1
    filename_bak='bak'
    
    csvf=file('/home/customer/document/lung-nodule-detection/challenge/evaluationScript/test/'+ProjName+'_Submission'+str(thresh)+'.csv','w')
    
    fieldnames=['seriesuid','coordX','coordY','coordZ','probability']
    writer = csv.DictWriter(csvf, fieldnames=fieldnames)
    writer.writeheader()
    bak='0'
    ClassRoot='/home/customer/document/lung-nodule-detection/challenge/ClassTrain/Data/'
    CompareRoot='/home/customer/document/lung-nodule-detection/challenge/ClassTrain/Data/val_compare_Final_3/'
    pid=0
    #os.mkdir(CompareRoot)
    for line in reader:
        if p<0:
            p=p+1
            continue
        else:
            #if line[0]!=bak:
                filename=line[0]+'.npy'
                pred = sitk.ReadImage('/media/customer/Disk1/LUNG/challenge/'+ProjName+'/output/'+filename+'3.mhd')
                pred = sitk.GetArrayFromImage(pred)
                pred=pred.astype(np.float32)
                seg = sitk.ReadImage('/media/customer/Disk1/LUNG/challenge/'+ProjName+'/reseg/'+filename+'.mhd')
                seg = sitk.GetArrayFromImage(seg)
                seg2 = sitk.ReadImage('/media/customer/Disk1/LUNG/challenge/'+ProjName+'/seg/'+filename+'.mhd')
                seg2 = sitk.GetArrayFromImage(seg2)
                img = sitk.ReadImage('/media/customer/Disk1/LUNG/challenge/'+ProjName+'/img/'+filename+'.mhd')  
                img = sitk.GetArrayFromImage(img)
                #mean=np.load(ClassRoot+'train/'+'mean.npy')/255.0
                #std=np.load(ClassRoot+'train/'+'std.npy')/255.0
                mask=seg>0
                mx=np.ma.array(seg,mask=mask)
                seg=mx.mask.astype(np.uint8)
                pred=pred*seg
                
                mask=pred>thresh_seg
                mx=np.ma.array(pred,mask=mask)
                pred_temp=mx.mask.astype(np.uint8)*255
                

                #mean=np.average(img,weights=seg2)
                #mean=mean/255.0
                
                connect_map=label(pred_temp, connectivity= 2)
                props = regionprops(connect_map)
                m=np.zeros([len(props),1,2])
                prob=np.zeros([len(props),1,2])
                if ProjName!='new_test' and ProjName!='test':
                    center=np.zeros([3],np.int)
                    center[0]=int(line[3])
                    center[1]=int(line[2])
                    center[2]=int(line[1])
                    Diam=int(line[4])
                for i in range(len(props)):
                    location=props[i]['bbox']
                    startz=location[0]
                    starty=location[1]
                    startx=location[2]
                    endz=location[3]
                    endy=location[4]
                    endx=location[5]
                    pcenter_o=[(startz+endz)/2,(starty+endy)/2,(startx+endx)/2]
                    pcenter=[(startz+endz)/2,(starty+endy)/2,(startx+endx)/2]
                    Diam=np.max([endz-startz,endy-starty,endx-startx])
                    #Diam=np.max([endy-starty,endx-startx])
                    for t in range(3):
                        if pcenter[t]<w/2:
                            pcenter[t]=w/2
                        elif pcenter[t]>pred.shape[t]-w/2:
                            pcenter[t]=pred.shape[t]-w/2
                    
                    #pcenter_o=props[i]['weighted_centroid']
                            
                    #patch=img[pcenter[0]-w/2:pcenter[0]+w/2,pcenter[1]-w/2:pcenter[1]+w/2,pcenter[2]-w/2:pcenter[2]+w/2].astype(np.float32)/255
                    pred_patch=pred_temp[startz:endz,starty:endy,startx:endx].astype(np.float32)/255
                    pred_patch2=pred[startz:endz,starty:endy,startx:endx]             

                    #patch_center=np.where(pred_patch==np.max(pred_patch))
                    patch_center=np.where(pred_patch2>thresh_seg)
                    l_center=np.zeros([3])
                    for j in range(3):
                        #l_center[j]=np.average(patch_center[j])#.astype(np.int)
                        l_center[j]=np.average(patch_center[j],weights=pred_patch2[patch_center])
                    pcenter_o[0]=startz+l_center[0]
                    pcenter_o[1]=starty+l_center[1]
                    pcenter_o[2]=startx+l_center[2]
                    
                    for t in range(3):
                        if pcenter_o[t]<w/2:
                            pcenter[t]=w/2
                        elif pcenter_o[t]>pred.shape[t]-w/2:
                            pcenter[t]=pred.shape[t]-w/2
                        else:
                            pcenter[t]=int(pcenter_o[t])
                    #move=int(pcenter[0]-pcenter_o[0])
                    
                    #if Diam<12:
                    #patch1=img[pcenter[0]-12:pcenter[0]+12,pcenter[1]-12:pcenter[1]+12,pcenter[2]-12:pcenter[2]+12].astype(np.float32)/255
                    #patch1=transform.resize(patch1,(50,50,50),mode='nearest')[1:49,1:49,1:49]
                    #patch1-=mean[0]
                    #patch1/=std[0]
                    #else:
                    
                    #patch-=mean[0]
                    #patch/=std[0]
                    
                    #patch1=np.reshape(patch1,[1,1,48,48,48])
                    #patch=np.reshape(patch,[1,1,48,48,48])
                    #m[i] = model.predict(patch, verbose=0)#0.3*model48.predict(patch, verbose=0)+0.4*model36.predict(patch, verbose=0)+0.3*model24.predict(patch, verbose=0)#model.predict(patch, verbose=0)
                    #prob[i]= model.predict(patch1, verbose=0)
                    
                    if pcenter_o[0]>w/2 and np.max(pred_patch2)>thresh and Diam>=3 and Diam<40:# and pcenter_o[0]<pred.shape[0]-w/2 and np.max(pred_patch2)>thresh:# and pcenter_o[1]>w/2 and pcenter_o[1]<pred.shape[1]-w/2 and pcenter_o[2]>w/2 and pcenter_o[2]<pred.shape[2]-w/2:
                        if Diam*3<w:
                            D=3*Diam#int(Diam*2.5)
                            D2=int(2.5*Diam)
                            D3=int(2*Diam)
                        else:
                            D=48
                            D2=48
                            D3=48
                        for t in range(3):
                            if pcenter[t]-D/2<0:
                                D=pcenter[t]*2
                            elif pcenter[t]+D/2>pred.shape[t]:
                                D=(pred.shape[t]-pcenter[t])*2
                        
                        patch=img[pcenter[0]-D/2:pcenter[0]+D/2,pcenter[1]-D/2:pcenter[1]+D/2,pcenter[2]-D/2:pcenter[2]+D/2]#.astype(np.float32)/255
                        patch2=img[pcenter[0]-D2/2:pcenter[0]+D2/2,pcenter[1]-D2/2:pcenter[1]+D2/2,pcenter[2]-D2/2:pcenter[2]+D2/2]
                        patch3=img[pcenter[0]-D3/2:pcenter[0]+D3/2,pcenter[1]-D3/2:pcenter[1]+D3/2,pcenter[2]-D3/2:pcenter[2]+D3/2]
                        #patch_temp=np.zeros(patch.shape,dtype=np.uint8)                        
                        if D!=48:
                            patch=transform.resize(patch,[w,w,w],mode='nearest')    
                            patch2=transform.resize(patch2,[w,w,w],mode='nearest') 
                            patch3=transform.resize(patch3,[w,w,w],mode='nearest') 
                            
                            patch=patch.astype(np.float32)
                            patch2=patch2.astype(np.float32)
                            patch3=patch3.astype(np.float32)
                            #patch=(patch*255).astype(np.uint8)
                        else:
                            patch=patch.astype(np.float32)
                            patch2=patch2.astype(np.float32)
                            patch3=patch3.astype(np.float32)
                            patch=patch/255.0
                            patch2=patch2/255.0
                            patch3=patch3/255.0
                            
                        patch=np.reshape(patch,[1,1,48,48,48])
                        patch2=np.reshape(patch2,[1,1,48,48,48])
                        patch3=np.reshape(patch3,[1,1,48,48,48])
                        #patch_temp=np.zeros([5,1,48,48,48],dtype=np.float32)
                        #patch_temp[0,0]=patch[0,0,:,:,:]
                        #patch_temp[1,0]=patch[0,0,:,:,::-1]
                        #patch_temp[2,0]=patch[0,0,:,::-1,:]
                        #patch_temp[3,0]=patch[0,0,:,::-1,::-1]
                        #patch_temp[4,0]=patch.transpose([0,1,2,4,3])
                        m_a=model.predict(patch, verbose=0)+model.predict(patch2, verbose=0)+model.predict(patch3, verbose=0)#np.zeros([1,1,2])
                        m_a/=6.0
                        m_b=model2.predict(patch, verbose=0)+model2.predict(patch2, verbose=0)+model2.predict(patch3, verbose=0)
                        m_b/=6.0
                        #m_a=0
                        #for l in range(5):
                        #    m_a+=model.predict(patch_temp[l:l+1], verbose=0)
                        #m_a/=5.0
                        #patch_temp2=np.zeros([5*5,1,48,48,48],dtype=np.uint8)
                        #for l in range(5):
                        #    patch_temp2[]
                        #patch=patch#-mean
#                        if Diam<=9:
#                            m1=model2.predict(patch, verbose=0)
#                        if Diam>=6:
#                            m2=model.predict(patch, verbose=0)
#                            
#                        if Diam<=9 and Diam>=6:
#                            m[i:i+1]=m1*0.5+m2*0.5
#                        elif Diam<6:
#                            m[i:i+1]=m1
#                        elif Diam>9:
#                            m[i:i+1]=m2
#                        m3 = model2.predict(patch, verbose=0)#*0.5+model2.predict(patch, verbose=0)*0.5
                        m[i]=m_a[0]+m_b[0]#m[i:i+1]*0.5+m3*0.5
                        #m[i] = model.predict(patch, verbose=0)#*0.5+model2.predict(patch, verbose=0)*0.5
#                        if Diam>=3 and Diam<=5:
#                            m[i,0,0]*=0.1
#                        elif Diam>5 and Diam<=10:
#                            m[i,0,0]*=0.3
#                        else:
#                            pass
                        
#                        if Diam==3:
#                            m[i,0,0]*=0.1
#                        elif Diam==4:
#                            m[i,0,0]*=0.15
#                        elif Diam==4:
#                            m[i,0,0]*=0.2
#                        elif Diam==5:
#                            m[i,0,0]*=0.4
#                        elif Diam==6:
#                            m[i,0,0]*=0.6
#                        elif Diam==7:
#                            m[i,0,0]*=0.7
#                        elif Diam==8:
#                            m[i,0,0]*=0.9    
#                        else:
#                            pass
                        if line[0]!=bak and m[i,0,0]>0:#0.05:
                            writer.writerow({'seriesuid':line[0],'coordX':str(pcenter_o[2]),'coordY':str(pcenter_o[1]),'coordZ':str(pcenter_o[0]),'probability':str(m[i,0,0])})
                        #output
                        #compare=Image.fromarray((patch[0,0,23]*255).astype(np.uint8))
                        #compare.save('/home/customer/document/lung-nodule-detection/challenge/ClassTrain/Data/val_compare/'+str(p)+'_'+str(i)+'_'+str(m[i,0,0])+'.gif')
                        
                        if ProjName!='new_test' and ProjName!='test' and Display:
                            showed=0
        
                            if dis(center,pcenter_o)<Diam+1 and m[i,0,0]<0.1:#center[0]>startz-5 and center[0]<endz+5 and center[1]>starty-5 and center[1]<endy+5 and center[2]>startx-5 and center[2]<endx+5 and showed==0:
                                compare=Image.fromarray((patch[0,0,23]*255).astype(np.uint8))
                                compare.save(CompareRoot+'_'+filename+'_'+str(m[i,0,0])+'_'+str(pcenter_o)+'_'+str(Diam)+'_'+'FN'+'.gif')
                            
                                #print(m[i,0,0])
                                #print("False Negative!")
                                #pl.imshow(patch[0,0,23],cmap='gray')
                                #pl.show()
                            elif dis(center,pcenter_o)<Diam+1 and m[i,0,0]>0.1:#center[0]>startz-5 and center[0]<endz+5 and center[1]>starty-5 and center[1]<endy+5 and center[2]>startx-5 and center[2]<endx+5 and showed==1:
                                compare=Image.fromarray((patch[0,0,23]*255).astype(np.uint8))
                                compare.save(CompareRoot+'_'+filename+'_'+str(m[i,0,0])+'_'+str(pcenter_o)+'_'+str(Diam)+'_'+'TP'+'.gif')
                                if os.path.exists(CompareRoot+'_'+filename+'_'+str(m[i,0,0])+'_'+str(pcenter_o)+'_'+str(Diam)+'_'+'FP'+'.gif'):
                                    os.remove(CompareRoot+'_'+filename+'_'+str(m[i,0,0])+'_'+str(pcenter_o)+'_'+str(Diam)+'_'+'FP'+'.gif')
                                #print(m[i,0,0])
                                #print("True Positive!")
                                #pl.imshow(patch[0,0,23],cmap='gray')
                                #pl.show()
    
                            elif m[i,0,0]>0.1:
                                compare=Image.fromarray((patch[0,0,23]*255).astype(np.uint8))
                                if os.path.exists(CompareRoot+'_'+filename+'_'+str(m[i,0,0])+'_'+str(pcenter_o)+'_'+str(Diam)+'_'+'TP'+'.gif'):
                                    pass
                                else:                                        
                                    compare.save(CompareRoot+'_'+filename+'_'+str(m[i,0,0])+'_'+str(pcenter_o)+'_'+str(Diam)+'_'+'FP'+'.gif')              
                                #print(m[i,0,0])
                                #print("False Positive!")
                                #pl.imshow(patch[0,0,23-move],cmap='gray')
                                #pl.show()
                            
                        
                    #redraw
                    #pred[pcenter[0]-w/2:pcenter[0]+w/2,pcenter[1]-w/2:pcenter[1]+w/2,pcenter[2]-w/2:pcenter[2]+w/2]=pred_temp[pcenter[0]-w/2:pcenter[0]+w/2,pcenter[1]-w/2:pcenter[1]+w/2,pcenter[2]-w/2:pcenter[2]+w/2]*m[i,0,0]
                print(p)
                p=p+1                
                pred=pred.astype(np.uint8)
                #write_mhd_file('/media/customer/Disk1/LUNG/output-refined/'+filename+'.mhd', pred, [pred.shape[2],pred.shape[1],pred.shape[0]])
                bak=line[0]
            #else:
                #p=p+1
            
    csvf.close()
    