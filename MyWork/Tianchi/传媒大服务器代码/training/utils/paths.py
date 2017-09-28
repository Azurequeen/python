# -*- coding: utf-8 -*-
import os

#数据主目录
main_path = 'D:\\ali\\'


PATH = {
#CSV文件目录
'annotations_train' : main_path + 'csv\\train\\',
'annotations_val' : main_path + 'csv\\val\\',
'annotations_test' : main_path + 'csv\\test\\',

#MDH文件目录
'src_train' : main_path + 'Data\\Download\\train\\',
'src_val' : main_path + 'Data\\Download\\val\\',
'src_test' : main_path + 'Data\\Download\\test\\',

#图片存放目录
'pic_train' : main_path + 'Data\\train_pic\\',

#分割模型训练数据
'model_train' : main_path + 'Data\\train_mask\\',    

#送入分割模型的数据存放目录
'model_train_lung' : main_path + 'Data\\train_mask\\lung\\',
'model_train_nodule' : main_path + 'Data\\train_mask\\nodule\\',
'model_train_pred' : main_path + 'Data\\train_mask\\train_pred\\',
'model_val_pred' : main_path + 'Data\\train_mask\\val_pred\\',    
'model_test_pred' : main_path + 'Data\\train_mask\\test_pred\\',
    
#分割、分类模型目录
'model_paths' : main_path + 'Data\\model\\',
}


#检查文件夹，如果没有就新建一个
for i in PATH:
    if not os.path.exists(PATH[i]):
        os.mkdir(PATH[i])
        print(PATH[i],u'maked')

#其他参数
TARGET_VOXEL_MM = 1.00
MEAN_PIXEL_VALUE_NODULE = 41
LUNA_SUBSET_START_INDEX = 0
SEGMENTER_IMG_SIZE = 512