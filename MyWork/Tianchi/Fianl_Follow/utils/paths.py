# -*- coding: utf-8 -*-
import os

#数据主目录
main_path = '/Volumes/solo/ali/'


PATH = {
#CSV文件目录
'annotations_train' : main_path + 'csv/train/',
'annotations_val' : main_path + 'csv/val/',
'annotations_test' : main_path + 'csv/test/',

#MDH文件目录
'src_train' : main_path + 'Data/train/',
'src_val' : main_path + 'Data/val/',
'src_test' : main_path + 'Data/test/',

#图片存放目录
'pic_train' : main_path + 'Data/train_pic/',
'pic_val' : main_path + 'Data/val_pic/',
'pic_test' : main_path + 'Data/test_pic/',
    
#标签存放目录
'label_train' : main_path + 'Data/train_pic/_labels/',
'label_val' : main_path + 'Data/val_pic/_labels/',
'label_test' : main_path + 'Data/test_pic/_labels/',

#送入模型的数据存放目录
'generated_train' : main_path + 'Data/train_generated/',
'generated_val' : main_path + 'Data/val_generated/',
'generated_test' : main_path + 'Data/test_generated/',
    
#分割、分类模型目录
'model_paths_temp' : main_path + 'Data/model/temp/',
'model_paths' : main_path + 'Data/model/',
}


#检查文件夹，如果没有就新建一个
for i in PATH:
    if not os.path.exists(PATH[i]):
        os.mkdir(PATH[i])
        print(PATH[i],'文件夹已新建')

#其他参数
TARGET_VOXEL_MM = 1.00
MEAN_PIXEL_VALUE_NODULE = 41
LUNA_SUBSET_START_INDEX = 0
SEGMENTER_IMG_SIZE = 512