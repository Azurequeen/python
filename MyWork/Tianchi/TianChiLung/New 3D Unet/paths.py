# -*- coding: utf-8 -*-
import os

#数据主目录
main_path = '/Volumes/solo/ali/'

#CSV文件目录
annotations_train_path = main_path + 'csv/train/'
annotations_val_path = main_path + 'csv/val/'
annotations_test_path = main_path + 'csv/test/'

#MDH文件目录
src_train = main_path + 'Data/train/'
src_val = main_path + 'Data/val/'
src_test = main_path + 'Data/test/'



#有数据处理目录
mask_train = main_path + 'Data/train_mask/'
mask_val = main_path + 'Data/val_mask/'
mask_test = main_path + 'Data/test_mask/'



#完整npy目录
full_lung_train = main_path + 'Data/train_mask/train_full_lung/'
full_lung_val = main_path + 'Data/val_mask/val_full_lung/'
full_lung_test = main_path + 'Data/test_mask/test_full_lung/'

#分割、分类模型目录
model_path = main_path + 'Data/model/'




def dir_check():
    # 有数据处理目录
    if os.path.isdir(mask_train):
        pass
    else:
        os.mkdir(mask_train)

    if os.path.isdir(mask_val):
        pass
    else:
        os.mkdir(mask_val)

    if os.path.isdir(mask_test):
        pass
    else:
        os.mkdir(mask_test)

    # 完整npy目录
    if os.path.isdir(full_lung_train):
        pass
    else:
        os.mkdir(full_lung_train)

    if os.path.isdir(full_lung_val):
        pass
    else:
        os.mkdir(full_lung_val)

    if os.path.isdir(full_lung_test):
        pass
    else:
        os.mkdir(full_lung_test)

    # 分割、分类模型目录
    if os.path.isdir(model_path):
        pass
    else:
        os.mkdir(model_path)

