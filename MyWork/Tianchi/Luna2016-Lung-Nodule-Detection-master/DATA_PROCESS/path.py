# -*- coding: utf-8 -*-
import os

# 数据主目录
main_path = '/Volumes/solo/ali/'

PATH = {
    # CSV文件目录
    'annotations_train': main_path + 'csv/train/',
    'annotations_val': main_path + 'csv/val/',
    'annotations_test': main_path + 'csv/test/',

    # MDH文件目录
    'src_train': main_path + 'Data/Download/train/',
    'src_val': main_path + 'Data/Download/val/',
    'src_test': main_path + 'Data/Download/test/',

    # 输出目录
    'output_train': main_path + 'Luna/output/',
    'output_test': main_path + 'Luna/output_test/',

    # 模型训练数据
    'model_train': main_path + 'Data/train_mask/',

    # 送入模型的数据存放目录
    'model_train_lung': main_path + 'Data/train_mask/lung/',
    'model_train_nodule': main_path + 'Data/train_mask/nodule/',
    'model_train_pred': main_path + 'Data/train_mask/train_pred/',

    'model_test_pred': main_path + 'Data/train_mask/test_pred/',

    # 分割、分类模型目录
    'model_paths': main_path + 'Data/model/',
}

# 检查文件夹，如果没有就新建一个
for i in PATH:
    if not os.path.exists(PATH[i]):
        os.mkdir(PATH[i])
        print(PATH[i], u'maked')