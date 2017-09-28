# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import pandas as pd
from skimage import morphology
from skimage import measure
from sklearn.cluster import KMeans
from skimage.transform import resize
from glob import glob
from tqdm import tqdm
import SimpleITK as sitk
import os
import time


def save_mask(npy_list):
    for img_file in tqdm(npy_list):
        imgs_to_process = np.load(img_file).astype(np.float64)
        for i in range(len(imgs_to_process)):
            img = imgs_to_process[i]
            mean = np.mean(img)
            std = np.std(img)
            img = img - mean
            img = img / std
            middle = img[100:-100, 100:-100]
            mean = np.mean(middle)
            max = np.max(img)
            min = np.min(img)
            img[img == max] = mean
            img[img == min] = mean
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
            mask = np.ndarray([imgs_to_process[i].shape[0], imgs_to_process[i].shape[0]], dtype=np.int8)
            mask[:] = 0
            for N in good_labels:
                mask = mask + np.where(labels == N, 1, 0)
            mask = morphology.dilation(mask, np.ones([10, 10]))  # one last dilation
            imgs_to_process[i] = mask
        np.save(img_file.replace("images", "lungmask"), imgs_to_process)
    print(u'肺部掩模已生成！')

def save_train_test(luna_path,working_path):
    npy_list = glob(luna_path + 'npy/' + "lungmask_*.npy")
    # file_list=glob(working_path+"lungmask_*.npy")
    out_images = []  # final set of images
    out_nodemasks = []  # final set of nodemasks
    for fname in tqdm(npy_list):

        imgs_to_process = np.load(fname.replace("lungmask", "images"))
        masks = np.load(fname)
        node_masks = np.load(fname.replace("lungmask", "masks"))
        for i in range(len(imgs_to_process)):
            mask = masks[i]
            node_mask = node_masks[i]
            img = imgs_to_process[i]
            new_size = [512, 512]  # we're scaling back up to the original size of the image
            img = mask * img  # apply lung mask
            
            #
            # renormalizing the masked image (in the mask region)
            #
            new_mean = np.mean(img[mask > 0])
            new_std = np.std(img[mask > 0])
            #
            #  Pulling the background color up to the lower end
            #  of the pixel range for the lungs
            #
            old_min = np.min(img)  # background color
            img[img == old_min] = new_mean - 1.2 * new_std  # resetting backgound color
            img = img - new_mean
            img = img / new_std
            # make image bounding box  (min row, min col, max row, max col)
            labels = measure.label(mask)
            regions = measure.regionprops(labels)
            #
            # Finding the global min and max row over all regions
            #
            min_row = 512
            max_row = 0
            min_col = 512
            max_col = 0
            for prop in regions:
                B = prop.bbox
                if min_row > B[0]:
                    min_row = B[0]
                if min_col > B[1]:
                    min_col = B[1]
                if max_row < B[2]:
                    max_row = B[2]
                if max_col < B[3]:
                    max_col = B[3]
            width = max_col - min_col
            height = max_row - min_row
            if width > height:
                max_row = min_row + width
            else:
                max_col = min_col + height
            #
            # cropping the image down to the bounding box for all regions
            # (there's probably an skimage command that can do this in one line)
            #
            img = img[min_row:max_row, min_col:max_col]
            mask = mask[min_row:max_row, min_col:max_col]
            if max_row - min_row < 5 or max_col - min_col < 5:  # skipping all images with no god regions
                pass
            else:
                # moving range to -1 to 1 to accomodate the resize function
                mean = np.mean(img)
                img = img - mean
                min = np.min(img)
                max = np.max(img)
                img = img / (max - min)
                new_img = resize(img, [512, 512])
                new_node_mask = resize(node_mask[min_row:max_row, min_col:max_col], [512, 512])
                out_images.append(new_img)
                out_nodemasks.append(new_node_mask)
                #out_nodemasks.append(new_node_mask*new_img)  #注意：这里是用结节mask*原图的，如果train mask只用mask的话就用上面那个
    print(u'(1/2) 肺部样本图片和结节掩模已生成！')


    num_images = len(out_images)
    #
    #  Writing out images and masks as 1 channel arrays for input into network
    #
    final_images = np.ndarray([num_images, 1, 512, 512], dtype=np.float32)
    final_masks = np.ndarray([num_images, 1, 512, 512], dtype=np.float32)
    for i in range(num_images):
        final_images[i, 0] = out_images[i]
        final_masks[i, 0] = out_nodemasks[i]

    rand_i = np.random.choice(range(num_images), size=num_images, replace=False)
    test_i = int(0.2 * num_images)
    np.save(working_path + "trainImages.npy", final_images[rand_i[test_i:]])
    np.save(working_path + "trainMasks.npy", final_masks[rand_i[test_i:]])
    np.save(working_path + "testImages.npy", final_images[rand_i[:test_i]])
    np.save(working_path + "testMasks.npy", final_masks[rand_i[:test_i]])
    print(u'(2/2) train和test数据已经保存！\n')
    print(u'\t训练数据集：',working_path + "trainImages.npy\t",u'训练样本：',len(rand_i)-test_i)
    print(u'\t训练掩模集：', working_path + "trainMasks.npy\t",u'样本占比：',int(len(rand_i)-test_i)*100/len(rand_i),'%\n')
    print(u'\t测试数据集：', working_path + "testImages.npy\t",u'测试样本：',test_i)
    print(u'\t测试掩模集：', working_path +  "testMasks.npy\t",u'样本占比：',int(test_i*100/len(rand_i)),'%')


def make_mask(center, diam, z, width, height, spacing, origin):  # 只显示结节
    '''
Center : 圆的中心 px -- list of coordinates x,y,z
diam : 圆的直径 px -- diameter
widthXheight : pixel dim of image
spacing = mm/px conversion rate np array x,y,z
origin = x,y,z mm np.array
z = z position of slice in world coordinates mm
    '''
    mask = np.zeros([height, width])  # 0's everywhere except nodule swapping x,y to match img
    # convert to nodule space from world coordinates

    # Defining the voxel range in which the nodule falls
    v_center = (center - origin) / spacing
    v_diam = int(diam / spacing[0] + 5)
    v_xmin = np.max([0, int(v_center[0] - v_diam) - 5])
    v_xmax = np.min([width - 1, int(v_center[0] + v_diam) + 5])
    v_ymin = np.max([0, int(v_center[1] - v_diam) - 5])
    v_ymax = np.min([height - 1, int(v_center[1] + v_diam) + 5])

    v_xrange = range(v_xmin, v_xmax + 1)
    v_yrange = range(v_ymin, v_ymax + 1)

    # Convert back to world coordinates for distance calculation
    x_data = [x * spacing[0] + origin[0] for x in range(width)]
    y_data = [x * spacing[1] + origin[1] for x in range(height)]

    # Fill in 1 within sphere around nodule
    for v_x in v_xrange:
        for v_y in v_yrange:
            p_x = spacing[0] * v_x + origin[0]
            p_y = spacing[1] * v_y + origin[1]
            if np.linalg.norm(center - np.array([p_x, p_y, z])) <= diam:
                mask[int((p_y - origin[1]) / spacing[1]), int((p_x - origin[0]) / spacing[0])] = 1.0
    return (mask)


def matrix2int16(matrix):
    '''
matrix must be a numpy array NXN
Returns uint16 version
    '''
    m_min = np.min(matrix)
    m_max = np.max(matrix)
    matrix = matrix - m_min
    return (np.array(np.rint((matrix - m_min) / float(m_max - m_min) * 65535.0), dtype=np.uint16))


#
# Helper function to get rows in data frame associated
# with each file
def get_filename(file_list, case):
    for f in file_list:
        if case in f:
            return (f)


#
# The locations of the nodes


def normalize(image, MIN_BOUND=-1000.0, MAX_BOUND=400.0):
    """数据标准化"""
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image
    # ---数据标准化


def set_window_width(image, MIN_BOUND=-1000.0, MAX_BOUND=400.0):
    """设置窗宽"""
    image[image > MAX_BOUND] = MAX_BOUND
    image[image < MIN_BOUND] = MIN_BOUND
    return image
    # ---设置窗宽

def loaddata(file_list,df_node,output_path):
    for fcount, img_file in enumerate(tqdm(file_list)):
        mini_df = df_node[df_node["file"] == img_file]  # get all nodules associate with file
        if mini_df.shape[0] > 0:  # some files may not have a nodule--skipping those
            # load the data once
            itk_img = sitk.ReadImage(img_file)
            img_array = sitk.GetArrayFromImage(itk_img)  # indexes are z,y,x (notice the ordering)
            num_z, height, width = img_array.shape  # heightXwidth constitute the transverse plane
            origin = np.array(itk_img.GetOrigin())  # x,y,z  Origin in world coordinates (mm)
            spacing = np.array(itk_img.GetSpacing())  # spacing of voxels in world coor. (mm)
            # go through all nodes (why just the biggest?)
            for node_idx, cur_row in mini_df.iterrows():
                node_x = cur_row["coordX"]
                node_y = cur_row["coordY"]
                node_z = cur_row["coordZ"]
                diam = cur_row["diameter_mm"]
                # just keep 3 slices
                imgs = np.ndarray([3, height, width], dtype=np.float32)
                masks = np.ndarray([3, height, width], dtype=np.uint8)
                center = np.array([node_x, node_y, node_z])  # nodule center
                v_center = np.rint((center - origin) / spacing)  # nodule center in voxel space (still x,y,z ordering)
                for i, i_z in enumerate(np.arange(int(v_center[2]) - 1,
                                                  int(v_center[2]) + 2).clip(0,
                                                                             num_z - 1)):  # clip prevents going out of bounds in Z
                    mask = make_mask(center, diam, i_z * spacing[2] + origin[2],
                                     width, height, spacing, origin)
                    masks[i] = mask
                    imgs[i] = img_array[i_z]

                np.save(os.path.join(output_path, "images_%04d_%04d.npy" % (fcount, node_idx)), imgs)
                np.save(os.path.join(output_path, "masks_%04d_%04d.npy" % (fcount, node_idx)), masks)







