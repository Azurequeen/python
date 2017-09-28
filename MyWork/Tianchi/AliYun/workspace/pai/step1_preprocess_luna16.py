import settings
import helpers
import SimpleITK  # conda install -c https://conda.anaconda.org/simpleitk SimpleITK
import numpy
import pandas
#import ntpath
import cv2  # conda install -c https://conda.anaconda.org/menpo opencv3
#import shutil
#import random
#import math
#import multiprocessing
#from bs4 import BeautifulSoup #  conda install beautifulsoup4, coda install lxml
import os
import glob

def normalize(image):
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image

def process_image(src_path):
    patient_id = src_path[-14:-4]
    print("Patient: ", patient_id)

    dst_dir = settings.LUNA16_EXTRACTED_IMAGE_DIR + patient_id + "/"
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    itk_img = SimpleITK.ReadImage(src_path)
    img_array = SimpleITK.GetArrayFromImage(itk_img)
    #print("Img array: ", img_array.shape)

    origin = numpy.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
    #print("Origin (x,y,z): ", origin)

    direction = numpy.array(itk_img.GetDirection())      # x,y,z  Origin in world coordinates (mm)
    #print("Direction: ", direction)


    spacing = numpy.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
    #print("Spacing (x,y,z): ", spacing)
    rescale = spacing / settings.TARGET_VOXEL_MM
    #print("Rescale: ", rescale)

    img_array = helpers.rescale_patient_images(img_array, spacing, settings.TARGET_VOXEL_MM)

    img_list = []
    for i in range(img_array.shape[0]):
        img = img_array[i]
        seg_img, mask = helpers.get_segmented_lungs(img.copy())
        img_list.append(seg_img)
        img = normalize(img)
        cv2.imwrite(dst_dir + "img_" + str(i).rjust(4, '0') + "_i.png", img * 255)
        cv2.imwrite(dst_dir + "img_" + str(i).rjust(4, '0') + "_m.png", mask * 255)


def process_images(delete_existing=False):
    if delete_existing and os.path.exists(settings.LUNA16_EXTRACTED_IMAGE_DIR):
        print("Removing old stuff..")
        if os.path.exists(settings.LUNA16_EXTRACTED_IMAGE_DIR):
            shutil.rmtree(settings.LUNA16_EXTRACTED_IMAGE_DIR)

    if not os.path.exists(settings.LUNA16_EXTRACTED_IMAGE_DIR):
        os.mkdir(settings.LUNA16_EXTRACTED_IMAGE_DIR)
        os.mkdir(settings.LUNA16_EXTRACTED_IMAGE_DIR + "_labels/")

    for subject_no in range(settings.LUNA_SUBSET_START_INDEX, settings.LUNA_SUBSET_END_INDEX):
        src_dir = settings.LUNA16_RAW_SRC_DIR + "LKDS_MHD_TRAIN/train_subset" + str(subject_no).zfill(2) + "/"
        src_paths = glob.glob(src_dir + "*.mhd")
        for src_path in src_paths:
            process_image(src_path)


def process_luna_candidates_patients():
    for subject_no in range(settings.LUNA_SUBSET_START_INDEX, settings.LUNA_SUBSET_END_INDEX):
        src_dir = settings.LUNA16_RAW_SRC_DIR + "LKDS_MHD_TRAIN/train_subset" + str(subject_no).zfill(2) + "/"
        for patient_index, src_path in enumerate(glob.glob(src_dir + "*.mhd")):            
            patient_id = src_path[-14:-4]
            print("Patient: ", patient_index, " ", patient_id)
            process_luna_candidates_patient(src_path, patient_id)

def process_luna_candidates_patient(src_path, patient_id):
    dst_dir = settings.LUNA16_EXTRACTED_IMAGE_DIR + "_labels/"
    img_dir = dst_dir + patient_id + "/"
    df_pos_annos = pandas.read_csv("../../data/csv/train/annotations.csv")
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    itk_img = SimpleITK.ReadImage(src_path)
    img_array = SimpleITK.GetArrayFromImage(itk_img)
    #print("Img array: ", img_array.shape)
    #print("Pos annos: ", len(df_pos_annos))
    num_z, height, width = img_array.shape        #heightXwidth constitute the transverse plane
    origin = numpy.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
    #print("Origin (x,y,z): ", origin)
    spacing = numpy.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
    #print("Spacing (x,y,z): ", spacing)
    rescale = spacing / settings.TARGET_VOXEL_MM
    #print("Rescale: ", rescale)
    direction = numpy.array(itk_img.GetDirection())      # x,y,z  Origin in world coordinates (mm)
    #print("Direction: ", direction)
    flip_direction_x = False
    flip_direction_y = False
    if round(direction[0]) == -1:
        origin[0] *= -1
        direction[0] = 1
        flip_direction_x = True
        #print("Swappint x origin")
    if round(direction[4]) == -1:
        origin[1] *= -1
        direction[4] = 1
        flip_direction_y = True
        #print("Swappint y origin")
    #print("Direction: ", direction)
    assert abs(sum(direction) - 3) < 0.01

    src_df = pandas.read_csv("../../data/csv/train/annotations.csv")
    src_df = src_df[src_df["seriesuid"] == patient_id]
    #src_df = src_df[src_df["class"] == 0]
    patient_imgs = helpers.load_patient_images(patient_id, settings.LUNA16_EXTRACTED_IMAGE_DIR, "*_i.png")
    candidate_list = []

    for df_index, candiate_row in src_df.iterrows():
        node_x = candiate_row["coordX"]
        if flip_direction_x:
            node_x *= -1
        node_y = candiate_row["coordY"]
        if flip_direction_y:
            node_y *= -1
        node_z = candiate_row["coordZ"]
        candidate_diameter = candiate_row["diameter_mm"]
        # print("Node org (x,y,z,diam): ", (round(node_x, 2), round(node_y, 2), round(node_z, 2), round(candidate_diameter, 2)))
        center_float = numpy.array([node_x, node_y, node_z])
        center_int = numpy.rint((center_float-origin) / spacing)
        # center_int = numpy.rint((center_float - origin) )
        # print("Node tra (x,y,z,diam): ", (center_int[0], center_int[1], center_int[2]))
        # center_int_rescaled = numpy.rint(((center_float-origin) / spacing) * rescale)
        center_float_rescaled = (center_float - origin) / settings.TARGET_VOXEL_MM
        center_float_percent = center_float_rescaled / patient_imgs.swapaxes(0, 2).shape
        # center_int = numpy.rint((center_float - origin) )
        # print("Node sca (x,y,z,diam): ", (center_float_rescaled[0], center_float_rescaled[1], center_float_rescaled[2]))
        coord_x = center_float_rescaled[0]
        coord_y = center_float_rescaled[1]
        coord_z = center_float_rescaled[2]
        candidate_list.append([len(candidate_list), round(center_float_percent[0], 4), round(center_float_percent[1], 4), round(center_float_percent[2], 4), round(candidate_diameter / patient_imgs.shape[0], 4), 0])

    df_candidates = pandas.DataFrame(candidate_list, columns=["anno_index", "coord_x", "coord_y", "coord_z", "diameter", "malscore"])
    df_candidates.to_csv(dst_dir + patient_id + "_candidates_luna.csv", index=False)
    
if __name__ == "__main__":    
    if True:
        print('process_images')
        process_images(delete_existing=False)
    if True:
        print('process_luna_candidates_patients')
        process_luna_candidates_patients()
