import os
USER_NAME = os.environ['USER']
print("User: ", USER_NAME)



TARGET_VOXEL_MM = 1.00
MEAN_PIXEL_VALUE_NODULE = 41
LUNA_SUBSET_START_INDEX = 0
SEGMENTER_IMG_SIZE = 320


BASE_DIR_SSD = '/Volumes/solo/ali/pic/'
BASE_DIR = '/Volumes/solo/ali/'
EXTRA_DATA_DIR = "/Volumes/solo/ali/resources/"
NDSB3_RAW_SRC_DIR = BASE_DIR + "ndsb_raw/stage12/"

TOTAL_TRAIN_SUBSET = 15
LUNA16_RAW_SRC_DIR = BASE_DIR + "Data/train/"

NDSB3_EXTRACTED_IMAGE_DIR = BASE_DIR_SSD + "ndsb3_extracted_images/"
LUNA16_EXTRACTED_IMAGE_DIR = BASE_DIR_SSD + "luna16_extracted_images/"
NDSB3_NODULE_DETECTION_DIR = BASE_DIR_SSD + "ndsb3_nodule_predictions/"

