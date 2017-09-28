import time
import gc
import os
import SimpleITK as sitk
import numpy as np
import csv
import scipy
from glob import glob
import pandas as pd

from tqdm import tqdm

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import dicom

from joblib import Parallel, delayed
from numba import autojit
import zarr
from PIL import Image
import cv2

import matplotlib.pyplot as plt

#from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
#from skimage.measure import label,regionprops, perimeter
#from skimage.filters import roberts, sobel
#from skimage import measure, feature,data
#from skimage.segmentation import clear_border
#from skimage import measure, morphology
from skimage.morphology import convex_hull_image

import scipy.spatial.distance as dist
from scipy.ndimage.morphology import binary_dilation,generate_binary_structure
from scipy import ndimage,misc


import tensorflow as tf
from keras import backend as K
from keras.models import Sequential,load_model,Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, SpatialDropout2D
from keras.layers import Input, merge, UpSampling2D, BatchNormalization
from keras.optimizers import Adam,SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.advanced_activations import PReLU
from keras.layers import Convolution3D, MaxPooling3D, SpatialDropout3D, UpSampling3D
from keras.preprocessing.image import ImageDataGenerator,img_to_array, load_img,array_to_img

K.set_image_dim_ordering('th')
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import warnings
warnings.filterwarnings("ignore")

from unet_utils import *
from unet_models import *
from preproc_utils import *
from models_3D import *

