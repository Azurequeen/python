
from utils.paths import *



import SimpleITK  # conda install -c https://conda.anaconda.org/simpleitk SimpleITK
import numpy
import pandas
import numpy as np
import pandas as pd
import scipy as sp
import ntpath
import cv2  # conda install -c https://conda.anaconda.org/menpo opencv3
import shutil
import random
import math
import multiprocessing
import os
import glob
import h5py
from joblib import Parallel, delayed
from tqdm import tqdm
import matplotlib.pyplot as plt
import skimage
import sklearn
import time
import random

from typing import List, Tuple
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from skimage.transform import resize
from skimage import morphology
from skimage import measure

from keras.optimizers import Adam, SGD
from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D, merge, Dropout, BatchNormalization,SpatialDropout2D,Convolution3D,MaxPooling3D, UpSampling3D, Flatten, Dense, AveragePooling3D, Activation
from keras.models import Model, load_model, model_from_json
from keras.metrics import binary_accuracy, binary_crossentropy, mean_squared_error, mean_absolute_error
from keras import backend as K
from keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler,EarlyStopping
from keras.backend.tensorflow_backend import set_session
from keras.utils.vis_utils import plot_model
from keras.layers.advanced_activations import PReLU
from keras.preprocessing.image import ImageDataGenerator


from sklearn.utils import shuffle
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold as KFold
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier as RF
#import xgboost as xgb

random.seed(1321)
numpy.random.seed(1321)

import warnings
warnings.filterwarnings("ignore")

K.set_image_dim_ordering('th') 