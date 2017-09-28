import utils.helpers as helpers
from utils.paths import *


import SimpleITK  # conda install -c https://conda.anaconda.org/simpleitk SimpleITK
import numpy
import pandas
import ntpath
import cv2  # conda install -c https://conda.anaconda.org/menpo opencv3
import shutil
import random
import math
import multiprocessing
#from bs4 import BeautifulSoup #  conda install beautifulsoup4, coda install lxml
import os
import glob
import h5py




from typing import List, Tuple
from keras.optimizers import Adam, SGD
from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D, merge, BatchNormalization,SpatialDropout2D,Convolution3D,MaxPooling3D, UpSampling3D, Flatten, Dense
from keras.models import Model, load_model, model_from_json
from keras.metrics import binary_accuracy, binary_crossentropy, mean_squared_error, mean_absolute_error
from keras import backend as K
from keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from keras.backend.tensorflow_backend import set_session

random.seed(1321)
numpy.random.seed(1321)

import warnings
warnings.filterwarnings("ignore")

