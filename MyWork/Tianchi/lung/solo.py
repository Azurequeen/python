import pandas as pd
import matplotlib.pyplot as plt
import SimpleITK as sitk
from tqdm import tqdm
from glob import glob



'''
This funciton reads a '.mhd' file using SimpleITK and return the image array, 
origin and spacing of the image.
'''

def get_filename(file_list, case):
    for f in file_list:
        if case in f:
            return(f)


def load_itk(filename):
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)

    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin())))

    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return ct_scan, origin, spacing


'''
This function is used to convert the world coordinates to voxel coordinates using 
the origin and spacing of the ct_scan
'''


def world_2_voxel(world_coordinates, origin, spacing):
    stretched_voxel_coordinates = np.absolute(world_coordinates - origin)
    voxel_coordinates = stretched_voxel_coordinates / spacing
    return voxel_coordinates


'''
This function is used to convert the voxel coordinates to world coordinates using 
the origin and spacing of the ct_scan.
'''


def voxel_2_world(voxel_coordinates, origin, spacing):
    stretched_voxel_coordinates = voxel_coordinates * spacing
    world_coordinates = stretched_voxel_coordinates + origin
    return world_coordinates


def seq(start, stop, step=1):
    n = int(round((stop - start) / float(step)))
    if n > 1:
        return ([start + step * i for i in range(n + 1)])
    else:
        return ([])


'''
This function is used to create spherical regions in binary masks
at the given locations and radius.
'''


def draw_circles(image, cands, origin, spacing):
    # make empty matrix, which will be filled with the mask
    RESIZE_SPACING = [1, 1, 1]


    image_mask = np.zeros(image.shape)

    # run over all the nodules in the lungs
    for ca in cands.values:
        # get middel x-,y-, and z-worldcoordinate of the nodule
        radius = np.ceil(ca[4]) / 2
        coord_x = ca[1]
        coord_y = ca[2]
        coord_z = ca[3]
        image_coord = np.array((coord_z, coord_y, coord_x))

        # determine voxel coordinate given the worldcoordinate
        image_coord = world_2_voxel(image_coord, origin, spacing)

        # determine the range of the nodule
        noduleRange = seq(-radius, radius, RESIZE_SPACING[0])

        # create the mask
        for x in noduleRange:
            for y in noduleRange:
                for z in noduleRange:
                    coords = world_2_voxel(np.array((coord_z + z, coord_y + y, coord_x + x)), origin, spacing)
                    if (np.linalg.norm(image_coord - coords) * RESIZE_SPACING[0]) < radius:
                        image_mask[np.round(coords[0]), np.round(coords[1]), np.round(coords[2])] = int(1)

    return image_mask

'''
This function takes the path to a '.mhd' file as input and
is used to create the nodule masks and segmented lungs after
rescaling to 1mm size in all directions. It saved them in the .npz
format. It also takes the list of nodule locations in that CT Scan as
input.
'''


def create_nodule_mask(imagePath, maskPath, cands):
    # if os.path.isfile(imagePath.replace('original',SAVE_FOLDER_image)) == False:
    img, origin, spacing = load_itk(imagePath)

    # calculate resize factor


    RESIZE_SPACING = [1, 1, 1]
    resize_factor = spacing / RESIZE_SPACING
    new_real_shape = img.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize = new_shape / img.shape
    new_spacing = spacing / real_resize

    # resize image
    lung_img = scipy.ndimage.interpolation.zoom(img, real_resize)

    # Segment the lung structure
    lung_img = lung_img + 1024
    lung_mask = segment_lung_from_ct_scan(lung_img)
    lung_img = lung_img - 1024

    # create nodule mask
    nodule_mask = draw_circles(lung_img, cands, origin, new_spacing)

    lung_img_512, lung_mask_512, nodule_mask_512 = np.zeros((lung_img.shape[0], 512, 512)), np.zeros(
        (lung_mask.shape[0], 512, 512)), np.zeros((nodule_mask.shape[0], 512, 512))

    original_shape = lung_img.shape
    for z in range(lung_img.shape[0]):
        offset = (512 - original_shape[1])
        upper_offset = np.round(offset / 2)
        lower_offset = offset - upper_offset

        new_origin = voxel_2_world([-upper_offset, -lower_offset, 0], origin, new_spacing)

        lung_img_512[z, upper_offset:-lower_offset, upper_offset:-lower_offset] = lung_img[z, :, :]
        lung_mask_512[z, upper_offset:-lower_offset, upper_offset:-lower_offset] = lung_mask[z, :, :]
        nodule_mask_512[z, upper_offset:-lower_offset, upper_offset:-lower_offset] = nodule_mask[z, :, :]

        # save images.
    np.save(imageName + '_lung_img.npz', lung_img_512)
    np.save(imageName + '_lung_mask.npz', lung_mask_512)
    np.save(imageName + '_nodule_mask.npz', nodule_mask_512)


def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

'''
The UNET model is compiled in this function.
'''
def unet_model():
    inputs = Input((1, 512, 512))
    conv1 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(1024, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Convolution2D(1024, 3, 3, activation='relu', border_mode='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Dropout(0.2)(conv8)
    conv8 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Dropout(0.2)(conv9)
    conv9 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)
    model.summary()
    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model


'''
This function takes the numpy array of CT_Scan and a list of coords from
which voxels are to be cut. The list of voxel arrays are returned. We keep the
voxels cubic because per pixel distance is same in all directions.
'''


def get_patch_from_list(lung_img, coords, window_size=10):
    shape = lung_img.shape
    output = []
    lung_img = lung_img + 1024
    for i in range(len(coords)):
        patch = lung_img[coords[i][0] - 18: coords[i][0] + 18,
                coords[i][1] - 18: coords[i][1] + 18,
                coords[i][2] - 18: coords[i][2] + 18]
        output.append(patch)

    return output


'''
Sample a random point from the image and return the coordinates.
'''


def get_point(shape):
    x = randint(50, shape[2] - 50)
    y = randint(50, shape[1] - 50)
    z = randint(20, shape[0] - 20)
    return np.asarray([z, y, x])


'''
This function reads the training csv file which contains the CT Scan names and
location of each nodule. It cuts many voxels around a nodule and takes random points as
negative samples. The voxels are dumped using pickle. It is to be noted that the voxels are
cut from original Scans and not the masked CT Scans generated while creating candidate
regions.
'''


def create_data(path, train_csv_path):
    coords, trainY = [], []

    with open(train_csv_path, 'rb') as f:
        lines = f.readlines()

        for line in lines:
            row = line.split(',')

            if os.path.isfile(path + row[0] + '.mhd') == False:
                continue

            lung_img = sitk.GetArrayFromImage(sitk.ReadImage(path + row[0] + '.mhd'))

            for i in range(-5, 5, 3):
                for j in range(-5, 5, 3):
                    for k in range(-2, 3, 2):
                        coords.append([int(row[3]) + k, int(row[2]) + j, int(row[1]) + i])
                        trainY.append(True)

            for i in range(60):
                coords.append(get_point(lung_img.shape))
                trainY.append(False)

            trainX = get_patch_from_list(lung_img, coords)

            print (trainX[0].shape, len(trainX))

            pickle.dump(np.asarray(trainX), open('traindata_' + str(counter) + '_Xtrain.p', 'wb'))
            pickle.dump(np.asarray(trainY, dtype=bool), open('traindata_' + str(counter) + '_Ytrain.p', 'wb'))

            counter = counter + 1
            coords, trainY = [], []


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution3D, MaxPooling3D
from keras.utils import np_utils
from keras import backend as K

'''
Creates a keras model with 3D CNNs and returns the model.
'''


def classifier(input_shape, kernel_size, pool_size):
    model = Sequential()

    model.add(Convolution3D(16, kernel_size[0], kernel_size[1], kernel_size[2],
                            border_mode='valid',
                            input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=pool_size))
    model.add(Convolution2D(32, kernel_size[0], kernel_size[1], kernel_size[2]))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=pool_size))
    model.add(Convolution3D(64, kernel_size[0], kernel_size[1], kernel_size[2]))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=pool_size))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    return model


def train_classifier(input_shape):
    model = classifier(input_shape, (3, 3, 3), (2, 2, 2))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    '''
    Read the preprocessed datafiles chunk by chunnk and train the model on that batch (trainX, trainY) using:'''
    model.train_on_batch(trainX, trainY, sample_weight=None)
    '''The model can be trained on many epochs using for loops'''

    '''
    AFter training the dataset we test our model of the test dataset, read the test file chunk by chunk and
    test on each chunk (trainX, trainY) using:'''
    print (model.test_on_batch(trainX, trainY, sample_weight=None))
    '''The average of all of this can be taken to obtain the final test score.'''

    '''After testing save the model using'''
    model.save('my_model.h5')



'''
A sample code to train xgboost.
The OptTrain class takes trainX(features) and trainY(labels) as input and initialise the class for
training the classifier. The optimize function then trains the classifier and the best model is saved
in 'model_best.pkl'.

A sample code is below:
trainX = np.array([[1,1,1], [2,3,4], [2,3,2], [1,1,1], [2,3,4], [2,3,2]])
trainY = np.array([0,1,0,0,0,1])
a = OptTrain(trainX, trainY)
a.optimize()
'''
import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.externals import joblib
from sklearn import model_selection
import numpy as np

class OptTrain:
    def __init__(self, trainX, trainY):
        self.trainX = trainX
        self.trainY = trainY

        self.level0 = xgb.XGBClassifier(learning_rate=0.325,
                                       silent=True,
                                       objective="binary:logistic",
                                       nthread=-1,
                                       gamma=0.85,
                                       min_child_weight=5,
                                       max_delta_step=1,
                                       subsample=0.85,
                                       colsample_bytree=0.55,
                                       colsample_bylevel=1,
                                       reg_alpha=0.5,
                                       reg_lambda=1,
                                       scale_pos_weight=1,
                                       base_score=0.5,
                                       seed=0,
                                       missing=None,
                                       n_estimators=1920, max_depth=6)
        self.h_param_grid = {'max_depth': hp.quniform('max_depth', 1, 13, 1),
                        'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
                        'learning_rate': hp.quniform('learning_rate', 0.025, 0.5, 0.025),
                        'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
                        'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
                        'n_estimators': hp.quniform('n_estimators', 100, 1000, 5),
                        }
        self.to_int_params = ['n_estimators', 'max_depth']

    def change_to_int(self, params, indexes):
        for index in indexes:
            params[index] = int(params[index])

    # Hyperopt Implementatation
    def score(self, params):
        self.change_to_int(params, self.to_int_params)
        self.level0.set_params(**params)
        score = model_selection.cross_val_score(self.level0, self.trainX, self.trainY, cv=5, n_jobs=-1)
        print('%s ------ Score Mean:%f, Std:%f' % (params, score.mean(), score.std()))
        return {'loss': score.mean(), 'status': STATUS_OK}

    def optimize(self):
        trials = Trials()
        print('Tuning Parameters')
        best = fmin(self.score, self.h_param_grid, algo=tpe.suggest, trials=trials, max_evals=200)

        print('\n\nBest Scoring Value')
        print(best)

        self.change_to_int(best, self.to_int_params)
        self.level0.set_params(**best)
        self.level0.fit(self.trainX, self.trainY)
        joblib.dump(self.level0,'model_best.pkl', compress=True)