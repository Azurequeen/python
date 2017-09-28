from imports import *


smooth = 1.0
img_rows = 512
img_cols = 512



def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def get_max_slices_fenlei(lung_path, start, end):
    subsets = os.listdir(lung_path)[start:end]
    full_slices = 0
    for i in range(len(subsets)):
        if subsets[i] != '.DS_Store':
            num_slices = np.array(Image.open(lung_path + subsets[i])).shape[0]
            full_slices += num_slices
    print('Number of 2D slices in CT image: {}'.format(full_slices))
    return full_slices


def unet_model(dropout_rate,width):
    inputs = Input((1, 512, 512))
    conv1 = Convolution2D(width, (3, 3), padding="same", activation="relu")(inputs)
    conv1 = BatchNormalization(axis = 1)(conv1)
    conv1 = Convolution2D(width, (3, 3), padding="same", activation="relu")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(width*2, (3, 3), padding="same", activation="relu")(pool1)
    conv2 = BatchNormalization(axis = 1)(conv2)
    conv2 = Convolution2D(width*2, (3, 3), padding="same", activation="relu")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(width*4, (3, 3), padding="same", activation="relu")(pool2)
    conv3 = BatchNormalization(axis = 1)(conv3)
    conv3 = Convolution2D(width*4, (3, 3), padding="same", activation="relu")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(width*8, (3, 3), padding="same", activation="relu")(pool3)
    conv4 = BatchNormalization(axis = 1)(conv4)
    conv4 = Convolution2D(width*8, (3, 3), padding="same", activation="relu")(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(width*16, (3, 3), padding="same", activation="relu")(pool4)
    conv5 = BatchNormalization(axis = 1)(conv5)
    conv5 = Convolution2D(width*16, (3, 3), padding="same", activation="relu")(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = SpatialDropout2D(dropout_rate)(up6)
    conv6 = Convolution2D(width*8, (3, 3), padding="same", activation="relu")(conv6)
    conv6 = Convolution2D(width*8, (3, 3), padding="same", activation="relu")(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = SpatialDropout2D(dropout_rate)(up7)
    conv7 = Convolution2D(width*4, (3, 3), padding="same", activation="relu")(conv7)
    conv7 = Convolution2D(width*4, (3, 3), padding="same", activation="relu")(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = SpatialDropout2D(dropout_rate)(up8)
    conv8 = Convolution2D(width*2, (3, 3), padding="same", activation="relu")(conv8)
    conv8 = Convolution2D(width*2, (3, 3), padding="same", activation="relu")(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = SpatialDropout2D(dropout_rate)(up9)
    conv9 = Convolution2D(width, (3, 3), padding="same", activation="relu")(conv9)
    conv9 = Convolution2D(width, (3, 3), padding="same", activation="relu")(conv9)
    conv10 = Convolution2D(1, (1, 1), activation="sigmoid")(conv9)

    model = Model(input=inputs, output=conv10)
    #model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9, nesterov=True), loss=dice_coef_loss, metrics=[dice_coef])
    return model



def unet_fit(name, unet_model,PATH,load_check = False, check_name = None):

    lung_path = PATH['lung_path']
    nodule_path = PATH['nodule_path']
    model_path = PATH['model_path']

    batch_size = PATH['batch_size']
    epochs = PATH['epochs']
    dropout_rate = PATH['dropout_rate']
    width = PATH['width']



    start_t, end_t = PATH['t'][0], PATH['t'][1]
    start_v, end_v = end_t, PATH['t'][2]
    f1 = get_max_slices(lung_path, start_t, end_t)
    f2 = get_max_slices(lung_path, start_v, end_v)
    f1 = f1
    f2 = f2//2
    print('Training samples:', f1, 'Validation samples:', f2)




    t = time.time()
    callbacks = [EarlyStopping(monitor='val_loss', patience = 5, verbose = 1),
                 ModelCheckpoint((model_path + '{}.h5').format(name),
                                 monitor='val_loss',
                                 verbose = 0,
                                 save_best_only = True)]
    if load_check:
        check_model = (model_path + '{}.h5').format(check_name)
        model = load_model(check_model, custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
    else:
        model = unet_model(dropout_rate,width)

    model.fit_generator(generate_train(lung_path, nodule_path, start_t, end_t, batch_size), epochs = epochs, verbose = 1,
                        validation_data = generate_val(lung_path, nodule_path, batch_size), callbacks = callbacks,
                        steps_per_epoch = f1, validation_steps = f2)
    return model


