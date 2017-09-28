from imports import *
from functions import *


def train_generator(PATH):
    file_list_true = get_dirfiles(PATH['pics_path_true_train'])
    file_list_false = get_dirfiles(PATH['pics_path_false_train'])
    nb_true = len(file_list_true) + len(file_list_false)
    sample = np.zeros([nb_true,64,64,64])
    labels = np.zeros([nb_true,2])
    for i in tqdm(range(len(file_list_true))):
        cc= np.load(file_list_true[i]).reshape([1,64,64,64])
        sample[i] = cc[0]
        labels[i][0] = 1
    for j in tqdm(range(len(file_list_false))):
        bb= np.load(file_list_true[i]).reshape([1,64,64,64])
        sample[j+len(file_list_true)] = bb[0]
        labels[j+len(file_list_true)][1] = 1    
    sample = np.expand_dims(sample, axis=1)        
    return sample,labels
            
def val_generator(PATH):
    file_list_true = get_dirfiles(PATH['pics_path_true_val'])
    file_list_false = get_dirfiles(PATH['pics_path_false_val'])
    nb_true = len(file_list_true) + len(file_list_false)
    sample = np.zeros([nb_true,64,64,64])
    labels = np.zeros([nb_true,2])
    for i in tqdm(range(len(file_list_true))):
        cc= np.load(file_list_true[i]).reshape([1,64,64,64])
        sample[i] = cc[0]
        labels[i][0] = 1
    for j in tqdm(range(len(file_list_false))):
        bb= np.load(file_list_true[i]).reshape([1,64,64,64])
        sample[j+len(file_list_true)] = bb[0]
        labels[j+len(file_list_true)][1] = 1
    
    sample = np.expand_dims(sample, axis=1) 
    return sample,labels

def test_generator(PATH):
    file_list_true = get_dirfiles(PATH['pics_path_true_test'])
    nb_true = len(file_list_true)
    sample = np.zeros([nb_true,64,64,64])
    for i in tqdm(range(len(file_list_true))):
        cc= np.load(file_list_true[i]).reshape([1,64,64,64])
        sample[i] = cc[0]    
    sample = np.expand_dims(sample, axis=1) 
    np.save(PATH['model_path']+"X_test.npy",sample)
    return sample

def load_fenge_csv(PATH,load_data = False):
    if load_data:
        X1 = np.load(PATH['model_path'] + "X_train.npy")
        Y1 = np.load(PATH['model_path'] + "Y_train.npy") 
        X2 = np.load(PATH['model_path'] + "X_val.npy") 
        Y2 = np.load(PATH['model_path'] + "Y_val.npy") 
        X_train = np.vstack((X1,X2))
        Y_train = np.vstack((Y1,Y2))
        
    else:
        X1,Y1 = train_generator(PATH)
        X2,Y2 = val_generator(PATH)
        X_train = np.vstack((X1,X2))
        Y_train = np.vstack((Y1,Y2))
        
        np.save(PATH['model_path'] + "X_train.npy",X1)
        np.save(PATH['model_path'] + "Y_train.npy",Y1) 
        np.save(PATH['model_path'] + "X_val.npy",X2) 
        np.save(PATH['model_path'] + "Y_val.npy",Y2) 
    return X_train,Y_train

def preds3d_dense(width):
    
    learning_rate = 5e-5
    #optimizer = SGD(lr=learning_rate, momentum = 0.9, decay = 1e-3, nesterov = True)
    optimizer = Adam(lr=learning_rate)
    
    inputs = Input(shape=(1, 64, 64, 64))
    conv1 = Convolution3D(width, 3, 3, 3, activation = 'relu', border_mode='same')(inputs)
    conv1 = BatchNormalization(axis = 1)(conv1)
    conv1 = Convolution3D(width*2, 3, 3, 3, activation = 'relu', border_mode='same')(conv1)
    conv1 = BatchNormalization(axis = 1)(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2), border_mode='same')(conv1)
    
    conv2 = Convolution3D(width*2, 3, 3, 3, activation = 'relu', border_mode='same')(pool1)
    conv2 = BatchNormalization(axis = 1)(conv2)
    conv2 = Convolution3D(width*4, 3, 3, 3, activation = 'relu', border_mode='same')(conv2)
    conv2 = BatchNormalization(axis = 1)(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2), border_mode='same')(conv2)

    conv3 = Convolution3D(width*4, 3, 3, 3, activation = 'relu', border_mode='same')(pool2)
    conv3 = BatchNormalization(axis = 1)(conv3)
    conv3 = Convolution3D(width*8, 3, 3, 3, activation = 'relu', border_mode='same')(conv3)
    conv3 = BatchNormalization(axis = 1)(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2), border_mode='same')(conv3)
    
    conv4 = Convolution3D(width*8, 3, 3, 3, activation = 'relu', border_mode='same')(pool3)
    conv4 = BatchNormalization(axis = 1)(conv4)
    conv4 = Convolution3D(width*16, 3, 3, 3, activation = 'relu', border_mode='same')(conv4)
    conv4 = BatchNormalization(axis = 1)(conv4)
    pool4 = MaxPooling3D(pool_size=(8, 8, 8), border_mode='same')(conv4)
    
    output = Flatten(name='flatten')(pool4)
    output = Dropout(0.2)(output)
    output = Dense(128)(output)
    output = PReLU()(output)
    output = BatchNormalization()(output)
    output = Dropout(0.2)(output)
    output = Dense(128)(output)
    output = PReLU()(output)
    output = BatchNormalization()(output)
    output = Dropout(0.3)(output)
    output = Dense(2, activation='softmax', name = 'predictions')(output)
    model3d = Model(inputs, output)
    model3d.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    return model3d

def fenlei_fit(name, PATH,load_data = False,load_check = False,batch_size=2, epochs=10,check_name = None):

    t = time.time()
    callbacks = [EarlyStopping(monitor='val_loss', patience = 5, verbose = 1),
                 ModelCheckpoint((PATH['model_path'] + '{}.h5').format(name),
                                 monitor='val_loss',
                                 verbose = 0,
                                 save_best_only = True)]
    if load_check:
        check_model = (PATH['model_path'] + '{}.h5').format(check_name)
        model = load_model(check_model)
    else:
        #model = Resnet3DBuilder.build_resnet_18((64, 64, 64, 1), 2)
        model = preds3d_dense(16)
    x,y = load_fenge_csv(PATH,load_data)
    model.fit(x=x, y=y, batch_size=batch_size, epochs=epochs,
              validation_split = 0.3,verbose=1, callbacks=callbacks, shuffle=True)
    return model