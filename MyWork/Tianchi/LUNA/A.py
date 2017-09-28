
# coding: utf-8

# In[1]:


from utils.imports import *


# In[2]:


def make_cls_true_false_old2(img_array, v_center, diam, times, patient_id, node_idx, dst):
    new_x = int(v_center[0])
    new_y = int(v_center[1])
    new_z = int(v_center[2])
    diam = int(diam + 1)
    if times == 1:
        trainX_1 = img_array[new_z - diam: new_z + diam,
                             new_y - diam: new_y + diam,
                             new_x - diam: new_x + diam]
        if trainX_1.shape == (diam*2, diam*2, diam*2) and trainX_1.sum() > 5./255 and trainX_1.mean() > 10./255:
            trainX_1 = resize(trainX_1,[32,32,32])
            np.save(dst + str(patient_id) + '_' + str(node_idx) + str('_oa') + '.npy', trainX_1)
            np.save(dst + str(patient_id) + '_' + str(node_idx) + str('_ob') + '.npy', np.fliplr(trainX_1))
            np.save(dst + str(patient_id) + '_' + str(node_idx) + str('_oc') + '.npy', np.flipud(trainX_1))            
    else:
        for j in range(times):
            rand_nb = int(diam/2)
            rand_list=range(-rand_nb,rand_nb+1)
            rand_list.remove(0)
            new_z1 = new_z + random.choice(rand_list)
            new_y1 = new_y + random.choice(rand_list)
            new_x1 = new_x + random.choice(rand_list)
            trainX_2 = img_array[new_z1 - diam: new_z1 + diam,
                                 new_y1 - diam: new_y1 + diam,
                                 new_x1 - diam: new_x1 + diam]
            if trainX_2.shape == (diam*2, diam*2, diam*2) and trainX_2.sum() > 5./255 and trainX_2.mean() > 10./255:
                trainX_2 = resize(trainX_2,[32,32,32])
                np.save(dst + str(patient_id) + '_' + str(node_idx) + '_' + str(j) + str('a') + '.npy', trainX_2)
                np.save(dst + str(patient_id) + '_' + str(node_idx) + '_' + str(j) + str('b') + '.npy', np.fliplr(trainX_2))
                np.save(dst + str(patient_id) + '_' + str(node_idx) + '_' + str(j) + str('c') + '.npy', np.flipud(trainX_2))                
    return

def make_cls_true_false(img_array, v_center, diam, times, patient_id, node_idx, dst):
    new_x = int(v_center[0])
    new_y = int(v_center[1])
    new_z = int(v_center[2])
    diam = int(diam)
    if times == 1:
        
        z1 = np.max([new_z - 3*diam,0])
        y1 = np.max([new_y - 3*diam,0])
        x1 = np.max([new_x - 3*diam,0])
            
        z2 = np.min([new_z + 3*diam,img_array.shape[0]])
        y2 = np.min([new_y + 3*diam,img_array.shape[1]])
        x2 = np.min([new_x + 3*diam,img_array.shape[2]])

        trainX_1 = img_array[z1 : z2,y1 : y2,x1 : x2]        

        if trainX_1.shape[0] != 0 and trainX_1.shape[1] != 0 and trainX_1.shape[2] != 0:
            trainX_1 = resize(trainX_1,[32,32,32])
            np.save(dst + str(patient_id) + '_' + str(node_idx) + str('_oa') + '.npy', trainX_1)
            np.save(dst + str(patient_id) + '_' + str(node_idx) + str('_ob') + '.npy', np.fliplr(trainX_1))
            np.save(dst + str(patient_id) + '_' + str(node_idx) + str('_oc') + '.npy', np.flipud(trainX_1))            
    else:
        for j in range(times):
            rand_nb = int(diam/2)
            rand_list=range(-rand_nb,rand_nb+1)
            rand_list.remove(0)
            new_z1 = new_z + random.choice(rand_list)
            new_y1 = new_y + random.choice(rand_list)
            new_x1 = new_x + random.choice(rand_list)
            trainX_2 = img_array[new_z1 - 3*diam: new_z1 + 3*diam,
                                 new_y1 - 3*diam: new_y1 + 3*diam,
                                 new_x1 - 3*diam: new_x1 + 3*diam]
            if trainX_2.shape == (diam*6, diam*6, diam*6) and trainX_2.sum() > 5./255 and trainX_2.mean() > 10./255:
                trainX_2 = resize(trainX_2,[32,32,32])
                np.save(dst + str(patient_id) + '_' + str(node_idx) + '_' + str(j) + str('a') + '.npy', trainX_2)
                np.save(dst + str(patient_id) + '_' + str(node_idx) + '_' + str(j) + str('b') + '.npy', np.fliplr(trainX_2))
                np.save(dst + str(patient_id) + '_' + str(node_idx) + '_' + str(j) + str('c') + '.npy', np.flipud(trainX_2))                
    return

def make_cls_true_false_old(img_array, v_center, times, patient_id, node_idx, dst):
    new_x = int(v_center[0])
    new_y = int(v_center[1])
    new_z = int(v_center[2])
    if times == 1:
        trainX_1 = img_array[new_z - 32: new_z + 32,
                   new_y - 32: new_y + 32,
                   new_x - 32: new_x + 32]
        if trainX_1.shape == (32, 32, 32) and trainX_1.sum() > 5 and trainX_1.mean() > 10:
            np.save(dst + str(patient_id) + '_' + str(node_idx) + str('_oa') + '.npy', trainX_1)
            np.save(dst + str(patient_id) + '_' + str(node_idx) + str('_ob') + '.npy', np.fliplr(trainX_1))
            np.save(dst + str(patient_id) + '_' + str(node_idx) + str('_oc') + '.npy', np.flipud(trainX_1))            
    else:
        for j in range(times):
            new_z1 = new_z + random.choice([-3, -2, -1, 1, 2, 3])
            new_y1 = new_y + random.choice([-3, -2, -1, 1, 2, 3])
            new_x1 = new_x + random.choice([-3, -2, -1, 1, 2, 3])
            trainX_2 = img_array[new_z1 - 16: new_z1 + 16,
                       new_y1 - 16: new_y1 + 16,
                       new_x1 - 16: new_x1 + 16]
            if trainX_2.shape == (32, 32, 32) and trainX_2.sum() > 5 and trainX_2.mean() > 10:
                np.save(dst + str(patient_id) + '_' + str(node_idx) + '_' + str(j) + str('a') + '.npy', trainX_2)
                np.save(dst + str(patient_id) + '_' + str(node_idx) + '_' + str(j) + str('b') + '.npy', np.fliplr(trainX_2))
                np.save(dst + str(patient_id) + '_' + str(node_idx) + '_' + str(j) + str('c') + '.npy', np.flipud(trainX_2))                
    return

def create_cls_sample(df_anno,df_pred,img_file,data_path,output_true,output_false):
    mini_df_anno = df_anno[df_anno["file"]==img_file] 
    mini_df_pred = df_pred[df_pred["file"]==img_file]
    if mini_df_anno.shape[0]>0:
        patient_id = img_file[:-9]
        img_array = np.load(data_path + img_file)
        img_array = normalize(img_array)
        pos_annos = pd.read_csv(data_path + img_file[:-9] + '_annos_pos.csv')
        origin = np.array([pos_annos.loc[0]['origin_x'],pos_annos.loc[0]['origin_y'],pos_annos.loc[0]['origin_z']]) 
        spacing = np.array([pos_annos.loc[0]['spacing_x'],pos_annos.loc[0]['spacing_y'],pos_annos.loc[0]['spacing_z']]) 
        
        for node_idx1, cur_row1 in mini_df_anno.iterrows():       
            node_x = cur_row1["coordX"]
            node_y = cur_row1["coordY"]
            node_z = cur_row1["coordZ"]
            diam = cur_row1["diameter_mm"]
            center = np.array([node_x, node_y, node_z])   
            v_center1 = np.rint(np.absolute(center-origin)/spacing) 
            make_cls_true_false(img_array, v_center1, diam, 1, patient_id, node_idx1, output_true)
            make_cls_true_false(img_array, v_center1, diam, 5, patient_id, node_idx1, output_true)
        for node_idx2, cur_row2 in mini_df_pred.iterrows():       
            node_x = cur_row2["coordX"]
            node_y = cur_row2["coordY"]
            node_z = cur_row2["coordZ"]
            diam = cur_row2["diameter_mm"]
            center = np.array([node_x, node_y, node_z])   
            v_center2 = np.rint(np.absolute(center-origin)/spacing)  
            make_cls_true_false(img_array, v_center2, diam, 1, patient_id, node_idx2, output_false)
            make_cls_true_false(img_array, v_center2, diam, 2, patient_id, node_idx2, output_false)            
    return


# In[3]:


csv_path = PATH['annotations_train']
output_true = PATH['cls_train_cube_30_true']
output_false = PATH['cls_train_cube_30_false']
pred_csv_path = PATH['model_train_pred']
data_path = PATH['model_train_pred']
anno_csv_new = pd.read_csv(pred_csv_path + "anno_true_final.csv")
pred_csv_new = pd.read_csv(pred_csv_path + "anno_false_final.csv")
#pred_csv_new = pd.read_csv(pred_csv_path + "anno_false_final.csv")


# In[4]:


len(anno_csv_new),len(pred_csv_new)


# In[5]:


para = (len(pred_csv_new)*3)/(len(anno_csv_new)*6)


# In[6]:


pred_csv_new = pred_csv_new[pred_csv_new.index%para == 0]


# In[7]:


patients = [x for x in os.listdir(data_path) if 'orig.npy' in x]


# In[8]:


anno_csv_new["file"] = anno_csv_new["seriesuid"].map(lambda file_name: get_filename(patients, file_name))
anno_csv_new = anno_csv_new.dropna()
pred_csv_new["file"] = pred_csv_new["seriesuid"].map(lambda file_name: get_filename(patients, file_name))
pred_csv_new = pred_csv_new.dropna()


# In[9]:


Parallel(n_jobs=-1)(delayed(create_cls_sample)(anno_csv_new,pred_csv_new,patient,data_path,output_true,output_false) for patient in tqdm(sorted(patients)))


# In[ ]:




