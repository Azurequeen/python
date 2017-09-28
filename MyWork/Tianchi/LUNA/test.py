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
                np.save(dst + str(patient_id) + '_' + str(node_idx) + '_' + str(j) + str('b') + '.npy',
                        np.fliplr(trainX_2))
                np.save(dst + str(patient_id) + '_' + str(node_idx) + '_' + str(j) + str('c') + '.npy',
                        np.flipud(trainX_2))
    return


def create_cls_sample(df_anno, df_pred, img_file, data_path, output_true, output_false):
    mini_df_anno = df_anno[df_anno["file"] == img_file]
    mini_df_pred = df_pred[df_pred["file"] == img_file]
    if mini_df_anno.shape[0] > 0:
        patient_id = img_file[:-9]
        img_array = np.load(data_path + img_file)
        img_array = normalize(img_array)
        pos_annos = pd.read_csv(data_path + img_file[:-9] + '_annos_pos.csv')
        origin = np.array([pos_annos.loc[0]['origin_x'], pos_annos.loc[0]['origin_y'], pos_annos.loc[0]['origin_z']])
        spacing = np.array(
            [pos_annos.loc[0]['spacing_x'], pos_annos.loc[0]['spacing_y'], pos_annos.loc[0]['spacing_z']])

        for node_idx1, cur_row1 in mini_df_anno.iterrows():
            node_x = cur_row1["coordX"]
            node_y = cur_row1["coordY"]
            node_z = cur_row1["coordZ"]
            diam = cur_row1["diameter_mm"]
            center = np.array([node_x, node_y, node_z])
            v_center1 = np.rint(np.absolute(center - origin) / spacing)
            make_cls_true_false(img_array, v_center1, diam, 1, patient_id, node_idx1, output_true)
            make_cls_true_false(img_array, v_center1, diam, 5, patient_id, node_idx1, output_true)
        for node_idx2, cur_row2 in mini_df_pred.iterrows():
            node_x = cur_row2["coordX"]
            node_y = cur_row2["coordY"]
            node_z = cur_row2["coordZ"]
            diam = cur_row2["diameter_mm"]
            center = np.array([node_x, node_y, node_z])
            v_center2 = np.rint(np.absolute(center - origin) / spacing)
            make_cls_true_false(img_array, v_center2, diam, 1, patient_id, node_idx2, output_false)
            make_cls_true_false(img_array, v_center2, diam, 2, patient_id, node_idx2, output_false)
    return


