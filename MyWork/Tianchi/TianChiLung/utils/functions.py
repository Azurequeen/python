# -*- coding: utf-8 -*-
from imports import *


threshold_min = -1200
threshold_max = 600
height_mask = 512
width_mask = 512



def load_train(PATH):
    src = PATH['src']
    data_path = src
    folders = [x for x in os.listdir(data_path) if 'subset' in x]
    os.chdir(data_path)
    patients = []
    for i in folders:
        os.chdir(data_path + i)
        # print('Changing folder to: {}'.format(data_path + i))
        patient_ids = [x for x in os.listdir(data_path + i) if '.mhd' in x]
        for id in patient_ids:
            j = '{}/{}'.format(i, id)
            patients.append(j)
    return patients


def get_dirfiles(dir):
    file_list = []
    subset_path = os.listdir(dir)
    for _ in range(len(subset_path)):
        if subset_path[_] != '.DS_Store':
            file_list.append(dir + subset_path[_])
    return file_list


def get_filename(file_list, case):
    for f in file_list:
        if case in f:
            return (f)



def seq(start, stop, step=1):
    n = int(round((stop - start) / float(step)))
    if n > 1:
        return ([start + step * i for i in range(n + 1)])
    else:
        return ([])


def load_itk(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
    return numpyImage, numpyOrigin, numpySpacing


def world_2_voxel(world_coordinates, origin, spacing):
    stretched_voxel_coordinates = np.absolute(world_coordinates - origin)
    voxel_coordinates = stretched_voxel_coordinates / spacing
    return voxel_coordinates


def voxel_2_world(voxel_coordinates, origin, spacing):
    stretched_voxel_coordinates = voxel_coordinates * spacing
    world_coordinates = stretched_voxel_coordinates + origin
    return world_coordinates


def get_pixels_hu(image):
    image = image.astype(np.int16)
    image[image == threshold_min] = 0
    return np.array(image, dtype=np.int16)


def get_nodule_slices(lung_mask, nodule_mask, lung_raw):
    indexes = np.unique(np.nonzero(nodule_mask)[0])
    # print('Nodule_present on slices: {}'.format(indexes))
    lung_mask_pres = lung_mask[indexes, :, :]
    nod_mask_pres = nodule_mask[indexes, :, :]
    lung_raw_pres = lung_raw[indexes, :, :]
    return lung_mask_pres, nod_mask_pres, lung_raw_pres


def reshape_3d(image_3d):
    reshaped_img = image_3d.reshape([image_3d.shape[0], 1, 512, 512])
    # print('Reshaped image shape:', reshaped_img.shape)
    return reshaped_img

def intersect(coord1,diam1,coord2,diam2): 
    aa = coord1 - diam1
    bb = coord1 + diam1
    cc = coord2 - diam2
    dd = coord2 + diam2
    
    if ( max(aa, bb)<min(cc, dd) ):
        return False;        
    if ( max(cc, dd)<min(aa, bb) ):
        return False;        
      
    return True;  

def indiam(coord1,coord2,diam2): 
    aa = coord1
    bb = coord1
    cc = coord2 - diam2
    dd = coord2 + diam2
    
    if ( max(aa, bb)<min(cc, dd) ):
        return False;        
    if ( max(cc, dd)<min(aa, bb) ):
        return False;        
      
    return True;


def in64mm(coord1, coord2, diam2):
    aa = coord1 - 64
    bb = coord1 + 64
    cc = coord2 - diam2
    dd = coord2 + diam2

    if (max(aa, bb) < min(cc, dd)):
        return False;
    if (max(cc, dd) < min(aa, bb)):
        return False;

    return True;



def create_masks_for_patient_watershed(img_file,df_node,PATH,save=True):
    src = PATH['src']
    dst_nodules = PATH['dst_nodules']
    dst_full_lungs = PATH['dst_full_lungs']


    def draw_nodule_mask(node_idx, cur_row):
        # print('Working on node: {}, row: {}'.format(node_idx, cur_row), '\n')
        coord_x = cur_row["coordX"]
        coord_y = cur_row["coordY"]
        coord_z = cur_row["coordZ"]
        diam = cur_row["diameter_mm"]
        radius = np.ceil(diam / 2)
        noduleRange = seq(-radius, radius, RESIZE_SPACING[0])
        # print('Nodule range:', noduleRange)
        world_center = np.array((coord_z, coord_y, coord_x))  # nodule center
        voxel_center = world_2_voxel(world_center, origin, new_spacing)
        image_mask = np.zeros(lung_img.shape)
        for x in noduleRange:
            for y in noduleRange:
                for z in noduleRange:
                    coords = world_2_voxel(np.array((coord_z + z, coord_y + y, coord_x + x)), origin, new_spacing)
                    if (np.linalg.norm(voxel_center - coords) * RESIZE_SPACING[0]) < radius:
                        image_mask[int(np.round(coords[0])), int(np.round(coords[1])), int(np.round(coords[2]))] = int(
                            1)
        # print(np.max(image_mask))

        return image_mask

    #print("Getting mask for image file {}".format(img_file))
    patient_id = img_file.split('/')[-1][:-4]
    mini_df = df_node[df_node["file"] == img_file]
    if mini_df.shape[0] > 0:  # some files may not have a nodule--skipping those
        img, origin, spacing = load_itk(src + img_file)
        height, width = img.shape[1], img.shape[2]
        # calculate resize factor
        RESIZE_SPACING = [1, 1, 1]
        resize_factor = spacing / RESIZE_SPACING
        new_real_shape = img.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize = new_shape / img.shape
        new_spacing = spacing / real_resize

        lung_img = scipy.ndimage.interpolation.zoom(img, real_resize)

        # print('Original image shape: {}'.format(img.shape))
        # print('Resized image shape: {}'.format(lung_img.shape))

        lung_img = get_pixels_hu(lung_img)
        # lung_mask = segment_lung_from_ct_scan(lung_img)
        # lung_mask[lung_mask >= threshold_max] = threshold_max
        # lung_img[lung_img >= threshold_max] = threshold_max
        # lung_img[lung_img == 0] = threshold_min



        lung_mask = lung_img.copy()
        # lung_mask[lung_mask == 0] = threshold_min
        lung_mask[lung_mask >= threshold_max] = threshold_max
        lung_img[lung_img >= threshold_max] = threshold_max

        lung_masks_512 = lung_full_masks_512 = np.zeros([lung_img.shape[0], height_mask, width_mask], dtype=np.float32)
        nodule_masks_512 = nodule_full_masks_512 = np.zeros([lung_img.shape[0], height_mask, width_mask], dtype=np.float32)
        lung_masks_512[lung_masks_512 == 0] = threshold_min
        lung_full_masks_512[lung_full_masks_512 == 0] = threshold_min

        i = 0
        for node_idx, cur_row in mini_df.iterrows():
            nodule_mask = draw_nodule_mask(node_idx, cur_row)
            lung_img_512, lung_mask_512, nodule_mask_512 = np.zeros((lung_img.shape[0], 512, 512)), np.zeros(
                (lung_mask.shape[0], 512, 512)), np.zeros((nodule_mask.shape[0], 512, 512))
            lung_mask_512[lung_mask_512 == 0] = threshold_min
            lung_img_512[lung_img_512 == 0] = threshold_min
            original_shape = lung_img.shape

            for z in range(lung_img.shape[0]):
                offset = (512 - original_shape[1])
                upper_offset = int(np.round(offset / 2))
                lower_offset = int(offset - upper_offset)

                new_origin = voxel_2_world([-upper_offset, -lower_offset, 0], origin, new_spacing)

                lung_img_512[z, upper_offset:-lower_offset, upper_offset:-lower_offset] = lung_img[z, :, :]
                lung_mask_512[z, upper_offset:-lower_offset, upper_offset:-lower_offset] = lung_mask[z, :, :]
                nodule_mask_512[z, upper_offset:-lower_offset, upper_offset:-lower_offset] = nodule_mask[z, :, :]
            nodule_masks_512 += nodule_mask_512

        # print('Offsets shape for node index {} - main: {}, upper: {}, lower: {}'.format(node_idx, offset, upper_offset, lower_offset), '\n')


        lung_mask_pres, nod_mask_pres, lung_raw_pres = get_nodule_slices(lung_mask_512, nodule_masks_512, lung_img_512)
        # print('Nodules present on slices: ', np.unique(np.nonzero(nodule_masks_512)[0]))



        del lung_mask_512, nodule_masks_512, lung_img_512
        gc.collect()

        lung_mask_pres = reshape_3d(lung_mask_pres)
        nod_mask_pres = reshape_3d(nod_mask_pres)
        lung_mask_pres[lung_mask_pres <= threshold_min] = threshold_min
        lung_mask_pres[lung_mask_pres >= threshold_max] = threshold_max

        lung_mask_preproc = my_PreProc(lung_mask_pres)
        lung_mask_preproc = lung_mask_preproc.astype(np.float32)
        nod_mask_pres = (nod_mask_pres > 0.0).astype(np.float32)
        nod_mask_pres[nod_mask_pres == 1.0] = 255.

        np.save('{}lung_mask/{}.npy'.format(dst_nodules, patient_id), lung_mask_preproc)
        np.save('{}nodule_mask/{}.npy'.format(dst_nodules, patient_id), nod_mask_pres)

        del lung_mask_pres, lung_mask_preproc, nod_mask_pres


        nodule_mask = np.ones(lung_img.shape)
        # print('nodule_mask shape: {}'.format(nodule_mask.shape))
        lung_full_img_512, lung_full_mask_512, nodule_full_mask_512 = np.zeros((lung_img.shape[0], 512, 512)), np.zeros(
            (lung_mask.shape[0], 512, 512)), np.zeros((nodule_mask.shape[0], 512, 512))
        lung_full_mask_512[lung_full_mask_512 == 0] = threshold_min
        lung_full_img_512[lung_full_img_512 == 0] = threshold_min
        original_full_shape = lung_img.shape

        for z in range(lung_img.shape[0]):
            offset_full = (512 - original_full_shape[1])
            upper_full_offset = int(np.round(offset_full / 2))
            lower_full_offset = int(offset_full - upper_full_offset)

            new_origin_full = voxel_2_world([-upper_full_offset, -lower_full_offset, 0], origin, new_spacing)

            lung_full_img_512[z, upper_full_offset:-lower_full_offset, upper_full_offset:-lower_full_offset] = lung_img[z, :, :]
            lung_full_mask_512[z, upper_full_offset:-lower_full_offset, upper_full_offset:-lower_full_offset] = lung_mask[z, :, :]
            nodule_full_mask_512[z, upper_full_offset:-lower_full_offset, upper_full_offset:-lower_full_offset] = nodule_mask[z, :, :]
        nodule_full_masks_512 += nodule_full_mask_512
        # print('Offsets shape for node index {} - main: {}, upper: {}, lower: {}'.format(node_idx, offset, upper_offset, lower_offset), '\n')

        # print('nodule_masks_512 shape: {}'.format(nodule_masks_512.shape))

        lung_full_mask_pres, nod_full_mask_pres, lung_full_raw_pres = get_nodule_slices(lung_full_mask_512, nodule_full_masks_512, lung_full_img_512)
        # print('Nodules present on slices: ', np.unique(np.nonzero(nodule_masks_512)[0]))



        del lung_full_mask_512, nodule_full_masks_512, lung_full_img_512


        lung_full_mask_pres = reshape_3d(lung_full_mask_pres)
        nod_full_mask_pres = reshape_3d(nod_full_mask_pres)
        lung_full_mask_pres[lung_full_mask_pres <= threshold_min] = threshold_min
        lung_full_mask_pres[lung_full_mask_pres >= threshold_max] = threshold_max

        lung_full_mask_preproc = my_PreProc(lung_full_mask_pres)
        lung_full_mask_preproc = lung_full_mask_preproc.astype(np.float32)
        nod_full_mask_pres = (nod_full_mask_pres > 0.0).astype(np.float32)
        nod_full_mask_pres[nod_full_mask_pres == 1.0] = 255.
        
        #if dst_full_lungs[-16:-11] !='train':
        #    np.save('{}{}.npy'.format(dst_full_lungs, patient_id), lung_full_mask_preproc)
        # np.save('{}/nodule_mask/{}.npy'.format(dst_nodules, patient_id), nod_mask_pres)



        del lung_full_mask_pres, lung_full_mask_preproc, nod_full_mask_pres

        var = (patient_id, origin[0], origin[1], origin[2], new_spacing[0], new_spacing[1], new_spacing[2], offset,
               img.shape[0], img.shape[1], img.shape[2], spacing[0], spacing[1], spacing[2])

        # return lung_mask_preproc, nod_mask_pres
        return var
    else:
        print('\n', 'No nodules found for patient: {}'.format(patient_id), '\n')
        return

def segment_HU_scan_ira(x, threshold=-350, min_area=300):
    mask = np.asarray(x < threshold, dtype='int8')

    for zi in xrange(mask.shape[0]):
        skimage.segmentation.clear_border(mask[zi, :, :], in_place=True)

    # noise reduction
    mask = skimage.morphology.binary_opening(mask, skimage.morphology.cube(2))
    mask = np.asarray(mask, dtype='int8')

    # label regions
    label_image = skimage.measure.label(mask)
    region_props = skimage.measure.regionprops(label_image)
    sorted_regions = sorted(region_props, key=lambda x: x.area, reverse=True)
    lung_label = sorted_regions[0].label
    lung_mask = np.asarray((label_image == lung_label), dtype='int8')

    # convex hull mask
    lung_mask_convex = np.zeros_like(lung_mask)
    for i in range(lung_mask.shape[2]):
        if np.any(lung_mask[:, :, i]):
            lung_mask_convex[:, :, i] = skimage.morphology.convex_hull_image(lung_mask[:, :, i])

    # old mask inside the convex hull
    mask *= lung_mask_convex
    label_image = skimage.measure.label(mask)
    region_props = skimage.measure.regionprops(label_image)
    sorted_regions = sorted(region_props, key=lambda x: x.area, reverse=True)

    for r in sorted_regions[1:]:
        if r.area > min_area:
            # make an image only containing that region
            label_image_r = label_image == r.label
            # grow the mask
            label_image_r = scipy.ndimage.binary_dilation(label_image_r,
                                                          structure=scipy.ndimage.generate_binary_structure(3, 2))
            # compute the overlap with true lungs
            overlap = label_image_r * lung_mask
            if not np.any(overlap):
                for i in range(label_image_r.shape[0]):
                    if np.any(label_image_r[i]):
                        label_image_r[i] = skimage.morphology.convex_hull_image(label_image_r[i])
                lung_mask_convex *= 1 - label_image_r

    return lung_mask_convex

def create_masks_for_test(img_file,PATH,save=True):
    src = PATH['src']
    dst_nodules = PATH['dst_nodules']
    dst_full_lungs = PATH['dst_full_lungs']

    patient_id = img_file.split('/')[-1][:-4]
    img, origin, spacing = load_itk(src + img_file)
    lung_mask = segment_HU_scan_ira(img)

    lung_mask[lung_mask<0.8] = int(0)
    lung_mask[lung_mask>=0.8] = int(1)
    img = lung_mask*img
    
    RESIZE_SPACING = [1, 1, 1]
    resize_factor = spacing / RESIZE_SPACING
    new_real_shape = img.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize = new_shape / img.shape
    new_spacing = spacing / real_resize

    lung_img = scipy.ndimage.interpolation.zoom(img, real_resize)


    # print('Original image shape: {}'.format(img.shape))
    # print('Resized image shape: {}'.format(lung_img.shape))

    lung_img = get_pixels_hu(lung_img)
    # lung_mask = segment_lung_from_ct_scan(lung_img)
    # lung_mask[lung_mask >= threshold_max] = threshold_max
    # lung_img[lung_img >= threshold_max] = threshold_max
    # lung_img[lung_img == 0] = threshold_min



    lung_mask = lung_img.copy()
    # lung_mask[lung_mask == 0] = threshold_min
    lung_mask[lung_mask >= threshold_max] = threshold_max
    lung_img[lung_img >= threshold_max] = threshold_max
    lung_img = lung_mask*lung_img

    lung_masks_512 = np.zeros([lung_img.shape[0], height_mask, width_mask], dtype=np.float32)
    nodule_masks_512 = np.zeros([lung_img.shape[0], height_mask, width_mask], dtype=np.float32)
    lung_masks_512[lung_masks_512 == 0] = threshold_min

    i = 0

    nodule_mask = np.ones(lung_img.shape)
    # print('nodule_mask shape: {}'.format(nodule_mask.shape))
    lung_img_512, lung_mask_512, nodule_mask_512 = np.zeros((lung_img.shape[0], 512, 512)), np.zeros(
        (lung_mask.shape[0], 512, 512)), np.zeros((nodule_mask.shape[0], 512, 512))
    lung_mask_512[lung_mask_512 == 0] = threshold_min
    lung_img_512[lung_img_512 == 0] = threshold_min
    original_shape = lung_img.shape

    for z in range(lung_img.shape[0]):
        offset = (512 - original_shape[1])
        upper_offset = int(np.round(offset / 2))
        lower_offset = int(offset - upper_offset)

        new_origin = voxel_2_world([-upper_offset, -lower_offset, 0], origin, new_spacing)

        lung_img_512[z, upper_offset:-lower_offset, upper_offset:-lower_offset] = lung_img[z, :, :]
        lung_mask_512[z, upper_offset:-lower_offset, upper_offset:-lower_offset] = lung_mask[z, :, :]
        nodule_mask_512[z, upper_offset:-lower_offset, upper_offset:-lower_offset] = nodule_mask[z, :, :]
    nodule_masks_512 += nodule_mask_512
    # print('Offsets shape for node index {} - main: {}, upper: {}, lower: {}'.format(node_idx, offset, upper_offset, lower_offset), '\n')

    # print('nodule_masks_512 shape: {}'.format(nodule_masks_512.shape))

    lung_mask_pres, nod_mask_pres, lung_raw_pres = get_nodule_slices(lung_mask_512, nodule_masks_512, lung_img_512)
    # print('Nodules present on slices: ', np.unique(np.nonzero(nodule_masks_512)[0]))



    del lung_mask_512, nodule_masks_512, lung_img_512


    lung_mask_pres = reshape_3d(lung_mask_pres)
    nod_mask_pres = reshape_3d(nod_mask_pres)
    lung_mask_pres[lung_mask_pres <= threshold_min] = threshold_min
    lung_mask_pres[lung_mask_pres >= threshold_max] = threshold_max

    lung_mask_preproc = my_PreProc(lung_mask_pres)
    lung_mask_preproc = lung_mask_preproc.astype(np.float32)
    nod_mask_pres = (nod_mask_pres > 0.0).astype(np.float32)
    nod_mask_pres[nod_mask_pres == 1.0] = 255.

    np.save('{}{}.npy'.format(dst_full_lungs, patient_id), lung_mask_preproc)
    # np.save('{}/nodule_mask/{}.npy'.format(dst_nodules, patient_id), nod_mask_pres)

    del lung_mask_pres, lung_mask_preproc, nod_mask_pres

    var = (patient_id, origin[0], origin[1], origin[2], new_spacing[0], new_spacing[1], new_spacing[2], offset,
           img.shape[0], img.shape[1], img.shape[2], spacing[0], spacing[1], spacing[2])
        # return lung_mask_preproc, nod_mask_pres
    return var

def save_test_full_to_csv(i,PATH):
    patient_id_= []
    origin1_= []
    origin2_= []
    origin3_= []
    new_spacing1_= []
    new_spacing2_= []
    new_spacing3_= []
    offset_= []
    imgshape1_= []
    imgshape2_= []
    imgshape3_= []
    spacing1_= []
    spacing2_= []
    spacing3_= []


    for u in tqdm(range(len(i[0]))):
        patient_id_.append(i[0][u])
        origin1_.append(i[1][u])
        origin2_.append(i[2][u])
        origin3_.append(i[3][u])
        new_spacing1_.append(i[4][u])
        new_spacing2_.append(i[5][u])
        new_spacing3_.append(i[6][u])
        offset_.append(i[7][u])
        imgshape1_.append(i[8][u])
        imgshape2_.append(i[9][u])
        imgshape3_.append(i[10][u])
        spacing1_.append(i[11][u])
        spacing2_.append(i[12][u])
        spacing3_.append(i[13][u])

    df_node = pd.DataFrame({
        'seriesuid' : patient_id_,
        'origin1' : origin1_,
        'origin2' : origin2_,
        'origin3' : origin3_,

        'new_spacing1': new_spacing1_,
        'new_spacing2': new_spacing2_,
        'new_spacing3': new_spacing3_,

        'offset': offset_,
        'imgshape1': imgshape1_,
        'imgshape2': imgshape2_,
        'imgshape3': imgshape3_,

        'spacing1': spacing1_,
        'spacing2': spacing2_,
        'spacing3': spacing3_,

    })

    df_node = df_node[['seriesuid', 'origin1' ,'origin2','origin3','new_spacing1','new_spacing2','new_spacing3',
                       'offset','imgshape1','imgshape2','imgshape3','spacing1','spacing2','spacing3']]


    df_node.to_csv(PATH['annotations_path'] + 'annotations_full.csv', index=None)
    print(u'结果保存为：%s' %str(PATH['annotations_path'] + 'annotations_full.csv'))




def predict_on_scan(scan,model_path,dropout_rate,width):
    model = unet_model(dropout_rate,width)
    model.load_weights(model_path + 'FenGe.h5')

    num_test = scan.shape[0]
    scan = scan.reshape(num_test, 1, 512, 512)
    imgs_mask_test = np.ndarray([num_test, 1, 512, 512],dtype=np.float32)
    for i in range(num_test):
        imgs_mask_test[i] = model.predict([scan[i:i+1]], verbose=0)[0]

    del num_test, scan
    return imgs_mask_test


def predict(files,PATH):
    model_path = PATH['model_path']
    dropout_rate = PATH['dropout_rate']
    width = PATH['width']


    temp = np.load(files)
    

    patient_id = files.split('/')[-1][:-4]

    temp = predict_on_scan(temp, model_path,dropout_rate,width)
    temp = np.squeeze(temp)
    temp[temp < 1] = 0
    temp = skimage.morphology.binary_opening(np.squeeze(temp), np.ones([5, 5, 5]))
    labels = measure.label(np.squeeze(temp))
    props = regionprops(labels)

    patient_id_ = []
    Centroid3_ = []
    Centroid1_ = []
    Centroid2_ = []
    EquivDiameter_ = []

    for i in range(len(props)):
        if props[i]['EquivDiameter'] < 30:

            patient_id_.append(patient_id)
            Centroid3_.append(props[i]['Centroid'][2])
            Centroid1_.append(props[i]['Centroid'][0])
            Centroid2_.append(props[i]['Centroid'][1])
            EquivDiameter_.append(props[i]['EquivDiameter'])


    var = (patient_id_,Centroid3_,Centroid1_,Centroid2_,EquivDiameter_)
    return var

def save_to_csv(i,df_node,PATH):
    for u in tqdm(range(len(i[0]))):
        df_node['origin1'].replace(str(i[0][u]), i[1][u], inplace=True)
        df_node['origin2'].replace(str(i[0][u]), i[2][u], inplace=True)
        df_node['origin3'].replace(str(i[0][u]), i[3][u], inplace=True)
        df_node['new_spacing1'].replace(str(i[0][u]), i[4][u], inplace=True)
        df_node['new_spacing2'].replace(str(i[0][u]), i[5][u], inplace=True)
        df_node['new_spacing3'].replace(str(i[0][u]), i[6][u], inplace=True)
        df_node['offset'].replace(str(i[0][u]), i[7][u], inplace=True)
        df_node['imgshape1'].replace(str(i[0][u]), i[8][u], inplace=True)
        df_node['imgshape2'].replace(str(i[0][u]), i[9][u], inplace=True)
        df_node['imgshape3'].replace(str(i[0][u]), i[10][u], inplace=True)
        df_node['spacing1'].replace(str(i[0][u]), i[11][u], inplace=True)
        df_node['spacing2'].replace(str(i[0][u]), i[12][u], inplace=True)
        df_node['spacing3'].replace(str(i[0][u]), i[13][u], inplace=True)
    df_node.to_csv(PATH['annotations_path'] + 'annotations_full.csv', index=None)
    print(u'结果保存为：%s' %str(PATH['annotations_path'] + 'annotations_full.csv'))


def save_pred_to_csv(i,PATH):
    patient_id_ = []
    Centroid3_ = []
    Centroid1_ = []
    Centroid2_ = []
    EquivDiameter_ = []
    for u in range(len(i[0])):
        for v in tqdm(range(len(i[0][u]))):
            patient_id_.append(i[0][u][v])
            Centroid3_.append(i[1][u][v])
            Centroid1_.append(i[2][u][v])
            Centroid2_.append(i[3][u][v])
            EquivDiameter_.append(i[4][u][v])




    df_node = pd.DataFrame({
        'seriesuid' : patient_id_,
        'coordZ' : Centroid3_,
        'coordX' : Centroid1_,
        'coordY' : Centroid2_,
        'diameter_mm' : EquivDiameter_
    })
    df_node = df_node[['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm']]
    df_node.to_csv(PATH['annotations_path'] + 'annotations_pred.csv', index=None)
    
    


def df_add_column(df_node):
    df_node['origin1'] = df_node['origin2'] = df_node['origin3'] = df_node['seriesuid']
    df_node['new_spacing1'] = df_node['new_spacing2'] = df_node['new_spacing3'] = df_node['offset'] = df_node[
        'seriesuid']
    df_node['imgshape1'] = df_node['imgshape2'] = df_node['imgshape3'] = df_node['seriesuid']
    df_node['spacing1'] = df_node['spacing2'] = df_node['spacing3'] = df_node['seriesuid']
    return df_node


def result_v2w(final, PATH):
    seriesuid_ = []
    diameter_mm_ = []
    coordZ_w = []
    coordX_w = []
    coordY_w = []

    offset_ = []

    origin1_ = []
    origin2_ = []
    origin3_ = []
    new_spacing1_ = []
    new_spacing2_ = []
    new_spacing3_ = []

    voxel_x = []
    voxel_y = []
    voxel_z = []

    for ix, row in tqdm(final.iterrows()):
        img_shape = np.array([row['imgshape1'], row['imgshape2'], row['imgshape3']])
        origin = np.array([row['origin1'], row['origin2'], row['origin3']])
        spacing = np.array([row['spacing1'], row['spacing2'], row['spacing3']])

        RESIZE_SPACING = [1, 1, 1]
        resize_factor = spacing / RESIZE_SPACING

        new_real_shape = img_shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize = new_shape / img_shape

        new_spacing = spacing / real_resize
        lung_img_shape = img_shape * real_resize
        original_shape = lung_img_shape
        offset = (512 - original_shape[1])
        upper_offset = int(np.round(offset / 2))
        lower_offset = int(offset - upper_offset)

        new_origin = voxel_2_world([-upper_offset, -lower_offset, 0], origin, new_spacing)

        [a1, a2, a3] = img_shape
        [b1, b2, b3] = lung_img_shape
        aa = np.array([a1, a2, a3], dtype=np.float32)
        bb = np.array([b1, b2, b3], dtype=np.float32)

        offset = int(np.round(offset / 2))

        z = row['coordX']
        y = row['coordY'] - offset
        x = row['coordZ'] - offset

        voxel_center = np.array([z, y, x])

        world_center = voxel_2_world(voxel_center, origin, new_spacing)

        coordZ_w.append(world_center[0])
        coordY_w.append(world_center[1])
        coordX_w.append(world_center[2])

        origin1_.append(row['origin1'])
        origin2_.append(row['origin2'])
        origin3_.append(row['origin3'])
        new_spacing1_.append(row['new_spacing1'])
        new_spacing2_.append(row['new_spacing2'])
        new_spacing3_.append(row['new_spacing3'])
        offset_.append(offset)
        seriesuid_.append(row['seriesuid'])
        diameter_mm_.append(row['diameter_mm'])
        voxel_x.append(row['coordX'])
        voxel_y.append(row['coordY'])
        voxel_z.append(row['coordZ'])


    bb = pd.DataFrame({'seriesuid': seriesuid_,
                       'coordX': coordX_w,
                       'coordY': coordY_w,
                       'coordZ': coordZ_w,
                       'diameter_mm': diameter_mm_,
                       'origin1': origin1_,
                       'origin2': origin2_,
                       'origin3': origin3_,
                       'new_spacing1': new_spacing1_,
                       'new_spacing2': new_spacing2_,
                       'new_spacing3': new_spacing3_,
                       'offset': offset_,
                       'voxel_x' : voxel_x,
                       'voxel_y' : voxel_y,
                       'voxel_z' : voxel_z,
                       })
    bb = bb[['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm', 'origin1', 'origin2', 'origin3',
             'new_spacing1', 'new_spacing2', 'new_spacing3', 'offset','voxel_x','voxel_y','voxel_z']]

    return bb


def pred_result(file_list, PATH):
    pred = [[], [], [], [], []]
    for files in tqdm(file_list):
        r = predict(files, PATH)

        pred[0].append(r[0])
        pred[1].append(r[1])
        pred[2].append(r[2])
        pred[3].append(r[3])
        pred[4].append(r[4])
        del r
        K.clear_session()
    return pred


def merge_pred_origin_csv2(PATH):
    origin1_ = []
    origin2_ = []
    origin3_ = []
    new_spacing1_ = []
    new_spacing2_ = []
    new_spacing3_ = []
    offset_ = []
    imgshape1_ = []
    imgshape2_ = []
    imgshape3_ = []
    spacing1_ = []
    spacing2_ = []
    spacing3_ = []
  
    pred = pd.read_csv(PATH['annotations_path'] + "annotations_pred.csv", index_col=None)
    origin = pd.read_csv(PATH['annotations_path'] + "annotations_full.csv", index_col=None)
    for idx1,row1 in tqdm(pred.iterrows()):  
        for idx2,row2 in origin.iterrows():        
            if row1['seriesuid'] == row2['seriesuid']:
                origin1_.append(row2['origin1'])
                origin2_.append(row2['origin2'])
                origin3_.append(row2['origin3'])
                new_spacing1_.append(row2['new_spacing1'])
                new_spacing2_.append(row2['new_spacing2'])
                new_spacing3_.append(row2['new_spacing3'])
                offset_.append(row2['offset'])
                imgshape1_.append(row2['imgshape1'])
                imgshape2_.append(row2['imgshape2'])
                imgshape3_.append(row2['imgshape3'])
                spacing1_.append(row2['spacing1'])
                spacing2_.append(row2['spacing2'])
                spacing3_.append(row2['spacing3'])
    
    pred['origin1'] = origin1_
    pred['origin2'] = origin2_
    pred['origin3'] = origin3_
    pred['new_spacing1'] = new_spacing1_
    pred['new_spacing2'] = new_spacing2_
    pred['new_spacing3'] = new_spacing3_
    pred['offset'] = offset_
    pred['imgshape1'] = imgshape1_
    pred['imgshape2'] = imgshape2_
    pred['imgshape3'] = imgshape3_
    pred['spacing1'] = spacing1_
    pred['spacing2'] = spacing2_
    pred['spacing3'] = spacing3_
    
    pred.to_csv(PATH['annotations_path'] + 'annotations_final1.csv', index=None)
    print(u'结果保存为：%s' %str(PATH['annotations_path'] + 'annotations_final1.csv'))
    return pred

def merge_pred_origin_csv(PATH):

  
    pred = pd.read_csv(PATH['annotations_path'] + "annotations_pred.csv", index_col=None)
    origin = pd.read_csv(PATH['annotations_path'] + "annotations_full.csv", index_col=None)
    
    origin1_ = np.empty([pred['seriesuid'].count()])
    origin2_ = np.empty([pred['seriesuid'].count()])
    origin3_ = np.empty([pred['seriesuid'].count()])
    new_spacing1_ = np.empty([pred['seriesuid'].count()])
    new_spacing2_ = np.empty([pred['seriesuid'].count()])
    new_spacing3_  = np.empty([pred['seriesuid'].count()])
    offset_ = np.empty([pred['seriesuid'].count()])
    imgshape1_ = np.empty([pred['seriesuid'].count()])
    imgshape2_ = np.empty([pred['seriesuid'].count()])
    imgshape3_ = np.empty([pred['seriesuid'].count()])
    spacing1_ = np.empty([pred['seriesuid'].count()])
    spacing2_ = np.empty([pred['seriesuid'].count()])
    spacing3_ = np.empty([pred['seriesuid'].count()])
    
    
    for idx1,row1 in tqdm(pred.iterrows()):  
        for idx2,row2 in origin.iterrows():        
            if row1['seriesuid'] == row2['seriesuid']:
                origin1_[idx1] = row2['origin1']
                origin2_[idx1] = row2['origin2']
                origin3_[idx1] = row2['origin3']
                new_spacing1_[idx1] = row2['new_spacing1']
                new_spacing2_[idx1] = row2['new_spacing2']
                new_spacing3_[idx1] = row2['new_spacing3']
                offset_[idx1] = row2['offset']
                imgshape1_[idx1] = row2['imgshape1']
                imgshape2_[idx1] = row2['imgshape2']
                imgshape3_[idx1] = row2['imgshape3']
                spacing1_[idx1] = row2['spacing1']
                spacing2_[idx1] = row2['spacing2']
                spacing3_[idx1] = row2['spacing3']
    
    pred['origin1'] = origin1_
    pred['origin2'] = origin2_
    pred['origin3'] = origin3_
    pred['new_spacing1'] = new_spacing1_
    pred['new_spacing2'] = new_spacing2_
    pred['new_spacing3'] = new_spacing3_
    pred['offset'] = offset_
    pred['imgshape1'] = imgshape1_
    pred['imgshape2'] = imgshape2_
    pred['imgshape3'] = imgshape3_
    pred['spacing1'] = spacing1_
    pred['spacing2'] = spacing2_
    pred['spacing3'] = spacing3_
    
    pred.to_csv(PATH['annotations_path'] + 'annotations_final1.csv', index=None)
    print(u'结果保存为：%s' %str(PATH['annotations_path'] + 'annotations_final1.csv'))
    return pred





def ifin(PATH): 
    pred = pd.read_csv(PATH['annotations_path'] + "annotations_final2cls.csv", index_col=None)
    origin = pd.read_csv(PATH['annotations_path'] + "annotations.csv", index_col=None)
    
    intersect_ = np.zeros([pred['seriesuid'].count()])
    indiam_ = np.zeros([pred['seriesuid'].count()])
    intersect64_ = np.zeros([pred['seriesuid'].count()])

    intersect_origin = np.zeros([origin['seriesuid'].count()])
    indiam_origin = np.zeros([origin['seriesuid'].count()])
    intersect64_origin = np.zeros([origin['seriesuid'].count()])
    
    for idx1,row1 in tqdm(pred.iterrows()):  
        for idx2,row2 in origin.iterrows():      
            if row1['seriesuid'] == row2['seriesuid']:
                
                
                if intersect(row1['coordX'],row1['diameter_mm'], row2['coordX'],row2['diameter_mm']):
                    if intersect(row1['coordY'],row1['diameter_mm'], row2['coordY'],row2['diameter_mm']):
                        if intersect(row1['coordZ'],row1['diameter_mm'], row2['coordZ'],row2['diameter_mm']):
                            intersect_[idx1] = int(1)
                            intersect_origin[idx2] = int(1)
                if indiam(row1['coordX'], row2['coordX'],row2['diameter_mm']):
                    if indiam(row1['coordY'], row2['coordY'],row2['diameter_mm']):
                        if indiam(row1['coordZ'], row2['coordZ'],row2['diameter_mm']):
                            indiam_[idx1] = int(1)
                            indiam_origin[idx2] = int(1)
                if in64mm(row1['coordX'], row2['coordX'],row2['diameter_mm']):
                    if in64mm(row1['coordY'], row2['coordY'],row2['diameter_mm']):
                        if in64mm(row1['coordZ'], row2['coordZ'],row2['diameter_mm']):
                            intersect64_[idx1] = int(1)
                            intersect64_origin[idx2] = int(1)
    pred['intersect'] = intersect_
    pred['indiam'] = indiam_
    pred['intersect64'] = intersect64_
    
    
    pred.to_csv(PATH['annotations_path'] + 'annotations_final2cls.csv', index=None)
    
    print(u'结果保存为：%s' %str(PATH['annotations_path'] + 'annotations_final2cls.csv'))
    #print(u'预测结果与给定标注区域相交比例：%.2f%%' %(pred[pred['intersect']==1].seriesuid.count()*100.0/origin.seriesuid.count()))
    #print(u'预测质心在给定标注区域内的比例：%.2f%%' %(pred[pred['indiam']==1].seriesuid.count()*100.0/origin.seriesuid.count()))
    #print(u'预测质心在给定标注直径64内比例：%.2f%%' %(pred[pred['intersect64']==1].seriesuid.count()*100.0/origin.seriesuid.count()))

    print(u'预测结果与给定标注区域相交比例：%.2f%%' %(sum(intersect_origin)*100.0/origin.seriesuid.count()))
    print(u'预测质心在给定标注区域内的比例：%.2f%%' %(sum(indiam_origin)*100.0/origin.seriesuid.count()))
    print(u'预测质心在给定标注直径64内比例：%.2f%%' %(sum(intersect64_origin)*100.0/origin.seriesuid.count()))
    return

def generate_cls_sample3d(file_list,full,PATH,train = True):
    if train:
        for i in tqdm(range(len(file_list))):
            patient_id = file_list[i][-14:-4]
            aa = np.load(file_list[i])
            aa = np.squeeze(aa)

            for ix, row in full.iterrows():
                if row["seriesuid"] == patient_id:
                    coord_x = row["voxel_x"]
                    coord_y = row["voxel_y"]
                    coord_z = row["voxel_z"]
                    diam = row["diameter_mm"]

                    times = 0
                    if row['indiam'] == 1:
                        for i in range(1,int(diam//7) + 1):
                            rand_s = 3 * (np.random.binomial(1, 0.5, size=4) * 2 - 1)
                            for j in range(1,int(diam // 4) + 1):
                                a0 = general_sample3d(aa, coord_x, coord_y, coord_z)
                                a1 = general_sample3d(aa, coord_x + 2 * j , coord_y + i * rand_s[0], coord_z + i * rand_s[1])
                                a2 = general_sample3d(aa, coord_x - 2 * j , coord_y + i * rand_s[2], coord_z + i * rand_s[3])
                                np.save(PATH['pics_path_true'] + str(ix).zfill(6) + str('_z_') + str(i).zfill(
                                    2) + '.npy', a0)
                                np.save(PATH['pics_path_true'] + str(ix).zfill(6) + str('_u_') + str(i).zfill(
                                    2) + '.npy', a1)
                                np.save(PATH['pics_path_true'] + str(ix).zfill(6) + str('_d_') + str(i).zfill(
                                    2) + '.npy', a2)

                    else:
                        if row['intersect64'] != 1:
                            a3 = general_sample3d(aa, coord_x, coord_y, coord_z)
                            np.save(PATH['pics_path_false'] + str(ix).zfill(6) + str('_') + str(times).zfill(2) + '.npy',a3)
        print(u'生成完毕！')

    else:
        for i in tqdm(range(len(file_list))):
            patient_id = file_list[i][-14:-4]
            aa = np.load(file_list[i])
            aa = np.squeeze(aa)

            for ix, row in full.iterrows():
                if row["seriesuid"] == patient_id:
                    coord_x = row["voxel_x"]
                    coord_y = row["voxel_y"]
                    coord_z = row["voxel_z"]
                    diam = row["diameter_mm"]

                    np.save(PATH['pics_path_true'] + str(ix).zfill(6) + str(0).zfill(2) + '.npy',
                                    general_sample3d(aa,coord_x, coord_y, coord_z))


        print(u'生成完毕！')


def general_sample3d(aa,coord_x, coord_y, coord_z):
    x1 = xx1 = int(np.round(coord_x)) - 32
    x2 = xx2 = int(np.round(coord_x)) + 32
    y1 = yy1 = int(np.round(coord_y)) - 32
    y2 = yy2 = int(np.round(coord_y)) + 32
    z1 = zz1 = int(np.round(coord_z)) - 32
    z2 = zz2 = int(np.round(coord_z)) + 32
    sample_y1 = sample_z1 = sample_x1 = 0
    sample = np.ones([64, 64, 64])
    sample = sample * 170

    if x1 < 0:
        xx1 = 0
        sample_x1 = -x1
    if x2 >aa.shape[0]:
        sample_x1 = xx2 - aa.shape[0]
        xx2 = aa.shape[0]
    if y1 < 0:
        yy1 = 0
        sample_y1 = -y1-1
    if y2 > 512:
        sample_y1 = yy2 - 512
        yy2 = 512
    if z1 < 0:
        zz1 = 0
        sample_z1 = -z1-1
    if z2 > 512:
        sample_z1 = zz2 - 512
        zz2 = 512
    sample[sample_x1:, sample_y1:, sample_z1:] = aa[xx1:xx2, yy1:yy2, zz1:zz2]
    return sample



def generate_cls_sample2d(file_list,full,PATH,train = True):
    if train:
        for i in tqdm(range(len(file_list))):
            patient_id = file_list[i][-14:-4]
            aa = np.load(file_list[i])
            aa = np.squeeze(aa)

            for ix, row in full.iterrows():
                if row["seriesuid"] == patient_id:
                    coord_x = row["voxel_x"]
                    coord_y = row["voxel_y"]
                    coord_z = row["voxel_z"]
                    diam = row["diameter_mm"]

                    times = 0
                    if row['indiam'] == 1:
                        for i in range(1,int(diam//7) + 1):
                            rand_s = 3 * (np.random.binomial(1, 0.5, size=4) * 2 - 1)
                            for j in range(1,int(diam // 4) + 1):
                                a0 = general_sample2d(aa, coord_x, coord_y, coord_z)
                                a1 = general_sample2d(aa, coord_x + 2 * j , coord_y + i * rand_s[0], coord_z + i * rand_s[1])
                                a2 = general_sample2d(aa, coord_x - 2 * j , coord_y + i * rand_s[2], coord_z + i * rand_s[3])
                                misc.imsave(PATH['pics_path_true'] + str(ix).zfill(6) + str('_z_') + str(i).zfill(
                                    2) + '.jpg', a0[0])
                                misc.imsave(PATH['pics_path_true'] + str(ix).zfill(6) + str('_u_') + str(i).zfill(
                                    2) + '.jpg', a1[0])
                                misc.imsave(PATH['pics_path_true'] + str(ix).zfill(6) + str('_d_') + str(i).zfill(
                                    2) + '.jpg', a2[0])

                    else:
                        if row['intersect64'] != 1:
                            a3 = general_sample2d(aa, coord_x, coord_y, coord_z)
                            misc.imsave(PATH['pics_path_false'] + str(ix).zfill(6) + str('_') + str(times).zfill(2) + '.jpg',a3[0])
        print(u'生成完毕！')

    else:
        for i in tqdm(range(len(file_list))):
            patient_id = file_list[i][-14:-4]
            aa = np.load(file_list[i])
            aa = np.squeeze(aa)

            for ix, row in full.iterrows():
                if row["seriesuid"] == patient_id:
                    coord_x = row["voxel_x"]
                    coord_y = row["voxel_y"]
                    coord_z = row["voxel_z"]
                    diam = row["diameter_mm"]

                    misc.imsave(PATH['pics_path_true'] + str(ix).zfill(6) + str(0).zfill(2) + '.jpg',
                                    general_sample2d(coord_x, coord_y, coord_z))


        print(u'生成完毕！')


def general_sample2d(aa,coord_x, coord_y, coord_z):
    x = int(np.round(coord_x))
    y1 = yy1 = int(np.round(coord_y)) - 32
    y2 = yy2 = int(np.round(coord_y)) + 32
    z1 = zz1 = int(np.round(coord_z)) - 32
    z2 = zz2 = int(np.round(coord_z)) + 32
    sample_y1 = sample_z1 = 0
    sample = np.ones([1, 64, 64])
    sample = sample * 170

    if y1 < 0:
        yy1 = 0
        sample_y1 = -y1-1
    if y2 > 512:
        sample_y1 = yy2 - 512
        yy2 = 512
    if z1 < 0:
        zz1 = 0
        sample_z1 = -z1-1
    if z2 > 512:
        sample_z1 = zz2 - 512
        zz2 = 512
    sample[:, sample_y1:, sample_z1:] = aa[x, yy1:yy2, zz1:zz2]
    return sample



