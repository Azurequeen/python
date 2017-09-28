import numpy as np
import os
import time
import collections
import random
import torch
from detector.layers import iou
from scipy.ndimage import zoom
import warnings
from scipy.ndimage.interpolation import rotate


class Dataset(object):
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class DataBowl3Detector(Dataset):
    def __init__(self, data_dir, split_path, config, phase='train', split_comber=None):
        assert (phase == 'train' or phase == 'val' or phase == 'test')
        self.phase = phase                                                                              #阶段：train、val或test
        self.max_stride = config['max_stride']                                                          #最大滑动步长
        self.stride = config['stride']                                                                  #滑动步长
        sizelim = config['sizelim'] / config['reso']                                                    #尺寸限制
        sizelim2 = config['sizelim2'] / config['reso']
        sizelim3 = config['sizelim3'] / config['reso']
        self.blacklist = config['blacklist']                                                            #黑名单
        self.isScale = config['aug_scale']                                                              #图像增强缩放
        self.r_rand = config['r_rand_crop']                                                             #随机切割（获得难分样本，0.3）
        self.augtype = config['augtype']                                                                #是否翻转、旋转等
        self.pad_value = config['pad_value']                                                            #pad的大小（170，前期骨骼210改为170）
        self.split_comber = split_comber
        idcs = np.load(split_path)                                                                      #获得分割好的样本名数据集
        if phase != 'test':
            idcs = [f for f in idcs if (f not in self.blacklist)]                                       #去掉在黑名单的样本名

        self.filenames = [os.path.join(data_dir, '%s_clean.npy' % idx) for idx in idcs]                 #读取样本
        self.kagglenames = [f for f in self.filenames if len(f.split('/')[-1].split('_')[0]) > 20]      #读取的样本中，名字大于20个字符的是kaggle的数据
        self.lunanames = [f for f in self.filenames if len(f.split('/')[-1].split('_')[0]) < 20]        #名字小于20个字符的是luna的数据
        labels = []

        for idx in idcs:
            l = np.load(os.path.join(data_dir, '%s_label.npy' % idx))                                   #读入样本相应的anno坐标
            if np.all(l == 0):
                l = np.array([])
            labels.append(l)                                                                            #如果坐标都是0，则改l为空list，将所有l加入labels这个list
        self.sample_bboxes = labels                                                                     #sample_bboxs为anno坐标的集合
        if self.phase != 'test':                                                                        #如果是test阶段，将sample_bboxs重置为空
            self.bboxes = []
            for i, l in enumerate(labels):                                                              #如果是train和valid阶段，如果l不为0（有标注结节），
                if len(l) > 0:
                    for t in l:
                        if t[3] > sizelim:
                            self.bboxes.append([np.concatenate([[i], t])])                              #建立空数组bboxes，如果直径大于sizelim（6，最小），将labels序号和坐标、尺寸存入
                        if t[3] > sizelim2:
                            self.bboxes += [[np.concatenate([[i], t])]] * 2                             #如果大于sizelim2（30）,存入2遍
                        if t[3] > sizelim3:
                            self.bboxes += [[np.concatenate([[i], t])]] * 4                             #如果大于sizelim3（40）,存入4遍
            self.bboxes = np.concatenate(self.bboxes, axis=0)                                           #将bboxes从list转为numpy

        self.crop = Crop(config)                                                                        #切割
        self.label_mapping = LabelMapping(config, self.phase)                                           #坐标映射

    def __getitem__(self, idx, split=None):
        t = time.time()                                                                                 #时间
        np.random.seed(int(str(t % 1)[2:7]))  # seed according to time                                  #根据时间的随机种子

        isRandomImg = False
        if self.phase != 'test':                                                                        #isRandomImg先置为False
            if idx >= len(self.bboxes):                                                                 # 若不是test，idx比bboxes大，idx就是bboxes和idx的余数
                isRandom = True
                idx = idx % len(self.bboxes)
                isRandomImg = np.random.randint(2)                                                      #isRandomImg设置为随机的0或1
            else:
                isRandom = False                                                                        #否则isRandom为False，isRandomImg仍为False
        else:
            isRandom = False

        if self.phase != 'test':                                                                        #若不是test，且isRandomImg为False
            if not isRandomImg:
                bbox = self.bboxes[idx]                                                                 #bbox为bboxes的第idx个
                filename = self.filenames[int(bbox[0])]                                                 #在所有样本中，获取该样本的文件名
                imgs = np.load(filename)                                                                #获得图像文件
                bboxes = self.sample_bboxes[int(bbox[0])]                                               #在anno文件中获得该样本标注的结节位置坐标
                isScale = self.augtype['scale'] and (self.phase == 'train')                             #若是训练数据，获得是否要改变尺寸的设置
                sample, target, bboxes, coord = self.crop(imgs, bbox[1:], bboxes, isScale, isRandom)    #将图像、该样本结节坐标、所有结节坐标表、是否改变尺寸、是否随机送入crop函数进行切割
                if self.phase == 'train' and not isRandom:                                              #如果是train且不随机
                    sample, target, bboxes, coord = augment(sample, target, bboxes, coord,              #将切割获得的样本、目标、结节表和坐标送入增强函数
                                                            ifflip=self.augtype['flip'],
                                                            ifrotate=self.augtype['rotate'],
                                                            ifswap=self.augtype['swap'])
            else:                                                                                       #若不是test且isRandomImg为True
                randimid = np.random.randint(len(self.kagglenames))                                     #在kaggle文件名表中获取随机的index
                filename = self.kagglenames[randimid]                                                   #从kaggle文件表中获得样本文件名
                imgs = np.load(filename)                                                                #获得该样本图像
                bboxes = self.sample_bboxes[randimid]                                                   #获得样本结节位置坐标
                isScale = self.augtype['scale'] and (self.phase == 'train')                             #若是训练数据，获得是否需要改变尺寸
                sample, target, bboxes, coord = self.crop(imgs, [], bboxes, isScale=False, isRand=True) #将这些信息送入crop函数进行切割
            label = self.label_mapping(sample.shape[1:], target, bboxes)                                #获得样本的映射后的坐标
            sample = (sample.astype(np.float32) - 128) / 128                                            #以128为中心归一化？？？？？？
            # if filename in self.kagglenames and self.phase=='train':
            #    label[label==-1]=0
            return torch.from_numpy(sample), torch.from_numpy(label), coord
        else:
            imgs = np.load(self.filenames[idx])                                                         #若isRandomImg为True，获取样本图像
            bboxes = self.sample_bboxes[idx]                                                            #获取样本的结节标注坐标
            nz, nh, nw = imgs.shape[1:]                                                                 #以下暂时没看明白？？？？？？？？？？？？？？？
            pz = int(np.ceil(float(nz) / self.stride)) * self.stride
            ph = int(np.ceil(float(nh) / self.stride)) * self.stride
            pw = int(np.ceil(float(nw) / self.stride)) * self.stride
            imgs = np.pad(imgs, [[0, 0], [0, pz - nz], [0, ph - nh], [0, pw - nw]], 'constant',
                          constant_values=self.pad_value)

            xx, yy, zz = np.meshgrid(np.linspace(-0.5, 0.5, imgs.shape[1] / self.stride),
                                     np.linspace(-0.5, 0.5, imgs.shape[2] / self.stride),
                                     np.linspace(-0.5, 0.5, imgs.shape[3] / self.stride), indexing='ij')
            coord = np.concatenate([xx[np.newaxis, ...], yy[np.newaxis, ...], zz[np.newaxis, :]], 0).astype('float32')
            imgs, nzhw = self.split_comber.split(imgs)
            coord2, nzhw2 = self.split_comber.split(coord,
                                                    side_len=self.split_comber.side_len / self.stride,
                                                    max_stride=self.split_comber.max_stride / self.stride,
                                                    margin=self.split_comber.margin / self.stride)
            assert np.all(nzhw == nzhw2)
            imgs = (imgs.astype(np.float32) - 128) / 128
            return torch.from_numpy(imgs), bboxes, torch.from_numpy(coord2), np.array(nzhw)

    def __len__(self):                                                                                  #长度函数
        if self.phase == 'train':                                                                       #如果是train，长度为anno标注表长度/(1-难分样本比例)
            return int(len(self.bboxes) / (1 - self.r_rand))
        elif self.phase == 'val':                                                                       #如果是val，长度为anno标注表长度
            return int(len(self.bboxes))
        else:                                                                                           #如果是test，长度为sample_bboxes的长度
            return int(len(self.sample_bboxes))


def augment(sample, target, bboxes, coord, ifflip=True, ifrotate=True, ifswap=True):                    #增强函数，旋转、交换、翻转
    #                     angle1 = np.random.rand()*180
    if ifrotate:
        validrot = False
        counter = 0
        while not validrot:
            newtarget = np.copy(target)
            angle1 = np.random.rand() * 180
            size = np.array(sample.shape[2:4]).astype('float')
            rotmat = np.array([[np.cos(angle1 / 180 * np.pi), -np.sin(angle1 / 180 * np.pi)],
                               [np.sin(angle1 / 180 * np.pi), np.cos(angle1 / 180 * np.pi)]])
            newtarget[1:3] = np.dot(rotmat, target[1:3] - size / 2) + size / 2
            if np.all(newtarget[:3] > target[3]) and np.all(newtarget[:3] < np.array(sample.shape[1:4]) - newtarget[3]):
                validrot = True
                target = newtarget
                sample = rotate(sample, angle1, axes=(2, 3), reshape=False)
                coord = rotate(coord, angle1, axes=(2, 3), reshape=False)
                for box in bboxes:
                    box[1:3] = np.dot(rotmat, box[1:3] - size / 2) + size / 2
            else:
                counter += 1
                if counter == 3:
                    break
    if ifswap:
        if sample.shape[1] == sample.shape[2] and sample.shape[1] == sample.shape[3]:
            axisorder = np.random.permutation(3)
            sample = np.transpose(sample, np.concatenate([[0], axisorder + 1]))
            coord = np.transpose(coord, np.concatenate([[0], axisorder + 1]))
            target[:3] = target[:3][axisorder]
            bboxes[:, :3] = bboxes[:, :3][:, axisorder]

    if ifflip:
        #         flipid = np.array([np.random.randint(2),np.random.randint(2),np.random.randint(2)])*2-1
        flipid = np.array([1, np.random.randint(2), np.random.randint(2)]) * 2 - 1
        sample = np.ascontiguousarray(sample[:, ::flipid[0], ::flipid[1], ::flipid[2]])
        coord = np.ascontiguousarray(coord[:, ::flipid[0], ::flipid[1], ::flipid[2]])
        for ax in range(3):
            if flipid[ax] == -1:
                target[ax] = np.array(sample.shape[ax + 1]) - target[ax]
                bboxes[:, ax] = np.array(sample.shape[ax + 1]) - bboxes[:, ax]
    return sample, target, bboxes, coord


class Crop(object):
    def __init__(self, config):
        self.crop_size = config['crop_size']                                                            #【128，128，128】
        self.bound_size = config['bound_size']                                                          #12
        self.stride = config['stride']                                                                  #4
        self.pad_value = config['pad_value']                                                            #170

    def __call__(self, imgs, target, bboxes, isScale=False, isRand=False):
        if isScale:                                                                                     #若需要变换尺寸
            radiusLim = [8., 120.]                                                                      #半径限制：8~120
            scaleLim = [0.75, 1.25]                                                                     #尺寸限制：0.75~1.25
            scaleRange = [np.min([np.max([(radiusLim[0] / target[3]), scaleLim[0]]), 1])                #target：结节坐标，根据radius，确定坐标实际变换的范围
                , np.max([np.min([(radiusLim[1] / target[3]), scaleLim[1]]), 1])]
            scale = np.random.rand() * (scaleRange[1] - scaleRange[0]) + scaleRange[0]                  #在变换范围内获取一个随机变换尺寸
            crop_size = (np.array(self.crop_size).astype('float') / scale).astype('int')                #根据变换尺寸，改变原【128，128，128】的切割尺寸
        else:
            crop_size = self.crop_size                                                                  #若不需要进行尺寸变换，则切割尺寸为【128，128，128】
        bound_size = self.bound_size                                                                    #边缘距离：12
        target = np.copy(target)
        bboxes = np.copy(bboxes)

        start = []
        for i in range(3):
            if not isRand:
                r = target[3] / 2
                s = np.floor(target[i] - r) + 1 - bound_size
                e = np.ceil(target[i] + r) + 1 + bound_size - crop_size[i]
            else:
                s = np.max([imgs.shape[i + 1] - crop_size[i] / 2, imgs.shape[i + 1] / 2 + bound_size])
                e = np.min([crop_size[i] / 2, imgs.shape[i + 1] / 2 - bound_size])
                target = np.array([np.nan, np.nan, np.nan, np.nan])
            if s > e:
                start.append(np.random.randint(e, s))  # !
            else:
                start.append(int(target[i]) - crop_size[i] / 2 + np.random.randint(-bound_size / 2, bound_size / 2))

        normstart = np.array(start).astype('float32') / np.array(imgs.shape[1:]) - 0.5
        normsize = np.array(crop_size).astype('float32') / np.array(imgs.shape[1:])
        xx, yy, zz = np.meshgrid(np.linspace(normstart[0], normstart[0] + normsize[0], self.crop_size[0] / self.stride),
                                 np.linspace(normstart[1], normstart[1] + normsize[1], self.crop_size[1] / self.stride),
                                 np.linspace(normstart[2], normstart[2] + normsize[2], self.crop_size[2] / self.stride),
                                 indexing='ij')
        coord = np.concatenate([xx[np.newaxis, ...], yy[np.newaxis, ...], zz[np.newaxis, :]], 0).astype('float32')

        pad = []
        pad.append([0, 0])
        for i in range(3):
            leftpad = max(0, -start[i])
            rightpad = max(0, start[i] + crop_size[i] - imgs.shape[i + 1])
            pad.append([leftpad, rightpad])
        crop = imgs[:,
               max(start[0], 0):min(start[0] + crop_size[0], imgs.shape[1]),
               max(start[1], 0):min(start[1] + crop_size[1], imgs.shape[2]),
               max(start[2], 0):min(start[2] + crop_size[2], imgs.shape[3])]
        crop = np.pad(crop, pad, 'constant', constant_values=self.pad_value)
        for i in range(3):
            target[i] = target[i] - start[i]
        for i in range(len(bboxes)):
            for j in range(3):
                bboxes[i][j] = bboxes[i][j] - start[j]

        if isScale:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                crop = zoom(crop, [1, scale, scale, scale], order=1)
            newpad = self.crop_size[0] - crop.shape[1:][0]
            if newpad < 0:
                crop = crop[:, :-newpad, :-newpad, :-newpad]
            elif newpad > 0:
                pad2 = [[0, 0], [0, newpad], [0, newpad], [0, newpad]]
                crop = np.pad(crop, pad2, 'constant', constant_values=self.pad_value)
            for i in range(4):
                target[i] = target[i] * scale
            for i in range(len(bboxes)):
                for j in range(4):
                    bboxes[i][j] = bboxes[i][j] * scale
        return crop, target, bboxes, coord


class LabelMapping(object):
    def __init__(self, config, phase):
        self.stride = np.array(config['stride'])
        self.num_neg = int(config['num_neg'])
        self.th_neg = config['th_neg']
        self.anchors = np.asarray(config['anchors'])
        self.phase = phase
        if phase == 'train':
            self.th_pos = config['th_pos_train']
        elif phase == 'val':
            self.th_pos = config['th_pos_val']

    def __call__(self, input_size, target, bboxes):
        stride = self.stride
        num_neg = self.num_neg
        th_neg = self.th_neg
        anchors = self.anchors
        th_pos = self.th_pos

        output_size = []
        for i in range(3):
            assert (input_size[i] % stride == 0)
            output_size.append(int(input_size[i] / stride))

        # label = -1 * np.ones(output_size + [len(anchors), 5], np.float32)
        
        print(output_size)
        print([len(anchors), 5])
        print(output_size + [len(anchors), 5])
        
        label = -1 * np.ones(output_size + [len(anchors), 5], np.float32)
        offset = ((stride.astype('float')) - 1) / 2
        oz = np.arange(offset, offset + stride * (output_size[0] - 1) + 1, stride)
        oh = np.arange(offset, offset + stride * (output_size[1] - 1) + 1, stride)
        ow = np.arange(offset, offset + stride * (output_size[2] - 1) + 1, stride)

        for bbox in bboxes:
            for i, anchor in enumerate(anchors):
                iz, ih, iw = select_samples(bbox, anchor, th_neg, oz, oh, ow)
                label[iz, ih, iw, i, 0] = 0

        if self.phase == 'train' and self.num_neg > 0:
            neg_z, neg_h, neg_w, neg_a = np.where(label[:, :, :, :, 0] == -1)
            neg_idcs = random.sample(range(len(neg_z)), min(num_neg, len(neg_z)))
            neg_z, neg_h, neg_w, neg_a = neg_z[neg_idcs], neg_h[neg_idcs], neg_w[neg_idcs], neg_a[neg_idcs]
            label[:, :, :, :, 0] = 0
            label[neg_z, neg_h, neg_w, neg_a, 0] = -1

        if np.isnan(target[0]):
            return label
        iz, ih, iw, ia = [], [], [], []
        for i, anchor in enumerate(anchors):
            iiz, iih, iiw = select_samples(target, anchor, th_pos, oz, oh, ow)
            iz.append(iiz)
            ih.append(iih)
            iw.append(iiw)
            ia.append(i * np.ones((len(iiz),), np.int64))
        iz = np.concatenate(iz, 0)
        ih = np.concatenate(ih, 0)
        iw = np.concatenate(iw, 0)
        ia = np.concatenate(ia, 0)
        flag = True
        if len(iz) == 0:
            pos = []
            for i in range(3):
                pos.append(max(0, int(np.round((target[i] - offset) / stride))))
            idx = np.argmin(np.abs(np.log(target[3] / anchors)))
            pos.append(idx)
            flag = False
        else:
            idx = random.sample(range(len(iz)), 1)[0]
            pos = [iz[idx], ih[idx], iw[idx], ia[idx]]
        dz = (target[0] - oz[pos[0]]) / anchors[pos[3]]
        dh = (target[1] - oh[pos[1]]) / anchors[pos[3]]
        dw = (target[2] - ow[pos[2]]) / anchors[pos[3]]
        dd = np.log(target[3] / anchors[pos[3]])
        label[pos[0], pos[1], pos[2], pos[3], :] = [1, dz, dh, dw, dd]
        return label


def select_samples(bbox, anchor, th, oz, oh, ow):
    z, h, w, d = bbox
    max_overlap = min(d, anchor)
    min_overlap = np.power(max(d, anchor), 3) * th / max_overlap / max_overlap
    if min_overlap > max_overlap:
        return np.zeros((0,), np.int64), np.zeros((0,), np.int64), np.zeros((0,), np.int64)
    else:
        s = z - 0.5 * np.abs(d - anchor) - (max_overlap - min_overlap)
        e = z + 0.5 * np.abs(d - anchor) + (max_overlap - min_overlap)
        mz = np.logical_and(oz >= s, oz <= e)
        iz = np.where(mz)[0]

        s = h - 0.5 * np.abs(d - anchor) - (max_overlap - min_overlap)
        e = h + 0.5 * np.abs(d - anchor) + (max_overlap - min_overlap)
        mh = np.logical_and(oh >= s, oh <= e)
        ih = np.where(mh)[0]

        s = w - 0.5 * np.abs(d - anchor) - (max_overlap - min_overlap)
        e = w + 0.5 * np.abs(d - anchor) + (max_overlap - min_overlap)
        mw = np.logical_and(ow >= s, ow <= e)
        iw = np.where(mw)[0]

        if len(iz) == 0 or len(ih) == 0 or len(iw) == 0:
            return np.zeros((0,), np.int64), np.zeros((0,), np.int64), np.zeros((0,), np.int64)

        lz, lh, lw = len(iz), len(ih), len(iw)
        iz = iz.reshape((-1, 1, 1))
        ih = ih.reshape((1, -1, 1))
        iw = iw.reshape((1, 1, -1))
        iz = np.tile(iz, (1, lh, lw)).reshape((-1))
        ih = np.tile(ih, (lz, 1, lw)).reshape((-1))
        iw = np.tile(iw, (lz, lh, 1)).reshape((-1))
        centers = np.concatenate([
            oz[iz].reshape((-1, 1)),
            oh[ih].reshape((-1, 1)),
            ow[iw].reshape((-1, 1))], axis=1)

        r0 = anchor / 2
        s0 = centers - r0
        e0 = centers + r0

        r1 = d / 2
        s1 = bbox[:3] - r1
        s1 = s1.reshape((1, -1))
        e1 = bbox[:3] + r1
        e1 = e1.reshape((1, -1))

        overlap = np.maximum(0, np.minimum(e0, e1) - np.maximum(s0, s1))

        intersection = overlap[:, 0] * overlap[:, 1] * overlap[:, 2]
        union = anchor * anchor * anchor + d * d * d - intersection

        iou = intersection / union

        mask = iou >= th
        # if th > 0.4:
        #   if np.sum(mask) == 0:
        #      print(['iou not large', iou.max()])
        # else:
        #    print(['iou large', iou[mask]])
        iz = iz[mask]
        ih = ih[mask]
        iw = iw[mask]
        return iz, ih, iw


def collate(batch):
    if torch.is_tensor(batch[0]):
        return [b.unsqueeze(0) for b in batch]
    elif isinstance(batch[0], np.ndarray):
        return batch
    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], collections.Iterable):
        transposed = zip(*batch)
        return [collate(samples) for samples in transposed]


