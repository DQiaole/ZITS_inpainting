import os
import random
import sys
from glob import glob

import cv2
import numpy as np
import torchvision.transforms.functional as F
from skimage.color import rgb2gray
from skimage.feature import canny
from torch.utils.data import Dataset
import pickle
import skimage.draw

sys.path.append('..')


def to_int(x):
    return tuple(map(int, x))


class ContinuousEdgeLineDatasetMask(Dataset):

    def __init__(self, pt_dataset, mask_path=None, test_mask_path=None, is_train=False, mask_rates=None,
                 image_size=256, line_path=None):

        self.is_train = is_train
        self.pt_dataset = pt_dataset

        self.image_id_list = []
        with open(self.pt_dataset) as f:
            for line in f:
                self.image_id_list.append(line.strip())

        if is_train:
            self.irregular_mask_list = []
            with open(mask_path[0]) as f:
                for line in f:
                    self.irregular_mask_list.append(line.strip())
            self.irregular_mask_list = sorted(self.irregular_mask_list, key=lambda x: x.split('/')[-1])
            self.segment_mask_list = []
            with open(mask_path[1]) as f:
                for line in f:
                    self.segment_mask_list.append(line.strip())
            self.segment_mask_list = sorted(self.segment_mask_list, key=lambda x: x.split('/')[-1])
        else:
            self.mask_list = glob(test_mask_path + '/*')
            self.mask_list = sorted(self.mask_list, key=lambda x: x.split('/')[-1])

        self.image_size = image_size
        self.training = is_train
        self.mask_rates = mask_rates
        self.line_path = line_path
        self.wireframe_th = 0.85

    def __len__(self):
        return len(self.image_id_list)

    def resize(self, img, height, width, center_crop=False):
        imgh, imgw = img.shape[0:2]

        if center_crop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        if imgh > height and imgw > width:
            inter = cv2.INTER_AREA
        else:
            inter = cv2.INTER_LINEAR
        img = cv2.resize(img, (height, width), interpolation=inter)
        return img

    def load_mask(self, img, index):
        imgh, imgw = img.shape[0:2]

        # test mode: load mask non random
        if self.training is False:
            mask = cv2.imread(self.mask_list[index], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (imgw, imgh), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.uint8) * 255
            return mask
        else:  # train mode: 40% mask with random brush, 40% mask with coco mask, 20% with additions
            rdv = random.random()
            if rdv < self.mask_rates[0]:
                mask_index = random.randint(0, len(self.irregular_mask_list) - 1)
                mask = cv2.imread(self.irregular_mask_list[mask_index],
                                  cv2.IMREAD_GRAYSCALE)
            elif rdv < self.mask_rates[1]:
                mask_index = random.randint(0, len(self.segment_mask_list) - 1)
                mask = cv2.imread(self.segment_mask_list[mask_index],
                                  cv2.IMREAD_GRAYSCALE)
            else:
                mask_index1 = random.randint(0, len(self.segment_mask_list) - 1)
                mask_index2 = random.randint(0, len(self.irregular_mask_list) - 1)
                mask1 = cv2.imread(self.segment_mask_list[mask_index1],
                                   cv2.IMREAD_GRAYSCALE).astype(np.float)
                mask2 = cv2.imread(self.irregular_mask_list[mask_index2],
                                   cv2.IMREAD_GRAYSCALE).astype(np.float)
                mask = np.clip(mask1 + mask2, 0, 255).astype(np.uint8)

            if mask.shape[0] != imgh or mask.shape[1] != imgw:
                mask = cv2.resize(mask, (imgw, imgh), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.uint8) * 255  # threshold due to interpolation
            return mask

    def to_tensor(self, img, norm=False):
        # img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        if norm:
            img_t = F.normalize(img_t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        return img_t

    def load_edge(self, img):
        return canny(img, sigma=2, mask=None).astype(np.float)

    def load_wireframe(self, idx, size):
        selected_img_name = self.image_id_list[idx]
        line_name = self.line_path + '/' + os.path.basename(selected_img_name).replace('.png', '.pkl').replace('.jpg', '.pkl')
        wf = pickle.load(open(line_name, 'rb'))
        lmap = np.zeros((size, size))
        for i in range(len(wf['scores'])):
            if wf['scores'][i] > self.wireframe_th:
                line = wf['lines'][i].copy()
                line[0] = line[0] * size
                line[1] = line[1] * size
                line[2] = line[2] * size
                line[3] = line[3] * size
                rr, cc, value = skimage.draw.line_aa(*to_int(line[0:2]), *to_int(line[2:4]))
                lmap[rr, cc] = np.maximum(lmap[rr, cc], value)
        return lmap

    def __getitem__(self, idx):
        selected_img_name = self.image_id_list[idx]
        img = cv2.imread(selected_img_name)
        while img is None:
            print('Bad image {}...'.format(selected_img_name))
            idx = random.randint(0, len(self.image_id_list) - 1)
            img = cv2.imread(self.image_id_list[idx])
        img = img[:, :, ::-1]

        img = self.resize(img, self.image_size, self.image_size, center_crop=False)
        img_gray = rgb2gray(img)
        edge = self.load_edge(img_gray)
        line = self.load_wireframe(idx, self.image_size)
        # load mask
        mask = self.load_mask(img, idx)
        # augment data
        if self.training is True:
            if random.random() < 0.5:
                img = img[:, ::-1, ...].copy()
                edge = edge[:, ::-1].copy()
                line = line[:, ::-1].copy()
            if random.random() < 0.5:
                mask = mask[:, ::-1, ...].copy()
            if random.random() < 0.5:
                mask = mask[::-1, :, ...].copy()

        img = self.to_tensor(img, norm=True)
        edge = self.to_tensor(edge)
        line = self.to_tensor(line)
        mask = self.to_tensor(mask)
        meta = {'img': img, 'mask': mask, 'edge': edge, 'line': line,
                'name': os.path.basename(selected_img_name)}
        return meta


class ContinuousEdgeLineDatasetMaskFinetune(ContinuousEdgeLineDatasetMask):

    def __init__(self, pt_dataset, mask_path=None, test_mask_path=None,
                 is_train=False, mask_rates=None, image_size=256, line_path=None):
        super().__init__(pt_dataset, mask_path, test_mask_path, is_train, mask_rates, image_size, line_path)

    def __getitem__(self, idx):
        selected_img_name = self.image_id_list[idx]
        img = cv2.imread(selected_img_name)
        while img is None:
            print('Bad image {}...'.format(selected_img_name))
            idx = random.randint(0, len(self.image_id_list) - 1)
            img = cv2.imread(self.image_id_list[idx])
        img = img[:, :, ::-1]

        img = self.resize(img, self.image_size, self.image_size, center_crop=False)
        img_gray = rgb2gray(img)
        edge = self.load_edge(img_gray)
        line = self.load_wireframe(idx, self.image_size)
        # load mask
        mask = self.load_mask(img, idx)
        # augment data
        if self.training is True:
            if random.random() < 0.5:
                img = img[:, ::-1, ...].copy()
                edge = edge[:, ::-1].copy()
                line = line[:, ::-1].copy()
            if random.random() < 0.5:
                mask = mask[:, ::-1, ...].copy()
            if random.random() < 0.5:
                mask = mask[::-1, :, ...].copy()

        erode = mask
        img = self.to_tensor(img, norm=True)
        edge = self.to_tensor(edge)
        line = self.to_tensor(line)
        mask = self.to_tensor(mask)
        mask_img = img * (1 - mask)

        # aug for mask-predict
        while True:
            if random.random() > 0.5:
                erode = self.to_tensor(erode)
                break
            k_size = random.randint(5, 25)
            erode2 = cv2.erode(erode // 255, np.ones((k_size, k_size), np.uint8), iterations=1)
            if np.sum(erode2) > 0:
                erode = self.to_tensor(erode2 * 255)
                break

        meta = {'img': img, 'mask_img': mask_img, 'mask': mask, 'erode_mask': erode, 'edge': edge, 'line': line,
                'name': os.path.basename(selected_img_name)}

        return meta
