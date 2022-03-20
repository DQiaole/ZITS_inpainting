import argparse
import os
import pickle
from glob import glob

import numpy as np
import torch
import torchvision.transforms as transforms
from skimage import io
from skimage.transform import resize
from torchvision.transforms import functional as F
from tqdm import tqdm

from src.lsm_hawp.detector import WireframeDetector


class ResizeImage(object):
    def __init__(self, image_height, image_width):
        self.image_height = image_height
        self.image_width = image_width

    def __call__(self, image):
        image = resize(image, (self.image_height, self.image_width))
        image = np.array(image, dtype=np.float32) / 255.0
        return image


class ToTensor(object):
    def __call__(self, image):
        return F.to_tensor(image)


class Normalize(object):
    def __init__(self, mean, std, to_255=True):
        self.mean = mean
        self.std = std
        self.to_255 = to_255

    def __call__(self, image):
        if self.to_255:
            image *= 255.0
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image


def to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device)
        return data
    if isinstance(data, list):
        return [to_device(d, device) for d in data]


class LSM_HAWP:
    def __init__(self, size=512):
        self.lsm_hawp = WireframeDetector(is_cuda=True).cuda()
        self.transform = transforms.Compose([ResizeImage(size, size), ToTensor(),
                                             Normalize(mean=[109.730, 103.832, 98.681],
                                                       std=[22.275, 22.124, 23.229],
                                                       to_255=True)])

    def wireframe_detect(self, img_paths, output_path):
        self.lsm_hawp.eval()
        results = {}
        with torch.no_grad():
            for img_path in tqdm(img_paths):
                image = io.imread(img_path).astype(float)[:, :, :3]
                image = self.transform(image).unsqueeze(0).cuda()
                output = self.lsm_hawp(image)
                output = to_device(output, 'cpu')
                for k in output:
                    if type(output[k]) == torch.Tensor:
                        output[k] = output[k].cpu().numpy()
                fname = os.path.basename(img_path).split('.')[0]
                results[fname] = output
        with open(os.path.join(output_path), 'wb') as w:
            pickle.dump(results, w)


parser = argparse.ArgumentParser(description='HAWP Testing')

parser.add_argument("--ckpt_path", type=str, required=True, help='ckpt path of HAWP')
parser.add_argument("--input_path", type=str, required=True, help='input file path of images')
parser.add_argument("--output_path", type=str, required=True, help='output pkl dir')
args = parser.parse_args()

model = LSM_HAWP(size=512)
model.lsm_hawp.load_state_dict(torch.load(args.ckpt_path)['model'])
img_paths = glob(args.input_path + '/*')
model.wireframe_detect(img_paths, args.output_path)
