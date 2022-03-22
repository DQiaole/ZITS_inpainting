from src.lsm_hawp.lsm_hawp_model import LSM_HAWP
from glob import glob
import torch
import os
import argparse

parser = argparse.ArgumentParser(description='HAWP Testing')

parser.add_argument("--ckpt_path", type=str, required=True, help='ckpt path of HAWP')
parser.add_argument("--input_path", type=str, required=True, help='input file path of images')
parser.add_argument("--output_path", type=str, required=True, help='output pkl dir')
parser.add_argument("--gpu_ids", type=str, default='0')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

if __name__ == '__main__':
    os.makedirs(args.output_path, exist_ok=True)

    model = LSM_HAWP(threshold=0.8, size=512)
    model.lsm_hawp.load_state_dict(torch.load(args.ckpt_path)['model'])
    img_paths = glob(args.input_path + '/*')
    model.wireframe_detect(img_paths, args.output_path)
