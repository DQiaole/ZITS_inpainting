import argparse
import os
import time

import cv2
import numpy as np
import torch
from tqdm import tqdm

from datasets.dataset_TSR import ContinuousEdgeLineDatasetMask
from src.models.TSR_model import EdgeLineGPTConfig, EdgeLineGPT256RelBCE
from src.utils import set_seed, SampleEdgeLineLogits

if __name__ == '__main__':
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--GPU_ids', type=str, default='0')
    parser.add_argument('--ckpt_path', type=str, default='./ckpt/places2_continous_edgeline/best.pth')
    parser.add_argument('--image_url', type=str, default=None, help='the folder of image')
    parser.add_argument('--mask_url', type=str, default=None)
    parser.add_argument('--test_line_path', type=str, default='', help='Indicate where is the wireframes of test set')
    parser.add_argument('--image_size', type=int, default=256, help='input sequence length: image_size*image_size')
    parser.add_argument('--n_layer', type=int, default=16)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--n_embd', type=int, default=256)
    parser.add_argument('--save_url', type=str, default=None, help='save the output results')
    parser.add_argument('--iterations', type=int, default=5)

    opts = parser.parse_args()

    os.makedirs(opts.save_url + '/edge', exist_ok=True)
    os.makedirs(opts.save_url + '/line', exist_ok=True)

    s_time = time.time()
    model_config = EdgeLineGPTConfig(embd_pdrop=0.0, resid_pdrop=0.0, n_embd=opts.n_embd, block_size=32,
                                     attn_pdrop=0.0, n_layer=opts.n_layer, n_head=opts.n_head)
    # Load model
    IGPT_model = EdgeLineGPT256RelBCE(model_config)
    checkpoint = torch.load(opts.ckpt_path)

    if opts.ckpt_path.endswith('.pt'):
        IGPT_model.load_state_dict(checkpoint)
    else:
        IGPT_model.load_state_dict(checkpoint['model'])

    IGPT_model.cuda()

    test_dataset = ContinuousEdgeLineDatasetMask(opts.image_url, test_mask_path=opts.mask_url, is_train=False,
                                                 image_size=opts.image_size, line_path=opts.test_line_path)

    for it in tqdm(range(test_dataset.__len__())):

        items = test_dataset.__getitem__(it)

        edge_pred, line_pred = SampleEdgeLineLogits(IGPT_model, context=[items['img'].unsqueeze(0),
                                                   items['edge'].unsqueeze(0), items['line'].unsqueeze(0)],
                              mask=items['mask'].unsqueeze(0), iterations=opts.iterations)
        # save separately
        edge_output = edge_pred[0, ...].cpu() * items['mask'] + items['edge'] * (1 - items['mask'])
        edge_output = edge_output.repeat(3, 1, 1).permute(1, 2, 0)
        line_output = line_pred[0, ...].cpu() * items['mask'] + items['line'] * (1 - items['mask'])
        line_output = line_output.repeat(3, 1, 1).permute(1, 2, 0)

        edge_output = (edge_output * 255).detach().numpy().astype(np.uint8)
        line_output = (line_output * 255).detach().numpy().astype(np.uint8)

        cv2.imwrite(opts.save_url + '/edge/' + items['name'], edge_output[:, :, ::-1])
        cv2.imwrite(opts.save_url + '/line/' + items['name'], line_output[:, :, ::-1])

    e_time = time.time()
    print("This inference totally costs %.5f seconds" % (e_time - s_time))
