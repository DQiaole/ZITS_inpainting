import argparse
import os
import random
from shutil import copyfile

import cv2
import numpy as np
import torch
from src.lsm_hawp.detector import WireframeDetector
from src.FTR_trainer import ZITS
from src.config import Config
from skimage.color import rgb2gray
import torchvision.transforms.functional as FF
import torch.nn.functional as F
from skimage.feature import canny
import skimage
from src.utils import stitch_images, SampleEdgeLineLogits


def load_masked_position_encoding(mask):
    ones_filter = np.ones((3, 3), dtype=np.float32)
    d_filter1 = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=np.float32)
    d_filter2 = np.array([[0, 0, 0], [1, 1, 0], [1, 1, 0]], dtype=np.float32)
    d_filter3 = np.array([[0, 1, 1], [0, 1, 1], [0, 0, 0]], dtype=np.float32)
    d_filter4 = np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]], dtype=np.float32)
    str_size = 256
    pos_num = 128

    ori_mask = mask.copy()
    ori_h, ori_w = ori_mask.shape[0:2]
    ori_mask = ori_mask / 255
    mask = cv2.resize(mask, (str_size, str_size), interpolation=cv2.INTER_AREA)
    mask[mask > 0] = 255
    h, w = mask.shape[0:2]
    mask3 = mask.copy()
    mask3 = 1. - (mask3 / 255.0)
    pos = np.zeros((h, w), dtype=np.int32)
    direct = np.zeros((h, w, 4), dtype=np.int32)
    i = 0
    while np.sum(1 - mask3) > 0:
        i += 1
        mask3_ = cv2.filter2D(mask3, -1, ones_filter)
        mask3_[mask3_ > 0] = 1
        sub_mask = mask3_ - mask3
        pos[sub_mask == 1] = i

        m = cv2.filter2D(mask3, -1, d_filter1)
        m[m > 0] = 1
        m = m - mask3
        direct[m == 1, 0] = 1

        m = cv2.filter2D(mask3, -1, d_filter2)
        m[m > 0] = 1
        m = m - mask3
        direct[m == 1, 1] = 1

        m = cv2.filter2D(mask3, -1, d_filter3)
        m[m > 0] = 1
        m = m - mask3
        direct[m == 1, 2] = 1

        m = cv2.filter2D(mask3, -1, d_filter4)
        m[m > 0] = 1
        m = m - mask3
        direct[m == 1, 3] = 1

        mask3 = mask3_

    abs_pos = pos.copy()
    rel_pos = pos / (str_size / 2)  # to 0~1 maybe larger than 1
    rel_pos = (rel_pos * pos_num).astype(np.int32)
    rel_pos = np.clip(rel_pos, 0, pos_num - 1)

    if ori_w != w or ori_h != h:
        rel_pos = cv2.resize(rel_pos, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)
        rel_pos[ori_mask == 0] = 0
        direct = cv2.resize(direct, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)
        direct[ori_mask == 0, :] = 0

    return rel_pos, abs_pos, direct


def resize(img, height, width, center_crop=False):
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


def to_tensor(img, norm=False):
    # img = Image.fromarray(img)
    img_t = FF.to_tensor(img).float()
    if norm:
        img_t = FF.normalize(img_t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    return img_t


def load_image(img_path, mask_path, sigma256=3.0):
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    input_size = min(h, w)
    img = img[:, :, ::-1]
    img = resize(img, input_size, input_size, center_crop=True)
    imgh, imgw = img.shape[0:2]
    img_256 = resize(img, 256, 256)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (imgw, imgh), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 127).astype(np.uint8) * 255
    mask_256 = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_AREA)
    mask_256[mask_256 > 0] = 255
    mask_512 = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_AREA)
    mask_512[mask_512 > 0] = 255

    gray_256 = rgb2gray(img_256)
    edge_256 = canny(gray_256, sigma=sigma256, mask=None).astype(np.float)

    # line
    img_512 = resize(img, 512, 512)

    rel_pos, abs_pos, direct = load_masked_position_encoding(mask)

    batch = dict()
    batch['image'] = to_tensor(img.copy()).unsqueeze(0)
    batch['img_256'] = to_tensor(img_256, norm=True).unsqueeze(0)
    batch['mask'] = to_tensor(mask).unsqueeze(0)
    batch['mask_256'] = to_tensor(mask_256).unsqueeze(0)
    batch['mask_512'] = to_tensor(mask_512).unsqueeze(0)
    batch['edge_256'] = to_tensor(edge_256).unsqueeze(0)
    batch['img_512'] = to_tensor(img_512).unsqueeze(0)
    batch['rel_pos'] = torch.LongTensor(rel_pos).unsqueeze(0)
    batch['abs_pos'] = torch.LongTensor(abs_pos).unsqueeze(0)
    batch['direct'] = torch.LongTensor(direct).unsqueeze(0)
    batch['h'] = imgh
    batch['w'] = imgw

    return batch


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


def wf_inference_test(wf, images, h, w, masks, obj_remove=False, valid_th=0.925, mask_th=0.925):
    lcnn_mean = torch.tensor([109.730, 103.832, 98.681]).to(0).reshape(1, 3, 1, 1)
    lcnn_std = torch.tensor([22.275, 22.124, 23.229]).to(0).reshape(1, 3, 1, 1)
    with torch.no_grad():
        images = images * 255.
        origin_masks = masks
        masks = F.interpolate(masks, size=(images.shape[2], images.shape[3]), mode='nearest')
        # the mask value of lcnn is 127.5
        masked_images = images * (1 - masks) + torch.ones_like(images) * masks * 127.5
        images = (images - lcnn_mean) / lcnn_std
        masked_images = (masked_images - lcnn_mean) / lcnn_std

        def to_int(x):
            return tuple(map(int, x))

        lines_tensor = []
        target_mask = origin_masks.cpu().numpy()  # origin_masks, masks size不同
        for i in range(images.shape[0]):
            lmap = np.zeros((h, w))

            output_nomask = wf(images[i].unsqueeze(0))
            output_nomask = to_device(output_nomask, 'cpu')
            if output_nomask['num_proposals'] == 0:
                lines_nomask = []
                scores_nomask = []
            else:
                lines_nomask = output_nomask['lines_pred'].numpy()
                lines_nomask = [[line[1] * h, line[0] * w, line[3] * h, line[2] * w]
                                for line in lines_nomask]
                scores_nomask = output_nomask['lines_score'].numpy()

            output_masked = wf(masked_images[i].unsqueeze(0))
            output_masked = to_device(output_masked, 'cpu')
            if output_masked['num_proposals'] == 0:
                lines_masked = []
                scores_masked = []
            else:
                lines_masked = output_masked['lines_pred'].numpy()
                lines_masked = [[line[1] * h, line[0] * w, line[3] * h, line[2] * w]
                                for line in lines_masked]
                scores_masked = output_masked['lines_score'].numpy()

            target_mask_ = target_mask[i, 0]
            if obj_remove:
                for line, score in zip(lines_nomask, scores_nomask):
                    line = np.clip(line, 0, 255)
                    if score > valid_th and (
                            target_mask_[to_int(line[0:2])] == 0 or target_mask_[to_int(line[2:4])] == 0):
                        rr, cc, value = skimage.draw.line_aa(*to_int(line[0:2]), *to_int(line[2:4]))
                        lmap[rr, cc] = np.maximum(lmap[rr, cc], value)
                for line, score in zip(lines_masked, scores_masked):
                    line = np.clip(line, 0, 255)
                    if score > mask_th and target_mask_[to_int(line[0:2])] == 1 and target_mask_[
                        to_int(line[2:4])] == 1:
                        rr, cc, value = skimage.draw.line_aa(*to_int(line[0:2]), *to_int(line[2:4]))
                        lmap[rr, cc] = np.maximum(lmap[rr, cc], value)
            else:
                for line, score in zip(lines_masked, scores_masked):
                    if score > mask_th:
                        rr, cc, value = skimage.draw.line_aa(*to_int(line[0:2]), *to_int(line[2:4]))
                        lmap[rr, cc] = np.maximum(lmap[rr, cc], value)

            lmap = np.clip(lmap * 255, 0, 255).astype(np.uint8)
            lines_tensor.append(to_tensor(lmap).unsqueeze(0))

        lines_tensor = torch.cat(lines_tensor, dim=0)
    return lines_tensor.detach().to(0)


def test(model, wf, img_path, mask_path, save_path, valid_th, sigma256=3.0):
    items = load_image(img_path, mask_path, sigma256)
    input_size = min(items['h'], items['w'])
    line = wf_inference_test(wf, items['img_512'].cuda(), h=256, w=256, masks=items['mask_512'].cuda(),
                             valid_th=valid_th, mask_th=valid_th)
    items['line_256'] = line

    with torch.no_grad():
        for k in items:
            if type(items[k]) is torch.Tensor:
                items[k] = items[k].to(0)
        edge_pred, line_pred = SampleEdgeLineLogits(model.inpaint_model.transformer,
                                                    context=[items['img_256'], items['edge_256'], items['line_256']],
                                                    mask=items['mask_256'].clone(), iterations=5,
                                                    add_v=0.05, mul_v=4, device=0)
        edge_pred, line_pred = edge_pred.detach().to(torch.float32), line_pred.detach().to(torch.float32)
        if input_size != 256 and input_size > 256:
            while edge_pred.shape[2] < input_size:
                edge_pred = model.inpaint_model.structure_upsample(edge_pred)[0]
                edge_pred = torch.sigmoid((edge_pred + 2) * 2)

                line_pred = model.inpaint_model.structure_upsample(line_pred)[0]
                line_pred = torch.sigmoid((line_pred + 2) * 2)

            edge_pred = F.interpolate(edge_pred, size=(input_size, input_size), mode='bilinear', align_corners=False)
            line_pred = F.interpolate(line_pred, size=(input_size, input_size), mode='bilinear', align_corners=False)
        elif input_size < 256:
            print('input size must >= 256!')
            raise NotImplementedError

        items['edge'] = edge_pred.detach()
        items['line'] = line_pred.detach()
        # inpaint model
        items = model.inpaint_model(items)
        outputs_merged = (items['predicted_image'] * items['mask']) + (items['image'] * (1 - items['mask']))

    image_per_row = 1
    images = stitch_images(
        model.postprocess((outputs_merged).cpu()),
        img_per_row=image_per_row
    )
    print('\nsaving sample ' + os.path.basename(img_path))
    images.save(save_path + '/' + os.path.basename(img_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '--checkpoints', type=str, default=None,
                        help='model checkpoints path (default: ./checkpoints)')
    parser.add_argument('--config_file', type=str, default=None,
                        help='The config file of each experiment ')
    parser.add_argument('--GPU_ids', type=str, default='0')
    parser.add_argument('--img_path', type=str, help='test image path')
    parser.add_argument('--mask_path', type=str, help='test mask path')
    parser.add_argument('--save_path', type=str, help='the path to save the results')

    args = parser.parse_args()
    config_path = os.path.join(args.path, 'config.yml')

    # create checkpoints path if does't exist
    os.makedirs(args.path, exist_ok=True)

    # copy config template if does't exist
    if not os.path.exists(config_path):
        copyfile(args.config_file, config_path)  ## Training, always copy

    args.config_path = config_path

    os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU_ids
    args.world_size = 1

    torch.cuda.set_device(0)

    # load config file
    config = Config(args.config_path)
    config.MODE = 1
    config.gpus = 1
    config.GPU_ids = args.GPU_ids
    config.world_size = 1

    torch.backends.cudnn.benchmark = True  # cudnn auto-tuner
    cv2.setNumThreads(0)

    # initialize random seed
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    # build the model and load the best model for eval
    model = ZITS(config, 0, 0, True, True)
    model.inpaint_model.eval()

    # load hawp
    print("load HAWP")
    wf = WireframeDetector(is_cuda=True)
    wf = wf.to(0)
    wf.load_state_dict(torch.load('./ckpt/best_lsm_hawp.pth', map_location='cpu')['model'])
    wf.eval()

    test(model, wf, args.img_path, args.mask_path, args.save_path, 0.85, sigma256=3.0)
