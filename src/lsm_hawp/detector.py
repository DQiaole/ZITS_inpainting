import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .model_config import get_config
from .multi_task_head import MultitaskHead
from .stacked_hg import HourglassNet, Bottleneck2D


def build_hg(cfg):
    inplanes = cfg.MODEL.HGNETS.INPLANES
    num_feats = cfg.MODEL.OUT_FEATURE_CHANNELS // 2
    depth = cfg.MODEL.HGNETS.DEPTH
    num_stacks = cfg.MODEL.HGNETS.NUM_STACKS
    num_blocks = cfg.MODEL.HGNETS.NUM_BLOCKS
    head_size = cfg.MODEL.HEAD_SIZE

    out_feature_channels = cfg.MODEL.OUT_FEATURE_CHANNELS
    input_channel = 3

    num_class = sum(sum(head_size, []))
    model = HourglassNet(
        input_channel=input_channel,
        block=Bottleneck2D,
        inplanes=inplanes,
        num_feats=num_feats,
        depth=depth,
        head=lambda c_in, c_out: MultitaskHead(c_in, c_out, head_size=head_size),
        num_stacks=num_stacks,
        num_blocks=num_blocks,
        num_classes=num_class)

    model.out_feature_channels = out_feature_channels

    return model


def non_maximum_suppression(a):
    ap = F.max_pool2d(a, 3, stride=1, padding=1)
    mask = (a == ap).float().clamp(min=0.0)
    return a * mask


def get_junctions(jloc, joff, topk=300, th=0):
    height, width = jloc.size(1), jloc.size(2)
    jloc = jloc.reshape(-1)
    joff = joff.reshape(2, -1)

    scores, index = torch.topk(jloc, k=topk)
    y = (index // width).float() + torch.gather(joff[1], 0, index) + 0.5
    x = (index % width).float() + torch.gather(joff[0], 0, index) + 0.5

    junctions = torch.stack((x, y)).t()

    return junctions[scores > th], scores[scores > th]


class WireframeDetector(nn.Module):
    def __init__(self, is_cuda=True):
        super(WireframeDetector, self).__init__()
        cfg = get_config()
        self.backbone = build_hg(cfg)
        self.cfg = cfg

        self.n_dyn_junc = cfg.MODEL.PARSING_HEAD.N_DYN_JUNC
        self.n_dyn_posl = cfg.MODEL.PARSING_HEAD.N_DYN_POSL
        self.n_dyn_negl = cfg.MODEL.PARSING_HEAD.N_DYN_NEGL
        self.n_dyn_othr = cfg.MODEL.PARSING_HEAD.N_DYN_OTHR
        self.n_dyn_othr2 = cfg.MODEL.PARSING_HEAD.N_DYN_OTHR2
        self.n_pts0 = cfg.MODEL.PARSING_HEAD.N_PTS0
        self.n_pts1 = cfg.MODEL.PARSING_HEAD.N_PTS1
        self.dim_loi = cfg.MODEL.PARSING_HEAD.DIM_LOI
        self.dim_fc = cfg.MODEL.PARSING_HEAD.DIM_FC
        self.n_out_junc = cfg.MODEL.PARSING_HEAD.N_OUT_JUNC
        self.n_out_line = cfg.MODEL.PARSING_HEAD.N_OUT_LINE
        self.use_residual = cfg.MODEL.PARSING_HEAD.USE_RESIDUAL
        # self.
        if is_cuda:
            self.register_buffer('tspan', torch.linspace(0, 1, self.n_pts0)[None, None, :].cuda())
        else:
            self.register_buffer('tspan', torch.linspace(0, 1, self.n_pts0)[None, None, :])
        self.loss = nn.BCEWithLogitsLoss(reduction='none')

        self.fc1 = nn.Conv2d(256, self.dim_loi, 1)
        self.pool1d = nn.MaxPool1d(self.n_pts0 // self.n_pts1, self.n_pts0 // self.n_pts1)
        self.fc2 = nn.Sequential(
            nn.Linear(self.dim_loi * self.n_pts1, self.dim_fc),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim_fc, self.dim_fc),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim_fc, 1),
        )
        self.train_step = 0

    def pooling(self, features_per_image, lines_per_im):
        h, w = features_per_image.size(1), features_per_image.size(2)
        U, V = lines_per_im[:, :2], lines_per_im[:, 2:]
        sampled_points = U[:, :, None] * self.tspan + V[:, :, None] * (1 - self.tspan) - 0.5
        sampled_points = sampled_points.permute((0, 2, 1)).reshape(-1, 2)
        px, py = sampled_points[:, 0], sampled_points[:, 1]
        px0 = px.floor().clamp(min=0, max=w - 1)
        py0 = py.floor().clamp(min=0, max=h - 1)
        px1 = (px0 + 1).clamp(min=0, max=w - 1)
        py1 = (py0 + 1).clamp(min=0, max=h - 1)
        px0l, py0l, px1l, py1l = px0.long(), py0.long(), px1.long(), py1.long()

        xp = ((features_per_image[:, py0l, px0l] * (py1 - py) * (px1 - px) + features_per_image[:, py1l, px0l] * (
                py - py0) * (px1 - px) + features_per_image[:, py0l, px1l] * (py1 - py) * (
                       px - px0) + features_per_image[:, py1l, px1l] * (py - py0) * (px - px0)).reshape(128, -1, 32)
              ).permute(1, 0, 2)

        # if self.pool1d is not None:
        xp = self.pool1d(xp)
        features_per_line = xp.view(-1, self.n_pts1 * self.dim_loi)
        logits = self.fc2(features_per_line).flatten()
        return logits

    def forward(self, images):
        return self.forward_test(images)

    def forward_test(self, images):
        
        try:
            outputs, features = self.backbone(images)

            loi_features = self.fc1(features)
            output = outputs[0]
            md_pred = output[:, :3].sigmoid()
            dis_pred = output[:, 3:4].sigmoid()
            res_pred = output[:, 4:5].sigmoid()
            jloc_pred = output[:, 5:7].softmax(1)[:, 1:]
            joff_pred = output[:, 7:9].sigmoid() - 0.5

            batch_size = md_pred.size(0)
            assert batch_size == 1

            if self.use_residual:
                lines_pred = self.proposal_lines_new(md_pred[0], dis_pred[0], res_pred[0]).view(-1, 4)
            else:
                lines_pred = self.proposal_lines_new(md_pred[0], dis_pred[0], None).view(-1, 4)

            jloc_pred_nms = non_maximum_suppression(jloc_pred[0])
            topK = min(300, int((jloc_pred_nms > 0.008).float().sum().item()))

            juncs_pred, _ = get_junctions(non_maximum_suppression(jloc_pred[0]), joff_pred[0], topk=topK)
            dis_junc_to_end1, idx_junc_to_end1 = torch.sum((lines_pred[:, :2] - juncs_pred[:, None]) ** 2, dim=-1).min(0)
            dis_junc_to_end2, idx_junc_to_end2 = torch.sum((lines_pred[:, 2:] - juncs_pred[:, None]) ** 2, dim=-1).min(0)

            idx_junc_to_end_min = torch.min(idx_junc_to_end1, idx_junc_to_end2)
            idx_junc_to_end_max = torch.max(idx_junc_to_end1, idx_junc_to_end2)

            iskeep = (idx_junc_to_end_min < idx_junc_to_end_max)
            # idx_lines_for_junctions: [n_line', 2] 为筛选后的line起点终点idx
            idx_lines_for_junctions = torch.unique(torch.cat((idx_junc_to_end_min[iskeep, None],
                                                            idx_junc_to_end_max[iskeep, None]), dim=1), dim=0)
            # [n_line', 4]
            lines_adjusted = torch.cat((juncs_pred[idx_lines_for_junctions[:, 0]],
                                        juncs_pred[idx_lines_for_junctions[:, 1]]), dim=1)
            scores = self.pooling(loi_features[0], lines_adjusted).sigmoid()

            lines_final = lines_adjusted[scores > 0.05]
            score_final = scores[scores > 0.05]

            juncs_final = juncs_pred[idx_lines_for_junctions.unique()]
            juncs_score = _[idx_lines_for_junctions.unique()]

            sx = 1 / output.size(3)
            sy = 1 / output.size(2)

            lines_final[:, 0] *= sx
            lines_final[:, 1] *= sy
            lines_final[:, 2] *= sx
            lines_final[:, 3] *= sy

            juncs_final[:, 0] *= sx
            juncs_final[:, 1] *= sy

            output = {
                'lines_pred': lines_final,
                'lines_score': score_final,
                'juncs_pred': juncs_final,
                'juncs_score': juncs_score,
                'num_proposals': lines_adjusted.size(0)
            }

        except Exception:
            output = {
                'lines_pred': [],
                'lines_score': [],
                'juncs_pred': [],
                'juncs_score': [],
                'num_proposals': 0
            }

        return output

    def proposal_lines_new(self, md_maps, dis_maps, residual_maps, scale=5.0):
        """

        :param md_maps: 3xhxw, the range should be (0,1) for every element
        :param dis_maps: 1xhxw
        :return:
        """
        device = md_maps.device
        sign_pad = torch.tensor([-1, 0, 1], device=device, dtype=torch.float32).reshape(3, 1, 1)

        if residual_maps is None:
            dis_maps_new = dis_maps.repeat((1, 1, 1))
        else:
            dis_maps_new = dis_maps.repeat((3, 1, 1)) + sign_pad * residual_maps.repeat((3, 1, 1))
        height, width = md_maps.size(1), md_maps.size(2)
        _y = torch.arange(0, height, device=device).float()
        _x = torch.arange(0, width, device=device).float()

        y0, x0 = torch.meshgrid(_y, _x)
        md_ = (md_maps[0] - 0.5) * np.pi * 2
        st_ = md_maps[1] * np.pi / 2
        ed_ = -md_maps[2] * np.pi / 2

        cs_md = torch.cos(md_)
        ss_md = torch.sin(md_)

        cs_st = torch.cos(st_).clamp(min=1e-3)
        ss_st = torch.sin(st_).clamp(min=1e-3)

        cs_ed = torch.cos(ed_).clamp(min=1e-3)
        ss_ed = torch.sin(ed_).clamp(max=-1e-3)

        y_st = ss_st / cs_st
        y_ed = ss_ed / cs_ed

        x_st_rotated = (cs_md - ss_md * y_st)[None] * dis_maps_new * scale
        y_st_rotated = (ss_md + cs_md * y_st)[None] * dis_maps_new * scale

        x_ed_rotated = (cs_md - ss_md * y_ed)[None] * dis_maps_new * scale
        y_ed_rotated = (ss_md + cs_md * y_ed)[None] * dis_maps_new * scale

        x_st_final = (x_st_rotated + x0[None]).clamp(min=0, max=width - 1)
        y_st_final = (y_st_rotated + y0[None]).clamp(min=0, max=height - 1)

        x_ed_final = (x_ed_rotated + x0[None]).clamp(min=0, max=width - 1)
        y_ed_final = (y_ed_rotated + y0[None]).clamp(min=0, max=height - 1)

        lines = torch.stack((x_st_final, y_st_final, x_ed_final, y_ed_final)).permute((1, 2, 3, 0))

        return lines
