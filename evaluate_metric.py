import torch
import torch.nn as nn
import numpy as np
from utils.SE3 import decompose_trans, transform
from benchmark_utils_predator import computeTransformationErr


class TransformationLoss(nn.Module):
    def __init__(self, re_thre=15, te_thre=30):
        super().__init__()
        self.re_thre = re_thre
        self.te_thre = te_thre

    def forward(self, trans, gt_trans, src_keypts, tgt_keypts, gt_info=None):
        bs = trans.shape[0]
        R, t = decompose_trans(trans)
        gt_R, gt_t = decompose_trans(gt_trans)

        recall = 0
        re_total = torch.tensor(0.0, device=trans.device)
        te_total = torch.tensor(0.0, device=trans.device)
        rmse_total = torch.tensor(0.0, device=trans.device)

        for i in range(bs):
            re = torch.acos(torch.clamp((torch.trace(R[i].T @ gt_R[i]) - 1) / 2.0, min=-1, max=1))
            te = torch.sqrt(torch.sum((t[i] - gt_t[i]) ** 2))
            warped_src = transform(src_keypts[i], trans[i])
            rmse = torch.norm(warped_src - tgt_keypts[i], dim=-1).mean()
            re = re * 180 / np.pi
            te_cm = te * 100

            if gt_info is not None:
                pred = trans[i].detach().cpu().numpy()
                gt = gt_trans[i].detach().cpu().numpy()
                info = gt_info.detach().cpu().numpy() if torch.is_tensor(gt_info) else gt_info
                success = computeTransformationErr(np.linalg.inv(gt) @ pred, info) <= 0.04
            else:
                success = te_cm < self.te_thre and re < self.re_thre

            if success:
                recall += 1
            re_total += re
            te_total += te_cm
            rmse_total += rmse

        return recall * 100.0 / bs, re_total / bs, te_total / bs, rmse_total / bs


class ClassificationLoss(nn.Module):
    def __init__(self, inlier_threshold=0.10):
        super().__init__()
        self.inlier_threshold = inlier_threshold

    def forward(self, gt_trans, src_corr, tgt_corr, src_corr_input, tgt_corr_input):
        output_dist = torch.norm(transform(src_corr, gt_trans) - tgt_corr, dim=-1)
        input_dist = torch.norm(transform(src_corr_input, gt_trans) - tgt_corr_input, dim=-1)

        output_inliers = output_dist < self.inlier_threshold
        input_inliers = input_dist < self.inlier_threshold

        output_num = output_inliers.sum().float()
        output_total = torch.tensor(output_inliers.numel(), device=output_dist.device, dtype=torch.float32)
        input_num = input_inliers.sum().float()
        input_total = torch.tensor(input_inliers.numel(), device=input_dist.device, dtype=torch.float32)

        precision = output_num / torch.clamp(output_total, min=1.0)
        recall = output_num / torch.clamp(input_num, min=1.0)
        f1 = 2 * precision * recall / torch.clamp(precision + recall, min=1e-6)

        feature_match_recall = (input_num > 0).float()
        feature_match_recall_01 = ((input_num / torch.clamp(input_total, min=1.0)) > 0.1).float()
        feature_match_recall_001 = ((input_num / torch.clamp(input_total, min=1.0)) > 0.01).float()

        return {
            "output_inlier_number": output_num.item(),
            "precision": precision.item(),
            "recall": recall.item(),
            "f1": f1.item(),
            "feature_match_recall": feature_match_recall.item(),
            "feature_match_recall_0.1": feature_match_recall_01.item(),
            "feature_match_recall_0.01": feature_match_recall_001.item(),
            "IR_ratio": precision.item(),
            "INR": (output_num / torch.clamp(input_num, min=1.0)).item(),
            "NR": (output_total / torch.clamp(input_total, min=1.0)).item(),
            "inlier_nums": output_num.item(),
        }
