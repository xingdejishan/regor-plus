import torch
from common import knn, rigid_transform_3d
from utils.SE3 import transform
import numpy as np
import open3d
import copy
import math
import optimal_transport
import visualization
from utils.timer import Timer



class Regenerator():
    def __init__(self,
                 inlier_threshold=0.10,
                 num_node='all',
                 use_mutual=True,
                 d_thre=0.1,
                 num_iterations=10,
                 ratio=0.2,
                 nms_radius=0.1,
                 max_points=8000,
                 k1=30,
                 k2=20,
                 prior_lambda=0.1,
                 eta_l=0.5,
                 prior_eps=1e-4,
                 select_scene=None,
                 ):
        self.use_sampling = False
        self.inlier_threshold = inlier_threshold
        self.num_node = num_node
        self.use_mutual = use_mutual
        self.d_thre = d_thre
        self.num_iterations = num_iterations  # maximum iteration of power iteration algorithm
        self.ratio = ratio # the maximum ratio of seeds.
        self.max_points = max_points
        self.nms_radius = nms_radius
        self.k1 = k1
        self.k2 = k2
        self.prior_lambda = prior_lambda
        self.eta_l = eta_l
        self.prior_eps = prior_eps
        self.sampling_num = 0
        self.last_r2_candidates = []
        self.DEBUG = False

    def knn_search(self, x, y, k, ignore_self=False, normalized=True):
        """ find feature space knn neighbor of x
        Input:
            - x:       [bs, num_corr, num_channels],  input  source features
            - y:       [bs, num_corr, num_channels],  input features for knn
            - k:
            - ignore_self:  True/False, return knn include self or not.
            - normalized:   True/False, if the feature x normalized.
        Output:
            - idx:     [bs, num_corr, k], the indices of knn neighbors
        """
        bs, num_corr, num_channels = x.shape
        num_corr_knn = y.shape[1]
        x_expand = x[:, :, None, :].expand(bs, num_corr, num_corr_knn, num_channels)
        y_expand = y[:, None, :, :].expand(bs, num_corr, num_corr_knn, num_channels)
        x_y = x_expand - y_expand
        pairwise_distance = torch.sqrt(torch.sum(x_y ** 2, dim=3))


        if ignore_self is False:
            idx = pairwise_distance.topk(k=k, dim=-1, largest=False)[1]  # (batch_size, num_points, k)
        else:
            idx = pairwise_distance.topk(k=k + 1, dim=-1, largest=False)[1][:, :, 1:]
        return idx

    def radius_search(self, x, y, k, r):
        """ find feature space knn neighbor of x
        Input:
            - x:       [bs, num_corr, num_channels],  input  source features
            - y:       [bs, num_corr, num_channels],  input features for knn
            - k:
            - ignore_self:  True/False, return knn include self or not.
            - normalized:   True/False, if the feature x normalized.
        Output:
            - idx:     [bs, num_corr, k], the indices of knn neighbors
        """
        bs, num_corr_key, num_channels = x.shape
        num_corr = y.shape[1]

        distance = torch.norm((x[:, :, None, :] - y[:, None, :, :]), dim=-1)
        radius_mask = distance < r

        select_num = k
        indices_total = torch.zeros(num_corr_key, select_num, dtype=torch.int64).cuda()
        for i in range(num_corr_key):
            indices = torch.where(radius_mask[0, i, :])[0]
            print(indices.shape[0],select_num)
            if indices.shape[0] > select_num:
                sel_ind = np.random.choice(indices.shape[0], select_num)
                indices = indices[sel_ind]
            elif indices.shape[0] < select_num:
                sel_ind = np.random.choice(indices.shape[0], select_num-indices.shape[0])
                indices_pad = indices[sel_ind]
                indices = torch.cat([indices, indices_pad], dim=0)
            indices_total[i, :] = indices
        return indices_total.unsqueeze(0)

    def radius_search2(self, src_key, src_pts, tgt_key, tgt_pts, k):
        """ find feature space knn neighbor of x
        Input:
            - x:       [bs, num_corr, num_channels],  input  source features
            - y:       [bs, num_corr, num_channels],  input features for knn
            - k:
            - ignore_self:  True/False, return knn include self or not.
            - normalized:   True/False, if the feature x normalized.
        Output:
            - idx:     [bs, num_corr, k], the indices of knn neighbors
        """
        bs, num_corr_key, num_channels = src_key.shape
        num_corr = src_pts.shape[1]

        distance_src = torch.norm((src_key[:, :, None, :] - src_pts[:, None, :, :]), dim=-1)
        radius_mask_src = distance_src < 0.3
        distance_tgt = torch.norm((tgt_key[:, :, None, :] - tgt_pts[:, None, :, :]), dim=-1)
        radius_mask_tgt = distance_tgt < 0.3

        select_num = k
        # indices_total = torch.zeros(num_corr_key, select_num, dtype=torch.int64).cuda()
        indices_set_src = []
        indices_set_tgt = []
        for i in range(num_corr_key):
            indices_src = torch.where(radius_mask_src[0, i, :])[0]
            indices_tgt = torch.where(radius_mask_tgt[0, i, :])[0]
            if indices_src.shape[0] >= select_num and indices_tgt.shape[0] >= select_num:
                indices_src = indices_src[:select_num]
                indices_set_src.append(indices_src)
                indices_tgt = indices_tgt[:select_num]
                indices_set_tgt.append(indices_tgt)
            # elif indices.shape[0] < select_num:
            #     indices = torch.cat([indices, torch.ones(select_num-indices.shape[0], dtype=torch.int64).cuda()*indices[0]], dim=0)

        indices_total_src = torch.stack(indices_set_src)
        indices_total_tgt = torch.stack(indices_set_tgt)

        return indices_total_src.unsqueeze(0), indices_total_tgt.unsqueeze(0)

    def local_matching(self, src_key_corr, tgt_key_corr, src_corr_knn_idx, tgt_corr_knn_idx, src_point, tgt_point, src_feature, tgt_feature, guide=None, return_indices=False):
        # - src_key_corr: [bs, num_key_corr, 3]
        # - tgt_key_corr: [bs, num_key_corr, 3]

        bs, num_seeds, num_knn = src_corr_knn_idx.shape   # [bs, num_seeds,knn_num]
        src_feature_knn = self.idx_selection(src_corr_knn_idx, src_feature)[0]  # [num_seeds,knn_num,num_channel]
        tgt_feature_knn = self.idx_selection(tgt_corr_knn_idx, tgt_feature)[0]
        src_point_knn = self.idx_selection(src_corr_knn_idx, src_point)[0]
        tgt_point_knn = self.idx_selection(tgt_corr_knn_idx, tgt_point)[0]  # [num_seeds,knn_num,3]

        ot_timer = Timer()
        ot_timer.tic()

        ''' GMM matching '''
        match_mask = self.match_pair_GMM(
            src_feature_knn,
            tgt_feature_knn,
            src_point_knn,
            tgt_point_knn,
            src_corr_knn_idx[0],
            tgt_corr_knn_idx[0],
            guide=guide
        )
        bs_idx, src_idx, tgt_idx = torch.where(match_mask)
        src_local_corr_idx = src_corr_knn_idx[:, bs_idx, src_idx]
        tgt_local_corr_idx = tgt_corr_knn_idx[:, bs_idx, tgt_idx]
        local_corr_idx = torch.cat([src_local_corr_idx.unsqueeze(-1), tgt_local_corr_idx.unsqueeze(-1)], dim=-1)
        correspondences = local_corr_idx[0]
        unique_selected_corr_idx = torch.unique(correspondences.view(-1, 2), dim=0)
        unique_selected_src_corr_idx = unique_selected_corr_idx[:, 0]
        unique_selected_tgt_corr_idx = unique_selected_corr_idx[:, 1]
        src_point_corr_mm = src_point.gather(dim=1, index=unique_selected_src_corr_idx[None, :, None].expand(-1, -1, 3)).view([1, -1, 3])
        tgt_point_corr_mm = tgt_point.gather(dim=1, index=unique_selected_tgt_corr_idx[None, :, None].expand(-1, -1, 3)).view([1, -1, 3])

        correspondences = torch.zeros((num_seeds, num_knn*2, 2), dtype=torch.int64, device=src_point.device)
        correspondences_true_mask = torch.zeros((num_seeds, num_knn*2), dtype=torch.bool, device=src_point.device)
        for i in range(num_seeds):
            src_idx, tgt_idx = torch.where(match_mask[i])
            num = src_idx.shape[0]
            src_local_corr_idx = src_corr_knn_idx[0, i, src_idx]
            tgt_local_corr_idx = tgt_corr_knn_idx[0, i, tgt_idx]
            correspondences[i, :num, :] = torch.cat([src_local_corr_idx[:, None], tgt_local_corr_idx[:, None]], dim=-1)
            correspondences_true_mask[i, :num] = True

        src_correspondences_expand = correspondences[:, :, 0].contiguous().view([1, -1])
        src_correspondences_expand = src_correspondences_expand[:, :, None]
        src_correspondences_expand = src_correspondences_expand.expand(-1, -1, 3)
        tgt_correspondences_expand = correspondences[:, :, 1].contiguous().view([1, -1])
        tgt_correspondences_expand = tgt_correspondences_expand[:, :, None]
        tgt_correspondences_expand = tgt_correspondences_expand.expand(-1, -1, 3)

        src_point_corr1 = src_point.gather(dim=1, index=src_correspondences_expand).view([1, num_seeds, -1, 3])
        tgt_point_corr1 = tgt_point.gather(dim=1, index=tgt_correspondences_expand).view([1, num_seeds, -1, 3])

        ''' local consistency '''
        corr_mask, key_mask = self.center_based_three_points_consistency3(src_key_corr, tgt_key_corr, src_point_corr1, tgt_point_corr1, correspondences_true_mask[None])
        corr_mask = corr_mask*correspondences_true_mask[None]
        _, seed_idx, corr_idx = torch.where(corr_mask)
        correspondences_1 = correspondences[seed_idx, corr_idx, :]
        unique_selected_corr_idx = torch.unique(correspondences_1, dim=0)

        unique_selected_src_corr_idx = unique_selected_corr_idx[:, 0]
        unique_selected_tgt_corr_idx = unique_selected_corr_idx[:, 1]

        ''' global consistency '''
        src_point_corr1 = src_point.gather(dim=1, index=unique_selected_src_corr_idx[None, :, None].expand(-1, -1, 3)).view([1, -1, 3])
        tgt_point_corr1 = tgt_point.gather(dim=1, index=unique_selected_tgt_corr_idx[None, :, None].expand(-1, -1, 3)).view([1, -1, 3])

        corr_idx = self.global_spatial_fitering_topk_two(src_point_corr1, tgt_point_corr1)
        src_point_corr2 = src_point_corr1[:, corr_idx, :]
        tgt_point_corr2 = tgt_point_corr1[:, corr_idx, :]
        selected_corr_indices = unique_selected_corr_idx[corr_idx]
        if src_point_corr2.shape[1]<10:
            src_point_corr2 = src_point_corr1
            tgt_point_corr2 = tgt_point_corr1
            selected_corr_indices = unique_selected_corr_idx


        if self.DEBUG:
            src = open3d.geometry.PointCloud()
            src.points = open3d.utility.Vector3dVector(src_point.view(-1, 3).cpu().numpy())
            tgt = open3d.geometry.PointCloud()
            tgt.points = open3d.utility.Vector3dVector(tgt_point.view(-1, 3).cpu().numpy())

            src_key0 = open3d.geometry.PointCloud()
            src_key0.points = open3d.utility.Vector3dVector(src_point_knn.view(-1, 3).cpu().numpy())
            tgt_key0 = open3d.geometry.PointCloud()
            tgt_key0.points = open3d.utility.Vector3dVector(tgt_point_knn.view(-1, 3).cpu().numpy())


            src_key = open3d.geometry.PointCloud()
            src_key.points = open3d.utility.Vector3dVector(src_point_corr_mm.view(-1, 3).cpu().numpy())
            tgt_key = open3d.geometry.PointCloud()
            tgt_key.points = open3d.utility.Vector3dVector(tgt_point_corr_mm.view(-1, 3).cpu().numpy())

            src_key1 = open3d.geometry.PointCloud()
            src_key1.points = open3d.utility.Vector3dVector(src_point_corr1.view(-1, 3).cpu().numpy())
            tgt_key1 = open3d.geometry.PointCloud()
            tgt_key1.points = open3d.utility.Vector3dVector(tgt_point_corr1.view(-1, 3).cpu().numpy())

            src_key2 = open3d.geometry.PointCloud()
            src_key2.points = open3d.utility.Vector3dVector(src_point_corr2.view(-1, 3).cpu().numpy())
            tgt_key2 = open3d.geometry.PointCloud()
            tgt_key2.points = open3d.utility.Vector3dVector(tgt_point_corr2.view(-1, 3).cpu().numpy())

            visualization.draw_registration_corr2(src, tgt, src_key0, tgt_key0, np.eye(4))
            visualization.draw_registration_corr2(src, tgt, src_key, tgt_key, np.eye(4))
            visualization.draw_registration_corr2(src, tgt, src_key1, tgt_key1, np.eye(4))
            visualization.draw_registration_corr2(src, tgt, src_key2, tgt_key2, np.eye(4))

        if return_indices:
            return src_point_corr2, tgt_point_corr2, selected_corr_indices
        return src_point_corr2, tgt_point_corr2


    def center_based_three_points_consistency3(self, src_key_corr, tgt_key_corr, src_knn_corr, tgt_knn_corr, match_mask):
        """
        input:
            - src_key_corr: key points [bs,key_num,3]
            - tgt_key_corr: key points [bs,key_num,3]
            - src_knn_corr: knn points [bs,key_num, knn, 3]
            - tgt_knn_corr: knn points [bs,key_num, knn, 3]
        return:
            - selected_points: [bs, num_corr, num_knn, num_channel]
        """
        # 速度更快，也有全局信息
        # [bs, corr_num, 3]
        bs, key_num, knn_num, _ = src_knn_corr.shape
        diff_src_key = src_key_corr[:, :, None, :] - src_knn_corr  # [bs, key_num, knn, 3]
        diff_tgt_key = tgt_key_corr[:, :, None, :] - tgt_knn_corr  # [bs, key_num, knn, 3]
        distance_src_key = torch.norm(diff_src_key, dim=-1)  # [bs, key_num, knn]
        distance_tgt_key = torch.norm(diff_tgt_key, dim=-1)  # [bs, key_num, knn]

        cross_dist_key = torch.abs(distance_src_key - distance_tgt_key)
        key2knn_diff_mask = (cross_dist_key < self.d_thre/2).float()  # [bs, key_num, knn]
        # key2knn_diff_two = torch.einsum('abc,abd->abcd', key2knn_diff, key2knn_diff).bool()  # [bs, key_num, knn, knn]

        distance_src = torch.norm((src_knn_corr[:, :, :, None, :] - src_knn_corr[:, :, None, :, :]), dim=-1)  # [bs, key_num, knn]
        distance_tgt = torch.norm((tgt_knn_corr[:, :, :, None, :] - tgt_knn_corr[:, :, None, :, :]), dim=-1)  # [bs, key_num, knn]
        cross_dist = torch.abs(distance_src - distance_tgt)
        # print(cross_dist.shape, match_mask.shape)
        match_mask_two = match_mask[:, :, :, None].expand(-1, -1, -1, knn_num) * match_mask[:, :, None, :].expand(-1, -1, knn_num, -1)
        two_knn_point_diff = (cross_dist < self.d_thre/2) * match_mask_two   # [bs, key_num, knn, knn]
        two_knn_point_diff_sum = two_knn_point_diff.sum(dim=-1)  # [bs, key_num, knn]
        good_match_num = match_mask.sum(-1)  # 这里可以输入数量就不用计算了，可以提速 # [bs, key_num]
        # print(good_match_num.shape, two_knn_point_diff_sum.shape, good_match_num[:, :, :, None].expand(-1, -1, -1, knn_num).shape)
        two_knn_point_diff_mask = two_knn_point_diff_sum >= good_match_num[:, :, None].expand(-1, -1, knn_num) * 0.3

        corr_mask = key2knn_diff_mask.bool() | two_knn_point_diff_mask.bool()

        key_mask_false = torch.sum(corr_mask, dim=2) < good_match_num * 0.5  # [bs, key_num]
        if not torch.all(key_mask_false):
            corr_mask[:, key_mask_false.view(-1), :] = False

        return corr_mask, key_mask_false

    def global_spatial_fitering_topk_two(self, src_knn_corr, tgt_knn_corr):
        # 有效果
        """
        input:
            - src_key_corr: key points [bs,key_num,3]
            - tgt_key_corr: key points [bs,key_num,3]
            - src_knn_corr: knn points [bs,key_num, knn, 3]
            - tgt_knn_corr: knn points [bs,key_num, knn, 3]
        return:
            - selected_points: [bs, num_corr, num_knn, num_channel]
        """
        # print(src_knn_corr.shape, mutual_mask.shape)
        # 速度更快，也有全局信息
        # [bs, corr_num, 3]
        bs, knn_num, _ = src_knn_corr.shape

        distance_src = torch.norm((src_knn_corr[:, :, None, :] - src_knn_corr[:, None, :, :]), dim=-1)
        distance_tgt = torch.norm((tgt_knn_corr[:, :, None, :] - tgt_knn_corr[:, None, :, :]), dim=-1)
        cross_dist = torch.abs(distance_src - distance_tgt)
        local_measure = (cross_dist < 0.1).float()
        SC2_measure = torch.matmul(local_measure, local_measure) * local_measure
        SC2_measure_sum = torch.sum(SC2_measure, dim=-1)
        # _, corr_idx = torch.topk(SC2_measure_sum, largest=True, dim=-1, k=int(knn_num * 0.3))  # 筛选出corr_idx
        _, corr_idx = torch.topk(SC2_measure_sum, largest=True, dim=-1, k=int(knn_num * 0.5))  # 筛选出corr_idx

        src_knn_corr_s = src_knn_corr[:, corr_idx[0], :]
        tgt_knn_corr_s = tgt_knn_corr[:, corr_idx[0], :]

        # src = open3d.geometry.PointCloud()
        # src.points = open3d.utility.Vector3dVector(src_knn_corr_s.view(-1, 3).cpu().numpy())
        # tgt = open3d.geometry.PointCloud()
        # tgt.points = open3d.utility.Vector3dVector(tgt_knn_corr_s.view(-1, 3).cpu().numpy())

        distance_src = torch.norm((src_knn_corr_s[:, :, None, :] - src_knn_corr_s[:, None, :, :]), dim=-1)
        distance_tgt = torch.norm((tgt_knn_corr_s[:, :, None, :] - tgt_knn_corr_s[:, None, :, :]), dim=-1)
        cross_dist = torch.abs(distance_src - distance_tgt)
        local_measure = (cross_dist < 0.1).float()
        SC2_measure = torch.matmul(local_measure, local_measure) * local_measure
        SC2_measure_sum = torch.sum(SC2_measure, dim=-1)
        _, corr_idx_idx = torch.topk(SC2_measure_sum, largest=True, dim=-1, k=int(knn_num * 0.2))  # 筛选出corr_idx
        corr_idx = corr_idx[:, corr_idx_idx[0]]

        return corr_idx[0]

    def match_pair_GMM(self, src_features, tgt_features, src_points=None, tgt_points=None, src_indices=None, tgt_indices=None, guide=None):
        """
        input:
            - src_key_corr: key points [bs,key_num,3]
            - tgt_key_corr: key points [bs,key_num,3]
            - src_knn_corr: knn points [bs,key_num, knn, 3]
            - tgt_knn_corr: knn points [bs,key_num, knn, 3]
        return:
            - selected_points: [bs, num_corr, num_knn, num_channel]
        """

        num_seeds, num_knn, _ = src_features.shape
        num_k=3

        # match points in feature space with batch size. from src->tgt
        distance = torch.sqrt(2 - 2 * torch.matmul(src_features, tgt_features.transpose(1, 2)) + 1e-6)
        if guide is not None:
            distance = self.apply_guided_similarity(distance, src_points, tgt_points, src_indices, tgt_indices, guide)
        _, tgt_idx1 = torch.topk(distance, dim=2, largest=False, k=num_k)
        src_idx1 = torch.arange(num_knn, device=distance.device)[None, :, None].expand(num_seeds,-1, num_k)
        bs_idx1 = torch.arange(num_seeds, device=distance.device)[:, None, None].expand(-1, num_knn, num_k)

        mask_2_p2q = torch.zeros_like(distance, dtype=bool, device=distance.device)
        mask_2_p2q[bs_idx1.reshape(-1), src_idx1.reshape(-1), tgt_idx1.reshape(-1)] = True
        mask_1_p2q = torch.zeros_like(distance, dtype=bool, device=distance.device)
        mask_1_p2q[bs_idx1[:, :, 0].reshape(-1), src_idx1[:, :, 0].reshape(-1), tgt_idx1[:, :, 0].reshape(-1)] = True

        # corr1 = torch.cat([src_idx1[:, :, None], tgt_idx1[:, :, None]], dim=-1)
        # match points in feature space with batch size. from tgt->src
        _, src_idx2 = torch.topk(distance, dim=1, largest=False, k=num_k)
        tgt_idx2 = torch.arange(num_knn, device=distance.device)[None, None, :].expand(num_seeds, num_k, -1)
        bs_idx2 = torch.arange(num_seeds, device=distance.device)[:, None, None].expand(-1, num_k, num_knn)

        mask_2_q2p = torch.zeros_like(distance, dtype=bool, device=distance.device)
        mask_2_q2p[bs_idx2.reshape(-1), src_idx2.reshape(-1), tgt_idx2.reshape(-1)] = True
        mask_1_q2p = torch.zeros_like(distance, dtype=bool, device=distance.device)
        mask_1_q2p[bs_idx2[:, 0, :].reshape(-1), src_idx2[:, 0, :].reshape(-1), tgt_idx2[:, 0, :].reshape(-1)] = True

        # corr2 = torch.cat([src_idx2[:, :, None], tgt_idx2[:, :, None]], dim=-1)

        mutual_mask_p2q = mask_1_p2q * mask_2_q2p
        mutual_mask_q2p = mask_2_p2q * mask_1_q2p
        mutual_mask = mutual_mask_p2q | mutual_mask_q2p

        return mutual_mask

    def apply_guided_similarity(self, distance, src_points, tgt_points, src_indices, tgt_indices, guide):
        if src_points is None or tgt_points is None or src_indices is None or tgt_indices is None:
            return distance

        eps = guide.get('eps', self.prior_eps)
        prior_lambda = guide.get('lambda', self.prior_lambda)
        eta_l = guide.get('eta_l', self.eta_l)

        src_prior = guide.get('source_prior')
        src_under = guide.get('source_under_support')
        tgt_under = guide.get('target_under_support')
        c_ref = guide.get('c_ref')
        r_src = guide.get('r_src')

        if src_prior is None or src_under is None or tgt_under is None or c_ref is None or r_src is None:
            return distance

        src_prior_local = src_prior[src_indices].clamp(eps, 1.0)
        src_under_local = src_under[src_indices].clamp(eps, 1.0)
        tgt_under_local = tgt_under[tgt_indices].clamp(eps, 1.0)

        lever_d = torch.norm(src_points - c_ref.view(1, 1, 3), dim=-1) / (r_src + eps)
        lever = 1.0 + eta_l * lever_d.clamp(0.0, 1.0)
        lever = torch.maximum(lever, torch.tensor(eps, device=distance.device, dtype=distance.dtype))

        src_log_prior = (
            torch.log(src_prior_local)
            + torch.log(src_under_local)
            + torch.log(lever)
        )
        tgt_log_prior = torch.log(tgt_under_local)
        log_prior = src_log_prior[:, :, None] + tgt_log_prior[:, None, :]
        guided_score = -distance + prior_lambda * log_prior
        adjusted = -guided_score
        return torch.nan_to_num(adjusted, nan=1e6, posinf=1e6, neginf=-1e6)

    def generalized_mutual_mask(self, distance, top_k):
        src_count = distance.shape[1]
        tgt_count = distance.shape[2]
        src_k = min(max(int(top_k), 1), tgt_count)
        tgt_k = min(max(int(top_k), 1), src_count)

        src_topk = distance.topk(k=src_k, dim=2, largest=False).indices
        src_topk_mask = torch.zeros_like(distance, dtype=torch.bool)
        src_topk_mask.scatter_(2, src_topk, True)
        src_top1_mask = torch.zeros_like(distance, dtype=torch.bool)
        src_top1_mask.scatter_(2, src_topk[:, :, :1], True)

        tgt_topk = distance.topk(k=tgt_k, dim=1, largest=False).indices
        tgt_topk_mask = torch.zeros_like(distance, dtype=torch.bool)
        tgt_topk_mask.scatter_(1, tgt_topk, True)
        tgt_top1_mask = torch.zeros_like(distance, dtype=torch.bool)
        tgt_top1_mask.scatter_(1, tgt_topk[:, :1, :], True)

        return (src_top1_mask & tgt_topk_mask) | (src_topk_mask & tgt_top1_mask)

    def paired_local_matching(self, src_key_corr, tgt_key_corr, src_point, tgt_point, src_feature, tgt_feature, guide):
        guide = guide or {}
        radius = float(guide.get('local_radius', 0.8))
        max_local_points = int(guide.get('local_max_points', self.max_points))
        mutual_k = int(guide.get('generalized_mutual_k', 3))
        max_matches = int(guide.get('max_matches_per_seed', self.max_points))
        src_all = src_point[0]
        tgt_all = tgt_point[0]
        src_desc_all = src_feature[0]
        tgt_desc_all = tgt_feature[0]
        seed_count = min(src_key_corr.shape[1], tgt_key_corr.shape[1])
        pair_scores = {}
        candidate_records = []
        if seed_count > 0:
            src_center_distances = torch.cdist(src_key_corr[0, :seed_count], src_all)
            tgt_center_distances = torch.cdist(tgt_key_corr[0, :seed_count], tgt_all)
            src_local_count = min(max_local_points, src_all.shape[0])
            tgt_local_count = min(max_local_points, tgt_all.shape[0])
            src_distance, src_indices = torch.topk(src_center_distances, src_local_count, dim=1, largest=False)
            tgt_distance, tgt_indices = torch.topk(tgt_center_distances, tgt_local_count, dim=1, largest=False)
            src_valid = src_distance <= radius
            tgt_valid = tgt_distance <= radius
            src_desc = src_desc_all[src_indices]
            tgt_desc = tgt_desc_all[tgt_indices]
            distance = torch.sqrt(torch.clamp(2 - 2 * torch.matmul(src_desc, tgt_desc.transpose(1, 2)), min=1e-6))
            distance = self.apply_guided_similarity(
                distance,
                src_all[src_indices],
                tgt_all[tgt_indices],
                src_indices,
                tgt_indices,
                guide
            )
            valid_pair = src_valid[:, :, None] & tgt_valid[:, None, :]
            distance = distance.masked_fill(~valid_pair, 1e6)
            mutual_mask = self.generalized_mutual_mask(distance, mutual_k) & valid_pair
            for seed_index in range(seed_count):
                matched = torch.where(mutual_mask[seed_index])
                if matched[0].numel() == 0:
                    continue
                scores = -distance[seed_index][matched]
                if scores.numel() > max_matches:
                    keep = torch.topk(scores, max_matches, largest=True).indices
                    matched = (matched[0][keep], matched[1][keep])
                    scores = scores[keep]
                candidate_src = src_all[src_indices[seed_index, matched[0]]]
                candidate_tgt = tgt_all[tgt_indices[seed_index, matched[1]]]
                candidate_record = {
                    'seed_id': seed_index,
                    'seed_src_index': int(src_center_distances[seed_index].argmin().item()),
                    'seed_tgt_index': int(tgt_center_distances[seed_index].argmin().item()),
                    's2': float(scores.mean().item()),
                    's2_sum': float(scores.sum().item()),
                    'correspondence_count': int(scores.numel()),
                    'model_inlier_count': 0,
                    'valid_pose': int(scores.numel() >= 3),
                    'candidate_trans': None,
                }
                if scores.numel() >= 3:
                    weights = torch.softmax(scores, dim=0) * scores.shape[0]
                    candidate_trans = rigid_transform_3d(
                        candidate_src[None],
                        candidate_tgt[None],
                        weights[None],
                    ).detach().cpu()
                    residual = torch.norm(
                        transform(candidate_src[None], candidate_trans.to(candidate_src.device)) - candidate_tgt[None],
                        dim=-1,
                    )
                    candidate_record['candidate_trans'] = candidate_trans
                    candidate_record['model_inlier_count'] = int((residual < self.inlier_threshold).sum().item())
                candidate_records.append(candidate_record)
                for local_src, local_tgt, score in zip(matched[0].tolist(), matched[1].tolist(), scores.tolist()):
                    key = (int(src_indices[seed_index, local_src]), int(tgt_indices[seed_index, local_tgt]))
                    pair_scores[key] = max(pair_scores.get(key, -float('inf')), float(score))

        if not pair_scores:
            self.last_r2_candidates = candidate_records
            empty_src = src_point[:, :0, :]
            empty_tgt = tgt_point[:, :0, :]
            self.last_match_weights = torch.empty(0, dtype=src_point.dtype, device=src_point.device)
            return empty_src, empty_tgt, torch.eye(4, dtype=src_point.dtype, device=src_point.device)[None]

        pairs = sorted(pair_scores.items(), key=lambda item: item[1], reverse=True)
        src_indices = torch.tensor([item[0][0] for item in pairs], dtype=torch.long, device=src_point.device)
        tgt_indices = torch.tensor([item[0][1] for item in pairs], dtype=torch.long, device=tgt_point.device)
        scores = torch.tensor([item[1] for item in pairs], dtype=src_point.dtype, device=src_point.device)
        src_corr = src_all[src_indices][None]
        tgt_corr = tgt_all[tgt_indices][None]
        self.last_r2_candidates = candidate_records
        self.last_match_weights = torch.softmax(scores, dim=0) * scores.shape[0]
        if src_corr.shape[1] < 3:
            final_tran = torch.eye(4, dtype=src_point.dtype, device=src_point.device)[None]
        else:
            final_tran = rigid_transform_3d(src_corr, tgt_corr, self.last_match_weights[None])
        return src_corr, tgt_corr, final_tran

    def regenerate_seed_group(self, src_key_corr, tgt_key_corr, src_point, tgt_point, src_feature, tgt_feature, guide):
        return self.paired_local_matching(
            src_key_corr,
            tgt_key_corr,
            src_point,
            tgt_point,
            src_feature,
            tgt_feature,
            guide
        )


    def idx_selection(self, idx, points):
        """
        input:
            - idx: [bs, num_corr, num_knn]
            - points: [bs, num_corr, num_channel]
        return:
            - selected_points: [bs, num_corr, num_knn, num_channel]
        """
        bs, num_corr, num_knn = idx.shape
        num_channel = points.shape[-1]
        idx1 = idx.contiguous().view([1, -1])
        idx1 = idx1[:, :, None].expand(-1, -1, num_channel)
        selected_points = points.gather(dim=1, index=idx1).view([1, -1, num_knn, num_channel])
        # [bs, num_seeds, knn_num, num_channel]

        return selected_points

    def cal_leading_eigenvector(self, M, method='power'):
        """
        Calculate the leading eigenvector using power iteration algorithm or torch.symeig
        Input:
            - M:      [bs, num_corr, num_corr] the compatibility matrix
            - method: select different method for calculating the learding eigenvector.
        Output:
            - solution: [bs, num_corr] leading eigenvector
        """
        if method == 'power':
            # power iteration algorithm
            leading_eig = torch.ones_like(M[:, :, 0:1])
            leading_eig_last = leading_eig
            for i in range(self.num_iterations):
                leading_eig = torch.bmm(M, leading_eig)
                leading_eig = leading_eig / (torch.norm(leading_eig, dim=1, keepdim=True) + 1e-6)
                if torch.allclose(leading_eig, leading_eig_last):
                    break
                leading_eig_last = leading_eig
            leading_eig = leading_eig.squeeze(-1)
            return leading_eig
        elif method == 'eig':  # cause NaN during back-prop
            e, v = torch.symeig(M, eigenvectors=True)
            leading_eig = v[:, :, -1]
            return leading_eig
        else:
            exit(-1)

    def pick_seeds(self, dists, scores, R, max_num):
        """
        Select seeding points using Non Maximum Suppression. (here we only support bs=1)
        Input:
            - dists:       [bs, num_corr, num_corr] src keypoints distance matrix
            - scores:      [bs, num_corr]     initial confidence of each correspondence
            - R:           float              radius of nms
            - max_num:     int                maximum number of returned seeds
        Output:
            - picked_seeds: [bs, num_seeds]   the index to the seeding correspondences
        """
        assert scores.shape[0] == 1

        # parallel Non Maximum Suppression (more efficient)
        score_relation = scores.T >= scores  # [num_corr, num_corr], save the relation of leading_eig
        # score_relation[dists[0] >= R] = 1  # mask out the non-neighborhood node
        score_relation = score_relation.bool() | (dists[0] >= R).bool()
        is_local_max = score_relation.min(-1)[0].float()

        score_local_max = scores * is_local_max
        sorted_score = torch.argsort(score_local_max, dim=1, descending=True)

        # max_num = scores.shape[1]

        return_idx = sorted_score[:, 0: max_num].detach()

        return return_idx

    def seed_sampling(self, src_keypts, tgt_keypts, sampling_num):
        """
        Input:
            - src_keypts: [bs, num_corr, 3]
            - tgt_keypts: [bs, num_corr, 3]
        Output:
            - pred_trans:   [bs, 4, 4], the predicted transformation matrix.
            - pred_labels:  [bs, num_corr], the predicted inlier/outlier label (0,1), for classification loss calculation.
        """
        bs, num_corr = src_keypts.shape[0], tgt_keypts.shape[1]

        #################################
        # downsample points
        #################################
        max_points = 1000
        if num_corr > max_points:
            sel_ind = np.random.choice(src_keypts.shape[1], max_points)
            src_keypts = src_keypts[:, sel_ind, :]
            tgt_keypts = tgt_keypts[:, sel_ind, :]
            num_corr = max_points

        #################################
        # compute cross dist
        #################################
        src_dist = torch.norm((src_keypts[:, :, None, :] - src_keypts[:, None, :, :]), dim=-1)
        target_dist = torch.norm((tgt_keypts[:, :, None, :] - tgt_keypts[:, None, :, :]), dim=-1)
        cross_dist = torch.abs(src_dist - target_dist)

        #################################
        # compute first order measure
        #################################
        SC_dist_thre = self.d_thre
        SC_measure = torch.clamp(1.0 - cross_dist ** 2 / SC_dist_thre ** 2, min=0)
        hard_SC_measure = (cross_dist < SC_dist_thre).float()

        #################################
        # select reliable seed correspondences
        #################################
        confidence = self.cal_leading_eigenvector(SC_measure, method='power')
        seeds = self.pick_seeds(src_dist, confidence, R=self.nms_radius, max_num=sampling_num)   # 第重叠的时候 nms_radius应该调小
        src_keypts_best_corr = src_keypts[:, seeds[0]]
        tgt_keypts_best_corr = tgt_keypts[:, seeds[0]]

        return src_keypts_best_corr, tgt_keypts_best_corr

    def regenerate(self, src_key_corr, tgt_key_corr, src_point, tgt_point, src_feature, tgt_feature, knn_num=100, sampling_num=100, knn_radius=0.8, use_sampling=False, guide=None, mode="local", return_indices=False):
        """
        Input:
            - src_key_corr: [bs, num_key_corr, 3]
            - tgt_key_corr: [bs, num_key_corr, 3]
            - src_corr: [bs, num_corr, 3]
            - tgt_corr: [bs, num_corr, 3]
            - src_corr_feature: [bs, num_corr, 3]
            - tgt_corr_feature: [bs, num_corr, 3]
        Output:
            - pred_trans:   [bs, 4, 4], the predicted transformation matrix
            - src_corr_final:  [bs, num_corr, 3], the source points in the matched correspondences
            - src_corr_final:  [bs, num_corr, 3], the target points in the matched correspondences
        """
        self.use_sampling = use_sampling
        self.last_match_weights = None
        self.last_r2_candidates = []
        if mode == "paired_local":
            if return_indices:
                raise ValueError("paired_local regeneration does not expose local correspondence indices.")
            return self.regenerate_seed_group(
                src_key_corr,
                tgt_key_corr,
                src_point,
                tgt_point,
                src_feature,
                tgt_feature,
                guide,
            )
        if mode != "local":
            raise ValueError("mode must be local or paired_local; guided_global is removed because it bypassed seeds.")
        #################################
        # knn & regenerate correspondences
        #################################
        # [1,num_seeds,knn_num]
        self.knn_num = min(knn_num, min(src_point.shape[1], tgt_point.shape[1]))  # knn的近邻点数量
        self.sampling_num = sampling_num  # knn的近邻点数量

        if src_key_corr.shape[1] > self.sampling_num:
            sel_ind = np.random.choice(src_key_corr.shape[1], self.sampling_num, replace=False)
            src_key_corr = src_key_corr[:, sel_ind, :]
            tgt_key_corr = tgt_key_corr[:, sel_ind, :]

        src_corr_knn_idx = self.knn_search(src_key_corr, src_point, self.knn_num)
        tgt_corr_knn_idx = self.knn_search(tgt_key_corr, tgt_point, self.knn_num)

        matching = self.local_matching(
            src_key_corr,
            tgt_key_corr,
            src_corr_knn_idx,
            tgt_corr_knn_idx,
            src_point,
            tgt_point,
            src_feature,
            tgt_feature,
            guide=guide,
            return_indices=return_indices,
        )
        if return_indices:
            src_corr, tgt_corr, correspondence_indices = matching
        else:
            src_corr, tgt_corr = matching

        final_tran = rigid_transform_3d(src_corr, tgt_corr)

        if return_indices:
            return src_corr, tgt_corr, final_tran, correspondence_indices
        return src_corr, tgt_corr, final_tran
