"""
Loss functions

Author: Shengyu Huang
Last modified: 30.11.2020
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from lib.utils import square_distance
from sklearn.metrics import precision_recall_fscore_support

class MetricLoss(nn.Module):
    """
    We evaluate both contrastive loss and circle loss
    """
    def __init__(self,configs,log_scale=16, pos_optimal=0.1, neg_optimal=1.4):
        super(MetricLoss,self).__init__()
        self.log_scale = log_scale
        self.pos_optimal = pos_optimal
        self.neg_optimal = neg_optimal

        self.pos_margin = configs.pos_margin
        self.neg_margin = configs.neg_margin
        self.max_points = configs.max_points

        self.safe_radius = configs.safe_radius 
        self.matchability_radius = configs.matchability_radius
        self.pos_radius = configs.pos_radius # just to take care of the numeric precision
    
    def get_circle_loss(self, coords_dist, feats_dist):
        """
        coords_dist 是点云P, Q之间的L2距离, 维度是 [N, M]
        feats_dist 是点云P的特征和Q的特征之间的L2距离，维度是[N, M]
        每一行表示 P 中的一个元素到 Q 中所有元素的距离
        每一列则是表示 Q 中的一个元素到 P 中所有元素的距离
        为了方便描述，我把属于类内相似性的点对、特征对记为positive，相反，记为negative
        Modified from: https://github.com/XuyangBai/D3Feat.pytorch
        """
        # 距离小于半径的属于 positive
        pos_mask = coords_dist < self.pos_radius
        # 距离大于半径的属于 negative
        neg_mask = coords_dist > self.safe_radius 

        ## get anchors that have both positive and negative pairs
        row_sel = ((pos_mask.sum(-1)>0) * (neg_mask.sum(-1)>0)).detach()
        col_sel = ((pos_mask.sum(-2)>0) * (neg_mask.sum(-2)>0)).detach()

        # get alpha for both positive and negative pairs
        # feats_dist 中属于 negative 的减去 1e5，就变成了负值
        pos_weight = feats_dist - 1e5 * (~pos_mask).float() # mask the non-positive
        # self.pos_optimal 是一个经验值
        pos_weight = (pos_weight - self.pos_optimal) # mask the uninformative positive
        # 比较torch.zeros_like(pos_weight), pos_weight
        # pos_weight 中的负数都为0，也就是 negative 的赋值为0，只留下了 positive
        pos_weight = torch.max(torch.zeros_like(pos_weight), pos_weight).detach() 

        # 原理同上
        neg_weight = feats_dist + 1e5 * (~neg_mask).float() # mask the non-negative
        neg_weight = (self.neg_optimal - neg_weight) # mask the uninformative negative
        neg_weight = torch.max(torch.zeros_like(neg_weight),neg_weight).detach()

        # β: pos_weight, neg_weight
        # 类内相似性
        lse_pos_row = torch.logsumexp(self.log_scale * (feats_dist - self.pos_margin) * pos_weight,dim=-1)
        lse_pos_col = torch.logsumexp(self.log_scale * (feats_dist - self.pos_margin) * pos_weight,dim=-2)

        # 类间相似性
        lse_neg_row = torch.logsumexp(self.log_scale * (self.neg_margin - feats_dist) * neg_weight,dim=-1)
        lse_neg_col = torch.logsumexp(self.log_scale * (self.neg_margin - feats_dist) * neg_weight,dim=-2)

        loss_row = F.softplus(lse_pos_row + lse_neg_row)/self.log_scale
        loss_col = F.softplus(lse_pos_col + lse_neg_col)/self.log_scale

        circle_loss = (loss_row[row_sel].mean() + loss_col[col_sel].mean()) / 2

        return circle_loss

    def get_recall(self,coords_dist,feats_dist):
        """
        Get feature match recall, divided by number of true inliers
        """
        pos_mask = coords_dist < self.pos_radius
        n_gt_pos = (pos_mask.sum(-1)>0).float().sum()+1e-12
        _, sel_idx = torch.min(feats_dist, -1)
        sel_dist = torch.gather(coords_dist,dim=-1,index=sel_idx[:,None])[pos_mask.sum(-1)>0]
        n_pred_pos = (sel_dist < self.pos_radius).float().sum()
        recall = n_pred_pos / n_gt_pos
        return recall

    def get_weighted_bce_loss(self, prediction, gt):
        # 初始化BCELoss
        loss = nn.BCELoss(reduction='none')

        # 计算BCELoss
        class_loss = loss(prediction, gt)
        # 设置权重惩罚, 为了平衡分类结果的占比
        weights = torch.ones_like(gt)
        w_negative = gt.sum()/gt.size(0) 
        w_positive = 1 - w_negative  
        
        weights[gt >= 0.5] = w_positive
        weights[gt < 0.5] = w_negative
        w_class_loss = torch.mean(weights * class_loss)

        #######################################
        # get classification precision and recall
        # 得到分类准确率签 和 召回率 ,分类标签，四舍五入
        predicted_labels = prediction.detach().cpu().round().numpy()
        cls_precision, cls_recall, _, _ = precision_recall_fscore_support(gt.cpu().numpy(),predicted_labels, average='binary')

        return w_class_loss, cls_precision, cls_recall
            

    def forward(self, src_pcd, tgt_pcd, src_feats, tgt_feats, correspondence, rot, trans,scores_overlap,scores_saliency):
        """
        Circle loss for metric learning, here we feed the positive pairs only
        Input:
            src_pcd:        [N, 3]  
            tgt_pcd:        [M, 3]
            rot:            [3, 3]
            trans:          [3, 1]
            src_feats:      [N, C]
            tgt_feats:      [M, C]
        """
        src_pcd = (torch.matmul(rot,src_pcd.transpose(0,1))+trans).transpose(0,1)
        stats=dict()

        # 从correspondence中得到匹配点 correspondence [N,2]
        src_idx = list(set(correspondence[:,0].int().tolist()))
        tgt_idx = list(set(correspondence[:,1].int().tolist()))

        #######################
        # get BCE loss for overlap, here the ground truth label is obtained from correspondence information
        # 获得重叠的BCE损失，这里地面真理标签是从对应信息获得的
        src_gt = torch.zeros(src_pcd.size(0))
        src_gt[src_idx]=1.
        tgt_gt = torch.zeros(tgt_pcd.size(0))
        tgt_gt[tgt_idx]=1.
        gt_labels = torch.cat((src_gt, tgt_gt)).to(torch.device('cuda'))

        class_loss, cls_precision, cls_recall = self.get_weighted_bce_loss(scores_overlap, gt_labels)
        stats['overlap_loss'] = class_loss
        stats['overlap_recall'] = cls_recall
        stats['overlap_precision'] = cls_precision

        #######################
        # get BCE loss for saliency part, here we only supervise points in the overlap region
        # 获得显着部分的BCE损失，在这里我们只监督重叠区域中的点
        # 只关系重叠区域
        src_feats_sel, src_pcd_sel = src_feats[src_idx], src_pcd[src_idx]
        tgt_feats_sel, tgt_pcd_sel = tgt_feats[tgt_idx], tgt_pcd[tgt_idx]
        # 计算分数，分数越高说明src_feats_sel中特征和tgt_feats_sel中的特征相似
        scores = torch.matmul(src_feats_sel, tgt_feats_sel.transpose(0,1))
        # 得到和src_feats_sel最相似的tgt_feats_sel
        _, idx = scores.max(1)
        # 由上步就可以计算两个特征相似点对之间的距离
        distance_1 = torch.norm(src_pcd_sel - tgt_pcd_sel[idx], p=2, dim=1)

        # 得到和tgt_feats_sel最相似的src_feats_sel
        _, idx = scores.max(0)
        distance_2 = torch.norm(tgt_pcd_sel - src_pcd_sel[idx], p=2, dim=1)

        # 设置gt_labels， 距离<self.matchability_radius,gt_labels=1,反之为0
        gt_labels = torch.cat(((distance_1<self.matchability_radius).float(), (distance_2<self.matchability_radius).float()))

        src_saliency_scores = scores_saliency[:src_pcd.size(0)][src_idx]
        tgt_saliency_scores = scores_saliency[src_pcd.size(0):][tgt_idx]
        scores_saliency = torch.cat((src_saliency_scores, tgt_saliency_scores))

        # 和 overlap loss 一样
        class_loss, cls_precision, cls_recall = self.get_weighted_bce_loss(scores_saliency, gt_labels)
        stats['saliency_loss'] = class_loss
        stats['saliency_recall'] = cls_recall
        stats['saliency_precision'] = cls_precision

        #######################################
        # filter some of correspondence as we are using different radius for "overlap" and "correspondence"
        # 过滤一些correspondence 因为"overlap" and "correspondence"的半径不一样
        # 计算两个点云对应点对之间的距离
        c_dist = torch.norm(src_pcd[correspondence[:,0]] - tgt_pcd[correspondence[:,1]], dim = 1)
        # 选择符合条件的对应点对
        c_select = c_dist < self.pos_radius - 0.001
        correspondence = correspondence[c_select]
        # 如果更新的correspondence数量过多，就随机选择self.max_points个点对
        if(correspondence.size(0) > self.max_points):
            choice = np.random.permutation(correspondence.size(0))[:self.max_points]
            correspondence = correspondence[choice]
        src_idx = correspondence[:,0]
        tgt_idx = correspondence[:,1]
        src_pcd, tgt_pcd = src_pcd[src_idx], tgt_pcd[tgt_idx]
        src_feats, tgt_feats = src_feats[src_idx], tgt_feats[tgt_idx]

        #######################
        # get L2 distance between source / target point cloud
        # 计算距离
        coords_dist = torch.sqrt(square_distance(src_pcd[None,:,:], tgt_pcd[None,:,:]).squeeze(0))
        feats_dist = torch.sqrt(square_distance(src_feats[None,:,:], tgt_feats[None,:,:],normalised=True)).squeeze(0)

        ##############################
        # get FMR and circle loss
        ##############################
        recall = self.get_recall(coords_dist, feats_dist)
        circle_loss = self.get_circle_loss(coords_dist, feats_dist)

        stats['circle_loss']= circle_loss
        stats['recall']=recall

        return stats
