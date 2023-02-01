# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import rich

from entmax import sparsemax, entmax15
from utils.distributed import DistributedGather


class LabelSmoothingLoss(nn.Module):
    """
    Implementation of cross entropy loss with label smoothing.
    Follows the implementation of the two followings:
        https://github.com/pytorch/pytorch/issues/7455#issuecomment-513062631
        https://github.com/pytorch/pytorch/issues/7455#issuecomment-513735962
    """
    def __init__(self,
                 num_classes: int,
                 smoothing: float = .0,
                 dim: int = 1,
                 reduction: str = 'mean',
                 class_weights: torch.FloatTensor = None):
        """
        Arguments:
            num_classes: int, specifying the number of target classes.
            smoothing: float, default value of 0 is equal to general cross entropy loss.
            dim: int, aggregation dimension.
            reduction: str, default 'mean'.
            class_weights: 1D tensor of shape (C, ) or (C, 1).
        """
        super(LabelSmoothingLoss, self).__init__()
        assert 0 <= smoothing < 1
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.dim = dim

        assert reduction in ['sum', 'mean']
        self.reduction = reduction

        self.class_weights = class_weights

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Arguments:
            pred: 2D torch tensor of shape (B, C)
            target: 1D torch tensor of shape (B, )
        """
        pred = F.log_softmax(pred, dim=self.dim)
        true_dist = self.smooth_one_hot(target, self.num_classes, self.smoothing)
        multiplied = -true_dist * pred

        if self.class_weights is not None:
            weights = self.class_weights.to(multiplied.device)
            summed = torch.matmul(multiplied, weights.view(self.num_classes, 1))  # (B, C) @ (C, 1) -> (B, 1)
            summed = summed.squeeze()                                             # (B, 1) -> (B, )
        else:
            summed = torch.sum(multiplied, dim=self.dim)                          # (B, C) -> sum -> (B, )

        if self.reduction == 'sum':
            return summed
        elif self.reduction == 'mean':
            return torch.mean(summed)
        else:
            raise NotImplementedError

    @staticmethod
    def smooth_one_hot(target: torch.Tensor, num_classes: int, smoothing=0.):
        assert 0 <= smoothing < 1
        confidence = 1. - smoothing
        label_shape = torch.Size((target.size(0), num_classes))
        with torch.no_grad():
            true_dist = torch.zeros(label_shape, device=target.device)
            true_dist.fill_(smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), confidence)
        return true_dist  # (B, C)


class SimCLRLoss(nn.Module):
    """
    Modified implementation of the following:
        https://github.com/HobbitLong/SupContrast
    """
    def __init__(self,
                 temperature: float = 0.07,
                 distributed: bool = False,
                 local_rank: int = 0):
        super(SimCLRLoss, self).__init__()

        self.temperature = temperature
        self.distributed = distributed
        self.local_rank = local_rank

    def forward(self, features: torch.FloatTensor):
        """For SimCLR."""

        _, num_views, _ = features.size()

        # Normalize features to lie on a unit hypersphere.
        features = F.normalize(features, dim=-1)
        features = torch.cat(torch.unbind(features, dim=1), dim=0)       # (B, N, F) -> (NB, F)
        if self.distributed:
            contrasts = torch.cat(DistributedGather.apply(features), dim=0)  # (NB x world_size, F)
        else:
            contrasts = features

        # Compute logits (aka. similarity scores) & numerically stabilize them
        logits = features @ contrasts.T  # (BN, F) x (F, NB * world_size)
        logits = logits.div(self.temperature)

        # Compute masks
        _, pos_mask, neg_mask = self.create_masks(logits.size(), self.local_rank, num_views)

        # Compute loss
        numerator = logits * pos_mask  # FIXME
        denominator = torch.exp(logits) * pos_mask.logical_or(neg_mask)
        denominator = denominator.sum(dim=1, keepdim=True)
        log_prob = numerator - torch.log(denominator)
        mean_log_prob = (log_prob * pos_mask) / pos_mask.sum(dim=1, keepdim=True)
        loss = torch.neg(mean_log_prob)
        loss = loss.sum(dim=1).mean()

        return loss, logits, pos_mask

    @staticmethod
    @torch.no_grad()
    def create_masks(shape, local_rank: int, num_views: int = 2):

        device = local_rank
        nL, nG = shape

        local_mask = torch.eye(nL // num_views, device=device).repeat(2, 2)  # self+positive indicator
        local_pos_mask = local_mask - torch.eye(nL, device=device)           # positive indicator
        local_neg_mask = torch.ones_like(local_mask) - local_mask            # negative indicator

        # Global mask of self+positive indicators
        global_mask = torch.zeros(nL, nG, device=device)
        global_mask[:, nL*local_rank:nL*(local_rank+1)] = local_mask

        # Global mask of positive indicators
        global_pos_mask = torch.zeros_like(global_mask)
        global_pos_mask[:, nL*local_rank:nL*(local_rank+1)] = local_pos_mask

        # Global mask of negative indicators
        global_neg_mask = torch.ones_like(global_mask)
        global_neg_mask[:, nL*local_rank:nL*(local_rank+1)] = local_neg_mask

        return global_mask, global_pos_mask, global_neg_mask


    @staticmethod
    def semisupervised_mask(unlabeled_size: int, labels: torch.Tensor):
        """Create mask for semi-supervised contrastive learning."""

        labels = labels.view(-1, 1)
        labeled_size = labels.size(0)
        mask_size = unlabeled_size + labeled_size
        mask = torch.zeros(mask_size, mask_size, dtype=torch.float32).to(labels.device)

        L = torch.eq(labels, labels.T).float()
        mask[unlabeled_size:, unlabeled_size:] = L
        U = torch.eye(unlabeled_size, dtype=torch.float32).to(labels.device)
        mask[:unlabeled_size, :unlabeled_size] = U
        mask.clamp_(0, 1)  # Just in case. This might not be necessary.

        return mask
