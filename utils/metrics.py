# import numpy as np
# from scipy.spatial.distance import cdist
import torch

def torch_nanmean(x):
    num = torch.where(torch.isnan(x), torch.full_like(x, 0), torch.full_like(x, 1)).sum()
    value = torch.where(torch.isnan(x), torch.full_like(x, 0), x).sum()
    return value / num

def Pixel_Accuracy(confusion_matrix):
    Acc = torch.diag(confusion_matrix).sum() / confusion_matrix.sum()
    Acc_ = torch.diag(confusion_matrix)[:-2].sum() / confusion_matrix[:-2,:-2].sum()
    return Acc, Acc_

def Pixel_Accuracy_Class(confusion_matrix):
    Acc = torch.diag(confusion_matrix) / confusion_matrix.sum(dim=1)
    Acc = torch_nanmean(Acc)
    Acc_ = torch.diag(confusion_matrix)[:-2] / confusion_matrix[:-2,:-2].sum(dim=1)
    Acc_ = torch_nanmean(Acc_)
    return Acc, Acc_

def Mean_Intersection_over_Union(confusion_matrix):
    MIoU = torch.diag(confusion_matrix) / (
                torch.sum(confusion_matrix, dim=1) + torch.sum(confusion_matrix, dim=0) -
                torch.diag(confusion_matrix))
    MIoU = torch_nanmean(MIoU)
    return MIoU

def Frequency_Weighted_Intersection_over_Union(confusion_matrix):
    freq = torch.sum(confusion_matrix, dim=1) / torch.sum(confusion_matrix)
    iu = torch.diag(confusion_matrix) / (
                torch.sum(confusion_matrix, dim=1) + torch.sum(confusion_matrix, dim=0) -
                torch.diag(confusion_matrix))

    FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
    return FWIoU

class Evaluator(object):
    def __init__(self, seg_channel, vertex_channel):
        self.seg_channel = seg_channel
        self.seg_confusion_matrix = torch.zeros((self.seg_channel, )*2)
        
        self.vertex_channel = vertex_channel
        self.vertex_confusion_matrix = torch.zeros((self.vertex_channel, )*2)
        
        self.mask_channel = 2
        self.mask_confusion_matrix = torch.zeros((self.mask_channel, )*2)

    def _generate_matrix(self, gt_mask, pred_mask, channel):
        gt_mask = gt_mask.cpu()
        pred_mask = pred_mask.cpu()
        mask = (gt_mask >= 0) & (gt_mask < channel)
        label = channel * gt_mask[mask].long() + pred_mask[mask].long()
        count = torch.bincount(label, minlength=channel**2)
        confusion_matrix = count.reshape(channel, channel)
        return confusion_matrix.float()

    def add_seg_batch(self, seg_gt, seg_pred):
        assert seg_gt.shape == seg_pred.shape
        self.seg_confusion_matrix += self._generate_matrix(seg_gt, seg_pred, self.seg_channel)

    def add_vertex_batch(self, vertex_gt, vertex_pred):
        assert vertex_gt.shape == vertex_pred.shape
        # self.vertex_confusion_matrix += self._generate_matrix(vertex_gt, vertex_pred, self.vertex_channel)
        
    def add_mask_batch(self, mask_gt, mask_pred):
        assert mask_gt.shape == mask_pred.shape
        self.mask_confusion_matrix += self._generate_matrix(mask_gt, mask_pred, self.mask_channel)

    def reset(self):
        self.seg_confusion_matrix = torch.zeros((self.seg_channel,) * 2)
        self.vertex_confusion_matrix = torch.zeros((self.vertex_channel,) * 2)
        self.mask_confusion_matrix = torch.zeros((self.mask_channel,) * 2)




