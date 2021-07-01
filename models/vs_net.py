import torch
import torch.nn as nn
import torch.nn.functional as F
from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from models.aspp import build_aspp
from models.decoder import build_decoder
from models.backbone import build_backbone


class VSNet(nn.Module):
    def __init__(self, backbone="resnet", seg_decoder="v1",
                 vertex_decoder="v2", output_stride=16, seg_channel=12,
                 vertex_channel=2, sync_bn=True):
        super(VSNet, self).__init__()

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)

        self.seg_decoder, self.vertex_decoder = None, None
        if seg_decoder:
            self.seg_decoder = build_decoder(
                seg_channel, backbone, BatchNorm, seg_decoder)
        if vertex_decoder:
            self.vertex_decoder = build_decoder(
                vertex_channel, backbone, BatchNorm, vertex_decoder)

    def forward(self, x):
        x32s, x16s, x8s, x4s, x2s = self.backbone(x)
        x32s = self.aspp(x32s)

        seg_pred, vertex_pred = None, None
        if self.seg_decoder:
            seg_pred, seg_pred_x4s = self.seg_decoder(
                x32s, x16s, x8s, x4s, x2s, x)
        if self.vertex_decoder:
            vertex_pred, _ = self.vertex_decoder(x32s, x16s, x8s, x4s, x2s, x)

        return seg_pred, vertex_pred, seg_pred_x4s

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.seg_decoder, self.vertex_decoder]
        for i in range(len(modules)):
            if modules[i] is None:
                continue
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p
