import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


class decoder_v1(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm):
        super(decoder_v1, self).__init__()

        if backbone == "resnet":
            low_feat_inplanes = 256
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(low_feat_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3,
                                                 stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self._init_weight()

    def forward(self, x32s, x16s, x8s, x4s, x2s, x):
        x4s = self.conv1(x4s)
        x4s = self.bn1(x4s)
        x4s = self.relu(x4s)

        x32s = F.interpolate(x32s, size=x4s.size()[
                             2:], mode="bilinear", align_corners=True)
        x4s = torch.cat((x32s, x4s), dim=1)
        x4s = self.last_conv(x4s)

        return F.interpolate(x4s, size=x.size()[2:], mode="bilinear", align_corners=True), x4s.detach()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class decoder_v2(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm):
        super(decoder_v2, self).__init__()
        if backbone != "resnet":
            raise NotImplementedError

        s32dim, s8dim, s4dim, s2dim, raw_dim = 256, 256, 128, 64, 64
        # x8s
        self.conv8s = nn.Sequential(
            nn.Conv2d(512+s32dim, s8dim, 3, 1, 1, bias=False),
            BatchNorm(s8dim),
            nn.LeakyReLU(0.1, True)
        )

        # x4s
        self.conv4s = nn.Sequential(
            nn.Conv2d(256+s8dim, s4dim, 3, 1, 1, bias=False),
            BatchNorm(s4dim),
            nn.LeakyReLU(0.1, True)
        )

        # x2s
        self.conv2s = nn.Sequential(
            nn.Conv2d(64+s4dim, s2dim, 3, 1, 1, bias=False),
            BatchNorm(s2dim),
            nn.LeakyReLU(0.1, True)
        )

        # xraw

        self.convraw = nn.Sequential(
            nn.Conv2d(3+s2dim, raw_dim, 3, 1, 1, bias=False),
            BatchNorm(raw_dim),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(raw_dim, num_classes, 1, 1)
        )

        self._init_weight()

    def forward(self, x32s, x16s, x8s, x4s, x2s, x):
        fm = F.interpolate(x32s, size=x8s.size()[
                           2:], mode="bilinear", align_corners=True)

        fm = self.conv8s(torch.cat([fm, x8s], 1))
        fm = F.interpolate(fm, size=x4s.size()[
                           2:], mode="bilinear", align_corners=True)
        # fm = self.up8sto4s(fm)

        x4s = self.conv4s(torch.cat([fm, x4s], 1))
        fm = F.interpolate(x4s, size=x2s.size()[
                           2:], mode="bilinear", align_corners=True)
        # fm = self.up4sto2s(fm)

        fm = self.conv2s(torch.cat([fm, x2s], 1))
        fm = F.interpolate(fm, size=x.size()[
                           2:], mode="bilinear", align_corners=True)
        # fm = self.up2storaw(fm)

        return self.convraw(torch.cat([fm, x], 1)), x4s.detach()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_decoder(channel, backbone, BatchNorm, decoder_type="v1"):
    if decoder_type == "v1":
        return decoder_v1(channel, backbone, BatchNorm)
    elif decoder_type == "v2":
        return decoder_v2(channel, backbone, BatchNorm)
    else:
        raise NotImplementedError
