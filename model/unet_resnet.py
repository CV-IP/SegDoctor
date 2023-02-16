
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.spixel import SpixelNet

from model.resnet import resnet50

class UNet_ResNet(nn.Module):
    """
    The UNet implementation based on torchtorch.

    The original article refers to
    Olaf Ronneberger, et, al. "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    (https://arxiv.org/abs/1505.04597).

    Args:
        num_classes (int): The unique number of target classes.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.  Default: False.
        use_deconv (bool, optional): A bool value indicates whether using deconvolution in upsampling.
            If False, use resize_bilinear. Default: False.
        pretrained (str, optional): The path or url of pretrained model for fine tuning. Default: None.
    """

    def __init__(self,
                 num_classes,
                 align_corners=False,
                 use_deconv=False,
                 pretrained=True):
        super().__init__()

        self.encode = resnet50(pretrained=pretrained)
        # print('Resnet', self.encode.load_state_dict(torch.load('model_data/resnet50s-a75c83cf.pth')))
        self.decode = Decoder(align_corners)
        self.cls = self.conv = nn.Conv2d(
            in_channels=256,
            out_channels=num_classes,
            kernel_size=3,
            padding=1
        )
        self.spix_branch = SpixelNet(batchNorm=True)
        
        self.pretrained = pretrained

    def forward(self, x):
        bs, c, h, w = x.shape
        feats = self.encode(x)
        
        # [bs, 256, 128, 128]   4x
        # [bs, 512, 64, 64]     8x
        # [bs, 1024, 32, 32]    16x
        # [bs, 2048, 16, 16]    16x
        
        x, xs = self.decode(feats[-1], feats[:-1])
        if self.training:
            prob = self.spix_branch(feats[:-1])
            prob = F.interpolate(prob, size=(h, w))
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        logit = self.cls(x)
        if self.training: 
            return logit, xs, prob
        else:
            return logit, xs, 0


class ConvBNReLU(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 **kwargs):
        super().__init__()

        self._conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self._batch_norm = nn.BatchNorm2d(out_channels)
        self._relu = nn.ReLU()

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        x = self._relu(x)
        return x


class Decoder(nn.Module):
    def __init__(self, align_corners):
        super().__init__()

        # up_channels = [[512, 256], [256, 128], [128, 64], [64, 64]]
        up_channels = [[2048, 1024], [1024, 512], [512, 256]]

        self.up_sample_list = nn.ModuleList([
            UpSampling(channel[0], channel[1], align_corners)
            for channel in up_channels
        ])

    def forward(self, x, short_cuts):
        xs = [x]  # [16x]
        # xs = [short_cuts[-1]]  # [16x]
        for i in range(len(short_cuts)):
            x = self.up_sample_list[i](x, short_cuts[-(i + 1)])
            xs.append(x)
        return x, xs


class UpSampling(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 align_corners):
        super().__init__()

        self.align_corners = align_corners

        self.conv1 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

        self.double_conv = nn.Sequential(
            ConvBNReLU(out_channels * 2, out_channels, 3, padding=1),
            ConvBNReLU(out_channels, out_channels, 3, padding=1)
        )

    def forward(self, x, short_cut):

        x = F.interpolate(
                x,
                short_cut.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners
            )
        x = self.conv1(x)
        
        x = torch.concat([x, short_cut], dim=1)
        x = self.double_conv(x)
        return x
