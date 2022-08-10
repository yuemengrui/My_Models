# *_*coding:utf-8 *_*
# @Author : yuemengrui
import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import resnet50
from .aspp import ASPP
from .decoder import Decoder


class Model(nn.Module):
    def __init__(self, output_stride=8, num_classes=2):
        super().__init__()

        if output_stride == 8:
            replace_stride_with_dilation = [False, True, True]
            aspp_dilations = [12, 24, 36]
        else:
            replace_stride_with_dilation = [False, False, True]
            aspp_dilations = [6, 12, 18]

        self.backbone = resnet50(replace_stride_with_dilation=replace_stride_with_dilation)

        inplanes = 2048
        low_level_planes = 256

        self.aspp = ASPP(inplanes, aspp_dilations)

        self.decoder = Decoder(low_level_planes, num_classes)

    def forward(self, input):
        x, low_level_feat = self.backbone(input)

        x = self.aspp(x)

        x = self.decoder(x, low_level_feat)

        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x


if __name__ == '__main__':
    model = Model()
    model.eval()

    input = torch.randn((1, 3, 224, 224))

    out = model(input)

    print(out.shape)
    print(out)
