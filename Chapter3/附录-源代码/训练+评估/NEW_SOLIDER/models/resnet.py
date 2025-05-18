from collections import OrderedDict

import torch.nn.functional as F
import torchvision
from torch import nn

from .backbone import resnet50

class Backbone(nn.Sequential):
    def __init__(self, resnet):
        super(Backbone, self).__init__(
            OrderedDict(
                [
                    ["conv1", resnet.conv1],
                    ["bn1", resnet.bn1],
                    ["relu", resnet.relu],
                    ["maxpool", resnet.maxpool],
                    ["layer1", resnet.layer1],  # res2
                    ["layer2", resnet.layer2],  # res3
                    ["layer3", resnet.layer3],  # res4
                ]
            )
        )
        self.out_channels = 1024

    def forward(self, x):
        # using the forward method from nn.Sequential
        feat = super(Backbone, self).forward(x)
        return OrderedDict([["feat_res4", feat]])                                   # 有序字典作为特征表示的输出


class Res5Head(nn.Sequential):
    def __init__(self, resnet):
        super(Res5Head, self).__init__(OrderedDict([["layer4", resnet.layer4]]))  # res5
        self.out_channels = [1024, 2048]

    def forward(self, x):
        feat = super(Res5Head, self).forward(x)
        x = F.adaptive_max_pool2d(x, 1)                                             # 对输入x和特征feat进行自适应最大池化操作，输出的特征图的大小为 1x1 (batch_size, num_channels, 1, 1)
        feat = F.adaptive_max_pool2d(feat, 1)                                       # 整个特征图的一个综合的表示，而不关心具体的空间位置
        print(x.shape,feat.shape)
        return OrderedDict([["feat_res4", x], ["feat_res5", feat]])


def build_resnet(name="resnet50", pretrained=True):
    #resnet = torchvision.models.resnet.__dict__[name](pretrained=pretrained)
    resnet = resnet50(pretrained=True)                  # 选择对应的 ResNet 模型，加载预训练好的权重

    # freeze layers，冻结了 ResNet 的第一层（卷积层和批归一化层）的权重，使其在训练过程中不被更新。
    resnet.conv1.weight.requires_grad_(False)
    resnet.bn1.weight.requires_grad_(False)
    resnet.bn1.bias.requires_grad_(False)

    return Backbone(resnet), Res5Head(resnet)
