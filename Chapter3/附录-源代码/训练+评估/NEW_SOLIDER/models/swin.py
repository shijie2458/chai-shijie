from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn

from .swin_transformer import swin_tiny_patch4_window7_224,swin_small_patch4_window7_224,swin_base_patch4_window7_224

bonenum = 3         # 迭代的阶段数量
# semantic_weight=0.6
class Backbone(nn.Sequential):
    def __init__(self, swin, out_channels=384):
        super().__init__()
        self.swin = swin
        self.out_channels = out_channels

    def forward(self, x):                                               # torch.Size([3, 3, 928, 1504])
        if self.swin.semantic_weight >= 0:
            w = torch.ones(x.shape[0],1) * self.swin.semantic_weight    # 创建了一个形状为 (x.shape[0], 1) 的张量，其中每个元素都是1
            w = torch.cat([w, 1-w], axis=-1)                            # 两个张量 w 和 1-w 沿着列维度合并在一起，形成一个新的张量 w
            semantic_weight = w.cuda()                                  # 将张量移动到GPU
        x, hw_shape = self.swin.patch_embed(x)                          # 对输入特征 x 进行backbone特征提取，并将处理后的数据存储在 x：torch.Size([3, 87232, 96]) 中，同时记录数据的形状信息在 hw_shape (232, 376)变量中
        if self.swin.use_abs_pos_embed:                                 # 是否使用绝对位置编码
            x = x + self.swin.absolute_pos_embed                        #  x 与预先计算好的绝对位置编码相加
        x = self.swin.drop_after_pos(x)                                 # 丢弃（dropout）操作torch.Size([3, 87232, 96])


        outs = []
        for i, stage in enumerate(self.swin.stages[:bonenum]):          # 当前阶段的索引 i 和阶段对象 stage,索引从0到bonenum-1
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)         # 输入数据,数据形状的变量.stage 函数的输出,数据形状;x:torch.Size([3, 20304, 192])out:torch.Size([3, 5076, 384])
            if self.swin.semantic_weight >= 0:
                sw = self.swin.semantic_embed_w[i](semantic_weight).unsqueeze(1)  # 维度1上添加一个额外的维度;torch.Size([3, 1, 768])
                sb = self.swin.semantic_embed_b[i](semantic_weight).unsqueeze(1)  # torch.Size([3, 1, 768])
                x = x * self.swin.softplus(sw) + sb                      # softplus ReLU函数的平滑近似;torch.Size([3, 1269, 768])
            if i == bonenum-1:
                norm_layer = getattr(self.swin, f'norm{i}')              # 获取名称为 norm{i} 的属性，并将其存储在 norm_layer 中
                out = norm_layer(out)                                    # torch.Size([3, 5452, 384])
                # 规范化处理
                out = out.view(-1, *out_hw_shape,
                               self.swin.num_features[i]).permute(0, 3, 1,
                                                             2).contiguous()        # 对数据 out 进行形状重塑和维度重排，然后确保它在内存中是连续的torch.Size([3, 384, 58, 94])
                outs.append(out)                                            # list
        return OrderedDict([["feat_res4", outs[-1]]])                   # -1最后一个元素,返回字典类型

class Res5Head(nn.Sequential):
    def __init__(self, swin, out_channels=384):
        super().__init__()  # last block
        self.swin = swin
        self.out_channels = [out_channels, out_channels*2]

    def forward(self, x):
        if self.swin.semantic_weight >= 0:                                  # torch.Size([384, 384,14,14])
            w = torch.ones(x.shape[0],1) * self.swin.semantic_weight        # (batch_size, 1)
            w = torch.cat([w, 1-w], axis=-1)                                # (batch_size, 2)第一列是 w，第二列是 1-w,torch.Size([384, 1])
            semantic_weight = w.cuda()                                      # 将张量 w 移动到 GPU 上,torch.Size([384, 2]),[0.6600, 0.3400]]


        feat = x                    # feat.shape torch.Size([384, 384, 14, 14])
        hw_shape = x.shape[-2:]     # [-2:] 获取了形状信息的倒数第二和最后一个维度的大小,hw_shape torch.Size([14, 14]),hw_shape.shape <class 'torch.Size'>
        x = torch.flatten(x, 2)     # 将输入张量 x 在第2个维度上展平;torch.Size([384, 384, 196])
        x = x.permute(0, 2, 1)      # 重新排列;torch.Size([384, 196, 384])
        x,hw_shape = self.swin.stages[bonenum-1].downsample(x,hw_shape)     # 下采样;x torch.Size([384, 49, 768]);tuple hw_shape (7, 7)
        if self.swin.semantic_weight >= 0:
            sw = self.swin.semantic_embed_w[bonenum-1](semantic_weight).unsqueeze(1)        # unsqueeze(1)增加一个维度(batch_size, 1, ...)sw.shape torch.Size([384, 1, 768])
            sb = self.swin.semantic_embed_b[bonenum-1](semantic_weight).unsqueeze(1)        # sb.shape torch.Size([384, 1, 768])
            x = x * self.swin.softplus(sw) + sb                                             # torch.Size([384, 49, 768])
        for i, stage in enumerate(self.swin.stages[bonenum:]):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if self.swin.semantic_weight >= 0:
                sw = self.swin.semantic_embed_w[bonenum+i](semantic_weight).unsqueeze(1)        # 获取权重和偏置
                sb = self.swin.semantic_embed_b[bonenum+i](semantic_weight).unsqueeze(1)
                x = x * self.swin.softplus(sw) + sb
            if i == len(self.swin.stages) - bonenum - 1:                                        # 如果当前是最后一个 stage
                norm_layer = getattr(self.swin, f'norm{bonenum+i}')                             # 获取规范化层（norm_layer）
                out = norm_layer(out)                                                           # 对输出 out 进行规范化
                out = out.view(-1, *out_hw_shape,                                               # 重新形状 out
                               self.swin.num_features[bonenum+i]).permute(0, 3, 1,

                                                             2).contiguous()
        feat = self.swin.avgpool(feat)                  # 全局平均池化操作有助于减少参数数量，提取整体特征([384, 384, 14, 14])》torch.Size([384, 384, 1, 1])
        out = self.swin.avgpool(out)                    # torch.Size([384, 768, 7, 7])》torch.Size([384, 768, 1, 1])
        return OrderedDict([["feat_res4", feat], ["feat_res5", out]])

def build_swin(name="swin_tiny", semantic_weight=1.0):
    if 'tiny' in name:
        swin = swin_tiny_patch4_window7_224(drop_path_rate=0.1,semantic_weight=semantic_weight)         # 采用swin tiny的网络，丢弃通道的概率，语义权重
        out_channels = 384
    elif 'small' in name:
        swin = swin_small_patch4_window7_224(drop_path_rate=0.1,semantic_weight=semantic_weight)
        out_channels = 384
    elif 'base' in name:
        swin = swin_base_patch4_window7_224(drop_path_rate=0.1,semantic_weight=semantic_weight)
        out_channels = 512

    return Backbone(swin,out_channels), Res5Head(swin,out_channels), out_channels*2
