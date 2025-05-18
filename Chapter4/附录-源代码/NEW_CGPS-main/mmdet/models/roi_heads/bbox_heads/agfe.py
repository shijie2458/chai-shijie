import torch
import torch.nn as nn


class AdaptiveGlobalFeatureExtractor(nn.Module):
    def __init__(self, channels, dim, norm_cfg=dict(type='BatchNorm2d', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='SiLU')):
        super(AdaptiveGlobalFeatureExtractor, self).__init__()

        # 修改 proj_1 的通道数，确保输入通道为 2048，输出通道为 128
        self.proj_1 = nn.Conv2d(2048, 128, 1)  # 修改这里，确保输入输出通道匹配
        self.act = nn.SiLU()

        # 通道数调整层：用于调整 shortcut 的通道数以匹配 x 的通道数
        self.channel_adjust = nn.Conv2d(2048, 128, 1)

        self.spatial_gating_unit = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128, bias=False),
            getattr(nn, norm_cfg['type'])(128, momentum=norm_cfg['momentum'], eps=norm_cfg['eps']),
            nn.SiLU(),
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False),
            getattr(nn, norm_cfg['type'])(128, momentum=norm_cfg['momentum'], eps=norm_cfg['eps']),
            nn.Sigmoid()
        )
        self.proj_2 = nn.Conv2d(128, 128, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0, bias=False),
            getattr(nn, norm_cfg['type'])(channels, momentum=norm_cfg['momentum'], eps=norm_cfg['eps']),
            nn.SiLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0, bias=False),
            getattr(nn, norm_cfg['type'])(channels, momentum=norm_cfg['momentum'], eps=norm_cfg['eps']),
            nn.SiLU()
        )
        self.attn_act = nn.Sigmoid()

        self.global_info_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, 1, 0, bias=False),
            getattr(nn, norm_cfg['type'])(channels, momentum=norm_cfg['momentum'], eps=norm_cfg['eps']),
            nn.SiLU()
        )

    def forward(self, x):
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.act(x)
        gating = self.spatial_gating_unit(x)
        x = x * gating
        x = self.proj_2(x)

        # 将 shortcut 的通道数调整为与 x 一致
        shortcut = self.channel_adjust(shortcut)

        x = x + shortcut  # 现在通道数一致，可以进行相加

        avg_x = self.avg_pool(x)
        max_x = self.max_pool(x)

        avg_x = self.conv1(avg_x)
        max_x = self.conv1(max_x)

        global_x = torch.cat([avg_x, max_x], dim=1)
        global_x = self.global_info_conv(global_x)

        attn_factor = self.attn_act(self.conv2(global_x))

        enhanced_x = x * attn_factor.expand_as(x)

        return enhanced_x


