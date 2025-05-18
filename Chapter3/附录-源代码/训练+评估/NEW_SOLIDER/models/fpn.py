# ./maskrcnn_benchmark/modeling/backbone/fpn.py

import torch
import torch.nn.functional as F
from torch import nn

class FPN(nn.Module):
    # 在一系列的 feature map (实际上就是stage2~5的最后一层输出)添加 FPN
    # 这些 feature maps 的 depth 假定是不断递增的, 并且 feature maps 必须是连续的(从stage角度)

    def __init__(self, in_channels_list, out_channels, top_blocks=None):
        # in_channels_list (list[int]): 指示了送入 fpn 的每个 feature map 的通道数
        # out_channels (int): FPN表征的通道数, 所有的特征图谱最终都会转换成这个通道数大小
        # top_blocks (nn.Module or None): 当提供了 top_blocks 时, 就会在 FPN 的最后
        # 的输出上进行一个额外的 op, 然后 result 会扩展成 result list 返回
        super(FPN, self).__init__()

        # 创建两个空列表
        self.inner_blocks = []
        self.layer_blocks = []

        # 假设我们使用的是 ResNet-50-FPN 和配置, 则 in_channels_list 的值为:
        # [256, 512, 1024, 2048]
        for idx, in_channels in enumerate(in_channels_list, 1): # 下标从1开始
            # 用下表起名: fpn_inner1, fpn_inner2, fpn_inner3, fpn_inner4
            inner_block = "fpn_inner{}".format(idx)

            # fpn_layer1, fpn_layer2, fpn_layer3, fpn_layer4
            layer_block = "fpn_layer{}".format(idx)

            # 创建 inner_block 模块, 这里 in_channels 为各个stage输出的通道数
            # out_channels 为 256, 定义在用户配置文件中
            # 这里的卷积核大小为1, 该卷积层主要作用为改变通道数到 out_channels(降维)
            inner_block_module = nn.Conv2d(in_channels, out_channels, 1)

            # 改变 channels 后, 在每一个 stage 的特征图谱上再进行 3×3 的卷积计算, 通道数不变
            layer_block_module = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

            for module in [inner_block_module, layer_block_module]:
                # Caffe2 的实现使用了 XavierFill,
                # 实际上相当于 PyTorch 中的 kaiming_uniform_
                nn.init.kaiming_uniform_(module.weight, a=1)
                nn.init.constant_(module.bias, 0)

            # 在当前的特征图谱上添加 FPN
            self.add_module(inner_block, inner_block_module) #name, module
            self.add_module(layer_block, layer_block_module)

            # 将当前 stage 的 fpn 模块的名字添加到对应的列表当中
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)

        # 将top_blocks作为 FPN 类的成员变量
        self.top_blocks = top_blocks

    def forward(self, x):
        # x (list[Tensor]): 每个 feature level 的 feature maps,
        # ResNet的计算结果正好满足 FPN 的输入要求, 也因此可以使用 nn.Sequential 将二者直接结合
        # results (tuple[Tensor]): 经过FPN后的特征图谱组成的列表, 排列顺序是高分辨率的在前

        # 先计算最后一层(分辨率最低)特征图谱的fpn结果.
        last_inner = getattr(self, self.inner_blocks[-1])(x[-1])

        # 创建一个空的结果列表
        results=[]

        # 将最后一层的计算结果添加到 results 中
        results.append(getattr(self, self.layer_blocks[-1])(last_inner))


        # [:-1] 获取了前三项, [::-1] 代表从头到尾切片, 步长为-1, 效果为列表逆置
        # 举例来说, zip里的操作 self.inner_block[:-1][::-1] 的运行结果为
        # [fpn_inner3, fpn_inner2, fpn_inner1], 相当于对列表进行了逆置
        for feature, inner_block, layer_block in zip(
            x[:-1][::-1], self.inner_block[:-1][::-1], self.layer_blocks[:-1][::-1]
        ):
            # 根据给定的scale参数对特征图谱进行放大/缩小, 这里scale=2, 所以是放大
            inner_top_down = F.interpolate(last_inner, scale_factor=2, mode="nearest")

            # 获取 inner_block 的计算结果
            inner_lateral = getattr(self, inner_block)(feature)

            # 将二者叠加, 作为当前stage的输出 同时作为下一个stage的输入
            last_inner = inner_lateral + inner_top_down

            # 将当前stage输出添加到结果列表中, 注意还要用 layer_block 执行卷积计算
            # 同时为了使得分辨率最大的在前, 我们需要将结果插入到0位置
            results.insert(0, getattr(self, layer_block)(last_inner))

        # 如果 top_blocks 不为空, 则执行这些额外op
        if self.top_blocks is not None:
            last_results = self.top_blocks(results[-1])
            results.extend(last_results) # 将新计算的结果追加到列表中

        # 以元组(只读)形式返回
        return tuple(results)

# 最后一级的 max pool 层
class LastLevelMaxPool(nn.Module):
    def forward(self, x):
        return [F.max_pool2d(x, 1, 2, 0)]
