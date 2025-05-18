# Ge et al. Self-paced Contrastive Learning with Hybrid Memory for Domain Adaptive Object Re-ID.  # noqa
# Written by Yixiao Ge.

import torch
import torch.nn.functional as F
from torch import autograd, nn

from mmdet.utils import all_gather_tensor

try:
    # PyTorch >= 1.6 supports mixed precision training
    from torch.cuda.amp import custom_fwd, custom_bwd
    class HM(autograd.Function):

        @staticmethod
        @custom_fwd(cast_inputs=torch.float32)
        def forward(ctx, inputs, indexes, features, momentum):
            ctx.features = features
            ctx.momentum = momentum
            outputs = inputs.mm(ctx.features.t())
            all_inputs = all_gather_tensor(inputs)
            all_indexes = all_gather_tensor(indexes)
            ctx.save_for_backward(all_inputs, all_indexes)
            return outputs

        @staticmethod
        @custom_bwd
        def backward(ctx, grad_outputs):
            inputs, indexes = ctx.saved_tensors
            grad_inputs = None
            if ctx.needs_input_grad[0]:
                grad_inputs = grad_outputs.mm(ctx.features)

            # momentum update
            for x, y in zip(inputs, indexes):
                ctx.features[y] = ctx.momentum * ctx.features[y] + (1.0 - ctx.momentum) * x
                ctx.features[y] /= ctx.features[y].norm()

            return grad_inputs, None, None, None
except:
    class HM(autograd.Function):

        @staticmethod
        def forward(ctx, inputs, indexes, features, momentum):
            ctx.features = features
            ctx.momentum = momentum
            outputs = inputs.mm(ctx.features.t())
            all_inputs = all_gather_tensor(inputs)
            all_indexes = all_gather_tensor(indexes)
            ctx.save_for_backward(all_inputs, all_indexes)
            return outputs

        @staticmethod
        def backward(ctx, grad_outputs):
            inputs, indexes = ctx.saved_tensors
            grad_inputs = None
            if ctx.needs_input_grad[0]:
                grad_inputs = grad_outputs.mm(ctx.features)

            # momentum update
            for x, y in zip(inputs, indexes):
                ctx.features[y] = ctx.momentum * ctx.features[y] + (1.0 - ctx.momentum) * x
                ctx.features[y] /= ctx.features[y].norm()

            return grad_inputs, None, None, None


def hm(inputs, indexes, features, momentum=0.5):
    return HM.apply(
        inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device)
    )

class HybridMemory(nn.Module):
    def __init__(self, num_features, num_memory, temp=0.05, momentum=0.2):
        super(HybridMemory, self).__init__()
        self.num_features = num_features
        self.num_memory = num_memory

        self.momentum = momentum
        self.temp = temp

        self.idx = torch.zeros(num_memory).long()

        self.register_buffer("features", torch.zeros(num_memory, num_features))
        self.register_buffer("labels", torch.zeros(num_memory).long())
    
    @torch.no_grad()
    def _init_ids(self, ids):
        self.idx.data.copy_(ids.long().to(self.labels.device))

    @torch.no_grad()
    def _update_feature(self, features):
        features = F.normalize(features, p=2, dim=1)
        self.features.data.copy_(features.float().to(self.features.device))

    @torch.no_grad()
    def _update_label(self, labels):
        self.labels.data.copy_(labels.long().to(self.labels.device))
    
    @torch.no_grad()
    def get_cluster_ids(self, indexes):
        return self.labels[indexes].clone()

    def forward(self, results, indexes):
        inputs = results
        inputs = F.normalize(inputs, p=2, dim=1)

        # inputs: B*2048, features: N*2048
        inputs = hm(inputs, indexes, self.features, self.momentum) #B*N, similarity
        inputs /= self.temp
        B = inputs.size(0)

        def masked_softmax(vec, mask, dim=1, epsilon=1e-6):
            exps = torch.exp(vec)
            masked_exps = exps * mask.float().clone()
            masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
            return masked_exps / masked_sums

        targets = self.labels[indexes].clone()
        labels = self.labels.clone() #shape: N, unique label num: u

        sim = torch.zeros(labels.max() + 1, B).float().cuda() #u*B
        sim.index_add_(0, labels, inputs.t().contiguous()) #
        nums = torch.zeros(labels.max() + 1, 1).float().cuda() #many instances belong to a cluster, so calculate the number of instances in a cluster
        nums.index_add_(0, labels, torch.ones(self.num_memory, 1).float().cuda()) #u*1
        mask = (nums > 0).float()
        sim /= (mask * nums + (1 - mask)).clone().expand_as(sim) #average features in each cluster, u*B
        mask = mask.expand_as(sim)
        masked_sim = masked_softmax(sim.t().contiguous(), mask.t().contiguous()) #sim: u*B, mask:u*B, masked_sim: B*u
        return F.nll_loss(torch.log(masked_sim + 1e-6), targets)


try:
    # PyTorch >= 1.6 supports mixed precision training
    from torch.cuda.amp import custom_fwd, custom_bwd
    class HMUniqueUpdate(autograd.Function):

        @staticmethod
        @custom_fwd(cast_inputs=torch.float32)
        def forward(ctx, inputs, indexes, features, momentum):
            ctx.features = features
            ctx.momentum = momentum
            outputs = inputs.mm(ctx.features.t())
            all_inputs = all_gather_tensor(inputs)
            all_indexes = all_gather_tensor(indexes)
            ctx.save_for_backward(all_inputs, all_indexes)
            return outputs

        @staticmethod
        @custom_bwd
        def backward(ctx, grad_outputs):
            inputs, indexes = ctx.saved_tensors
            grad_inputs = None
            if ctx.needs_input_grad[0]:
                grad_inputs = grad_outputs.mm(ctx.features)

            # momentum update
            unique = set()
            for x, y in zip(inputs, indexes):
                if y.item() in unique:
                    continue
                else:
                    unique.add(y.item())
                ctx.features[y] = ctx.momentum * ctx.features[y] + (1.0 - ctx.momentum) * x
                ctx.features[y] /= ctx.features[y].norm()

            return grad_inputs, None, None, None
except:
    class HMUniqueUpdate(autograd.Function):

        @staticmethod
        def forward(ctx, inputs, indexes, features, momentum):
            ctx.features = features
            ctx.momentum = momentum
            outputs = inputs.mm(ctx.features.t())
            all_inputs = all_gather_tensor(inputs)
            all_indexes = all_gather_tensor(indexes)
            ctx.save_for_backward(all_inputs, all_indexes)
            return outputs

        @staticmethod
        def backward(ctx, grad_outputs):
            inputs, indexes = ctx.saved_tensors
            grad_inputs = None
            if ctx.needs_input_grad[0]:
                grad_inputs = grad_outputs.mm(ctx.features)

            # momentum update
            unique = set()
            for x, y in zip(inputs, indexes):
                if y.item() in unique:
                    continue
                else:
                    unique.add(y.item())
                ctx.features[y] = ctx.momentum * ctx.features[y] + (1.0 - ctx.momentum) * x
                ctx.features[y] /= ctx.features[y].norm()

            return grad_inputs, None, None, None


def hmuniqueupdate(inputs, indexes, features, momentum=0.5):
    return HMUniqueUpdate.apply(
        inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device)
    )

# 混合记忆的多焦点注意力,处理类间特征聚类，并使用了一个类似于对比学习（contrastive learning）的方法。该模型的目的是通过特征存储和聚类机制提升分类任务的性能
class LabelPropagation(nn.Module):
    def __init__(self, num_classes, num_neighbors=10, alpha=0.1, max_iter=100):
        super(LabelPropagation, self).__init__()
        self.num_classes = num_classes
        self.num_neighbors = num_neighbors  # 每个样本考虑的邻居数量
        self.alpha = alpha  # 标签传播的步长
        self.max_iter = max_iter  # 标签传播的最大迭代次数

    def forward(self, features, labels, mask):
        """
        features: 输入的特征矩阵 (B, D)
        labels: 当前标签 (B,)
        mask: 样本掩码 (B,) 0 表示未标记样本，1 表示已标记样本
        """
        # 构建相似性图
        similarity_matrix = self.build_similarity_graph(features)

        # 初始化标签（未标记的样本标签为0，已标记样本保留标签）
        propagated_labels = labels.clone().float()
        propagated_labels = propagated_labels * mask.float()  # 只传播未标记样本的标签

        for _ in range(self.max_iter):
            # 通过相似性矩阵传播标签
            propagated_labels = (1 - self.alpha) * propagated_labels + self.alpha * torch.matmul(similarity_matrix, propagated_labels)

            # 确保已标记样本的标签保持不变
            propagated_labels = propagated_labels * mask.float() + labels.float() * (1 - mask.float())

        # 将传播后的标签转为最终预测标签（使用最大概率标签）
        predicted_labels = propagated_labels.argmax(dim=1)
        
        return predicted_labels

    def build_similarity_graph(self, features):
        """
        构建样本间的相似性图。
        :param features: 输入特征矩阵 (B, D)
        :return: 相似性矩阵
        """
        similarity_matrix = torch.matmul(features, features.T)
        similarity_matrix = similarity_matrix / torch.norm(features, p=2, dim=1).view(-1, 1)  # 归一化
        return similarity_matrix


class HybridMemoryMultiFocalPercent(nn.Module):
    def __init__(self, num_features, num_memory, temp=0.05, momentum=0.2, top_percent=0.1, num_classes=80):
        super(HybridMemoryMultiFocalPercent, self).__init__()
        self.num_features = num_features
        self.num_memory = num_memory
        self.momentum = momentum
        self.temp = temp
        self.top_percent = top_percent
        self.num_classes = num_classes  # 类别数

        self.idx = torch.zeros(num_memory).long()

        self.register_buffer("features", torch.zeros(num_memory, num_features))
        self.register_buffer("labels", torch.zeros(num_memory).long())

        # 标签传播模块
        self.label_propagation = LabelPropagation(num_classes)

    @torch.no_grad()
    def _init_ids(self, ids):
        self.idx.data.copy_(ids.long().to(self.labels.device))

    @torch.no_grad()
    def _update_feature(self, features):
        features = F.normalize(features, p=2, dim=1)
        self.features.data.copy_(features.float().to(self.features.device))

    @torch.no_grad()
    def _update_label(self, labels):
        self.labels.data.copy_(labels.long().to(self.labels.device))

    @torch.no_grad()
    def get_cluster_ids(self, indexes):
        return self.labels[indexes].clone()

    def forward(self, results, indexes):
        print(f"[INFO] results.shape: {results.shape}")
        # 打印 targets 的形状，提前打印出 targets:targets.shape: torch.Size([14])
        targets = self.labels[indexes].clone()
        print(f"[INFO] targets.shape: {targets.shape}")
        
        # 将 results 赋值给 inputs; inputs.shape: torch.Size([14, 256])
        inputs = results
        print(f"[INFO] inputs.shape: {inputs.shape}")  # 打印 inputs 的形状
        
        # 对 inputs 进行 L2 归一化, torch.Size([14, 256])
        inputs = F.normalize(inputs, p=2, dim=1)
        print(f"[INFO] inputs after normalization.shape: {inputs.shape}")  # 打印归一化后的 inputs 形状

        # 计算相似度矩阵 torch.Size([14, 18048])
        inputs = hm(inputs, indexes, self.features, self.momentum)  # B*N, similarity
        print(f"[INFO] inputs after hm (similarity).shape: {inputs.shape}")  # 打印通过 hm 得到的相似度后的 inputs 形状
        
        # 对相似度进行温度归一化; torch.Size([14, 18048])
        inputs /= self.temp
        print(f"[INFO] inputs after temperature scaling.shape: {inputs.shape}")  # 打印归一化后的相似度

        B = inputs.size(0)  # 获取 B 的大小;14
        print(f"[INFO] B (batch size): {B}")  # 打印 B 的大小
        
        # 对输入特征执行标签传播（更新标签）
        mask = (targets != -1).float()  # 假设标签为 -1 的样本是未标记样本
        propagated_labels = self.label_propagation(inputs, targets, mask)
        print(f"[INFO] propagated_labels.shape: {propagated_labels.shape}")  # 打印传播后的标签形状

        # 使用多焦点策略计算加权 softmax
        def masked_softmax_multi_focal(vec, mask, dim=1, targets=None, epsilon=1e-6):
            exps = torch.exp(vec)
            one_hot_pos = torch.nn.functional.one_hot(targets, num_classes=exps.shape[1])
            one_hot_neg = one_hot_pos.new_ones(size=one_hot_pos.shape)
            one_hot_neg = one_hot_neg - one_hot_pos
            masked_exps = exps * mask.float().clone()
            neg_exps = exps.new_zeros(size=exps.shape)
            neg_exps[one_hot_neg > 0] = masked_exps[one_hot_neg > 0]
            ori_neg_exps = neg_exps
            neg_exps = neg_exps / neg_exps.sum(dim=1, keepdim=True)
            new_exps = masked_exps.new_zeros(size=exps.shape)
            new_exps[one_hot_pos > 0] = masked_exps[one_hot_pos > 0]

            sorted, indices = torch.sort(neg_exps, dim=1, descending=True)
            sorted_cum_sum = torch.cumsum(sorted, dim=1)
            sorted_cum_diff = (sorted_cum_sum - self.top_percent).abs()
            sorted_cum_min_indices = sorted_cum_diff.argmin(dim=1)
            min_values = sorted[torch.range(0, sorted.shape[0] - 1).long(), sorted_cum_min_indices]
            min_values = min_values.unsqueeze(dim=-1) * ori_neg_exps.sum(dim=1, keepdim=True)
            ori_neg_sum = ori_neg_exps.sum(dim=1, keepdim=True)
            ori_neg_exps[ori_neg_exps < min_values] = 0

            new_exps[one_hot_neg > 0] = ori_neg_exps[one_hot_neg > 0]
            masked_exps = new_exps

            masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
            return masked_exps / masked_sums

        sim = torch.zeros(self.num_classes, B).float().cuda()  # 类别数 * B
        sim.index_add_(0, propagated_labels, inputs.t().contiguous())
        nums = torch.zeros(self.num_classes, 1).float().cuda()
        nums.index_add_(0, propagated_labels, torch.ones(self.num_memory, 1).float().cuda())
        mask = (nums > 0).float()
        sim /= (mask * nums + (1 - mask)).clone().expand_as(sim)
        mask = mask.expand_as(sim)
        masked_sim = masked_softmax_multi_focal(sim.t().contiguous(), mask.t().contiguous(), targets=propagated_labels)
        return F.nll_loss(torch.log(masked_sim + 1e-6), targets)



