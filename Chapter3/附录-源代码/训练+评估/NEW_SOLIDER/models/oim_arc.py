import torch
import torch.nn.functional as F
from torch import autograd, nn
import math
# from utils.distributed import tensor_gather


class OIM(autograd.Function):
    @staticmethod
    def forward(ctx, inputs, targets, lut, cq, header, momentum):
        ctx.save_for_backward(inputs, targets, lut, cq, header, momentum)
        outputs_labeled = inputs.mm(lut.t())#输入minibatch和查找表做矩阵乘法
        outputs_unlabeled = inputs.mm(cq.t())
        return torch.cat([outputs_labeled, outputs_unlabeled], dim=1)

    @staticmethod
    def backward(ctx, grad_outputs):#更新梯度
        inputs, targets, lut, cq, header, momentum = ctx.saved_tensors

        # inputs, targets = tensor_gather((inputs, targets))

        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_outputs = grad_outputs.to(torch.half)
            lutt = lut.to(torch.half)
            cqq = cq.to(torch.half)
            grad_inputs = grad_outputs.mm(torch.cat([lutt, cqq], dim=0))
            if grad_inputs.dtype == torch.float16:
                grad_inputs = grad_inputs.to(torch.float32)

        for x, y in zip(inputs, targets):
            if y < len(lut):
                lut[y] = momentum * lut[y] + (1.0 - momentum) * x
                lut[y] /= lut[y].norm()
            else:
                cq[header] = x
                header = (header + 1) % cq.size(0)
        return grad_inputs, None, None, None, None, None


def oim(inputs, targets, lut, cq, header, momentum=0.5):
    return OIM.apply(inputs, targets, lut, cq, torch.tensor(header), torch.tensor(momentum))


class OIMLoss(nn.Module):
    def __init__(self, num_features, num_pids, num_cq_size, oim_momentum, oim_scalar):#特征数量
        super(OIMLoss, self).__init__()
        self.num_features = num_features
        self.num_pids = num_pids
        self.num_unlabeled = num_cq_size
        self.momentum = oim_momentum
        self.oim_scalar = oim_scalar

        self.register_buffer("lut", torch.zeros(self.num_pids, self.num_features))
        self.register_buffer("cq", torch.zeros(self.num_unlabeled, self.num_features))

        self.header_cq = 0
        self.s = 30.0
        self.m = 0.50
        self.arcface_loss_weight = 0.50
        self.arcface = ArcFace()

    def forward(self, inputs, roi_label):
        """
        inputs:输入特征
        roi_label:ROI标签
        """
        # merge into one batch, background label = 0
        targets = torch.cat(roi_label)
        label = targets - 1  # background label = -1

        inds = label >= 0
        label = label[inds]#筛选有效标签
        inputs = inputs[inds.unsqueeze(1).expand_as(inputs)].view(-1, self.num_features)#筛选有效标签对应的特征


        projected = oim(inputs, label, self.lut, self.cq, self.header_cq, momentum=self.momentum)
        arc_outputs = self.arcface(projected, label)

        projected *= self.oim_scalar#缩放处理

        self.header_cq = (
            self.header_cq + (label >= self.num_pids).long().sum().item()
        ) % self.num_unlabeled
        loss_oim = F.cross_entropy(projected, label, ignore_index=5554)

        loss_arcface = F.cross_entropy(arc_outputs, label, ignore_index=5554)
        loss = self.arcface_loss_weight * loss_arcface + (1 - self.arcface_loss_weight) * loss_oim
        return loss, inputs, label

class ArcFace(nn.Module):
    def init(self, in_features=256, out_features=482, s=30.0, m=0.50):
        super(ArcFace, self).init()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, projected, labels):
        # cos_theta = F.linear(F.normalize(inputs), F.normalize(self.weight))
        phi_theta = projected - self.m
        index = torch.where(labels >= 0)[0]
        output = torch.zeros_like(projected)
        if len(index) > 0:
            output[index] = projected[index] - self.s * phi_theta[index]
        return output
