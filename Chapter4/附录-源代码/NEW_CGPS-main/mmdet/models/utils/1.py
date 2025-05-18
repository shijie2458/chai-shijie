# 标签传播机制: 基于相似性矩阵传播标签
class HybridMemoryMultiFocalPercent(nn.Module):
    def __init__(self, num_features, num_memory, temp=0.05, momentum=0.2, top_percent=0.1, alpha=0.5):
        super(HybridMemoryMultiFocalPercent, self).__init__()
        self.num_features = num_features
        self.num_memory = num_memory
        self.momentum = momentum
        self.temp = temp
        self.top_percent = top_percent
        self.alpha = alpha  # 控制标签传播的权重

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
        # Normalize inputs:[71, 256]
        inputs = F.normalize(results, p=2, dim=1)
        print(f"[INFO] inputs.shape: {inputs.shape}")
        
        # Step 1: 计算相似性矩阵 (B * num_memory):[71, 55272]
        similarity_matrix = hm(inputs, indexes, self.features, self.momentum)
        print(f"[INFO] similarity_matrix.shape: {similarity_matrix.shape}")
        similarity_matrix /= self.temp
        
        # Step 2: 执行标签传播:[71]
        propagated_labels = self.label_propagation(similarity_matrix, indexes)
        print(f"[INFO] propagated_labels.shape: {propagated_labels.shape}")

        # Step 3: 使用传播后的标签进行目标计算:[71]
        targets = propagated_labels.long()
        print(f"[INFO] targets.shape: {targets.shape}")

        # Step 4: 计算损失 (基于传播后的标签)
        labels = self.labels.clone()  # N, 记录所有记忆样本的标签
        B = inputs.size(0)

         # 修复 sim 的维度问题;[49959, 90]
        num_clusters = labels.max().item() + 1  # 获取总的聚类数
        sim = torch.zeros(num_clusters, B).float().cuda()  # u * B
        print(f"[INFO] sim.shape: {sim.shape}")

        # 这里确保 labels 是正确的索引张量
        sim.scatter_add_(0, labels.unsqueeze(0).expand_as(inputs), inputs)
        # sim.index_add_(0, labels, inputs.t().contiguous())  # 聚合相似性
        nums = torch.zeros(num_clusters, 1).float().cuda()  # u * 1
        nums.index_add_(0, labels, torch.ones(self.num_memory, 1).float().cuda())  # 每个聚类样本的计数
        mask = (nums > 0).float()
        
        # 计算平均相似度
        sim /= (mask * nums + (1 - mask)).clone().expand_as(sim)
        mask = mask.expand_as(sim)

        # Step 5: 计算损失 (多焦点目标损失)
        masked_sim = self.masked_softmax_multi_focal(sim.t().contiguous(), mask.t().contiguous(), targets=targets)
        print(f"[INFO] masked_sim.shape: {masked_sim.shape}")
        
        return F.nll_loss(torch.log(masked_sim + 1e-6), targets)


    def label_propagation(self, similarity_matrix, indexes):
        """
        标签传播机制: 基于相似性矩阵传播标签。
        Args:
            similarity_matrix: 相似度矩阵 (B, num_memory)
            indexes: 当前批次的索引 (B,)
        Returns:
            final_labels: 传播后的标签 (B,)
        """
        # 标签扩展:[71, 55272]
        labels_expanded = self.labels.float().unsqueeze(1)  # (num_memory, 1)
        print(f"[INFO] labels_expanded.shape: {labels_expanded.shape}")

        # 标签传播计算 (B * num_memory) @ (num_memory * 1) -> (B, 1):[71, 1]
        propagated_labels = torch.matmul(similarity_matrix, labels_expanded) / similarity_matrix.sum(dim=1, keepdim=True)
        print(f"[INFO] propagated_labels.shape (before normalization): {propagated_labels.shape}")

        # 标签归一化:[71, 1]
        propagated_labels = propagated_labels / propagated_labels.sum(dim=1, keepdim=True)
        print(f"[INFO] propagated_labels.shape (after normalization): {propagated_labels.shape}")

        # 结合传播后的标签与内存标签;[71]
        final_labels = (1 - self.alpha) * self.labels[indexes].float() + self.alpha * propagated_labels.squeeze(1)
        print(f"[INFO] final_labels.shape: {final_labels.shape}")

        return final_labels


    def masked_softmax_multi_focal(self, vec, mask, dim=1, targets=None, epsilon=1e-6):
        """
        多焦点目标损失函数
        """
        exps = torch.exp(vec)
        one_hot_pos = torch.nn.functional.one_hot(targets, num_classes=exps.shape[1])
        one_hot_neg = one_hot_pos.new_ones(size=one_hot_pos.shape)
        one_hot_neg -= one_hot_pos

        masked_exps = exps * mask.float()
        neg_exps = exps.new_zeros(size=exps.shape)
        neg_exps[one_hot_neg > 0] = masked_exps[one_hot_neg > 0]
        ori_neg_exps = neg_exps.clone()
        neg_exps /= neg_exps.sum(dim=1, keepdim=True)

        new_exps = masked_exps.new_zeros(size=exps.shape)
        new_exps[one_hot_pos > 0] = masked_exps[one_hot_pos > 0]

        # 自适应选择负样本的最小值
        sorted, indices = torch.sort(neg_exps, dim=1, descending=True)
        sorted_cum_sum = torch.cumsum(sorted, dim=1)
        sorted_cum_diff = (sorted_cum_sum - self.top_percent).abs()
        sorted_cum_min_indices = sorted_cum_diff.argmin(dim=1)

        min_values = sorted[torch.arange(0, sorted.shape[0]).long(), sorted_cum_min_indices]
        min_values = min_values.unsqueeze(dim=-1) * ori_neg_exps.sum(dim=1, keepdim=True)
        ori_neg_sum = ori_neg_exps.sum(dim=1, keepdim=True)
        ori_neg_exps[ori_neg_exps < min_values] = 0

        new_exps[one_hot_neg > 0] = ori_neg_exps[one_hot_neg > 0]
        masked_exps = new_exps

        masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
        return masked_exps / masked_sums


