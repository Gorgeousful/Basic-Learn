# 实现了4种在线困难样本挖掘(OHEM)方法:
# 1. 基础版HardExampleMining:简单选取损失最大的k个样本
# 2. 标准版OHEM:继承nn.Module,支持前向传播计算损失
# 3. 高级版AdvancedOHEM:增加最小样本数限制和样本权重
# 4. 动态版DynamicOHEM:随训练进程动态调整采样比例
# 5. 平衡采样策略balanced_ohem_batch:保证每个类别都有足够样本被选中

import torch
import torch.nn as nn

class HardExampleMining:
    def __init__(self, model, k_ratio=0.7, loss_fn=nn.CrossEntropyLoss(reduction='none')):
        self.model = model
        self.k_ratio = k_ratio  # 保留的困难样本比例
        self.loss_fn = loss_fn
    
    def mine_hard_examples(self, data, labels, batch_size):
        # 计算每个样本的损失
        with torch.no_grad():
            outputs = self.model(data)
            losses = self.loss_fn(outputs, labels)
        
        # 计算要保留的样本数量
        k = int(batch_size * self.k_ratio)
        
        # 选择损失最大的k个样本
        _, indices = torch.topk(losses, k)
        
        # 返回困难样本
        return data[indices], labels[indices]
    
    def train_step(self, data, labels, optimizer):
        # 获取困难样本
        hard_data, hard_labels = self.mine_hard_examples(data, labels, len(data))
        
        # 训练模型
        optimizer.zero_grad()
        outputs = self.model(hard_data)
        loss = self.loss_fn(outputs, hard_labels).mean()
        loss.backward()
        optimizer.step()
        
        return loss.item()

class OHEM(nn.Module):
    def __init__(self, model, k_ratio=0.7, loss_fn=nn.CrossEntropyLoss(reduction='none')):
        super(OHEM, self).__init__()
        self.model = model
        self.k_ratio = k_ratio
        self.loss_fn = loss_fn
    
    def forward(self, features, targets):
        # 前向传播并计算每个样本的损失
        pred = self.model(features)
        losses = self.loss_fn(pred, targets)
        
        # 选择最困难的样本
        batch_size = features.shape[0]
        num_ohem = int(batch_size * self.k_ratio)
        
        # 获取最高损失的样本索引
        _, hard_indices = torch.topk(losses, num_ohem)
        
        # 只返回困难样本的平均损失
        return losses[hard_indices].mean()

class AdvancedOHEM(nn.Module):
    def __init__(self, model, k_ratio=0.7, min_samples=16, 
                 loss_fn=nn.CrossEntropyLoss(reduction='none')):
        super(AdvancedOHEM, self).__init__()
        self.model = model
        self.k_ratio = k_ratio
        self.min_samples = min_samples  # 最小保留样本数
        self.loss_fn = loss_fn
        
    def forward(self, features, targets, weights=None):
        batch_size = features.shape[0]
        
        # 计算预测和损失
        with torch.set_grad_enabled(True):
            pred = self.model(features)
            losses = self.loss_fn(pred, targets)
            
            if weights is not None:
                losses = losses * weights
            
            # 确定要保留的样本数量
            num_ohem = max(
                int(batch_size * self.k_ratio),
                self.min_samples
            )
            
            # 选择困难样本
            sorted_losses, indices = torch.sort(losses, descending=True)
            keep_losses = sorted_losses[:num_ohem]
            
            # 计算最终损失
            final_loss = keep_losses.mean()
            
            return final_loss, pred

class DynamicOHEM(OHEM):
    def __init__(self, model, initial_ratio=0.7, min_ratio=0.3):
        super().__init__(model)
        self.initial_ratio = initial_ratio
        self.min_ratio = min_ratio
        self.current_epoch = 0
        
    def update_ratio(self, epoch):
        # 随训练进程动态调整采样比例
        self.current_epoch = epoch
        self.k_ratio = max(
            self.initial_ratio * (1 - epoch/100),  # 线性衰减
            self.min_ratio
        )

def train_with_ohem(model, train_loader, epochs=10):
    ohem_loss = AdvancedOHEM(model)
    optimizer = torch.optim.Adam(model.parameters())
    
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # 计算OHEM损失
            loss, _ = ohem_loss(data, target)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 打印训练信息
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')

def balanced_ohem_batch(losses, labels, k_ratio=0.7, min_per_class=3):
    """
    平衡的在线困难样本挖掘（Balanced OHEM）
    
    原理：
    - 对每个类别分别进行困难样本挖掘
    - 确保每个类别都有足够的样本被选中
    - 避免某些类别的样本被完全忽略
    
    参数：
    - losses: 每个样本的损失值 [batch_size]
    - labels: 对应的标签 [batch_size]
    - k_ratio: 每个类别要保留的样本比例
    - min_per_class: 每个类别最少保留的样本数

    示例：
    losses = torch.tensor([0.5, 0.8, 0.2, 0.9, 0.3, 0.7])
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    selected = balanced_ohem_batch(losses, labels)

    结果分析：
    - 类别0的样本：[0.5, 0.8] -> 选择较大的损失值
    - 类别1的样本：[0.2, 0.9] -> 选择较大的损失值
    - 类别2的样本：[0.3, 0.7] -> 选择较大的损失值
每个类别都保证有样本被选中，避免了样本不平衡问题
    """
    
    # 获取数据集中的所有唯一标签
    unique_labels = torch.unique(labels)  # 例如：[0,1,2,3] 表示4个类别
    selected_indices = []
    
    for label in unique_labels:
        # 创建布尔掩码，标识属于当前类别的样本
        label_mask = (labels == label)  # 例如：[True, False, True, ...]
        
        # 获取当前类别的所有样本损失
        label_losses = losses[label_mask]  # 筛选出当前类别的损失值
        
        # 计算要选择的样本数量
        # 1. 根据比例计算：当前类别样本数 * k_ratio
        # 2. 确保不少于min_per_class个样本
        n_select = max(
            int(len(label_losses) * k_ratio),
            min_per_class
        )
        
        # 选择损失最大的n_select个样本
        # torch.topk返回(values, indices)，我们只需要indices
        _, indices = torch.topk(label_losses, 
                              k=min(n_select, len(label_losses)),  # 防止n_select大于样本数
                              largest=True)  # largest=True表示选择最大的值
        
        # 将选中的样本索引添加到结果列表中
        selected_indices.extend(indices)
    
    return selected_indices