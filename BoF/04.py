# CutMix数据增强:
# 1. 随机裁剪两张图片的部分区域进行混合,标签按面积比例混合
# 2. 使用Beta分布控制混合程度:β小时混合极端,β=1均匀混合,β大时混合平均
# 3. 代码实现了CutMix方法并展示了不同β值对混合效果的影响

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def cutmix(image, label, num_class=10, beta=1.0, cutmix_prob=0.5):
    """
    CutMix数据增强方法
    
    原理：随机裁剪两张图片的一部分区域进行混合，标签按照面积比例进行混合
    
    参数:
        image (torch.Tensor): 输入图像 shape=(N, C, H, W)
        label (torch.Tensor): 输入标签 shape=(N, )
        num_class (int): 类别数量
        beta (float): Beta分布的参数，控制混合程度 越大，混合程度越小
        cutmix_prob (float): 执行cutmix的概率
        
    返回:
        mixed_image (torch.Tensor): 混合后的图像
        mixed_label (torch.Tensor): 混合后的one-hot标签
    """
    # 转换标签为one-hot编码
    label = F.one_hot(label, num_classes=num_class)
    
    # 随机决定是否执行cutmix
    if np.random.random() > cutmix_prob:
        return image, label
    
    # 获取batch大小
    batch_size = len(image)
    
    # 生成混合权重 (保留原图的比例)
    lam = np.random.beta(beta, beta)
    
    # 随机选择要混合的图像索引
    rand_index = torch.randperm(batch_size)
    
    # 计算裁剪框大小
    W = image.size()[2]
    H = image.size()[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    # 随机选择裁剪框中心点
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    # 确保裁剪框在图像范围内
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    # 执行图像混合
    mixed_image = image.clone()
    mixed_image[:, :, bbx1:bbx2, bby1:bby2] = \
        image[rand_index, :, bbx1:bbx2, bby1:bby2]
    
    # 计算实际的混合比例
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    # 混合标签
    mixed_label = lam * label + (1 - lam) * label[rand_index]
    
    return mixed_image, mixed_label

# ----------------------------------------------------------------
# 创建示例数据
images = torch.randn(32, 3, 224, 224)  # 32张图片的batch
labels = torch.randint(0, 10, (32,))    # 随机标签

# 应用CutMix
# mixed_images: 混合后的图像 shape=(32, 3, 224, 224)
# mixed_labels: 混合后的标签 shape=(32, 10) (one-hot编码)
mixed_images, mixed_labels = cutmix(
    image=images,
    label=labels,
    num_class=10,     # 类别数量
    beta=1.0,         # Beta分布参数
    cutmix_prob=0.9   # 执行CutMix的概率
)
# ----------------------------------------------------------------
# Beta分布是一种连续概率分布,定义在区间[0,1]上
# 其概率密度函数为:
# f(x; α, β) = (x^(α-1) * (1-x)^(β-1)) / B(α,β)
# 其中B(α,β)是Beta函数,用于归一化
#
# Beta分布的特点:
# 1. α=β时,分布关于x=0.5对称
# 2. α,β都大于1时,分布为单峰
# 3. α,β都小于1时,分布为U型
# 4. α=β=1时,退化为均匀分布
#
# 在CutMix中,Beta分布用于生成混合比例λ
# 不同的β值会产生不同的混合效果:
# - β小: 倾向于生成接近0或1的λ值,混合更极端
# - β=1: 均匀分布,混合比例均等
# - β大: 倾向于生成接近0.5的λ值,混合更平均

# 生成不同beta值的样本
n_samples = 10000
beta_values = [0.2, 1.0, 2.0]

plt.figure(figsize=(12, 4))
for i, beta in enumerate(beta_values):
    samples = np.random.beta(beta, beta, n_samples)
    plt.subplot(1, 3, i+1)
    plt.hist(samples, bins=50, density=True)
    plt.title(f'Beta({beta}, {beta})')
    plt.xlabel('λ值')
    plt.ylabel('密度')

plt.tight_layout()
plt.show()