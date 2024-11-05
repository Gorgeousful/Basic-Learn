# MixUp数据增强方法:
# 1. 对两张图片进行线性插值混合,标签也按同样比例混合
# 2. 使用Beta分布控制混合程度:α小时混合极端,α=1均匀混合,α大时混合平均
# 3. 代码实现了MixUp方法并展示了不同α值对混合效果的影响

import numpy as np
import torch
import torch.nn.functional as F

def mixup(image, label, num_class=10, alpha=1.0, mixup_prob=0.5):
    """
    MixUp数据增强方法
    
    参数:
        image: 输入图像 shape=(N,C,H,W) 
        label: 输入标签 shape=(N,)
        num_class: 类别数
        alpha: Beta分布参数
        mixup_prob: 执行mixup的概率
    """
    # 转one-hot标签
    label = F.one_hot(label, num_classes=num_class)
    
    # 随机决定是否执行mixup
    if np.random.random() > mixup_prob:
        return image, label
        
    # 生成混合权重
    lam = np.random.beta(alpha, alpha)
    
    # 随机选择要混合的图像
    batch_size = len(image)
    rand_index = torch.randperm(batch_size)
    
    # 线性插值混合图像
    mixed_image = lam * image + (1 - lam) * image[rand_index]
    
    # 混合标签
    mixed_label = lam * label + (1 - lam) * label[rand_index]
    
    return mixed_image, mixed_label 

def test_mixup():
    """测试MixUp数据增强方法"""
    # 创建测试数据
    batch_size = 4
    channels = 3
    height = 32
    width = 32
    num_class = 10
    
    # 随机生成测试图像和标签
    test_images = torch.randn(batch_size, channels, height, width)
    test_labels = torch.randint(0, num_class, (batch_size,))
    
    print("原始图像形状:", test_images.shape)
    print("原始标签:", test_labels)
    
    # 测试不同alpha值
    for alpha in [0.2, 1.0, 4.0]:
        print(f"\n测试 alpha={alpha}:")
        mixed_images, mixed_labels = mixup(
            test_images, 
            test_labels,
            num_class=num_class,
            alpha=alpha,
            mixup_prob=1.0  # 确保执行mixup
        )
        
        print("混合后图像形状:", mixed_images.shape)
        print("混合后标签形状:", mixed_labels.shape)
        print("混合后标签和为1:", torch.allclose(mixed_labels.sum(dim=1), 
                                          torch.ones(batch_size)))

if __name__ == "__main__":
    test_mixup()