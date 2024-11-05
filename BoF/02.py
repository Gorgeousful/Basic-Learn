# 实现了三种Dropout变体:
# 1. Dropout: 随机丢弃单个神经元
# 2. DropConnect: 随机丢弃权重连接
# 3. DropBlock: 丢弃连续区域
# 包含完整的实现、测试和可视化代码

import numpy as np
import matplotlib.pyplot as plt

def dropout(feature_map, p=0.5, training=True):
    """
    对特征图进行Dropout操作
    
    原理：
    1. 在训练时，随机"丢弃"特征图中的一些神经元（设置为0）
    2. 对剩余神经元进行scale up（除以保留概率），以保持期望值不变 !!!
    3. 在测试时，不进行dropout，直接返回原特征图
    
    实现要点：
    - 训练时：每个神经元以概率p被置0，其余乘以1/(1-p)
    - 测试时：直接返回输入特征图
    - 这种机制可以防止神经元的共适应，减少过拟合
    
    参数：
        feature_map (numpy.ndarray): 输入特征图，形状为(N,C,H,W)
        p (float): dropout概率，范围[0,1]，表示被置0的概率
        training (bool): 是否处于训练模式
    
    返回：
        numpy.ndarray: dropout后的特征图
    """
    if not training:
        return feature_map
    
    # 生成随机mask 二项式分布
    mask = np.random.binomial(1, 1-p, feature_map.shape)
    
    # 对保留的值进行scale up
    output = feature_map * mask / (1-p)
    
    return output

def dropconnect(weights, p=0.5, training=True):
    """
    对权重矩阵进行DropConnect操作
    
    原理：
    1. 在训练时，随机"丢弃"权重矩阵中的连接（设置为0）
    2. 对保留的权重进行scale up，以保持期望值不变
    3. 在测试时，不进行dropconnect，直接返回原权重矩阵
    
    参数：
        weights (numpy.ndarray): 输入权重矩阵
        p (float): dropconnect概率，范围[0,1]
        training (bool): 是否处于训练模式
    
    返回：
        numpy.ndarray: dropconnect后的权重矩阵
    """
    if not training:
        return weights
    
    # 生成随机mask
    mask = np.random.binomial(1, 1-p, weights.shape)
    
    # 对保留的权重进行scale up
    output = weights * mask / (1-p)
    
    return output

def dropblock(feature_map, block_size=7, p=0.5, training=True):
    """
    对特征图进行DropBlock操作
    
    原理：
    1. 不同于Dropout随机丢弃单个神经元，DropBlock丢弃(概率p，但是p需要变化为gamma)连续的区域
    2. 对于每个特征图：
       - 随机选择中心点
       - 将以该点为中心的block_size×block_size区域置0
    3. 同样需要进行scale up以保持期望值不变
    
    参数：
        feature_map: shape (N, C, H, W)
        block_size: 要丢弃的块的大小
        p: 丢弃概率
        training: 是否在训练模式
    """
    if not training:
        return feature_map
    
    # 获取特征图的形状
    N, C, H, W = feature_map.shape
    
    # 计算gamma（需要丢弃的中心点概率）
    # 由于每个点会影响block_size×block_size的区域，需要调整概率
    # 设 gamma 为选择每个中心点的概率
    # # 则：选中的中心点数量 × 每个block的像素数 = 目标丢弃的像素数
    # gamma * valid_centers * pixels_per_block = H * W * p
    # # 解出 gamma
    # gamma = (H * W * p) / (pixels_per_block * valid_centers)
    # # 即
    # gamma = p * (H * W) / (block_size ** 2) / ((H - block_size + 1) * (W - block_size + 1))
    gamma = p * (H * W) / (block_size ** 2) / ((H - block_size + 1) * (W - block_size + 1))
    
    # 生成mask（初始全1）
    mask = np.ones_like(feature_map)
    
    for n in range(N):
        for c in range(C):
            # 随机生成中心点mask
            center_mask = np.random.binomial(1, gamma, (H-block_size+1, W-block_size+1))
            
            # 对每个被选中的中心点，将其周围block_size×block_size的区域置0
            for i in range(H-block_size+1):
                for j in range(W-block_size+1):
                    if center_mask[i,j]:
                        mask[n,c,i:i+block_size,j:j+block_size] = 0
    
    # 应用mask并scale
    output = feature_map * mask / (1-p)
    
    return output

# 使用示例
def test_dropout():
    """
    示例：如何在卷积神经网络中使用Dropout
    """
    # 假设有一个特征图
    feature_map = np.random.randn(1, 64, 32, 32)  # (batch_size, channels, height, width)
    
    # 训练时使用dropout
    train_output = dropout(feature_map, p=0.5, training=True)
    
    # 测试时不使用dropout
    test_output = dropout(feature_map, p=0.5, training=False)

def test_dropconnect():
    """
    示例：如何使用DropConnect
    """
    # 假设有一个权重矩阵
    weights = np.random.randn(512, 256)  # (输出特征数, 输入特征数)
    
    # 训练时使用dropconnect
    train_weights = dropconnect(weights, p=0.5, training=True)
    
    # 测试时不使用dropconnect
    test_weights = dropconnect(weights, p=0.5, training=False)

def test_dropblock():
    """
    测试DropBlock的效果
    包括：
    1. 基本功能测试
    2. 不同block_size的效果对比
    3. 可视化dropout的mask
    """
    # 1. 创建测试数据
    feature_map = np.ones((1, 1, 32, 32))  # 创建一个简单的特征图
    
    # 2. 测试不同的block_size
    block_sizes = [3, 5, 7]
    p = 0.5
    
    # 创建子图
    fig, axes = plt.subplots(1, len(block_sizes), figsize=(15, 5))
    
    for idx, block_size in enumerate(block_sizes):
        # 应用dropblock
        output = dropblock(feature_map, block_size=block_size, p=p, training=True)
        
        # 计算实际丢弃率
        actual_drop_rate = 1.0 - np.count_nonzero(output) / output.size
        
        # 可视化
        axes[idx].imshow(output[0, 0], cmap='gray')
        axes[idx].set_title(f'Block Size: {block_size}\nDrop Rate: {actual_drop_rate:.3f}')
        axes[idx].axis('off')
    
    plt.suptitle('DropBlock with Different Block Sizes (p=0.5)')
    plt.tight_layout()
    plt.show()
    
    # 3. 打印一些统计信息
    print("\nDropBlock Statistics:")
    print("-" * 50)
    for block_size in block_sizes:
        # 多次测试求平均
        drop_rates = []
        for _ in range(100):
            output = dropblock(feature_map, block_size=block_size, p=p, training=True)
            drop_rate = 1.0 - np.count_nonzero(output) / output.size
            drop_rates.append(drop_rate)
        
        avg_drop_rate = np.mean(drop_rates)
        std_drop_rate = np.std(drop_rates)
        
        print(f"Block Size: {block_size}")
        print(f"Average Drop Rate: {avg_drop_rate:.3f} ± {std_drop_rate:.3f}")
        print(f"Target Drop Rate: {p}")
        print("-" * 50)
    
    # 4. 测试训练模式和测试模式
    test_output = dropblock(feature_map, block_size=7, p=p, training=False)
    print("\nTest Mode Output:")
    print(f"All ones (expected): {np.allclose(test_output, feature_map)}")

if __name__ == "__main__":
    test_dropout()
    test_dropconnect()
    test_dropblock()
