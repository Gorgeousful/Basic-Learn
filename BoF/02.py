import numpy as np

def dropout(feature_map, p=0.5, training=True):
    """
    对特征图进行Dropout操作
    
    原理：
    1. 在训练时，随机"丢弃"特征图中的一些神经元（设置为0）
    2. 对剩余神经元进行scale up（除以保留概率），以保持期望值不变
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
    
    # 生成随机mask
    mask = np.random.binomial(1, 1-p, feature_map.shape)
    
    # 对保留的值进行scale up
    output = feature_map * mask / (1-p)
    
    return output

# 使用示例
def example_usage():
    """
    示例：如何在卷积神经网络中使用Dropout
    """
    # 假设有一个特征图
    feature_map = np.random.randn(1, 64, 32, 32)  # (batch_size, channels, height, width)
    
    # 训练时使用dropout
    train_output = dropout(feature_map, p=0.5, training=True)
    
    # 测试时不使用dropout
    test_output = dropout(feature_map, p=0.5, training=False)
