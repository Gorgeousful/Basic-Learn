import torch
import torch.nn.functional as F

def label_smoothing(labels, num_classes, epsilon=0.1):
    """
    标签平滑函数
    
    原理:
    1. 将one-hot标签转换为soft标签,避免模型过于自信
    2. 标签值从0/1变为ε/(K-1)和(1-ε),其中K是类别数,ε是平滑参数 
    3. 可以缓解过拟合,提高模型泛化能力
    
    参数:
        labels (torch.Tensor): 原始标签 shape=(N,) 或 (N,C)
        num_classes (int): 类别总数
        epsilon (float): 平滑参数,控制软化程度,一般取0.1
        
    返回:
        smoothed_labels (torch.Tensor): 平滑后的标签 shape=(N,C)
    """
    # 如果输入是普通标签,转换为one-hot编码
    if len(labels.shape) == 1:
        labels = F.one_hot(labels, num_classes=num_classes)
        
    # 计算平滑后的标签值
    smoothed_labels = (1 - epsilon) * labels + \
                     epsilon / num_classes
                     
    return smoothed_labels

def test_label_smoothing():
    """测试标签平滑效果"""
    # 创建测试数据
    batch_size = 4
    num_classes = 3
    labels = torch.tensor([0, 1, 2, 1])
    
    print("原始标签:", labels)
    
    # 测试不同的平滑参数
    for epsilon in [0.1, 0.2]:
        print(f"\n使用 epsilon={epsilon}:")
        smoothed = label_smoothing(
            labels,
            num_classes=num_classes,
            epsilon=epsilon
        )
        print("平滑后的标签:\n", smoothed)
        
        # 验证每行和为1
        print("每行和:", smoothed.sum(dim=1))
        
        # 验证最大值和最小值
        print("最大值:", smoothed.max())
        print("最小值:", smoothed.min())

if __name__ == "__main__":
    test_label_smoothing()
