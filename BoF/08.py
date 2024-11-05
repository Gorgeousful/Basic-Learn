import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def plot_boxes_comparison():
    """可视化不同IOU损失函数的特点和细节"""
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  
    plt.rcParams['axes.unicode_minus'] = False  

    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    fig.suptitle('IOU系列损失函数详细对比', fontsize=16)
    
    # 1. 基础IOU
    ax = axes[0, 0]
    ax.set_title('传统IOU损失')
    # 绘制目标框和预测框
    target_box = Rectangle((0.3, 0.3), 0.4, 0.4, fill=False, color='blue', label='目标框')
    pred_box = Rectangle((0.5, 0.5), 0.4, 0.4, fill=False, color='red', label='预测框')
    ax.add_patch(target_box)
    ax.add_patch(pred_box)
    # 标注交并集区域
    ax.fill([0.5, 0.7, 0.7, 0.5], [0.5, 0.5, 0.7, 0.7], alpha=0.3, color='green', label='交集区域')
    ax.fill([0.3, 0.7, 0.7, 0.5, 0.5, 0.3], 
            [0.3, 0.3, 0.9, 0.9, 0.7, 0.7], alpha=0.2, color='orange', label='并集区域')
    ax.text(0.1, 0.95, 'IOU = 交集面积/并集面积', fontsize=10)
    ax.text(0.1, 0.9, '优点:简单直观', fontsize=10)
    ax.text(0.1, 0.85, '缺点:当框不相交时梯度为0', color='red', fontsize=10)
    ax.legend()
    
    # 2. GIOU
    ax = axes[0, 1]
    ax.set_title('GIOU损失')
    target_box = Rectangle((0.2, 0.2), 0.3, 0.3, fill=False, color='blue', label='目标框')
    pred_box = Rectangle((0.6, 0.6), 0.3, 0.3, fill=False, color='red', label='预测框')
    closure = Rectangle((0.2, 0.2), 0.7, 0.7, fill=False, color='green', 
                       linestyle='--', label='最小闭包矩形')
    ax.add_patch(target_box)
    ax.add_patch(pred_box)
    ax.add_patch(closure)
    # 标注惩罚区域
    ax.fill([0.2, 0.9, 0.9, 0.2], [0.2, 0.2, 0.9, 0.9], alpha=0.1, color='gray', label='闭包区域')
    ax.text(0.1, 0.95, 'GIOU = IOU - (闭包面积-并集面积)/闭包面积', fontsize=10)
    ax.text(0.1, 0.9, '优点:解决不相交时的问题', fontsize=10)
    ax.text(0.1, 0.85, '缺点:收敛速度较慢', color='red', fontsize=10)
    ax.legend()
    
    # 3. DIOU
    ax = axes[1, 0]
    ax.set_title('DIOU损失')
    target_box = Rectangle((0.2, 0.2), 0.3, 0.3, fill=False, color='blue', label='目标框')
    pred_box = Rectangle((0.6, 0.6), 0.3, 0.3, fill=False, color='red', label='预测框')
    ax.add_patch(target_box)
    ax.add_patch(pred_box)
    # 绘制中心点距离和对角线
    ax.plot([0.35, 0.75], [0.35, 0.75], 'g--', label='中心点距离')
    ax.plot([0.2, 0.9], [0.2, 0.9], 'y--', label='对角线距离')
    ax.plot(0.35, 0.35, 'bo', label='目标框中心')
    ax.plot(0.75, 0.75, 'ro', label='预测框中心')
    ax.text(0.1, 0.95, 'DIOU = IOU - ρ²(b,bᵍᵗ)/c²', fontsize=10)
    ax.text(0.1, 0.9, '优点:直接最小化框中心距离', fontsize=10)
    ax.text(0.1, 0.85, '缺点:未考虑长宽比', color='red', fontsize=10)
    ax.legend()
    
    # 4. CIOU
    ax = axes[1, 1]
    ax.set_title('CIOU损失')
    target_box = Rectangle((0.2, 0.2), 0.4, 0.2, fill=False, color='blue', label='目标框')
    pred_box = Rectangle((0.6, 0.6), 0.2, 0.4, fill=False, color='red', label='预测框')
    ax.add_patch(target_box)
    ax.add_patch(pred_box)
    # 标注长宽比
    ax.plot([0.2, 0.6], [0.3, 0.3], 'b--', label='目标框宽')
    ax.plot([0.3, 0.3], [0.2, 0.4], 'b:', label='目标框高')
    ax.plot([0.6, 0.8], [0.7, 0.7], 'r--', label='预测框宽')
    ax.plot([0.7, 0.7], [0.6, 1.0], 'r:', label='预测框高')
    ax.text(0.1, 0.95, 'CIOU = DIOU - α*v', fontsize=10)
    ax.text(0.1, 0.9, 'v = 4/π² * (arctan(w_gt/h_gt)-arctan(w/h))²', fontsize=10)
    ax.text(0.1, 0.85, '优点:同时考虑重叠度、中心距离、长宽比', fontsize=10)
    ax.legend()
    
    for ax in axes.flat:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()

plot_boxes_comparison()
