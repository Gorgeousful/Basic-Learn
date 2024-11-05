# 实现了多种图像增强方法:
# 1. HSV和对比度调整
# 2. 高斯噪声和椒盐噪声
# 3. Random Erasing随机擦除
# 4. CutOut随机裁剪
# 5. Hide-and-Seek网格隐藏
# 6. Grid Mask网格掩码
# 包含完整的实现和测试代码

import cv2
import numpy as np
import random
import math


def adjust_image(image, hue=0, saturation=1.0, value=0, contrast=1.0):
    """
    调整图像的色调、饱和度、明度和对比度
    
    参数:
    image: 输入RGB图像
    hue: 色调调整值,范围[0,179]
    saturation: 饱和度调整系数,>0
    value: 明度调整值
    contrast: 对比度调整系数,>0
    
    返回:
    调整后的RGB图像
    """
    
    # 转换到HSV色彩空间
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # 调整色调h
    hsv[..., 0] = (hsv[..., 0] + hue) % 180
    
    # 调整饱和度s
    hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0, 255)
    
    # 调整明度v
    hsv[..., 2] = np.clip(hsv[..., 2] + value, 0, 255)
    
    # 转回RGB
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    # 调整对比度
    adjusted = cv2.convertScaleAbs(rgb, alpha=contrast)
    
    return adjusted

def add_noise(image, noise_type='gaussian', amount=0.05):
    """
    给RGB图像添加噪声
    
    参数:
    image: 输入RGB图像
    noise_type: 噪声类型,'gaussian'(高斯噪声)或'salt_pepper'(椒盐噪声)
    amount: 噪声强度,范围[0,1]
    
    返回:
    添加噪声后的RGB图像
    """
    
    image = image.astype(np.float32)
    
    if noise_type == "gaussian":
        # 添加高斯噪声
        mean = 0
        sigma = amount * 255
        noise = np.random.normal(mean, sigma, image.shape)
        noisy_image = image + noise
        
    elif noise_type == "salt_pepper":
        # 添加椒盐噪声
        noisy_image = np.copy(image)
        # 添加白点
        salt = np.random.random(image.shape) < amount/2
        noisy_image[salt] = 255
        # 添加黑点
        pepper = np.random.random(image.shape) < amount/2
        noisy_image[pepper] = 0
        
    # 确保像素值在有效范围内
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)


def random_erase(image, probability, sl, sh, r1, r2, value):
    # 1. 随机决定是否执行擦除
    if random.random() > probability:
        return image

    # 2. 计算擦除区域的大小
    h, w = image.shape[:2]
    area = h * w
    target_area = random.uniform(sl, sh) * area
    aspect_ratio = random.uniform(r1, r2)

    # 3. 计算擦除区域的宽高
    erase_h = int(round(math.sqrt(target_area / aspect_ratio)))
    erase_w = int(round(math.sqrt(target_area * aspect_ratio)))

    # 4. 随机选择擦除位置
    x = random.randint(0, w - erase_w)
    y = random.randint(0, h - erase_h)

    # 5. 执行擦除（填充随机值或指定值）
    if value == 'random':
        image[y:y+erase_h, x:x+erase_w] = np.random.randint(0, 255, size=(erase_h, erase_w, 3))
    else:
        image[y:y+erase_h, x:x+erase_w] = value
    return image

def cutout(image, n_holes=1, length=16):
    """
    对图像进行CutOut数据增强
    
    参数:
        image (numpy.ndarray): 输入图像 (H x W x C)
        n_holes (int): 裁剪区域的数量
        length (int): 裁剪正方形的边长
        
    返回:
        numpy.ndarray: 增强后的图像
    """
    h, w = image.shape[:2]
    mask = np.ones((h, w), np.float32)
    
    for _ in range(n_holes):
        # 随机选择中心点
        y = np.random.randint(h)
        x = np.random.randint(w)
        
        # 计算裁剪区域的范围
        y1 = np.clip(y - length // 2, 0, h)
        y2 = np.clip(y + length // 2, 0, h)
        x1 = np.clip(x - length // 2, 0, w)
        x2 = np.clip(x + length // 2, 0, w)
        
        # 将区域置为0
        mask[y1:y2, x1:x2] = 0
    
    # 扩展mask维度以匹配图像通道
    mask = mask[:, :, np.newaxis]
    
    # 应用mask
    image = image * mask
    
    return image.astype(np.uint8)

def hide_and_seek(image, grid_size=4, p_hide=0.5):
    """
    Hide-and-Seek数据增强
    
    参数:
        image: 输入图像
        grid_size: 网格大小（图像会被分成grid_size×grid_size个块）
        p_hide: 隐藏每个patch的概率
    """
    h, w = image.shape[:2]
    
    # 计算每个网格的大小
    patch_h = h // grid_size
    patch_w = w // grid_size
    
    # 创建掩码
    mask = np.ones((h, w), np.float32)
    
    # 对每个网格进行处理
    for i in range(grid_size):
        for j in range(grid_size):
            # 随机决定是否隐藏该网格
            if np.random.random() < p_hide:
                # 计算网格的坐标
                y1 = i * patch_h
                y2 = min((i + 1) * patch_h, h)
                x1 = j * patch_w
                x2 = min((j + 1) * patch_w, w)
                
                # 将该区域置为0
                mask[y1:y2, x1:x2] = 0
    
    # 应用掩码
    mask = mask[:, :, np.newaxis]
    augmented = image * mask
    
    return augmented.astype(np.uint8)

def grid_mask(image, d1=96, d2=32, rotate=1):
    """
    Grid Mask数据增强
    
    参数:
        image: 输入图像
        d1: 网格总大小
        d2: 保留区域大小
        rotate: 是否随机旋转
    """
    h, w = image.shape[:2]
    hh = int(np.ceil(h / d1) * d1)
    ww = int(np.ceil(w / d1) * d1)
    
    # 创建掩码
    mask = np.ones((hh, ww), np.float32)
    
    # 计算网格
    for i in range(0, hh, d1):
        for j in range(0, ww, d1):
            # 在每个d1×d1的网格中创建d2×d2的保留区域
            mask[i:min(i+d2, hh), j:min(j+d2, ww)] = 0
            
    # 如果需要旋转
    if rotate:
        angle = np.random.randint(0, 4) * 90
        mask = rotate_image(mask, angle)
    
    # 裁剪到原始大小
    mask = mask[:h, :w]
    
    # 应用掩码
    mask = mask[:, :, np.newaxis]
    augmented = image * mask
    
    return augmented.astype(np.uint8)

def rotate_image(image, angle):
    """旋转图像的辅助函数"""
    center = (image.shape[1] // 2, image.shape[0] // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
    return rotated


if __name__ == '__main__':
    # 测试调整图像 hsv contrast
    img = cv2.imread('./dataset/images/0001.jpeg')
    adjusted_img = adjust_image(img, hue=10, saturation=1.5, value=20, contrast=1.5)
    cv2.imshow('Original Image', img)
    cv2.imshow('Adjusted Image', adjusted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 测试添加噪声
    noisy_gaussian = add_noise(img, noise_type='gaussian', amount=0.1)
    noisy_salt_pepper = add_noise(img, noise_type='salt_pepper', amount=0.05)
    
    # 显示原图和添加噪声后的图像
    cv2.imshow('Original Image', img)
    cv2.imshow('Gaussian Noise', noisy_gaussian)
    cv2.imshow('Salt & Pepper Noise', noisy_salt_pepper)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 测试随机擦除
    erased_image = random_erase(img, probability=0.5, sl=0.02, sh=0.4, r1=0.3, r2=3.3, value=255)
    cv2.imshow('Erased Image', erased_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 测试CutOut
    cutout_image = cutout(img, n_holes=1, length=16)
    cv2.imshow('CutOut Image', cutout_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 测试Hide-and-Seek
    hide_and_seek_image = hide_and_seek(img, grid_size=4, p_hide=0.5)
    cv2.imshow('Hide-and-Seek Image', hide_and_seek_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 测试Grid Mask
    grid_mask_image = grid_mask(img)
    cv2.imshow('Grid Mask Image', grid_mask_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
