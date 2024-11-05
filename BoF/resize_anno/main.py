import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def resize_image_and_annotations(image_path, annotation_path, target_resolution, output_image_path, output_annotation_path):
    """调整图片大小和对应的YOLO格式标注
    
    Args:
        image_path (str): 输入图片路径
        annotation_path (str): 输入YOLO格式标注文件路径
        target_resolution (tuple): 目标分辨率,格式为(width, height)
        output_image_path (str): 输出图片保存路径
        output_annotation_path (str): 输出标注文件保存路径
        
    Returns:
        None: 函数将调整后的图片和标注保存到指定路径
    """
    # 读取图片
    image = Image.open(image_path)
    
    # 读取YOLO格式的标注
    with open(annotation_path, 'r') as f:
        annotations = f.readlines()
    
    # 获取原始图片的宽高
    original_width, original_height = image.size
    
    # 调整图片大小
    transform = transforms.Resize(target_resolution)
    resized_image = transform(image)
    
    # 计算宽高缩放比例
    scale_x = target_resolution[0] / original_width
    scale_y = target_resolution[1] / original_height
    
    # 调整YOLO格式的标注
    resized_annotations = []
    for annotation in annotations:
        class_id, x_center, y_center, width, height = map(float, annotation.split())
        x_center *= scale_x
        y_center *= scale_y
        width *= scale_x
        height *= scale_y
        resized_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}\n")
    
    # 检查并创建输出路径（如果不存在）
    if not os.path.exists(os.path.dirname(output_image_path)):
        os.makedirs(os.path.dirname(output_image_path))
    if not os.path.exists(os.path.dirname(output_annotation_path)):
        os.makedirs(os.path.dirname(output_annotation_path))
    
    # 保存调整后的图片
    resized_image.save(output_image_path)
    
    # 保存调整后的标注
    with open(output_annotation_path, 'w') as f:
        f.writelines(resized_annotations)
def visualize_image_and_annotations(image_path, annotation_path):
    """可视化图片和YOLO格式的标注框
    
    Args:
        image_path (str): 输入图片路径
        annotation_path (str): 输入YOLO格式标注文件路径
        
    Returns:
        None: 函数将显示带有标注框的图片
    """
    # 读取图片
    image = Image.open(image_path)
    original_width, original_height = image.size
    # 读取YOLO标注
    with open(annotation_path, 'r') as f:
        annotations = f.readlines()
    
    # 显示图片和标注框
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for annotation in annotations:
        class_id, x_center, y_center, width, height = map(float, annotation.split())
        x_min = (x_center - width / 2)*original_width
        y_min = (y_center - height / 2)*original_height
        rect = patches.Rectangle((x_min, y_min), width*original_width, height*original_height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()


if __name__ == "__main__":
    # 定义输入图片和标注文件的路径
    image_path = './dataset/images/0001.jpeg'
    annotation_path = './dataset/labels/0001.txt'
    
    # 定义调整大小后输出的图片和标注文件路径
    output_image_path = './dataset_resize/images/0001.jpeg'
    output_annotation_path = './dataset_resize/labels/0001.txt'
    
    # 设置目标分辨率为640x480
    target_resolution = (640, 480)
    
    # 调整图片大小和对应的标注
    resize_image_and_annotations(image_path, annotation_path, target_resolution, 
                                output_image_path, output_annotation_path)
    
    # 可视化原始图片和标注
    visualize_image_and_annotations(image_path, annotation_path)

    # 可视化调整大小后的图片和标注
    visualize_image_and_annotations(output_image_path, output_annotation_path)