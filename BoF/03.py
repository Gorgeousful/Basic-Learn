# 基于GAN的纹理生成模型:
# 1. 生成器: 从随机噪声生成纹理图像
# 2. 判别器: 使用PatchGAN判别真假纹理
# 3. 训练: 对抗训练生成逼真纹理
# 4. 应用: 生成新的纹理样本

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import numpy as np
from PIL import Image

# 设置随机种子保证可复现性
torch.manual_seed(42)

# 定义超参数
class Config:
    image_size = 256
    batch_size = 16
    latent_dim = 100
    num_epochs = 100
    lr = 0.0002
    beta1 = 0.5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 自定义数据集加载器
class TextureDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        """
        加载纹理图像数据集
        image_dir: 纹理图像所在文件夹
        transform: 图像预处理操作
        """
        self.image_paths = [...]  # 获取图像路径列表
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# 生成器网络
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        """
        生成器架构：
        1. 从随机噪声生成基础特征
        2. 通过多层卷积上采样生成图像
        3. 使用AdaIN进行风格注入
        """
        self.main = nn.Sequential(
            # 输入是latent_dim维随机向量
            nn.ConvTranspose2d(Config.latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            # 上采样层
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # 输出层
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

# 判别器网络
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        """
        判别器架构：
        1. 使用PatchGAN的思想
        2. 判断局部图像块的真实性
        """
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

# 训练函数
def train_texture_gan():
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize(Config.image_size),
        transforms.CenterCrop(Config.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # 加载数据集
    dataset = TextureDataset(image_dir='path/to/textures', transform=transform)
    dataloader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)

    # 初始化网络
    netG = Generator().to(Config.device)
    netD = Discriminator().to(Config.device)

    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=Config.lr, betas=(Config.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=Config.lr, betas=(Config.beta1, 0.999))

    # 训练循环
    for epoch in range(Config.num_epochs):
        for i, real_images in enumerate(dataloader):
            batch_size = real_images.size(0)
            real_images = real_images.to(Config.device)

            # 训练判别器
            netD.zero_grad()
            label_real = torch.ones(batch_size, 1).to(Config.device)
            label_fake = torch.zeros(batch_size, 1).to(Config.device)

            output_real = netD(real_images)
            d_loss_real = criterion(output_real, label_real)

            noise = torch.randn(batch_size, Config.latent_dim, 1, 1).to(Config.device)
            fake_images = netG(noise)
            output_fake = netD(fake_images.detach())
            d_loss_fake = criterion(output_fake, label_fake)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizerD.step()

            # 训练生成器
            netG.zero_grad()
            output_fake = netD(fake_images)
            g_loss = criterion(output_fake, label_real)
            g_loss.backward()
            optimizerG.step()

            # 打印训练进度
            if i % 100 == 0:
                print(f'Epoch [{epoch}/{Config.num_epochs}] Step [{i}/{len(dataloader)}] '
                      f'd_loss: {d_loss.item():.4f} g_loss: {g_loss.item():.4f}')

        # 保存生成的样本
        if epoch % 10 == 0:
            save_image(fake_images.data[:25], f'fake_samples_epoch_{epoch}.png',
                      normalize=True, nrow=5)

# 使用训练好的模型生成新的纹理
def generate_textures(num_samples=16):
    """
    使用训练好的生成器生成新的纹理
    """
    netG = Generator().to(Config.device)
    netG.load_state_dict(torch.load('generator.pth'))
    netG.eval()

    with torch.no_grad():
        noise = torch.randn(num_samples, Config.latent_dim, 1, 1).to(Config.device)
        fake_images = netG(noise)
        save_image(fake_images.data, 'generated_textures.png', normalize=True, nrow=4)

if __name__ == '__main__':
    # 训练模型
    train_texture_gan()
    
    # 生成新的纹理
    generate_textures()
