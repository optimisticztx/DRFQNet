import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整大小为224x224，适应网络输入
    transforms.ToTensor(),  # 将图像转换为 Tensor
])

# 加载数据集
dataset = datasets.ImageFolder(root='/home/ztx/code/quantumML-code/data/BrainTumorMRI/Training', transform=transform)

# 创建数据加载器
data_loader = DataLoader(dataset, batch_size=62, shuffle=False)

# 用于计算均值和标准差的累加器
mean = 0.
std = 0.
nb_samples = 0.

# 遍历数据集计算均值和标准差
for images, _ in data_loader:
    # 计算每个 batch 的均值和标准差
    batch_samples = images.size(0)  # 当前 batch 的样本数量
    images = images.view(batch_samples, images.size(1), -1)  # 将图片展平为 [batch, channels, height*width]

    # 累加均值和标准差
    mean += images.mean(2).sum(0)  # 按像素计算每个通道的均值
    std += images.std(2).sum(0)  # 按像素计算每个通道的标准差
    nb_samples += batch_samples

# 最终的均值和标准差
mean /= nb_samples
std /= nb_samples

# 打印计算结果
print(f'Mean: {mean}')
print(f'Standard Deviation: {std}')
