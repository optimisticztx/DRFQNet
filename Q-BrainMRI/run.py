
import torch
import torch.nn as nn
from torchvision import datasets, transforms

from models.inception import Net as Net1
from models.resnet18 import ResNet18 as ResNet18

from app.train import train_network

# load the dataset
# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整为 ResNet18 的输入尺寸
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 使用 ImageNet 的均值和标准差
    transforms.Normalize(mean=[0.1855, 0.1855, 0.1855], std=[0.1813, 0.1813, 0.1813]),  # 使用 ImageNet 的均值和标准差
])

# 加载数据集
train_set = datasets.ImageFolder(root='/home/ztx/code/quantumML-code/data/BrainTumorMRI/Training', transform=transform)
val_set = datasets.ImageFolder(root='/home/ztx/code/quantumML-code/data/BrainTumorMRI/Testing', transform=transform)
# output location/file names
# outdir = 'results_255_tr_mnist358'
# file_prefix = 'mnist_358'

# 选择设备
# device = torch.device('cpu')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 定义模型
preNet = ResNet18(num_classes=8)# 结果为8位
net1 = Net1(preNet=preNet)
net1.to(device)

prefix1 = 'test'


epochs = 200
bs = 16

criterion = nn.CrossEntropyLoss()  # loss function
optimizer = torch.optim.Adagrad(net1.parameters(), lr=0.02)  # optimizer
# optimizer = torch.optim.Adam(net1.parameters(), lr=0.02)  # optimizer


train_network( net=net1, train_set=train_set, val_set=val_set, device=device,
              epochs=epochs, bs=bs, optimizer=optimizer,
              criterion=criterion, file_prefix=prefix1)  # outdir = outdir, file_prefix = file_prefix)





