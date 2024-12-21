import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import ResNet18  # 假设你的模型文件名为 model.py
from resnet152 import ResNet152
from tqdm import tqdm  # 导入 tqdm

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整为 ResNet18 的输入尺寸
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 使用 ImageNet 的均值和标准差
    transforms.Normalize(mean=[0.1855, 0.1855, 0.1855], std=[0.1813, 0.1813, 0.1813]),  # 使用 ImageNet 的均值和标准差
])

# 加载数据集
train_dataset = datasets.ImageFolder(root='/home/ztx/code/quantumML-code/data/BrainTumorMRI/Training', transform=transform)
test_dataset = datasets.ImageFolder(root='/home/ztx/code/quantumML-code/data/BrainTumorMRI/Testing', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 创建一个 ResNet18 实例
model = ResNet18(num_classes=4)  # 假设你的数据集有 4 个类别
# model = ResNet152(num_classes=4)  # 假设你的数据集有 4 个类别

# 移动模型到 GPU（如果可用）
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 用于多分类的交叉熵损失
optimizer = optim.Adam(model.parameters(),
                       lr=0.0001,
                       betas=(0.9, 0.999),  # 动量系数
                       eps=1e-8,  # 数值稳定性常数
                       weight_decay=1e-5,  # 权重衰减（L2 正则化）
                       amsgrad=False  # 是否启用 AMSGrad
                       )  # 使用 Adam 优化器

# 训练函数
def train(model, train_loader, criterion, optimizer, device, num_epochs=200):
    model.train()  # 将模型设置为训练模式
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        # 使用 tqdm 包装 train_loader，显示进度条
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as tepoch:
            for i, (inputs, labels) in enumerate(tepoch, 0):
                inputs, labels = inputs.to(device), labels.to(device)
                print("inputs shape:",inputs.shape)
                optimizer.zero_grad()  # 清除之前计算的梯度

                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # 反向传播
                loss.backward()
                optimizer.step()

                # 统计损失
                running_loss += loss.item()

                # 统计准确度
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # 更新进度条的描述
                tepoch.set_postfix(loss=running_loss/(i+1), accuracy=100 * correct / total)

        # 打印每个 epoch 的训练损失和准确率
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.2f}%")

        # 训练完成后测试模型
        test(model, test_loader, device)

# 测试函数
def test(model, test_loader, device):
    model.eval()  # 将模型设置为评估模式
    correct = 0
    total = 0

    # 使用 tqdm 包装 test_loader，显示进度条
    with torch.no_grad():
        with tqdm(test_loader, desc="Testing", unit="batch") as tepoch:
            for inputs, labels in tepoch:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # 更新进度条的描述
                tepoch.set_postfix(accuracy=100 * correct / total)

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    model.train()  # 将模型设置为训练模式

    return accuracy

# 开始训练
num_epochs = 50
train(model, train_loader, criterion, optimizer, device, num_epochs=num_epochs)

# 训练完成后测试模型
test(model, test_loader, device)
