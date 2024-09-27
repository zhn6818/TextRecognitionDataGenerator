import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os
from PIL import Image

# 自定义数据集类
class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)
        # 过滤掉以 ._ 开头的文件
        self.imgs = [img for img in self.imgs if not os.path.basename(img[0]).startswith('._')]

    def __getitem__(self, index):
        img, target = super(CustomImageFolder, self).__getitem__(index)
        
        # 对于每个图像进行处理
        img = self.pad_and_resize(img)
        
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def pad_and_resize(self, img):
        # 将图像转为 PIL 格式以便处理
        img = Image.fromarray(img.numpy().astype('uint8').transpose(1, 2, 0))
        
        # 计算缩放比例
        max_size = 64
        width, height = img.size
        if width > height:
            new_width = max_size
            new_height = int((height / width) * max_size)
        else:
            new_height = max_size
            new_width = int((width / height) * max_size)

        # 等比例缩放
        img = img.resize((new_width, new_height), Image.ANTIALIAS)

        # 创建新的 64x64 黑色背景图像
        new_img = Image.new("RGB", (64, 64), (0, 0, 0))

        # 计算放置图像的位置
        x_offset = (64 - new_width) // 2
        y_offset = (64 - new_height) // 2

        # 将缩放后的图像粘贴到黑色背景上
        new_img.paste(img, (x_offset, y_offset))

        return new_img

# 数据预处理和数据增强
image_size = 64  # 输入尺寸
batch_size = 32  # 每批处理的图像数量

extract_path = "./out/"

# 使用自定义的数据集类
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

train_dataset = CustomImageFolder(root=extract_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 打印类别标签
class_names = train_dataset.classes
print(f"Classes: {class_names}")

# 使用预训练的 ResNet 模型
model = models.resnet50(pretrained=True)

# 冻结前两个残差块
for param in model.layer1.parameters():
    param.requires_grad = False
for param in model.layer2.parameters():
    param.requires_grad = False

# 修改最后的全连接层
num_classes = len(class_names)
model.fc = nn.Linear(model.fc.in_features, num_classes)  # 替换最后的全连接层

# 移动到设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 使用 DataParallel 包装模型以支持多卡训练
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

# 训练函数与验证函数（与之前类似）

def train(model, device, train_loader, optimizer, criterion, epoch, best_loss):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # 清零梯度
        output = model(data)
        loss = criterion(output, target)
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重

        running_loss += loss.item()
        if batch_idx % 10 == 0:
            print(f'Epoch [{epoch + 1}], Batch [{batch_idx + 1}], Loss: {loss.item():.4f}')
    
    avg_loss = running_loss / len(train_loader)
    print(f'Epoch {epoch + 1} complete, Average Loss: {avg_loss:.4f}')

    # 如果当前平均损失小于最佳损失，则保存模型
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), 'best_model.pth')
        print(f'保存新的最佳模型，损失: {best_loss:.4f}')

    return best_loss

# 初始化最佳损失为一个很大的值
best_loss = float('inf')

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    best_loss = train(model, device, train_loader, optimizer, criterion, epoch, best_loss)
