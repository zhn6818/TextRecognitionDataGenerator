import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models
import os
from torchvision import transforms
from datetime import datetime
import sys
sys.path.append("/data1/zhn/macdata/code/github/python/TextRecognitionDataGenerator")

from FontClassify.custom_dataset import CustomImageFolder  # 假设自定义数据集类在custom_dataset.py中

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (data, target, *_) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(output, 1)
        correct += (predicted == target).sum().item()
        total += target.size(0)

        if batch_idx % 10 == 0:
            print(f'Epoch [{epoch + 1}], Batch [{batch_idx + 1}], Loss: {loss.item():.4f}')
    
    train_accuracy = correct / total
    avg_loss = running_loss / len(train_loader)
    print(f'Epoch {epoch + 1} complete, Average Loss: {avg_loss:.4f}, Accuracy: {train_accuracy:.4f}')
    
    return train_accuracy, avg_loss

def validate(model, device, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)

    val_accuracy = correct / total
    avg_val_loss = val_loss / len(val_loader)
    print(f'Validation Accuracy: {val_accuracy:.4f}, Validation Loss: {avg_val_loss:.4f}')
    return val_accuracy, avg_val_loss


def main():
    # 设置参数
    extract_path = "./out/"
    image_size = 64
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # 加载数据集并划分为训练集和验证集
    dataset = CustomImageFolder(root=extract_path, transform=transform)
    train_size = int(0.9 * len(dataset))  # 80% 作为训练集
    val_size = len(dataset) - train_size  # 20% 作为验证集
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=32)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # class_names = dataset.classes
    num_classes = dataset.get_num_classes()
    print("num_classes:", num_classes)
    # print(f"Classes: {class_names}")

    # 加载模型
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
        
    checkpoint_path = ''  # 替换为你之前训练好的模型路径
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"成功加载之前的模型: {checkpoint_path}")
    else:
        print("找不到之前的模型，使用预训练模型进行训练")

    

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
# 创建保存模型的文件夹
    weights_dir = "FontClassify/weights/"
    os.makedirs(weights_dir, exist_ok=True)

    best_val_accuracy = 0.0  # 记录最佳验证准确率
    num_epochs = 100  # 设置训练 epoch 数
    for epoch in range(num_epochs):
        # 训练阶段
        train_accuracy, train_loss = train(model, device, train_loader, optimizer, criterion, epoch)

        # 验证阶段
        val_accuracy, val_loss = validate(model, device, val_loader, criterion)

        # 保存验证集上准确率最高的模型
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            model_filename = os.path.join(weights_dir, f'best_model_acc{val_accuracy:.4f}_{timestamp}.pth')
            torch.save(model.state_dict(), model_filename)
            print(f'保存新的最佳模型: {model_filename}, 验证准确率: {val_accuracy:.4f}')

if __name__ == "__main__":
    main()