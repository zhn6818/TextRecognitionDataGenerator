import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
import os
from torchvision import transforms
from datetime import datetime
import sys
sys.path.append("/data1/zhn/macdata/code/github/python/TextRecognitionDataGenerator")

from FontClassify.custom_dataset import CustomImageFolder  # 假设自定义数据集类在custom_dataset.py中

def train(model, device, train_loader, optimizer, criterion, epoch, best_loss, weights_dir):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 10 == 0:
            print(f'Epoch [{epoch + 1}], Batch [{batch_idx + 1}], Loss: {loss.item():.4f}')
    
    avg_loss = running_loss / len(train_loader)
    print(f'Epoch {epoch + 1} complete, Average Loss: {avg_loss:.4f}')

    if avg_loss < best_loss:
        best_loss = avg_loss
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_filename = os.path.join(weights_dir, f'model_epoch{epoch + 1}_loss{avg_loss:.4f}_{timestamp}.pth')
        torch.save(model.state_dict(), model_filename)
        print(f'保存新的最佳模型: {model_filename}, 损失: {best_loss:.4f}')

    return best_loss

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
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # 加载训练数据集
    train_dataset = CustomImageFolder(root=extract_path, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=32)

    class_names = train_dataset.classes
    print(f"Classes: {class_names}")

    # 加载模型
    model = models.resnet50(pretrained=True)
    num_classes = len(class_names)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    # 创建保存模型的文件夹
    weights_dir = "FontClassify/weights/"
    os.makedirs(weights_dir, exist_ok=True)

    best_loss = float('inf')
    target_loss = 0.005  # 目标损失
    num_epochs = 0  # 初始化 epoch 计数

    # 训练模型
    while True:
        num_epochs += 1
        best_loss = train(model, device, train_loader, optimizer, criterion, num_epochs - 1, best_loss, weights_dir)

        if best_loss < target_loss:
            print(f'训练完成，达到目标损失 {target_loss}，在 {num_epochs} 轮后停止训练。')
            break

if __name__ == "__main__":
    main()