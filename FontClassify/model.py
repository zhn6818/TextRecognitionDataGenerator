import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os
from PIL import Image
from datetime import datetime

# 自定义数据集类
class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)
        self.imgs = [img for img in self.imgs if not os.path.basename(img[0]).startswith('._')]

    def __getitem__(self, index):
        img, target = super(CustomImageFolder, self).__getitem__(index)
        img = self.pad_and_resize(img)
        
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def pad_and_resize(self, img):
        img = Image.fromarray(img.numpy().astype('uint8').transpose(1, 2, 0))
        max_size = 64
        width, height = img.size
        if width > height:
            new_width = max_size
            new_height = int((height / width) * max_size)
        else:
            new_height = max_size
            new_width = int((width / height) * max_size)

        img = img.resize((new_width, new_height), Image.LANCZOS)
        new_img = Image.new("RGB", (64, 64), (0, 0, 0))
        x_offset = (64 - new_width) // 2
        y_offset = (64 - new_height) // 2
        new_img.paste(img, (x_offset, y_offset))

        return new_img

# 数据预处理和数据增强
image_size = 64
batch_size = 32
extract_path = "./out/"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

train_dataset = CustomImageFolder(root=extract_path, transform=transform)

# 设置 num_workers 为 32
num_workers = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

class_names = train_dataset.classes
print(f"Classes: {class_names}")

model = models.resnet50(pretrained=True)

# for param in model.layer1.parameters():
#     param.requires_grad = False
# for param in model.layer2.parameters():
#     param.requires_grad = False

num_classes = len(class_names)
model.fc = nn.Linear(model.fc.in_features, num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

# 创建保存模型的文件夹
weights_dir = "FontClassify/weights/"
os.makedirs(weights_dir, exist_ok=True)

def train(model, device, train_loader, optimizer, criterion, epoch, best_loss):
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

best_loss = float('inf')
target_loss = 0.005  # 目标损失
num_epochs = 0  # 初始化 epoch 计数

while True:  # 无限循环，直到达到目标损失
    num_epochs += 1  # 递增 epoch 计数
    best_loss = train(model, device, train_loader, optimizer, criterion, num_epochs - 1, best_loss)
    
    # 检查当前损失是否低于目标损失
    if best_loss < target_loss:
        print(f'训练完成，达到目标损失 {target_loss}，在 {num_epochs} 轮后停止训练。')
        break