import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
import os
import sys
sys.path.append("/data1/zhn/macdata/code/github/python/TextRecognitionDataGenerator")
from custom_dataset import CustomImageFolder  # 假设自定义数据集类在custom_dataset.py中

def load_model_without_dataparallel(model, checkpoint_path):
    # 加载模型的 state_dict
    state_dict = torch.load(checkpoint_path)

    # 如果保存的模型是 DataParallel 模型，去掉 'module.' 前缀
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[len('module.'):]  # 去掉 'module.' 前缀
        else:
            new_key = key
        new_state_dict[new_key] = value

    # 加载新的 state_dict
    model.load_state_dict(new_state_dict)

    return model




def evaluate_model(model, device, data_loader, class_names):
    model.eval()  # 设置模型为评估模式
    total_correct = {class_name: 0 for class_name in class_names}
    total_samples = {class_name: 0 for class_name in class_names}
    
    # 存储分类不正确的文件路径
    incorrect_files = {class_name: [] for class_name in class_names}

    with torch.no_grad():
        for data, target, paths in data_loader:  # 从DataLoader中获取图像数据，标签和文件路径
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)

            # 统计正确的预测并记录错误分类的文件
            for i in range(len(target)):
                true_class = class_names[target[i].item()]
                predicted_class = class_names[predicted[i].item()]
                if predicted_class != true_class:
                    # 如果分类错误，记录文件路径到对应的真实类别中
                    incorrect_files[true_class].append(paths[i])
                else:
                    total_correct[true_class] += 1
                total_samples[true_class] += 1

    # 计算并打印每个类别的准确率
    accuracies = {}
    for class_name in class_names:
        accuracy = total_correct[class_name] / total_samples[class_name] if total_samples[class_name] > 0 else 0
        accuracies[class_name] = accuracy
        print(f'Class: {class_name}, Accuracy: {accuracy:.2f}')
    
    # 打印分类错误的文件路径
    for class_name, files in incorrect_files.items():
        print(f'Class: {class_name}, Incorrectly classified files:')
        for file in files:
            print(f'    {file}')

    return accuracies, incorrect_files

def main():
    # 设置参数
    extract_path = "./outTrainSingle/"
    image_size = 64
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # 加载测试数据集
    test_dataset = CustomImageFolder(root=extract_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_classes = test_dataset.get_num_classes()
    class_names = test_dataset.get_class_names()

    # 加载模型
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    checkpoint_path = 'FontClassify/weights/best_model_acc0.9927_20240930-095649.pth'
    if os.path.exists(checkpoint_path):
        model = load_model_without_dataparallel(model, checkpoint_path)
        print(f"成功加载之前的模型: {checkpoint_path}")
    else:
        print("找不到之前的模型，使用预训练模型进行训练")
    model.to(device)

    # 评估模型
    accuracies = evaluate_model(model, device, test_loader, class_names)

if __name__ == "__main__":
    main()