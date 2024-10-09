import torch
from torchvision import models, transforms
from PIL import Image
import os
from custom_dataset import CustomImageFolder  # 假设自定义数据集类在 custom_dataset.py 中

# 加载模型并去掉 'module.' 前缀（如果存在）
def load_model_without_dataparallel(model, checkpoint_path):
    state_dict = torch.load(checkpoint_path)
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key[len('module.'):] if key.startswith('module.') else key
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict)
    return model

# 图像处理函数，与训练时保持一致
def pad_and_resize(img):
    max_size = 64
    width, height = img.size
    if width > height:
        new_width = max_size
        new_height = int((height / width) * max_size)
    else:
        new_height = max_size
        new_width = int((width / height) * max_size)

    img = img.resize((new_width, new_height), Image.LANCZOS)
    new_img = Image.new("RGB", (64, 64), (0, 0, 0))  # 用黑色填充
    x_offset = (64 - new_width) // 2
    y_offset = (64 - new_height) // 2
    new_img.paste(img, (x_offset, y_offset))

    return new_img

# 单张图片推理
def predict_single_image(model, device, image_path, class_names):
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # 根据实际情况选择是否应用标准化
    ])

    # 加载并处理图片
    image = Image.open(image_path).convert('RGB')
    image = pad_and_resize(image)  # 与训练阶段保持一致
    image = transform(image).unsqueeze(0)  # 添加批次维度
    image = image.to(device)

    # 模型推理
    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    predicted_class = class_names[predicted.item()]
    print(f"预测类别: {predicted_class}")

    return predicted_class

def main():
    # 设置参数
    extract_path = "./outTrainSingle/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载测试数据集以获取类别名称
    test_dataset = CustomImageFolder(root=extract_path, transform=None)
    class_names = test_dataset.get_class_names()

    # 加载模型
    model = models.resnet50(pretrained=True)
    num_classes = len(class_names)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    
    checkpoint_path = 'FontClassify/weights/best_model_acc0.9927_20240930-095649.pth'
    if os.path.exists(checkpoint_path):
        model = load_model_without_dataparallel(model, checkpoint_path)
        print(f"成功加载模型: {checkpoint_path}")
    else:
        print("找不到模型，使用预训练模型进行推理")
    
    model.to(device)

    # 推理单张图片
    image_path = "./outEval/FangSong/ 货 服 蓉 尚 嚣 龄 野 抹_1758.jpg"  # 替换为你要推理的图片路径
    predict_single_image(model, device, image_path, class_names)

if __name__ == "__main__":
    main()
