import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class CustomImageFolder(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.classes, self.class_to_idx = self.find_classes(self.root)  # 获取类别名称和类别索引映射
        self.imgs, self.targets = self.load_images()

    def find_classes(self, directory):
        """查找类别并返回类别到索引的映射"""
        class_names = os.listdir(directory)
        class_names.sort()  # 确保类别有一致的顺序
        class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
        return class_names, class_to_idx

    def load_images(self):
        """加载图像和标签"""
        imgs = []
        targets = []
        
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = os.path.join(self.root, class_name)
            if not os.path.isdir(class_dir):
                continue  # 跳过非目录文件

            for img_name in os.listdir(class_dir):
                if img_name.startswith('._'):
                    continue  # 跳过以 '._' 开头的文件
                img_path = os.path.join(class_dir, img_name)
                imgs.append(img_path)
                targets.append(class_idx)

        return imgs, targets

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path = self.imgs[index]
        target = self.targets[index]
        
        # 加载图像
        img = Image.open(img_path).convert("RGB")
        
        # 调整图像大小并进行填充
        img = self.pad_and_resize(img)

        # 应用变换
        if self.transform is not None:
            img = self.transform(img)

        return img, target, img_path

    def pad_and_resize(self, img):
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

    def get_num_classes(self):
        """返回类别的数量"""
        return len(self.class_to_idx)
    def get_class_names(self):
        """返回类别名称"""
        return self.classes