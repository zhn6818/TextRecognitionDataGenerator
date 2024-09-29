import os
from torchvision import datasets
from PIL import Image

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