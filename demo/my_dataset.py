from PIL import Image
import torch
from torch.utils.data import Dataset

import image_to_img_mapping


class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, transform=None, num_classes=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        # 加载图像
        img = image_to_img_mapping.image_to_img_mapping.get(self.images_path[item], None)
        # label是多个分类，是个list,比如[0,1]
        label = torch.zeros(self.num_classes)
        for idx in self.images_class[item]:
            label[idx] = 1

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.stack(labels, dim=0)  # 将标签堆叠在一起
        #num_classes = labels[0].shape[0]
        #labels = [[1 if i in label else 0 for i in range(num_classes)] for label in labels]
        #labels = torch.as_tensor(labels)
        return images, labels