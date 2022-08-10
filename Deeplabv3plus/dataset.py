# *_*coding:utf-8 *_*
# @Author : yuemengrui
import os
import numpy as np
from torch.utils.data import Dataset
import cv2
import torch
import random
from torchvision import transforms


class RandomCrop(object):

    def __init__(self, size=512):
        self.size = size

    def __call__(self, img, label, **kwargs):
        h, w = img.shape[:2]
        h_range = h - self.size
        w_range = w - self.size

        x1 = random.randint(0, w_range)
        y1 = random.randint(0, h_range)

        x2 = x1 + self.size
        y2 = y1 + self.size

        img = img[y1:y2, x1:x2]

        label = label[y1:y2, x1:x2]

        return img, label


class ImageResize(object):

    def __init__(self, size=512):
        self.size = (size, size)

    def __call__(self, img, label=None, **kwargs):
        img = cv2.resize(img, self.size)
        if label:
            label = cv2.resize(label, self.size)

        return img, label


class SegDataset(Dataset):

    def __init__(self, dataset_dir, mode, **kwargs):
        assert mode in ['train', 'val']
        self.mode = mode

        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                             ])
        self.random_crop = RandomCrop()
        self.image_resize = ImageResize()

        self.data_list = self._load_data(dataset_dir, mode)

    def _load_data(self, dataset_dir, mode):
        if mode == 'train':
            img_dir = os.path.join(dataset_dir, 'images', 'training')
            label_dir = os.path.join(dataset_dir, 'annotations', 'training')
        else:
            img_dir = os.path.join(dataset_dir, 'images', 'validation')
            label_dir = os.path.join(dataset_dir, 'annotations', 'validation')

        file_list = os.listdir(img_dir)

        data_list = []
        for fil in file_list:
            file_name = fil.split('.')[0]
            img_path = os.path.join(img_dir, file_name + '.jpg')
            label_path = os.path.join(label_dir, file_name + '.png')

            data_list.append({'img_path': img_path, 'label_path': label_path})

        return data_list

    def __getitem__(self, idx):
        # try:
        data = self.data_list[idx]

        img = cv2.imread(data['img_path'])
        label = cv2.imread(data['label_path'], -1)

        if random.random() > 0.5:
            img, label = self.random_crop(img, label)

        img, label = self.image_resize(img, label)

        # img = self.transform(img)

        # label = torch.from_numpy(label)

        return img, label

    # except Exception as e:
    #     print(e)
    #     return self.__getitem__(np.random.randint(self.__len__()))

    def __len__(self):
        return len(self.data_list)


if __name__ == '__main__':
    dataset = SegDataset('/Users/yuemengrui/Desktop/train_dataset', mode='val')
    img, label = dataset[0]

    cv2.imshow('xx', img)
    cv2.waitKey(0)

    cv2.imshow('xxx', label * (255 / 6))
    cv2.waitKey(0)
