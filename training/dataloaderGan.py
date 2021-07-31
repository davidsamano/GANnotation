import os
import pandas as pd
import numpy as np
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2

class Dataset300VW(Dataset):
    def __init__(self, path, transform=None):
        self.dataset_csv = pd.read_csv(path)
        self.transform = transform

    def __len__(self):
        return len(self.dataset_csv)

    def __getitem__(self, idx):
        img_in_path = self.dataset_csv.iloc[idx, 0]
        img_out_path = self.dataset_csv.iloc[idx, 2]
        image_in = cv2.imread(img_in_path)
        image_out = cv2.imread(img_out_path)
        key_points = np.load(self.dataset_csv.iloc[idx, 1])
        sample = {'initialFrame':image_in, 'keypoints': key_points, 'targetFrame': image_out}
        if self.transform:
            sample = self.transform(sample)
        return sample

class Binarize(object):
    def __call__(self, sample):
        image, key_pts, target = sample['initialFrame'], sample['keypoints'], sample['targetFrame']

        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)/255.0
        image = np.reshape(image, (image.shape[0],image.shape[1],1))

        target = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)/255.0
        target = np.reshape(target, (target.shape[0],target.shape[1],1))

        return {'initialFrame': image, 'keypoints': key_pts, 'targetFrame': target}

class Rescale(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, key_pts, target = sample['initialFrame'], sample['keypoints'], sample['targetFrame']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        image = cv2.resize(image, (new_w, new_h))
        target = cv2.resize(target, (new_w, new_h))
        if len(image.shape) < 3:
            image = np.reshape(image, (new_w, new_h,1))
            target = np.reshape(target, (new_w, new_h,1))
        key_pts = key_pts * np.array([new_w / w, new_h / h])
        return {'initialFrame': image, 'keypoints': key_pts, 'targetFrame': target}

class RandomCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, key_pts, target = sample['initialFrame'], sample['keypoints'], sample['targetFrame']

        h, w = image.shape[:2]

        top = np.random.randint(0, h - self.output_size[0])
        left = np.random.randint(0, w - self.output_size[1])

        image = image[top: top + self.output_size[0],
                      left: left + self.output_size[1]]

        target = target[top: top + self.output_size[0],
                        left: left + self.output_size[1]]

        key_pts = key_pts - [left, top]

        return {'initialFrame': image, 'keypoints': key_pts, 'targetFrame': target}

class ToTensor(object):
    def __call__(self, sample):
        image, key_pts, target = sample['initialFrame'], sample['keypoints'], sample['targetFrame']
        image = image.transpose((2, 0, 1))
        target = target.transpose((2, 0, 1))
        return {'initialFrame': torch.from_numpy(image),
                'keypoints': torch.from_numpy(key_pts), 'targetFrame'}
                
# test_path = 'C:\\Users\\isaac\\Documents\\Classwork\\Graduate\\DL\\final\\300VW_Dataset_2015_12_14\\test\\test.csv'
# test_data = Dataset300VW(test_path)#,# /transform=transforms.Compose([
#                                         # Binarize(),
#                                         # Rescale((600,600)),
#                                         # ToTensor()]))
# test_data.__getitem__(0)
# test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)