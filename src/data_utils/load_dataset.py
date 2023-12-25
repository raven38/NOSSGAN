# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/data_utils/load_dataset.py


import os
import h5py as h5
import numpy as np
import random
from scipy import io
from PIL import ImageOps, Image

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, STL10
from torchvision.datasets import ImageFolder

from data_utils.cifar import CIFAR10NoisyLabels, CIFAR100NoisyLabels
from data_utils.tiny_imagenet import TinyNoisyLabels
from data_utils.clothing1m import Clothing1M
from data_utils.webvision import WebVision
from data_utils.osss import OSSDataset


class RandomCropLongEdge(object):
    """
    this code is borrowed from https://github.com/ajbrock/BigGAN-PyTorch
    MIT License
    Copyright (c) 2019 Andy Brock
    """
    def __call__(self, img):
        size = (min(img.size), min(img.size))
        # Only step forward along this edge if it's the long edge
        i = (0 if size[0] == img.size[0]
            else np.random.randint(low=0,high=img.size[0] - size[0]))
        j = (0 if size[1] == img.size[1]
            else np.random.randint(low=0,high=img.size[1] - size[1]))
        return transforms.functional.crop(img, j, i, size[0], size[1])

    def __repr__(self):
        return self.__class__.__name__


class CenterCropLongEdge(object):
    """
    this code is borrowed from https://github.com/ajbrock/BigGAN-PyTorch
    MIT License
    Copyright (c) 2019 Andy Brock
    """
    def __call__(self, img):
        return transforms.functional.center_crop(img, min(img.size))

    def __repr__(self):
        return self.__class__.__name__


class LoadDataset(Dataset):
    def __init__(self, dataset_name, data_path, train, download, resize_size, hdf5_path=None, random_flip=False):
        super(LoadDataset, self).__init__()
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.train = train
        self.download = download
        self.resize_size = resize_size
        self.hdf5_path = hdf5_path
        self.random_flip = random_flip
        self.norm_mean = [0.5,0.5,0.5]
        self.norm_std = [0.5,0.5,0.5]

        if self.hdf5_path is None:
            if self.dataset_name in ['nosscifar10', 'nosscifar100', 'cifar10', 'tiny_imagenet']:
                self.transforms = []
            elif self.dataset_name.startswith('nosstiny'):
                self.transforms = []
            else:
                if train:
                    self.transforms = [RandomCropLongEdge(), transforms.Resize(self.resize_size)]
                else:
                    self.transforms = [CenterCropLongEdge(), transforms.Resize(self.resize_size)]
        else:
            self.transforms = [transforms.ToPILImage()]

        if random_flip:
            self.transforms += [transforms.RandomHorizontalFlip()]

        self.transforms += [transforms.ToTensor(), transforms.Normalize(self.norm_mean, self.norm_std)]
        self.transforms = transforms.Compose(self.transforms)
        self.noise_rate = 0
        self.num_closedset_classes = 0
        if self.dataset_name == 'nosscifar10':
            self.noise_type = 'symmetric'
            self.noise_rate = 0.1
            self.num_closedset_classes = 5
            self.unlabeled_ratio = 0.8
        if self.dataset_name == 'nosscifar100':
            self.noise_type = 'symmetric'
            self.noise_rate = 0.3
            self.num_closedset_classes = 50
            self.unlabeled_ratio = 0.95
        if self.dataset_name.startswith('nosstiny'):
            self.noise_type = 'symmetric'
            num_closedset_classes, noise_rate, unlabeled_ratio, *args = list(map(int, self.dataset_name.split('_')[2:]))
            self.num_closedset_classes = num_closedset_classes
            self.noise_rate = noise_rate / 100
            self.unlabeled_ratio = unlabeled_ratio / 100
        if self.dataset_name.startswith('nossimage'):
            self.noise_type = 'symmetric'
            num_closedset_classes, noise_rate, labeled_ratio, usage_ratio = list(map(int, self.dataset_name.split('_')[1:]))
            self.num_closedset_classes = num_closedset_classes
            self.noise_rate = noise_rate / 100
            self.unlabeled_ratio = 1 - (labeled_ratio / 100)            

        self.load_dataset()

    
    def make_symmetric_T(self):
        T = (torch.eye(self.num_closedset_classes) * (1 - self.noise_rate) + (torch.ones([self.num_closedset_classes, self.num_closedset_classes]) / self.num_closedset_classes * self.noise_rate))
        return T


    def load_dataset(self):
        if self.hdf5_path is not None:
            print('Loading %s into memory...' % self.hdf5_path)
            
            with h5.File(self.hdf5_path, 'r') as f:
                self.data = f['imgs'][:]
                self.labels = f['labels'][:].astype(int)
                labels2 = f.get('labels2')
                if labels2:
                    self.labels2 = labels2[:].astype(int)
                T = f.get('T')
                self.T = T[:] if T else self.make_symmetric_T()
            return
            

        if self.dataset_name == 'cifar10':
            self.data = CIFAR10(root=self.data_path,
                                train=self.train,
                                download=self.download)
        elif self.dataset_name == 'nosscifar10':
            self.data = CIFAR10NoisyLabels(noise_type=self.noise_type,
                                           noise_rate=self.noise_rate if self.train is True else 0,
                                           seed=1021,
                                           root=self.data_path,
                                           train=self.train,
                                           download=self.download)
            self.data = OSSDataset(self.data, num_closedset_classes=self.num_closedset_classes, unlabeled_ratio=self.unlabeled_ratio, train=self.train)
        elif self.dataset_name == 'nosscifar100':
            self.data = CIFAR100NoisyLabels(noise_type=self.noise_type,
                                            noise_rate=self.noise_rate if self.train is True else 0,
                                            seed=1021,
                                            root=self.data_path,
                                            train=self.train,
                                            download=self.download)
            self.data = OSSDataset(self.data, num_closedset_classes=self.num_closedset_classes, unlabeled_ratio=self.unlabeled_ratio, train=self.train)
        elif self.dataset_name.startswith('nosstiny'):
            mode = 'train' if self.train is True else 'valid'
            root = os.path.join(self.data_path, mode)
            self.data = TinyNoisyLabels(noise_type=self.noise_type,
                                        noise_rate=self.noise_rate if self.train is True else 0,
                                        seed=1021,
                                        root=root,
                                        train=self.train)
            if mode is 'train':
                # self.labels2 = self.data.original_targets
                closedset_class_mask = np.where(np.array(self.data.original_targets) < self.num_closedset_classes, 1, 0)
                self.labels2 = self.data.original_targets * closedset_class_mask + (-1) * (1 - closedset_class_mask)
            self.data = OSSDataset(self.data, num_closedset_classes=self.num_closedset_classes, unlabeled_ratio=self.unlabeled_ratio, train=self.train)
        elif self.dataset_name.startswith('nossclothing1m') or self.dataset_name == 'clothing1m':
            self.data = Clothing1M(self.data_path, self.train)
        elif self.dataset_name.startswith('nosswebvision') or self.dataset_name == 'webvision':
            self.data = WebVision(self.data_path, self.train)
        else:
            mode = 'train' if self.train is True else 'valid'
            root = os.path.join(self.data_path, mode)
            self.data = ImageFolder(root=root)

    def __len__(self):
        if self.hdf5_path is not None:
            num_dataset = self.data.shape[0]
        else:
            num_dataset = len(self.data)
        return num_dataset

    def __getitem__(self, index):
        if self.hdf5_path is None:
            img, label = self.data[index]
            img, label = self.transforms(img), int(label)
        else:
            img, label = np.transpose(self.data[index], (1, 2, 0)), int(self.labels[index])
            img = self.transforms(img)

        if hasattr(self, 'labels2'):
            labels2 = int(self.labels2[index])
            return img, label, labels2
        return img, label
