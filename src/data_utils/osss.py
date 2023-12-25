import os
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import numpy as np
from numpy.testing import assert_equal

import torch
import torchvision.datasets as datasets
from torch.utils.data import Dataset


class OSSDataset(Dataset):
    def __init__(self,
                 base_dataset,
                 num_closedset_classes=8,
                 unlabeled_ratio=0.5,
                 train=True):
        self.data = base_dataset
        self.num_closedset_classes = num_closedset_classes
        self.unlabeled_ratio = unlabeled_ratio
        self.targets = self.data.targets
        self.train = train

        if self.train:
            openset_class_mask = np.where(np.array(self.data.targets) >= num_closedset_classes, 1, 0)
            unlabeled_mask = np.random.choice([True, False], size=len(self.data), p=[self.unlabeled_ratio, 1 - self.unlabeled_ratio])
            unlabeled_mask = unlabeled_mask | openset_class_mask
            self.targets = self.targets * (1 - unlabeled_mask) + (-1) * unlabeled_mask
        else:
            closedset_class_mask = np.where(np.array(self.data.targets) < num_closedset_classes, 1, 0)
            self.targets = self.targets * closedset_class_mask + (-1) * (1 - closedset_class_mask)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index][0], self.targets[index]

    def T(self, noise_type, noise_rate):
        T = self.data.T(noise_type, noise_rate)
        ncs = self.num_closedset_classes
        return T[:ncs, :ncs] + torch.sum(T[:ncs, ncs:], dim=1, keepdim=True) * torch.eye(ncs)
