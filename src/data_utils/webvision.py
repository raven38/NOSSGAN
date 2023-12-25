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

from typing import Tuple

from PIL import Image


def make_dataset_for_webvision(root, data_dir, datanames_file):
    class_num = 1000

    images = []
    classwise_images = [[] for i in range(class_num)]

    
    for df in datanames_file:
        with open(df, 'r') as f:
            line = f.readline().strip()

            while line:
                path, target = line.split()
                path = os.path.join(root, data_dir, path)

                if not os.path.exists(path):
                    print('does not exist : ' + path)
                    line = f.readline().strip()
                    continue

                target = int(target)

                classwise_images[target].append((path, target))
        
                line = f.readline().strip()

    classwise_idx = [0]

    images = []

    for i in range(class_num):
        classwise_idx.append(classwise_idx[i] + len(classwise_images[i]))
        images = images + classwise_images[i]

    classwise_idx = np.array(classwise_idx)

    print('dataset length : {}'.format(len(images)))

    return images, classwise_idx;


class WebVision(Dataset):
    def __init__(self, root, train):

        datanames_file = [os.path.join(root, 'info/train_filelist_google.txt'), os.path.join(root, 'info/train_filelist_flickr.txt')]
        if not train:
            datanames_file = [os.path.join(root, 'info/val_filelist.txt')]

        data_dir = 'val_images_256' if not train else './'
        imgs, classwise_idx = make_dataset_for_webvision(root, data_dir, datanames_file)
        if len(imgs) == 0:
            raise(RuntimeError('Found 0 images in subfolders of: ' + root + '\n'
                               'Supported image extensions are: ' + ','.join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.train = train
        self.classwise_idx = classwise_idx

    def __getitem__(self, idx : int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            index

        Returns:
            tuple: (input_tensor, target_tensor)
        """
        path, target = self.imgs[idx]
        target = torch.tensor(target, requires_grad = False)
        input_tensor = Image.open(path).convert('RGB')
        return input_tensor, target

    def __len__(self):
        return len(self.imgs)
