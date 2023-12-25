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

from data_utils.osss import OSSDataset


class TinyNoisyLabels(datasets.ImageFolder):
    """TinyImageNet Dataset with noisy labels.

    Args:
        noise_type (string): Noise type (default: 'symmetric').
            The value is either 'symmetric' or 'asymmetric'.
        noise_rate (float): Probability of label corruption (default: 0.0).
        seed (int): Random seed (default: 12345).

    This is a subclass of the `CIFAR100` Dataset.
    """

    def __init__(self,
                 root,
                 noise_type='symmetric',
                 noise_rate=0.0,
                 seed=12345,
                 **kwargs):
        super(TinyNoisyLabels, self).__init__(root=root)
        self.seed = seed
        self.root = root
        self.num_classes = 200
        self.original_targets = self.targets

        if noise_rate > 0:
            if noise_type == 'symmetric':
                self.symmetric_noise(noise_rate)
            elif noise_type == 'asymmetric':
                raise NotImplementedError('asymmetric noise type is not supported by TinyNoisyLabels')
            else:
                raise ValueError(
                    'expected noise_type is either symmetric or asymmetric '
                    '(got {})'.format(noise_type))

    def symmetric_noise(self, noise_rate):
        """Symmetric noise in TinyImageNet.

        For all classes, ground truth labels are replaced with uniform random
        classes.
        """
        np.random.seed(self.seed)
        targets = np.array(self.targets)
        mask = np.random.rand(len(targets)) <= noise_rate
        rnd_targets = np.random.choice(self.num_classes, mask.sum())
        targets[mask] = rnd_targets
        targets = [int(x) for x in targets]
        self.targets = targets

    def asymmetric_noise(self, noise_rate):
        """Insert asymmetric noise.

        Ground truth labels are flipped by mimicking real mistakes between
        similar classes. Following `Making Deep Neural Networks Robust to Label Noise: a Loss Correction Approach`_, 
        ground truth labels are flipped into the next class circularly within
        the same superclasses

        .. _Making Deep Neural Networks Robust to Label Noise: a Loss Correction Approach
            https://arxiv.org/abs/1609.03683
        """
        np.random.seed(self.seed)
        targets = np.array(self.targets)
        Tdata = self.T('asymmetric', noise_rate).numpy().astype(np.float64)
        Tdata = Tdata / np.sum(Tdata, axis=1)[:, None]
        for i, target in enumerate(targets):
            one_hot = np.random.multinomial(1, Tdata[target, :], 1)[0]
            targets[i] = np.where(one_hot == 1)[0]
        targets = [int(x) for x in targets]
        self.targets = targets

    def T(self, noise_type, noise_rate):
        if noise_type == 'symmetric':
            T = (torch.eye(self.num_classes) * (1 - noise_rate) +
                 (torch.ones([self.num_classes, self.num_classes]) /
                  self.num_classes * noise_rate))
        return T



if __name__ == '__main__':
    dataset = TinyNoisyLabels(noise_type='symmetric',
                              noise_rate=0.5,
                              seed=1234)
    N = len(dataset)
    num_classes = 10
    num_closedset_classes = 5
    unlabeled_ratio = 0.5

    nossdataset = OSSDataset(dataset, num_closedset_classes=num_closedset_classes, unlabeled_ratio=unlabeled_ratio)

    assert_equal(nossdataset.data, dataset.data)

    assert_equal(dataset.targets[nossdataset.targets != -1], nossdataset.targets[nossdataset.targets != -1])

    assert (nossdataset.targets == -1).sum() == int(N*unlabeled_ratio)
