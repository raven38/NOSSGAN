"""
MIT License

Copyright (c) 2021 Kai Katsumata
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import sys
import h5py as h5
import numpy as np
import PIL
import json
from argparse import ArgumentParser
from functools import partial
import shutil
from glob import glob
from pathlib import Path

# 0 inject noise
# 1 split open and closed class
# 2 split unlabeled and labeled in closed class
# 3 extract subset of open set class
def osss_subset(imgs, labels, rng, labeled_ratio=0.2, usage_ratio=0.1, closed_usage_ratio=1.0, noise_rate=0.1, subset_classes=[0]):
    N = len(labels)

    noise_mask = rng.random( len(labels)) <= noise_rate
    rnd_targets = rng.choice(1000, len(labels)).astype(labels.dtype)
    noised_labels = labels * (1 - noise_mask) + rnd_targets * noise_mask
    
    close_class_mask = np.isin(noised_labels,  subset_classes)
    open_class_mask = ~np.isin(noised_labels,  subset_classes)
    print('the number of closed-set sample in original data', np.sum(close_class_mask))
    print('the number of open-set sample in original data', np.sum(open_class_mask))
    unlabeled_mask = rng.choice([True, False], size=N, p=[1 - labeled_ratio, labeled_ratio])
    unlabeled_mask = unlabeled_mask | open_class_mask
    new_labels = noised_labels * (1 - unlabeled_mask) + (-1) * unlabeled_mask
    usage_mask = rng.choice([True, False], size=N, p=[usage_ratio, 1 - usage_ratio])
    closed_usage_mask = rng.choice([True, False], size=N, p=[closed_usage_ratio, 1 - closed_usage_ratio])
    mask = (close_class_mask & closed_usage_mask) | (open_class_mask & usage_mask)
    print(np.sum(mask))

    return imgs[mask], new_labels[mask], labels[mask]


def class_filtered_dataset(imgs, labels, subset_classes=[0]):
    mask = np.isin(labels,  subset_classes)
    return imgs[mask], labels[mask]


def replace_label(labels, classes):
    # Extract out keys and values
    k = np.array(list(classes.keys()))
    v = np.array(list(classes.values()))

    # Get argsort indices
    sidx = k.argsort()

    ks = k[sidx]
    vs = v[sidx]
    idx = np.searchsorted(ks, labels)
    idx[idx==len(vs)] = 0
    mask = ks[idx] == labels
    return np.where(mask, vs[idx], -1)


def make_semi_supervised_dataset(src_path, dst_path, config, rng, sub_f, T, subset_classes=[0]):
    with h5.File(src_path, 'r') as f:
        labels = f['labels'][...]
        imgs = f['imgs'][...]
    ss_imgs, ss_labels, old_labels = sub_f(imgs, labels, rng)
    ss_labels = replace_label(ss_labels, {**{c:i for i,c in enumerate(subset_classes)}, **{-1:-1}})
    old_labels = replace_label(old_labels, {c:i for i,c in enumerate(subset_classes)})
    assert old_labels.shape == ss_labels.shape

    with h5.File(dst_path, 'w') as df:
        df.create_dataset('closedclasses', data=subset_classes, dtype='int')
        df.create_dataset('labels2', data=old_labels, dtype='int64',
                          chunks=(config['chunk_size'],), compression=config['compression'])
        df.create_dataset('imgs', data=ss_imgs, dtype='uint8',
                          chunks=(config['chunk_size'], 3, config['img_size'], config['img_size']), compression=config['compression'])
        df.create_dataset('labels', data=ss_labels, dtype='int64',
                          chunks=(config['chunk_size'],), compression=config['compression'])
        # df.create_dataset('T', data=T, dtype='float', compression=config['compression'])

def make_filtered_dataset(src_path, dst_path, config, T, subset_classes=[0]):
    with h5.File(src_path, 'r') as f:
        labels = f['labels'][...]
        imgs = f['imgs'][...]

    ss_imgs, ss_labels = class_filtered_dataset(imgs, labels, subset_classes=subset_classes)
    ss_labels = replace_label(ss_labels, {c:i for i,c in enumerate(subset_classes)})

    with h5.File(dst_path, 'w') as df:
        df.create_dataset('closedclasses', data=subset_classes, dtype='int')
        df.create_dataset('imgs', data=ss_imgs, dtype='uint8',
                          chunks=(config['chunk_size'], 3, config['img_size'], config['img_size']), compression=config['compression'])
        df.create_dataset('labels', data=ss_labels, dtype='int64',
                          chunks=(config['chunk_size'],), compression=config['compression'])
        # df.create_dataset('T', data=T, dtype='float', compression=config['compression'])


def report_stats(hdf5_path):
    with h5.File(hdf5_path, 'r') as f:
        labels = f['labels'][...]

        classes = f['closedclasses'][...]
        imgs = f['imgs'][...]
        print(f'label shape: {labels.shape}')
        print(f'image shape: {imgs.shape}')
        print(f'The number of classes: {len(np.unique(labels))}, max: {np.max(labels)}, min: {np.min(labels)}')
        print(f'The number of labeled samples: {np.sum(labels != -1)}')
        print(f'The number of unlabeled samples: {np.sum(labels == -1)}')
        labels2 = f.get('labels2')
        if labels2:
            labels2 = labels2[...]
            print(f'Noise rate of dataset: {np.sum((labels != labels2) & (labels != -1)) / np.sum(labels != -1)}')
            print(f'The number of closed set and open set samples: {np.sum((labels == -1) & (labels2 != -1))} {np.sum((labels == -1) & (labels2 == -1))}')
        

if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('-c', '--config_path', type=str, default='./src/configs/CIFAR10/ContraGAN.json')
    parser.add_argument('--src', type=str, default=None)
    parser.add_argument('--dst', type=str, default=None)
    parser.add_argument('--ordered', action='store_true')
    parser.add_argument('--subset_class', type=int, default=-1)
    parser.add_argument('--ratio', type=float, default=0.2, help='The ratio of remaining labels')
    parser.add_argument('--usage', type=float, default=0.2, help='The ratio of remaining open-set samples')
    parser.add_argument('--closed_usage', type=float, default=1.0, help='The ratio of closed-set samples')
    parser.add_argument('--noise_rate', type=float, default=0.2, help='Noise rate')
    args = parser.parse_args()
    
    assert args.src and args.dst, 'argument src and dst is needed'

    rng = np.random.default_rng(1021)

    if args.config_path is not None:
        with open(args.config_path) as f:
            model_configs = json.load(f)
        train_configs = vars(args)
    else:
        raise NotImplementedError


    os.makedirs(args.dst, exist_ok=True)
    img_size = model_configs['data_processing']['img_size']
    src_path_list = list(glob(f'{args.src}/*{img_size}*train*.hdf5'))

    assert len(src_path_list) == 1, f'There are some datasets for train, {src_path_list}'
    src_path_train = src_path_list[0]
    dst_path_train_basename = '_'.join([model_configs['data_processing']['dataset_name']] + os.path.basename(src_path_train).split('_')[-2:])
    dst_path_train = f'{args.dst}/{dst_path_train_basename}'
    print(f'Source hdf5 dataset: {src_path_train}, Dest hdf5 dataset: {dst_path_train}')
    # shutil.copyfile(src_path_list[0], dst_path)
    # hdf5_path_train = dst_path
    assert not Path(dst_path_train).exists(), 'args.dst is existing path'

    if args.ordered:
        subset_classes = np.arange(args.subset_class)
    else:
        subset_classes = rng.choice(np.arange(1000), args.subset_class, replace=False)

    # make symmetric T
    T = np.eye(args.subset_class) * (1 - args.noise_rate) + (np.ones([args.subset_class, args.subset_class]) / args.subset_class * args.noise_rate)
    print(T)

    sub_f = partial(osss_subset, labeled_ratio=args.ratio, usage_ratio=args.usage, closed_usage_ratio=args.closed_usage, noise_rate=args.noise_rate, subset_classes=subset_classes)

    make_semi_supervised_dataset(src_path_train, dst_path_train, model_configs['data_processing'], rng, sub_f, T, subset_classes=subset_classes)
    report_stats(dst_path_train)
    src_path_list = list(glob(f'{args.src}/*{img_size}*eval*.hdf5'))
    assert len(src_path_list) == 1, 'There are some datasets for eval'
    src_path_eval = src_path_list[0]
    dst_path_eval_basename = '_'.join([model_configs['data_processing']['dataset_name']] + os.path.basename(src_path_eval).split('_')[-2:])
    dst_path_eval = f'{args.dst}/{dst_path_eval_basename}'
    print(f'Source hdf5 dataset: {src_path_eval}, Dest hdf5 dataset: {dst_path_eval}')
    make_filtered_dataset(src_path_eval, dst_path_eval, model_configs['data_processing'], T, subset_classes=subset_classes)
    report_stats(dst_path_eval)
