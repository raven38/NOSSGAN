# NOSSGAN: https://github.com/raven38/NOSSGAN
# The MIT License (MIT)
# See license file or visit https://github.com/raven38/NOSSGAN for details

# src/metrics/ClassAccuracy.py


import numpy as np
import math
from scipy import linalg
from tqdm import tqdm

from utils.sample import sample_latents
from utils.losses import latent_optimise
from utils.predict import pred_dis_out, pred_cls_out

import torch
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel


def calculate_class_accuracy(dataloader, generator, discriminator, D_loss, num_evaluate, truncated_factor, prior, latent_op,
                       latent_op_step, latent_op_alpha, latent_op_beta, device, cr, logger, eval_generated_sample=False):
    data_iter = iter(dataloader)
    batch_size = dataloader.batch_size
    disable_tqdm = device != 0

    if isinstance(generator, DataParallel) or isinstance(generator, DistributedDataParallel):
        z_dim = generator.module.z_dim
        num_classes = generator.module.num_classes
        conditional_strategy = discriminator.module.conditional_strategy
    else:
        z_dim = generator.z_dim
        num_classes = generator.num_classes
        conditional_strategy = discriminator.conditional_strategy

    total_batch = num_evaluate//batch_size

    if D_loss.__name__ == "loss_wgan_dis":
        raise NotImplementedError

    if device == 0: logger.info("Calculate Classifier Head Accuracies....")

    if eval_generated_sample:
        for batch_id in tqdm(range(total_batch), disable=disable_tqdm):
            zs, fake_labels = sample_latents(prior, batch_size, z_dim, truncated_factor, num_classes, None, device)
            if latent_op:
                zs = latent_optimise(zs, fake_labels, generator, discriminator, conditional_strategy, latent_op_step,
                                     1.0, latent_op_alpha, latent_op_beta, False, device)

            try:
                real_images, real_labels, *other = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                real_images, real_labels, *other = next(data_iter)
            real_images, real_labels = real_images.to(device), real_labels.to(device)

            fake_images = generator(zs, fake_labels, evaluation=True)
            real_labels2 = other[0]
            with torch.no_grad():
                try:
                    cls_out_real = pred_cls_out(discriminator, real_images, real_labels, conditional_strategy).cpu().numpy()
                    cls_out_fake = pred_cls_out(discriminator, fake_images, fake_labels, conditional_strategy).cpu().numpy()
                except NotImplementedError:
                    return np.array([0]), np.array([0]), np.array([0])

            if batch_id == 0:
                fake_confid = cls_out_fake
                real_confid = cls_out_real
                fake_confid_label = fake_labels.cpu().numpy()
                real_confid_label = real_labels.cpu().numpy()
                real_confid_label2 = real_labels2.cpu().numpy()
            else:
                fake_confid = np.concatenate((fake_confid, cls_out_fake), axis=0)
                real_confid = np.concatenate((real_confid, cls_out_real), axis=0)
                fake_confid_label = np.concatenate((fake_confid_label, fake_labels.cpu().numpy()), axis=0)
                real_confid_label = np.concatenate((real_confid_label, real_labels.cpu().numpy()), axis=0)
                real_confid_label2 = np.concatenate((real_confid_label2, real_labels2.cpu().numpy()), axis=0)

        # only_real_acc = (real_confid.argmax(axis=1) == real_confid_label).sum()/len(real_confid)
        only_label_acc = (real_confid.argmax(axis=1) == real_confid_label2)[real_confid_label != -1].sum()/len(real_confid[real_confid_label != -1])
        only_unlabel_acc = (real_confid.argmax(axis=1) == real_confid_label2)[(real_confid_label == -1) & (real_confid_label2 !=  -1)].sum()/len(real_confid[(real_confid_label == -1) & (real_confid_label2 !=  -1)])
        # only_real_acc = (real_confid.argmax(axis=1) == real_confid_label2)[real_confid_label2 != -1].sum()/len(real_confid[real_confid_label2 != -1])
        only_fake_acc = (fake_confid.argmax(axis=1) == fake_confid_label).sum()/len(fake_confid)

        # return only_real_acc, only_fake_acc
        return only_label_acc, only_unlabel_acc, only_fake_acc
    else:
        for batch_id in tqdm(range(total_batch), disable=disable_tqdm):
            try:
                real_images, real_labels, *other = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                real_images, real_labels, *other = next(data_iter)
            real_images, real_labels = real_images.to(device), real_labels.to(device)

            with torch.no_grad():
                try:
                    cls_out_real = pred_cls_out(discriminator, real_images, real_labels, conditional_strategy).cpu().numpy()
                except NotImplementedError:
                    return np.array(0)

            if batch_id == 0:
                real_confid = cls_out_real
                real_confid_label = real_labels.cpu().numpy()
            else:
                real_confid = np.concatenate((real_confid, cls_out_real), axis=0)
                real_confid_label = np.concatenate((real_confid_label, real_labels.cpu().numpy()), axis=0)

        only_real_acc = (real_confid.argmax(axis=1) == real_confid_label).sum()/len(real_confid)

        return only_real_acc
