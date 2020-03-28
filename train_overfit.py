import sys
import os
import time
import shutil
import re
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision import transforms as T

from data_manager import rgbd_gradients_dataset, rgbd_gradients_dataloader
from functions import torch2np_u8
import plot
from plot import post_epoch_plot
from models import SpecialFuseNetModel
from data_manager import rgbd_gradients_dataset, rgbd_gradients_dataloader
from train import FuseNetTrainer

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[I] - Using device: {device}')

    cwd = os.getcwd()
    dataset_dir = os.path.join(cwd, 'data/nyuv2_overfit')
    print(f'[I] - Dataset Directory: {dataset_dir}')

    BATCH_SIZE       = 4
    NUM_WORKERS      = 4
    TRAIN_TEST_RATIO = 0.5
    IMAGE_SIZE       = (64, 64)

    dl_train, dl_test = rgbd_gradients_dataloader(root=dataset_dir, use_transforms=True, overfit_mode=True,
                                                  batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                                                  train_test_ratio=TRAIN_TEST_RATIO,
                                                  image_size=IMAGE_SIZE)

    train_sample_batch = next(iter(dl_train))
    test_sample_batch = next(iter(dl_test))

    rgb_train_eq_test = not np.any(train_sample_batch['rgb'].numpy() - test_sample_batch['rgb'].numpy())
    depth_train_eq_test = not np.any(train_sample_batch['depth'].numpy() - test_sample_batch['depth'].numpy())
    x_train_eq_test = not np.any(train_sample_batch['x'].numpy() - test_sample_batch['x'].numpy())
    y_train_eq_test = not np.any(train_sample_batch['y'].numpy() - test_sample_batch['y'].numpy())

    print(f'RGB(train) == RGB(test) : {rgb_train_eq_test}')
    print(f'DEPTH(train) == DEPTH(test) : {depth_train_eq_test}')
    print(f'X(train) == X(test) : {x_train_eq_test}')
    print(f'Y(train) == Y(test) : {y_train_eq_test}')

    rgb_size = tuple(train_sample_batch['rgb'].shape[1:])
    depth_size = tuple(train_sample_batch['depth'].shape[1:])
    grads_size = tuple(train_sample_batch['x'].shape[1:])

    # fusenetmodel = SpecialFuseNetModel(rgb_size=rgb_size, depth_size=depth_size, grads_size=grads_size,
    #                                    device=device, dropout_p=0)
    fusenetmodel = SpecialFuseNetModel(rgb_size=rgb_size, depth_size=depth_size, grads_size=grads_size,
                                       device=device)

    train_sample_batch1 = next(iter(dl_train))
    train_sample_batch2 = next(iter(dl_train))
    print(
        f"Consecutive RGB        train mini-batchs equals: {not np.any((train_sample_batch1['rgb'] - train_sample_batch2['rgb']).numpy())}")
    print(
        f"Consecutive D          train mini-batchs equals: {not np.any((train_sample_batch1['depth'] - train_sample_batch2['depth']).numpy())}")
    print(
        f"Consecutive X          train mini-batchs equals: {not np.any((train_sample_batch1['x'] - train_sample_batch2['x']).numpy())}")
    print(
        f"Consecutive Y          train mini-batchs equals: {not np.any((train_sample_batch1['y'] - train_sample_batch2['y']).numpy())}")
    xy1 = fusenetmodel(train_sample_batch1['rgb'], train_sample_batch1['depth']).detach()
    xy2 = fusenetmodel(train_sample_batch2['rgb'], train_sample_batch2['depth']).detach()
    print(f"Outputs on consecutive train mini-batchs equals: {not np.any((xy1 - xy2).numpy())}")

    test_sample_batch1 = next(iter(dl_test))
    test_sample_batch2 = next(iter(dl_test))
    print(
        f"Consecutive RGB        test mini-batchs equals: {not np.any((test_sample_batch1['rgb'] - test_sample_batch2['rgb']).numpy())}")
    print(
        f"Consecutive D          test mini-batchs equals: {not np.any((test_sample_batch1['depth'] - test_sample_batch2['depth']).numpy())}")
    print(
        f"Consecutive X          test mini-batchs equals: {not np.any((test_sample_batch1['x'] - test_sample_batch2['x']).numpy())}")
    print(
        f"Consecutive Y          test mini-batchs equals: {not np.any((test_sample_batch1['y'] - test_sample_batch2['y']).numpy())}")
    xy1 = fusenetmodel(test_sample_batch1['rgb'], test_sample_batch1['depth']).detach()
    xy2 = fusenetmodel(test_sample_batch2['rgb'], test_sample_batch2['depth']).detach()
    print(f"Outputs on consecutive test mini-batchs equals: {not np.any((xy1 - xy2).numpy())}")

    train_sample_batch = next(iter(dl_train))
    test_sample_batch = next(iter(dl_test))
    print(
        f"Inputs RGB   train & test mini-batchs equals: {not np.any((train_sample_batch['rgb'] - test_sample_batch['rgb']).numpy())}")
    print(
        f"Inputs DEPTH train & test mini-batchs equals: {not np.any((train_sample_batch['depth'] - test_sample_batch['depth']).numpy())}")
    print(
        f"Inputs X     train & test mini-batchs equals: {not np.any((train_sample_batch['x'] - test_sample_batch['x']).numpy())}")
    print(
        f"Inputs Y     train & test mini-batchs equals: {not np.any((train_sample_batch['y'] - test_sample_batch['y']).numpy())}")

    xy1 = fusenetmodel(train_sample_batch['rgb'], train_sample_batch['depth']).detach()
    xy2 = fusenetmodel(test_sample_batch['rgb'], test_sample_batch['depth']).detach()
    print(f"Outputs on   train & test mini-batchs equals: {not np.any((xy1 - xy2).numpy())}")

    checkpoint_file = 'checkpoints/special_fusenet_overfit'
    print(f'[I] - checkpoint file: {checkpoint_file}')
    final_checkpoint_file = checkpoint_file + '_final'
    if os.path.isfile(f'{checkpoint_file}.pt'):
        print(f'[I] - checkpoint file exist ...',end='')
        os.remove(f'{checkpoint_file}.pt')
        print(' removed.')

    trainer = FuseNetTrainer(model=fusenetmodel, device=device, num_epochs=60)

    # fit_res = trainer.fit(dl_train=dl_train, dl_test=dl_test, print_every=10, post_epoch_fn=post_epoch_plot,
    #                       checkpoints=checkpoint_file)
    fit_res = trainer.fit(dl_train=dl_train, dl_test=dl_test, print_every=10,checkpoints=checkpoint_file)



