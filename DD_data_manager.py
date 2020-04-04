import numpy as np
import torch
import os
from copy import deepcopy
import cv2
import random
import PIL
from torchvision import transforms as T
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader

class rgb_to_depth_dataset(Dataset):
    def __init__(self, root, image_size):
        print(f'[I (rgbd_gradients_dataset)] - root={root}\n'
              f'                             - image_size={image_size}\n')
        self.root             = root
        self.image_size       = image_size

        # load all image files, sorting them to
        # ensure that they are aligned
        self.rgbs   = list(sorted(os.listdir(os.path.join(root, 'rgb'))))
        self.depths = list(sorted(os.listdir(os.path.join(root, 'depth'))))

        if not (len(self.rgbs) == len(self.depths)):
            raise Exception(f"Non-equal number of samples from each kind "
                            f"(|rgbs|={len(self.rgbs)} != |depths|={len(self.depths)})")

        self.len = len(self.rgbs)
        print(f'[I] - |self|={len(self)}')

    def transform(self, rgb, depth):
        # Resize to constant spatial dimensions
        rgb   = T.Resize(self.image_size)(rgb)
        depth = T.Resize(self.image_size)(depth)
            
        # PIL.Image -> torch.Tensor
        rgb   = T.ToTensor()(rgb)
        depth = T.ToTensor()(depth)
        
        # Dynamic range [0,1] -> [-1, 1]
        rgb   = T.Normalize(mean=(.5,.5,.5), std=(.5,.5,.5))(rgb)
        depth = T.Normalize(mean=(.5,), std=(.5,))(depth)
        
        return rgb, depth
    
    def __getitem__(self, index):
        rgb = Image.open(os.path.join(self.root, "rgb",   self.rgbs[index]))
        d   = Image.open(os.path.join(self.root, "depth", self.depths[index]))

        rgb, d = self.transform(rgb, d)

        return {'rgb': rgb,
                'depth': d}

    def __len__(self):
        return self.len


def rgb_to_depth_dataloader(root, batch_size, num_workers,image_size, seed=42):
    print(f'[I (rgb_to_depth_dataloader)] - root={root}\n'
          f'                              - batch_size={batch_size}\n'
          f'                              - num_workers={num_workers}\n'
          f'                              - image_size={image_size}\n'
          f'                              - seed={seed}\n')
    torch.manual_seed(seed)

    rgb_to_depth_ds = rgb_to_depth_dataset(root, image_size, use_transforms=use_transforms, overfit_mode=overfit_mode,
                                           constant_index=constant_index)

    train_test_ratio = 0.0
    split_lengths = [int(np.ceil(len(rgb_to_depth_ds)  *    train_test_ratio)),
                     int(np.floor(len(rgb_to_depth_ds) * (1-train_test_ratio)))]

    # NOTE: Don't forget to set the seed in order to maintain reproducibility over training experiments.
    ds_train, ds_test = random_split(rgb_to_depth_ds, split_lengths)

    print(f'[I (rgbd_gradients_dataloader)] - |Train Dataset|={len(ds_train)}, |Test Dataset|={len(ds_test)}')
    dl_train = torch.utils.data.DataLoader(ds_train,
                                           batch_size=batch_size,
                                           num_workers=num_workers,
                                           shuffle=True)
    dl_test  = torch.utils.data.DataLoader(ds_test,
                                           batch_size=batch_size,
                                           num_workers=num_workers,
                                           shuffle=True)
    return (dl_train, dl_test)
