import numpy as np
import torch
import os
import cv2
from torchvision import transforms as T
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader

def one_one_normalization(data):
    return 2*((data-np.min(data))/(np.max(data)-np.min(data)))-1


def calc_grads(img: torch.Tensor, ksize=5, grads_degree=1, normalization=True):
    assert isinstance(img,torch.Tensor)

    img_np = img.cpu().numpy().squeeze()
    grad_x = cv2.Sobel(img_np, cv2.CV_64F, grads_degree, 0, ksize=ksize)
    grad_y = cv2.Sobel(img_np, cv2.CV_64F, 0, grads_degree, ksize=ksize)

    if normalization:
        grad_x = one_one_normalization(grad_x)
        grad_y = one_one_normalization(grad_y)

    grad_x = np.expand_dims(grad_x, axis=2)
    grad_y = np.expand_dims(grad_y, axis=2)

    grad_x = T.ToTensor()(grad_x).type(torch.float32)
    grad_y = T.ToTensor()(grad_y).type(torch.float32)

    return grad_x,grad_y


class rgbd_gradients_dataset(Dataset):
    def __init__(self, root, rgb_transforms=None, depth_transforms=None):

        self.root             = root
        self.rgb_transforms   = rgb_transforms
        self.depth_transforms = depth_transforms

        # load all image files, sorting them to
        # ensure that they are aligned
        self.rgbs   = list(sorted(os.listdir(os.path.join(root, 'rgb'))))
        self.depths = list(sorted(os.listdir(os.path.join(root, 'depth'))))

        if not (len(self.rgbs) == len(self.depths)):
            raise Exception("Non-equal number of samples from each kind.")

        self.len = len(self.rgbs)

    # def __load_image__(self, path):
    #     # TODO: Ask Haim. I think it would harm performances due to memory shortage &
    #     #  the need to search the dictionary on each call instead of just to load an image when it needed)
    #     #  (manorz, 03/06/20)
    #     if path not in self.RAM_images: # Saves to RAM, to prevent a hard drive bottleneck.
    #         self.RAM_images[path] = Image.open(path + ".png")
    #         if self.transform:
    #             self.RAM_images[path] = self.transform(self.RAM_images[path])
    #     return self.RAM_images[path]

    # def __load_gradient__(self, path):
    #     if path not in self.RAM_gardients: # Saves to RAM, to prevent a hard drive bottleneck.
    #         self.RAM_gardients[path] = torch.from_numpy(np.load(path + ".npy"))
    #     return self.RAM_gardients[path]

    def __getitem__(self, index):
        # TODO: Uncomment if decided that the RAM thing above is better than standard implementation. (manorz, 03/06/20)
        # X_rgb   = self.__load_image__(self.rgb_dir + str(index))
        # X_depth = self.__load_image__(self.depth_dir + str(index))
        # Y_x     = self.__load_gradient__(self.gradx_dir + str(index))
        # Y_y     = self.__load_gradient__(self.grady_dir + str(index))

        rgb_path   = os.path.join(self.root, "rgb",   self.rgbs[index])
        d_path     = os.path.join(self.root, "depth", self.depths[index])

        rgb   = Image.open(rgb_path)
        d     = Image.open(d_path)

        if self.rgb_transforms:
            rgb = self.rgb_transforms(rgb)
        if self.depth_transforms:
            d   = self.depth_transforms(d)

        x,y = calc_grads(d)

        return {'rgb': rgb,
                'depth': d,
                'x'    : x,
                'y'    : y}

    def __len__(self):
        return self.len


def rgbd_gradients_dataloader(root, train_test_ration, batch_size, num_workers,
                              rgb_transforms=None, depth_transforms=None):
    rgbd_grads_ds = rgbd_gradients_dataset(root, rgb_transforms=rgb_transforms, depth_transforms=depth_transforms)
    split_lengths = [int(np.ceil(len(rgbd_grads_ds)  *    train_test_ration)),
                     int(np.floor(len(rgbd_grads_ds) * (1-train_test_ration)))]

    ds_train, ds_test = random_split(rgbd_grads_ds, split_lengths)

    dl_train = torch.utils.data.DataLoader(ds_train,
                                           batch_size=batch_size,
                                           num_workers=num_workers,
                                           shuffle=True)
    dl_test  = torch.utils.data.DataLoader(ds_test,
                                           batch_size=batch_size,
                                           num_workers=num_workers,
                                           shuffle=True)
    return (dl_train, dl_test)


if __name__ == '__main__':

    CONST_NUMBER_OF_GPUS = torch.cuda.device_count()
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'DEVICE={DEVICE}')
    print(f"GPU's={CONST_NUMBER_OF_GPUS}")

    pwd = os.getcwd()
    data_dir = os.path.join(pwd, 'data', 'nyuv2')

    rgb_dir = os.path.join(data_dir, 'rgb')
    depth_dir = os.path.join(data_dir, 'depth')
    gradx_dir = os.path.join(data_dir, 'x/')
    grady_dir = os.path.join(data_dir, 'y/')
    print(f'{rgb_dir}\n{depth_dir}\n{gradx_dir}\n{grady_dir}')

    batch_size = 64
    # Set CONST_NUMBER_OF_GPUS above,
    # found this formula in https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/5
    num_workers = 4 * CONST_NUMBER_OF_GPUS