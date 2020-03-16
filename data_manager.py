import numpy as np
import torch
import os
import cv2
import random
import PIL
from hyperparameters import *
from torchvision import transforms as T
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader

def one_one_normalization(data):
    return 2*((data-np.min(data))/(np.max(data)-np.min(data)))-1

def calc_grads(img: torch.Tensor, grads_degree=1, normalization=True):
    assert isinstance(img,torch.Tensor)
    
    # Use weighted average ksizes
    ksizes_and_weights = [{"ksize": 3, "weight": 0.7},
                          {"ksize": 5, "weight": 0.3}]
    img_np = img.cpu().numpy().squeeze()
    grad_x = np.zeros(img_np.shape, dtype=np.float64)
    grad_y = np.zeros(img_np.shape, dtype=np.float64)
    for ksize_and_weight in ksizes_and_weights:
        grad_x = np.add(grad_x, ksize_and_weight["weight"] * cv2.Sobel(img_np, cv2.CV_64F, grads_degree, 0, ksize=ksize_and_weight["ksize"]))
        grad_y = np.add(grad_y, ksize_and_weight["weight"] * cv2.Sobel(img_np, cv2.CV_64F, 0, grads_degree, ksize=ksize_and_weight["ksize"]))


    if normalization:
        grad_x = one_one_normalization(grad_x)
        grad_y = one_one_normalization(grad_y)

    grad_x = np.expand_dims(grad_x, axis=2)
    grad_y = np.expand_dims(grad_y, axis=2)

    grad_x = T.ToTensor()(grad_x).type(torch.float32)
    grad_y = T.ToTensor()(grad_y).type(torch.float32)

    return grad_x,grad_y


class rgbd_gradients_dataset(Dataset):
    def __init__(self, root, use_transforms=False):
        self.root             = root
        self.use_transforms   = use_transforms

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

    def transform(self, rgb, depth):
        # Resize to constant spatial dimensions
        rgb = T.Resize(IMAGE_SIZE)(rgb)
        depth = T.Resize(IMAGE_SIZE)(depth)
            
        # Random horizontal flipping
        if 0.5 < random.random():
            rgb = T.RandomHorizontalFlip(p=1.0)(rgb)
            depth = T.RandomHorizontalFlip(p=1.0)(depth)
        
        # Randomly changes the brightness, contrast and saturation of an image.
        # Example: https://discuss.pytorch.org/t/data-augmentation-in-pytorch/7925/15
        rgb = T.ColorJitter(
            brightness=abs(0.1 * torch.randn(1).item()),
            contrast=abs(0.1 * torch.randn(1).item()),
            saturation=abs(0.1 * torch.randn(1).item()),
            hue=abs(0.1 * torch.randn(1).item())
        )(rgb)
        
        # Crops and resizes back to IMAGE_SIZE
        minimum_image_precentage_to_look_at = 0.7
        image_precentage_to_look_at = random.uniform(minimum_image_precentage_to_look_at, 1.0)
        crop_width = int(image_precentage_to_look_at * IMAGE_SIZE[0])
        crop_height = int(image_precentage_to_look_at * IMAGE_SIZE[1])
        crop_top = random.randint(0, IMAGE_SIZE[0] - crop_height)
        crop_left = random.randint(0, IMAGE_SIZE[1] - crop_width)
        rgb = T.functional.resized_crop(rgb, crop_top, crop_left, crop_height, crop_width, IMAGE_SIZE)
        depth = T.functional.resized_crop(depth, crop_top, crop_left, crop_height, crop_width, IMAGE_SIZE)
        
        # Rotation
        degrees_range = 20 # (degrees_range X 2) = range
        padding = max(IMAGE_SIZE[0], IMAGE_SIZE[1])
        frame_removal_length = 3
        rgb = T.functional.resized_crop(rgb, frame_removal_length, frame_removal_length, IMAGE_SIZE[0] - 2 * frame_removal_length, IMAGE_SIZE[1] - 2 * frame_removal_length, IMAGE_SIZE)
        depth = T.functional.resized_crop(depth, frame_removal_length, frame_removal_length, IMAGE_SIZE[0] - 2 * frame_removal_length, IMAGE_SIZE[1] - 2 * frame_removal_length, IMAGE_SIZE)
        rgb = T.Pad(padding=padding, fill=0, padding_mode='edge')(rgb)
        depth = T.Pad(padding=padding, fill=0, padding_mode='edge')(depth)
        degree_to_rotate = random.uniform(-degrees_range, degrees_range)
        # some bug in torchvision, got the solution at:
        # https://github.com/pytorch/vision/issues/1759
        # https://www.gitmemory.com/issue/pytorch/vision/1759/575357711
        #filler = 0.0 if rgb.mode.startswith("F") else 0
        num_bands = len(rgb.getbands())
        rgb = T.functional.rotate(img=rgb, angle=degree_to_rotate, fill=(0,) * num_bands)
        #filler = 0.0 if depth.mode.startswith("F") else 0
        num_bands = len(depth.getbands())
        depth = T.functional.rotate(img=depth, angle=degree_to_rotate, fill=(0,) * num_bands)
        rgb = T.functional.resized_crop(rgb, padding, padding, IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_SIZE)
        depth = T.functional.resized_crop(depth, padding, padding, IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_SIZE)
    
        # USE THIS ??? torchvision.transforms.functional.perspective(img, startpoints, endpoints, interpolation=3)
        
        # PIL.Image -> torch.Tensor
        rgb = T.ToTensor()(rgb)
        depth = T.ToTensor()(depth)
        
        # Dynamic range [0,1] -> [-1, 1]
        rgb = T.Normalize(mean=(.5,.5,.5), std=(.5,.5,.5))(rgb)
        depth = T.Normalize(mean=(.5,), std=(.5,))(depth)
        
        return rgb, depth
    
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

        if self.use_transforms:
            rgb, d = self.transform(rgb, d)

        x,y = calc_grads(d)

        return {'rgb': rgb,
                'depth': d,
                'x'    : x,
                'y'    : y}

    def __len__(self):
        return self.len


def rgbd_gradients_dataloader(root, train_test_ration, batch_size, num_workers, use_transforms=False):
    rgbd_grads_ds = rgbd_gradients_dataset(root, use_transforms=use_transforms)
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