import numpy as np
import torch
import os
from copy import deepcopy
import cv2
import random
from torchvision import transforms as T
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import random_split
from gradients_to_navigation import Gradients_to_navigation

def one_one_normalization(data):
    return 2*((data-np.min(data))/(np.max(data)-np.min(data)))-1


def calc_grads(img: torch.Tensor, ksizes:list=[3,5], kweights=[0.7,0.5], grads_degree=1, normalization=True):
    assert isinstance(img,torch.Tensor)
    assert isinstance(ksizes,list) and isinstance(kweights, list)

    img_np = img.cpu().numpy().squeeze()
    # grad_x = cv2.Sobel(img_np, cv2.CV_64F, grads_degree, 0, ksize=ksizes[0])
    # grad_y = cv2.Sobel(img_np, cv2.CV_64F, 0, grads_degree, ksize=ksizes[0])

    for i,(size,weight) in enumerate(zip(ksizes,kweights)):
        if i == 0:
            grad_x = cv2.Sobel(img_np, cv2.CV_64F, grads_degree, 0, ksize=size) * weight
            grad_y = cv2.Sobel(img_np, cv2.CV_64F, 0, grads_degree, ksize=size) * weight
        else:
            grad_x += cv2.Sobel(img_np, cv2.CV_64F, grads_degree, 0, ksize=size) * weight
            grad_y += cv2.Sobel(img_np, cv2.CV_64F, 0, grads_degree, ksize=size) * weight

    if normalization:
        grad_x = one_one_normalization(grad_x)
        grad_y = one_one_normalization(grad_y)

    grad_x = np.expand_dims(grad_x, axis=2)
    grad_y = np.expand_dims(grad_y, axis=2)

    grad_x = T.ToTensor()(grad_x).type(torch.float32)
    grad_y = T.ToTensor()(grad_y).type(torch.float32)

    return grad_x,grad_y


class RandomCropAndResize(object):
    """Crop randomly the image in a sample, while keeping the aspect ratio, then resize
    back to the image's original size.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
        minimum_image_percentage_to_look_at(float): Minimum percentage of image to keep after cropping.
    """

    def __init__(self, output_size, minimum_image_percentage_to_look_at=0.7):
        assert isinstance(output_size, (int, tuple))
        assert isinstance(minimum_image_percentage_to_look_at, float)
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.minimum_image_percentage_to_look_at = minimum_image_percentage_to_look_at

    def __call__(self, rgb, depth):
        image_precentage_to_look_at = random.uniform(self.minimum_image_percentage_to_look_at, 1.0)
        crop_width  = int(image_precentage_to_look_at * self.output_size[1])
        crop_height = int(image_precentage_to_look_at * self.output_size[0])
        crop_top    = random.randint(0, self.output_size[0] - crop_height)
        crop_left   = random.randint(0, self.output_size[1] - crop_width)
        rgb         = T.functional.resized_crop(rgb,   crop_top, crop_left, crop_height, crop_width, self.output_size)
        depth       = T.functional.resized_crop(depth, crop_top, crop_left, crop_height, crop_width, self.output_size)
        return rgb, depth


class RotateAndFillCornersWithImageFrameColors(object):
    """Rotates both image by the same angle, and fills the created corners by the colors of the images' frames
    pixels.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
        degrees_range(float): Maximum degree to rotate to each side. (degrees_range X 2) = total range
        frame_removal_length(int): Number of pixels to remove from the images' frames, as they may contain useless colors.
    """

    def __init__(self, output_size, degrees_range=20, frame_removal_length=3):
        assert isinstance(output_size, (int, tuple))
        assert isinstance(degrees_range, (int, float))
        assert isinstance(frame_removal_length, int)
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.degrees_range = degrees_range
        self.frame_removal_length = frame_removal_length

    def __call__(self, rgb, depth):
        padding = max(self.output_size[0], self.output_size[1])
        rgb     = T.functional.resized_crop(rgb, self.frame_removal_length, self.frame_removal_length, self.output_size[0] - 2 * self.frame_removal_length, self.output_size[1] - 2 * self.frame_removal_length, self.output_size)
        depth   = T.functional.resized_crop(depth, self.frame_removal_length, self.frame_removal_length, self.output_size[0] - 2 * self.frame_removal_length, self.output_size[1] - 2 * self.frame_removal_length, self.output_size)
        rgb     = T.Pad(padding=padding, fill=0, padding_mode='edge')(rgb)
        depth   = T.Pad(padding=padding, fill=0, padding_mode='edge')(depth)
        degree_to_rotate = random.uniform(-self.degrees_range, self.degrees_range)
        # some bug in torchvision, got the solution at:
        # https://github.com/pytorch/vision/issues/1759
        # https://www.gitmemory.com/issue/pytorch/vision/1759/575357711
        #filler = 0.0 if rgb.mode.startswith("F") else 0
        num_bands = len(rgb.getbands())
        rgb       = T.functional.rotate(img=rgb, angle=degree_to_rotate, fill=(0,) * num_bands)
        # filler    = 0.0 if depth.mode.startswith("F") else 0
        num_bands = len(depth.getbands())
        depth     = T.functional.rotate(img=depth, angle=degree_to_rotate, fill=(0,) * num_bands)
        rgb       = T.functional.resized_crop(rgb,   padding, padding, self.output_size[0], self.output_size[1], self.output_size)
        depth     = T.functional.resized_crop(depth, padding, padding, self.output_size[0], self.output_size[1], self.output_size)
        return rgb, depth


class rgbd_gradients_dataset(Dataset):
    def __init__(self, root, image_size, goto_pixel,
                 use_transforms=False, overfit_mode=False):
        print(f'[I (rgbd_gradients_dataset)] - root={root}\n'
              f'                             - image_size={image_size}\n'
              f'                             - use_transforms={use_transforms}\n'
              f'                             - overfit_mode={overfit_mode}\n'
              f'                             - goto_pixel={goto_pixel}\n')


        self.root             = root
        self.use_transforms   = use_transforms
        self.overfit_mode     = overfit_mode
        self.image_size       = image_size
        self.goto_pixel       = goto_pixel

        if self.goto_pixel:
            self.grads2nav = Gradients_to_navigation()

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
            
        if not self.overfit_mode:
            # Random horizontal flipping
            if 0.5 < random.random():
                rgb   = T.RandomHorizontalFlip(p=1.0)(rgb)
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
            rgb, depth = RandomCropAndResize(output_size=self.image_size,
                                             minimum_image_percentage_to_look_at=0.7)(rgb, depth)
            # Rotation
            rgb, depth = RotateAndFillCornersWithImageFrameColors(output_size=self.image_size,
                                                                  degrees_range=20, frame_removal_length=3)(rgb, depth)

        # PIL.Image -> torch.Tensor
        rgb   = T.ToTensor()(rgb)
        depth = T.ToTensor()(depth)
        
        # Dynamic range [0,1] -> [-1, 1]
        rgb   = T.Normalize(mean=(.5,.5,.5), std=(.5,.5,.5))(rgb)
        depth = T.Normalize(mean=(.5,), std=(.5,))(depth)
        
        return rgb, depth
    
    def __getitem__(self, index):

        rgb_path   = os.path.join(self.root, "rgb",   self.rgbs[index])
        d_path     = os.path.join(self.root, "depth", self.depths[index])

        rgb   = Image.open(rgb_path)
        d     = Image.open(d_path)

        if self.use_transforms:
            rgb, d = self.transform(rgb, d)

        x,y = calc_grads(d)

        ret_val = {
            'rgb': rgb,
            'depth': d,
            'x': x,
            'y': y
        }

        if self.goto_pixel:
            navigation_image       = self.grads2nav.calculate_navigation_image(x=x.squeeze(0), y=y.squeeze(0))
            navigate2x, navigate2y = self.grads2nav.calculate_goto_pixel(navigation_image=navigation_image)
            goto_pixel = torch.from_numpy(np.array([navigate2x, navigate2y])).type(torch.float32)
            ret_val['goto_pixel'] = goto_pixel

        return ret_val

    def __len__(self):
        return self.len


class rgbd_gradients_inference_dataset(Dataset):
    def __init__(self, device,  root, image_size, goto_pixel, inference='both'):
        print(f'[I (rgbd_gradients_inference_dataset)] - root={root}\n'
              f'                                       - device={device}\n'
              f'                                       - image_size={image_size}\n'
              f'                                       - inference={inference}\n'
              f'                                       - goto_pixel={goto_pixel}\n')

        assert inference in ['both', 'zero_depth', 'zero_rgb', 'noise_depth', 'noise_rgb'], \
            f"inference ({inference}) not in " \
            f"['both', 'zero_depth', 'zero_rgb', 'noise_depth', 'noise_rgb']"

        self.device     = device
        self.root       = root
        self.image_size = image_size
        self.inference  = inference
        self.goto_pixel = goto_pixel

        if self.goto_pixel:
            self.grads2nav = Gradients_to_navigation()

        # load all image files, sorting them to
        # ensure that they are aligned
        self.rgbs = list(sorted(os.listdir(os.path.join(root, 'rgb'))))
        self.depths = list(sorted(os.listdir(os.path.join(root, 'depth'))))

        if not (len(self.rgbs) == len(self.depths)):
            raise Exception(f"Non-equal number of samples from each kind "
                            f"(|rgbs|={len(self.rgbs)} != |depths|={len(self.depths)})")

        self.len = len(self.rgbs)
        print(f'[I] - |self|={len(self)}')

    def __getitem__(self, index):

        rgb_path = os.path.join(self.root, "rgb", self.rgbs[index])
        d_path = os.path.join(self.root, "depth", self.depths[index])

        if self.inference == 'both':
            rgb = Image.open(rgb_path)
            d   = Image.open(d_path)
            # Resize to constant spatial dimensions
            rgb = T.Resize(self.image_size)(rgb)
            d = T.Resize(self.image_size)(d)
            # PIL.Image -> torch.Tensor
            rgb = T.ToTensor()(rgb)
            d = T.ToTensor()(d)
            # Dynamic range [0,1] -> [-1, 1]
            rgb = T.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5))(rgb)
            d = T.Normalize(mean=(.5,), std=(.5,))(d)
            x, y = calc_grads(d)
        elif self.inference == 'zero_rgb' or self.inference == 'noise_rgb':
            d = Image.open(d_path)
            d = T.Resize(self.image_size)(d)
            d = T.ToTensor()(d)
            d = T.Normalize(mean=(.5,), std=(.5,))(d)
            if self.inference == 'zero_rgb':
                rgb = torch.zeros((3,d.shape[1],d.shape[2]),dtype=torch.float32)
            else:

                rgb = torch.rand((3,d.shape[1],d.shape[2]))
                rgb = T.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5))(rgb)
            x, y = calc_grads(d)
        elif self.inference == 'zero_depth' or self.inference == 'noise_depth':
            rgb = Image.open(rgb_path)
            rgb = T.Resize(self.image_size)(rgb)
            rgb = T.ToTensor()(rgb)
            rgb = T.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5))(rgb)
            # Load Depth Image for Ground Truth grads calculations only
            d = Image.open(d_path)
            d = T.Resize(self.image_size)(d)
            d = T.ToTensor()(d)
            d = T.Normalize(mean=(.5,), std=(.5,))(d)
            x, y = calc_grads(d)
            if self.inference == 'zero_depth':
                d = torch.zeros((1,rgb.shape[1],rgb.shape[2]),dtype=torch.float32)
            else:
                d = torch.rand((1, rgb.shape[1], rgb.shape[2]))
                d = T.Normalize(mean=(.5,), std=(.5,))(d)

        ret_val = {
            'rgb': rgb,
            'depth': d,
            'x': x,
            'y': y
        }

        if self.goto_pixel:
            navigation_image = self.grads2nav.calculate_navigation_image(x=x.squeeze(0), y=y.squeeze(0))
            navigate2x, navigate2y = self.grads2nav.calculate_goto_pixel(navigation_image=navigation_image)
            goto_pixel = torch.from_numpy(np.array([navigate2x, navigate2y])).type(torch.float32)
            ret_val['goto_pixel'] = goto_pixel

        return ret_val

    def __len__(self):
        return self.len


class rgb2depth_dataset(Dataset):
    def __init__(self, root, image_size=(128,128), fusenet_image_size=(64,64)):
        print(f'[I (rgb2depth_dataset)] - root={root}\n'
              f'                        - image_size={image_size}\n'
              f'                        - fusenet_image_size={fusenet_image_size}\n')

        self.root = root
        self.image_size = image_size
        self.fusenet_image_size = fusenet_image_size

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

    def transform4fusenet(self, rgb, depth):
        # Resize to constant spatial dimensions
        rgb   = T.Resize(self.fusenet_image_size)(rgb)
        depth = T.Resize(self.fusenet_image_size)(depth)

        # PIL.Image -> torch.Tensor
        rgb   = T.ToTensor()(rgb)
        depth = T.ToTensor()(depth)

        # Dynamic range [0,1] -> [-1, 1]
        rgb   = T.Normalize(mean=(.5,.5,.5), std=(.5,.5,.5))(rgb)
        depth = T.Normalize(mean=(.5,), std=(.5,))(depth)

        return rgb, depth

    def __getitem__(self, index):
        rgb_pil = Image.open(os.path.join(self.root, "rgb",   self.rgbs[index]))
        d_pil   = Image.open(os.path.join(self.root, "depth", self.depths[index]))

        rgb, d = self.transform(rgb_pil, d_pil)
        rgb4fusenet, d4fusenet = self.transform4fusenet(rgb_pil, d_pil)
        x4fusenet, y4fusenet = calc_grads(d4fusenet)

        return {'rgb'           : rgb,
                'depth'         : d,
                'rgb4fusenet'   : rgb4fusenet,
                'depth4fusenet' : d4fusenet,
                'x4fusenet'     : x4fusenet,
                'y4fusenet'     : y4fusenet}

    def __len__(self):
        return self.len


def rgbd_gradients_dataloader(device, root, batch_size, num_workers, train_test_ratio, image_size,
                              use_transforms=False, overfit_mode=False, seed=42, goto_pixel=True):
    print(f'[I (rgbd_gradients_dataloader)] - root={root}\n'
          f'                                - device={device}\n'
          f'                                - batch_size={batch_size}\n'
          f'                                - num_workers={num_workers}\n'
          f'                                - train_test_ratio={train_test_ratio}\n'
          f'                                - image_size={image_size}\n'
          f'                                - use_transforms={use_transforms}\n'
          f'                                - overfit_mode={overfit_mode}\n'
          f'                                - seed={seed}\n'
          f'                                - goto_pixel={goto_pixel}\n')


    torch.manual_seed(seed)

    rgbd_grads_ds = rgbd_gradients_dataset(root, image_size, use_transforms=use_transforms,
                                            overfit_mode=overfit_mode, goto_pixel=goto_pixel)

    if not overfit_mode:
        split_lengths = [int(np.ceil(len(rgbd_grads_ds)  *    train_test_ratio)),
                         int(np.floor(len(rgbd_grads_ds) * (1-train_test_ratio)))]

        # NOTE: Don't forget to set the seed in order to maintain reproducibility over training experiments.
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
    else:
        dl_overfit = torch.utils.data.DataLoader(rgbd_grads_ds,
                                                 batch_size=batch_size,
                                                 num_workers=num_workers,
                                                 shuffle=True)
        print(f'[D (rgbd_gradients_dataloader)] - |Dataset|={len(rgbd_grads_ds)}')
        return (dl_overfit, deepcopy(dl_overfit)) # dl_train & dl_test equals and consist of a single image.
