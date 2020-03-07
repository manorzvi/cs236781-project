import numpy as np
import torch
import os
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset

# Hardware constants
CONST_NUMBER_OF_GPUS = torch.cuda.device_count()
# Software constants
base_data_path = 'data/'
train_data_path = base_data_path + 'train/'
test_data_path = base_data_path + 'test/'
rgb_data_dir_name = 'rgb/'
depth_data_dir_name = 'depth/'
x_data_dir_name = 'x/'
y_data_dir_name = 'y/'
DataLoaders_batch_size = 64
DataLoaders_num_workers = 4 * CONST_NUMBER_OF_GPUS # Set CONST_NUMBER_OF_GPUS above, found this formula in https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/5
# End all constants

train_rgb_data_path = train_data_path + rgb_data_dir_name
train_depth_data_path = train_data_path + depth_data_dir_name
train_x_data_path = train_data_path + x_data_dir_name
train_y_data_path = train_data_path + y_data_dir_name
test_rgb_data_path = test_data_path + rgb_data_dir_name
test_depth_data_path = test_data_path + depth_data_dir_name
test_x_data_path = test_data_path + x_data_dir_name
test_y_data_path = test_data_path + y_data_dir_name
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def number_of_files_in_folder(folder_path):
    return len(next(os.walk(folder_path))[2])

class Hidden_Spaces_Edges_Dataset(Dataset):
    def __init__(self, rgb_data_path, depth_data_path, x_data_path, y_data_path, transform=None):
        number_of_files_in_rgb_folder = number_of_files_in_folder(rgb_data_path)
        number_of_files_in_depth_folder = number_of_files_in_folder(depth_data_path)
        number_of_files_in_x_folder = number_of_files_in_folder(x_data_path)
        number_of_files_in_y_folder = number_of_files_in_folder(y_data_path)
        if number_of_files_in_rgb_folder == number_of_files_in_depth_folder and\
            number_of_files_in_depth_folder == number_of_files_in_x_folder and\
            number_of_files_in_x_folder == number_of_files_in_y_folder:
            self.len = number_of_files_in_rgb_folder
        else:
            raise Exception("Not all of the dataset's subfolders contain the same number of examples.")
        self.rgb_data_path = rgb_data_path
        self.depth_data_path = depth_data_path
        self.x_data_path = x_data_path
        self.y_data_path = y_data_path
        self.transform = transform
        self.loaded_to_RAM_images = {}
        self.loaded_to_RAM_gardients = {}

    def __load_image_and_maybe_transform__(self, path):
        if path not in self.loaded_to_RAM_images: # Saves to RAM, to prevent a hard drive bottleneck.
            self.loaded_to_RAM_images[path] = Image.open(path + ".png")
            if self.transform:
                self.loaded_to_RAM_images[path] = self.transform(self.loaded_to_RAM_images[path]).to(device)
        return self.loaded_to_RAM_images[path]

    def __load_gardient__(self, path):
        if path not in self.loaded_to_RAM_gardients: # Saves to RAM, to prevent a hard drive bottleneck.
            self.loaded_to_RAM_gardients[path] = torch.from_numpy(np.load(path + ".npy")).to(device)
        return self.loaded_to_RAM_gardients[path]

    def __getitem__(self, index):
        X_rgb = self.__load_image_and_maybe_transform__(path=self.rgb_data_path + str(index))
        X_depth = self.__load_image_and_maybe_transform__(path=self.depth_data_path + str(index))
        Y_x = self.__load_gardient__(self.x_data_path + str(index))
        Y_y = self.__load_gardient__(self.y_data_path + str(index))
        return (X_rgb, X_depth, (Y_x, Y_y))

    def __len__(self):
        return self.len

def load_Hidden_Spaces_Edges_train_and_test_datasets():
    def create_a_Dataset_in_a_DataLoader(rgb_data_path, depth_data_path, x_data_path, y_data_path, transform=None):
        hidden_Spaces_Edges_Dataset = Hidden_Spaces_Edges_Dataset(rgb_data_path=rgb_data_path,
                                                                  depth_data_path=depth_data_path,
                                                                  x_data_path=x_data_path,
                                                                  y_data_path=y_data_path,
                                                                  transform=transform)
        return torch.utils.data.DataLoader(
            hidden_Spaces_Edges_Dataset,
            batch_size=DataLoaders_batch_size,
            num_workers=DataLoaders_num_workers,
            shuffle=True
        )

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_DataLoader = create_a_Dataset_in_a_DataLoader(rgb_data_path=train_rgb_data_path,
                                                        depth_data_path=train_depth_data_path,
                                                        x_data_path=train_x_data_path,
                                                        y_data_path=train_y_data_path,
                                                        transform=transform)
    test_DataLoader = create_a_Dataset_in_a_DataLoader(rgb_data_path=test_rgb_data_path,
                                                       depth_data_path=test_depth_data_path,
                                                       x_data_path=test_x_data_path,
                                                       y_data_path=test_y_data_path,
                                                       transform=transform)
    return (train_DataLoader, test_DataLoader)
    # ---------------------------------------------------------------------------------CAN DELETE THE COMMENTED BELOW
    # train_x_rgb_dataset = torchvision.datasets.ImageFolder(
    #     root=train_rgb_data_path,
    #     transform=torchvision.transforms.ToTensor()
    # )
    # print(len(train_x_rgb_dataset))
    # # train_loader = torch.utils.data.DataLoader(
    # #     train_images_dataset,
    # #     batch_size=64,
    # #     num_workers=0,
    # #     shuffle=True
    # # )
    # train_x_depth_dataset = torchvision.datasets.ImageFolder(
    #     root=train_depth_data_path,
    #     transform=torchvision.transforms.ToTensor()
    # )
    # print(len(train_x_depth_dataset))
    # def npy_loader(path):
    #     sample = torch.from_numpy(np.load(path))
    #     return sample
    #
    # train_y_x_dataset = torchvision.datasets.DatasetFolder(
    #     root=train_x_data_path,
    #     loader=npy_loader,
    #     extensions='.npy')
    #
    # train_y_y_dataset = torchvision.datasets.DatasetFolder(
    #     root=train_y_data_path,
    #     loader=npy_loader,
    #     extensions='.npy')
    #
    # # train = torch.utils.data.TensorDataset(train_x_rgb_dataset, train_x_depth_dataset, train_y_x_dataset, train_y_y_dataset)
    # train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=64, shuffle=True, num_workers=0)
    # return train_DataLoader, test_DataLoader

if __name__ == '__main__':
    (train_DataLoader, test_DataLoader) = load_Hidden_Spaces_Edges_train_and_test_datasets()
    for batch_idx, (X_rgb, X_depth, (Y_x, Y_y)) in enumerate(train_DataLoader):
        # Train the network
        a = 1
    for batch_idx, (X_rgb, X_depth, (Y_x, Y_y)) in enumerate(test_DataLoader):
        # Test the network
        a = 1