import torch
from torchvision import models
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from copy import deepcopy
from functions import init_weights
from torch.nn import DataParallel

def make_encoder_cbr(names: list,model_dict: dict,bn_dim: float,
                     existing_layer=None, bn_momentum=0.1):
    if existing_layer:
        # Existing layer, followed by <names> layers
        layers = [existing_layer,
                  nn.BatchNorm2d(bn_dim, momentum=0.1),
                  nn.ReLU(inplace=True)]
    else:
        layers = []

    for name in names:
        layers += [deepcopy(model_dict[name]),
                   nn.BatchNorm2d(bn_dim, momentum=bn_momentum),
                   nn.ReLU(inplace=True)]

    return nn.Sequential(*layers)

def make_decoder_cbr(sizes: list, kernel_size=3, padding=1, bn_momentum=0.1):
    layers = []
    for size in sizes:
        layers += [nn.Conv2d(size[0], size[1], kernel_size=kernel_size, padding=padding),
                   nn.BatchNorm2d(size[1], momentum=bn_momentum),
                   nn.ReLU(inplace=True)]

    return nn.Sequential(*layers)


class SpecialFuseNet(nn.Module):
    def __init__(self, warm_start=True, bn_momentum=0.1):
        super().__init__()

        # Extract Conv2d layers only from VGG16 model (Encoder Warm Start, according to the paper)
        layers_names = ["conv1_1", "conv1_2", "conv2_1", "conv2_2", "conv3_1", "conv3_2", "conv3_3", "conv4_1",
                        "conv4_2", "conv4_3", "conv5_1", "conv5_2", "conv5_3"]
        layers = list(models.vgg16(pretrained=warm_start).features.children())
        layers = [x for x in layers if isinstance(x, nn.Conv2d)]
        layers_dict = dict(zip(layers_names, layers))

        self.need_initialization = []

        # -------------------------------------------------------------------
        # --------------------------- RGB Encoder ---------------------------
        # -------------------------------------------------------------------
        self.CBR1_RGB_ENC = make_encoder_cbr(["conv1_1", "conv1_2"], layers_dict, 64)
        self.RGB_POOL1    = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.CBR2_RGB_ENC = make_encoder_cbr(["conv2_1", "conv2_2"], layers_dict, 128)
        self.RGB_POOL2    = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.CBR3_RGB_ENC = make_encoder_cbr(["conv3_1", "conv3_2", "conv3_3"], layers_dict, 256)
        self.RGB_POOL3    = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.RGB_DROPOUT3 = nn.Dropout(p=0.4)

        self.CBR4_RGB_ENC = make_encoder_cbr(["conv4_1", "conv4_2", "conv4_3"], layers_dict, 512)
        self.RGB_POOL4    = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.RGB_DROPOUT4 = nn.Dropout(p=0.4)

        self.CBR5_RGB_ENC = make_encoder_cbr(["conv5_1", "conv5_2", "conv5_3"], layers_dict, 512)
        self.RGB_POOL5    = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.RGB_DROPOUT5 = nn.Dropout(p=0.4)

        # ---------------------------------------------------------------------
        # --------------------------- Depth Encoder ---------------------------
        # ---------------------------------------------------------------------
        # Depth image is 1D image,
        # therefore we average VGG16's first convolution layer weight on the channel dimension
        avg = torch.mean(layers_dict['conv1_1'].weight.data, dim=1)
        avg = avg.unsqueeze(1)
        conv11d = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        conv11d.weight.data = avg

        # We don't use VGG16's conv1_1 layer because we used it for our previous layer (conv11d)
        self.CBR1_DEPTH_ENC = make_encoder_cbr(["conv1_2"], layers_dict, 64, existing_layer=conv11d)
        self.DEPTH_POOL1    = nn.MaxPool2d(kernel_size=2, stride=2)

        self.CBR2_DEPTH_ENC = make_encoder_cbr(["conv2_1", "conv2_2"], layers_dict, 128)
        self.DEPTH_POOL2    = nn.MaxPool2d(kernel_size=2, stride=2)

        self.CBR3_DEPTH_ENC = make_encoder_cbr(["conv3_1", "conv3_2", "conv3_3"], layers_dict, 256)
        self.DEPTH_POOL3    = nn.MaxPool2d(kernel_size=2, stride=2)
        self.DEPTH_DROPOUT3 = nn.Dropout(p=0.4)

        self.CBR4_DEPTH_ENC = make_encoder_cbr(["conv4_1", "conv4_2", "conv4_3"], layers_dict, 512)
        self.DEPTH_POOL4    = nn.MaxPool2d(kernel_size=2, stride=2)
        self.DEPTH_DROPOUT4 = nn.Dropout(p=0.4)

        # According to the paper, no MaxPool & Dropout on Depth's last block
        self.CBR5_DEPTH_ENC = make_encoder_cbr(["conv5_1", "conv5_2", "conv5_3"], layers_dict, 512)

        # ---------------------------------------------------------------------
        # ---------------------------- RGB Decoder ----------------------------
        # ---------------------------------------------------------------------
        self.UNPOOL5      = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.CBR5_RGB_DEC = make_decoder_cbr([(512, 512), (512, 512), (512, 512)])
        self.DROPOUT5_DEC = nn.Dropout(p=0.4)

        self.need_initialization.append('CBR5_RGB_DEC')

        self.UNPOOL4      = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.CBR4_RGB_DEC = make_decoder_cbr([(512, 512), (512, 512), (512, 256)])
        self.DROPOUT4_DEC = nn.Dropout(p=0.4)

        self.need_initialization.append('CBR4_RGB_DEC')

        self.UNPOOL3      = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.CBR3_RGB_DEC = make_decoder_cbr([(256, 256), (256, 256), (256, 128)])
        self.DROPOUT3_DEC = nn.Dropout(p=0.4)

        self.need_initialization.append('CBR3_RGB_DEC')

        self.UNPOOL2      = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.CBR2_RGB_DEC = make_decoder_cbr([(128, 128), (128, 64)])

        self.need_initialization.append('CBR2_RGB_DEC')

        self.UNPOOL1      = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.CBR1_RGB_DEC = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=3, padding=1),  # 2 Output Channels: one for X and one for Y
            nn.Tanh()
        )

        self.need_initialization.append('CBR1_RGB_DEC')


    def forward(self, rgb_inputs, depth_inputs):
        # ---------------------------------------------------------------------
        # --------------------------- Depth Encoder ---------------------------
        # ---------------------------------------------------------------------
        # --------------------------- Stage 1 ------------------------------
        x_1 = self.CBR1_DEPTH_ENC(depth_inputs)
        x   = self.DEPTH_POOL1(x_1)
        # --------------------------- Stage 2 ------------------------------
        x_2 = self.CBR2_DEPTH_ENC(x)
        x   = self.DEPTH_POOL2(x_2)
        # --------------------------- Stage 3 ------------------------------
        x_3 = self.CBR3_DEPTH_ENC(x)
        x   = self.DEPTH_POOL3(x_3)
        x   = self.DEPTH_DROPOUT3(x)
        # --------------------------- Stage 4 ------------------------------
        x_4 = self.CBR4_DEPTH_ENC(x)
        x   = self.DEPTH_POOL4(x_4)
        x   = self.DEPTH_DROPOUT3(x)
        # --------------------------- Stage 5 ------------------------------
        x_5 = self.CBR5_DEPTH_ENC(x)

        # ---------------------------------------------------------------------
        # ---------------------------- RGB Encoder ----------------------------
        # ---------------------------------------------------------------------
        # --------------------------- Stage 1 ------------------------------
        y      = self.CBR1_RGB_ENC(rgb_inputs)
        y      = torch.add(y, x_1)
        y      = torch.div(y, 2) # NOTE: Not in the paper! found in the official pytorch implementation. (manorz, 03/07/20)
        y, id1 = self.RGB_POOL1(y)
        # --------------------------- Stage 2 ------------------------------
        y      = self.CBR2_RGB_ENC(y)
        y      = torch.add(y, x_2)
        y      = torch.div(y, 2)
        y, id2 = self.RGB_POOL2(y)
        # --------------------------- Stage 3 ------------------------------
        y      = self.CBR3_RGB_ENC(y)
        y      = torch.add(y, x_3)
        y      = torch.div(y, 2)
        y, id3 = self.RGB_POOL3(y)
        y      = self.RGB_DROPOUT3(y)
        # --------------------------- Stage 4 ------------------------------
        y      = self.CBR4_RGB_ENC(y)
        y      = torch.add(y, x_4)
        y      = torch.div(y, 2)
        y, id4 = self.RGB_POOL4(y)
        y      = self.RGB_DROPOUT4(y)
        # --------------------------- Stage 5 ------------------------------
        y      = self.CBR5_RGB_ENC(y)
        y      = torch.add(y, x_5)
        y      = torch.div(y, 2)
        y_size = y.size() # y_size needed for un-pooling in the decoder
        y, id5 = self.RGB_POOL5(y)
        y      = self.RGB_DROPOUT5(y)

        # ---------------------------------------------------------------------
        # ---------------------------- RGB Decoder ----------------------------
        # ---------------------------------------------------------------------
        # --------------------------- Stage 5 ------------------------------
        y = self.UNPOOL5(y, id5, output_size=y_size)
        y = self.CBR5_RGB_DEC(y)
        y   = self.DROPOUT5_DEC(y)
        # --------------------------- Stage 4 ------------------------------
        y = self.UNPOOL4(y, id4)
        y = self.CBR4_RGB_DEC(y)
        y = self.DROPOUT4_DEC(y)
        # --------------------------- Stage 3 ------------------------------
        y = self.UNPOOL3(y, id3)
        y = self.CBR3_RGB_DEC(y)
        y = self.DROPOUT3_DEC(y)
        # --------------------------- Stage 2 ------------------------------
        y = self.UNPOOL2(y, id2)
        y = self.CBR2_RGB_DEC(y)
        # --------------------------- Stage 1 ------------------------------
        y = self.UNPOOL1(y, id1)
        y = self.CBR1_RGB_DEC(y)

        return y


class SpecialFuseNetModel():
    def __init__(self, device=None, rgb_size=None,depth_size=None,grads_size=None,
                 seed=42, optimizer=None, scheduler=None):
        assert rgb_size and depth_size and grads_size, "Please provide inputs sizes"
        assert len(rgb_size) == len(depth_size) == len(grads_size) == 3, "Please emit the batch dimension"
        assert isinstance(device, torch.device), "Please provide device as torch.device"

        torch.manual_seed(seed)

        self.rgb_size   = rgb_size
        self.depth_size = depth_size
        self.grads_size = grads_size
        self.device     = device

        self.net = SpecialFuseNet()
        self._check_features()
        self.initialize()
        self.net = DataParallel(self.net).to(self.device)

        self.loss_func = nn.MSELoss()

        if optimizer:
            self.optimizer = optimizer
        else:
            lr           = 0.001 # HyperParameters from the paper
            momentum     = 0.9
            weight_decay = 0.0005
            print(f'[debug] - default optimizer set: SGD(lr={lr},momentum={momentum},weight_decay={weight_decay})')
            self.optimizer = optim.SGD(self.net.parameters(),
                                       lr=lr, momentum=momentum, weight_decay=weight_decay)
        if scheduler:
            self.scheduler = scheduler
        else:
            step_size = 1000
            gamma     = 0.1
            print(f'[debug] - default scheduler set: StepSR(step_size={step_size},gamma={gamma})')
            self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)

    def initialize(self, init_type='xavier', init_gain=0.02):
        self.net.to(self.device)

        # TODO: Not sure about that implementation.
        #  It don't take care about the order of the layers (important for Xavier). (manorz, 03/08/20)
        for child_name, child in self.net.named_children():
            # print(f'[debug] - child={child_name}, type(child)={type(child)}')
            if child_name in self.net.need_initialization: # CBR-s in the Decoder
                init_weights(child, init_type, init_gain=init_gain)

    def _check_features(self):
        with torch.no_grad():
            # Make sure encoder and decoder are compatible
            rgb_input   = torch.randn(1, *self.rgb_size,   device=self.device)
            depth_input = torch.randn(1, *self.depth_size, device=self.device)
            grads_output = self.net(rgb_input,depth_input)
            two_grads_size = tuple([2] + list(self.grads_size)[1:])
            assert tuple(grads_output.shape[1:]) == two_grads_size, f"Input gradients does not match in size to Output gradients\n" \
                                                                    f"(|{self.grads_size}|!=|{tuple(grads_output.shape[1:])}|)"

    def __call__(self, rgb_batch, depth_batch):
        return self.net(rgb_batch, depth_batch)

    def loss(self,ground_truth_grads: torch.Tensor, approximated_grads: torch.Tensor):
        # print(f'[debug] - shapes: |tags|={ground_truth_grads.shape}, |output|={approximated_grads.shape}')
        assert ground_truth_grads.shape == approximated_grads.shape
        return self.loss_func(approximated_grads, ground_truth_grads)





