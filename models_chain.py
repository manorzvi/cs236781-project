import torch
import torch.nn as nn
import torchvision
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler

from DD_model import DenseDepth

from models import SpecialFuseNet
from functions import init_weights
from torch.nn import DataParallel

class ModelsChain(nn.Module):
    def __init__(self, sgd_lr=0.001, sgd_momentum=0.9, sgd_wd=0.0005,
                 device=None, rgb_size=None, depth_size=None, grads_size=None,
                 seed=42, dropout_p=0.4, optimizer=None, scheduler=None, overfit_mode=False):
        super(ModelsChain, self).__init__()
#     def __init__(self, models : list, image_size):
#         super(ModelsChain, self).__init__()
#         self.models = models
#         self.num_models = len(self.models)
#         self.image_size = image_size
        assert rgb_size and depth_size and grads_size, "Please provide inputs sizes"
        assert len(rgb_size) == len(depth_size) == len(grads_size) == 3, "Please emit the batch dimension"
        assert isinstance(device, torch.device), "Please provide device as torch.device"
        print(f'[I] - device={device}\n'
              f'    - seed={seed}\n'
              f'    - dropout_p={dropout_p}\n'
              f'    - optimizer={optimizer}\n'
              f'    - scheduler={scheduler}'
              f'    - overfit_mode={overfit_mode}')
        torch.manual_seed(seed)

        self.rgb_size   = rgb_size
        self.depth_size = depth_size
        self.grads_size = grads_size
        self.device     = device
        
        self.densedepth = DenseDepth().cuda()
#         self.densedepth.to(self.device)
        # Training parameters
        self.optimizer_densedepth = torch.optim.Adam( self.densedepth.parameters(), 0.0001 )
        batch_size_densedepth = 4
        prefix = 'densenet_' + str(batch_size_densedepth)
        # Loss
        self.loss_func_densedepth = nn.L1Loss()
        
        
        self.dropout_p  = dropout_p
        self.special_fusenet = SpecialFuseNet(dropout_p=self.dropout_p, overfit_mode=overfit_mode)
        self.special_fusenet.to(self.device)
        self._check_features()
        self.initialize()
        self.special_fusenet = DataParallel(self.special_fusenet).to(self.device)
        self.loss_func_special_fusenet = nn.MSELoss()
        if optimizer:
            self.optimizer_fusenet = optimizer
        else:
            lr           = sgd_lr
            momentum     = sgd_momentum
            weight_decay = sgd_wd
            print(f'[I] - default optimizer set: SGD(lr={lr},momentum={momentum},weight_decay={weight_decay})')
            self.optimizer_fusenet = optim.SGD(self.special_fusenet.parameters(),
                                               lr=lr, momentum=momentum, weight_decay=weight_decay)
        if scheduler:
            self.scheduler = scheduler
        else:
            step_size = 1000
            gamma     = 0.1
            print(f'[I] - default scheduler set: StepSR(step_size={step_size},gamma={gamma})')
            self.scheduler = lr_scheduler.StepLR(self.optimizer_fusenet, step_size=step_size, gamma=gamma)

    def initialize(self, init_type='xavier', init_gain=0.02):
        self.special_fusenet.to(self.device)
        print(f'[I] - Initialize Net.\n'
              f'    - Init type={init_type}\n'
              f'    - Init gain={init_gain}\n')
        # TODO: Not sure about that implementation.
        #  It don't take care about the order of the layers (important for Xavier). (manorz, 03/08/20)
        for child_name, child in self.special_fusenet.named_children():
            # print(f'[debug] - child={child_name}, type(child)={type(child)}')
            if child_name in self.special_fusenet.need_initialization: # CBR-s in the Decoder
                init_weights(child, init_type, init_gain=init_gain)

    def _check_features(self):
        with torch.no_grad():
            # Make sure encoder and decoder are compatible
            rgb_input   = torch.randn(1, *self.rgb_size,   device=self.device)
            depth_input = torch.randn(1, *self.depth_size, device=self.device)
            grads_output = self.special_fusenet(rgb_input,depth_input)
            two_grads_size = tuple([2] + list(self.grads_size)[1:])
            assert tuple(grads_output.shape[1:]) == two_grads_size, f"Input gradients does not match in size to Output gradients\n" \
                                                                    f"(|{self.grads_size}|!=|{tuple(grads_output.shape[1:])}|)"

    def __call__(self, rgb_rgb2d_batch, rgb_batch, depth_batch):
        d = None
        if rgb_rgb2d_batch is not None:
            d = self.densedepth(rgb_rgb2d_batch)
        xy = None
        if rgb_batch is not None and depth_batch is not None:
            xy = self.special_fusenet(rgb_batch, depth_batch)
        return (d, xy)

    def loss_densedepth(self, ground_truth_depth: torch.Tensor, approximated_depth: torch.Tensor):
        # print(f'[debug] - shapes: |tags|={ground_truth_depth.shape}, |output|={approximated_depth.shape}')
        assert ground_truth_depth.shape == approximated_depth.shape
        return self.loss_func_densedepth(approximated_depth, ground_truth_depth)
    
    def loss_special_fusenet(self, ground_truth_grads: torch.Tensor, approximated_grads: torch.Tensor):
        # print(f'[debug] - shapes: |tags|={ground_truth_grads.shape}, |output|={approximated_grads.shape}')
        assert ground_truth_grads.shape == approximated_grads.shape
        return self.loss_func_special_fusenet(approximated_grads, ground_truth_grads)

    def set_requires_grad(self, requires_grad=False):
        for param in self.special_fusenet.parameters():
            param.requires_grad = requires_grad

    def set_dropout_train(self, train):
        for child_name, child in self.special_fusenet.named_children():
            # print(f'[D] - child={child_name}, type(child)={type(child)}')
            for child_name_, child_ in child.named_children():
                # print(f'    [D] - child_={child_name_}, type(child_)={type(child_)}')
                if child_name_ in child.dropouts: # Dropout Layer
                    # print(f'    [D] - child_={child_name_}, type(child_)={type(child_)}')
                    child_.training = train
                    # NOTE: The following probably unnecessary, but can't harm.
                    if train == False:
                        child_.p = 0
                    else:
                        child_.p = self.dropout_p

    def set_batchnorms_train(self, train):
        for child_name, child in self.special_fusenet.named_children():
            # print(f'[D] - child={child_name}, type(child)={type(child)}')
            for child_name_, child_ in child.named_children():
                if child_name_ in child.cbrs:
                    # print(f'    [D] - child_={child_name_}, type(child_)={type(child_)}')
                    for child_name__, child__ in child_.named_children():
                            if isinstance(child__, nn.BatchNorm2d):
                                # print(f'        [D] - child__={child_name__}, type(child__)={type(child__)}')
                                child__.train(train)

        
    def forward(self, Xs : list):
        assert len(Xs) == self.__len__(), "Error, bad number of parameters."
        
        with torch.no_grad():
            depth_pred = self.models[0](Xs[0])
        
        depth_pred = depth_pred.cpu()[0][0]
        
        # Depth -> [0, 1]
        depth_pred = depth_pred - torch.min(depth_pred)
        depth_pred = depth_pred / torch.max(depth_pred)
        
        # Inverse colors, to match the GT depth colors' style.
        depth_pred = 1.0 - depth_pred

        #resize
        depth_pred = torchvision.transforms.ToPILImage()(depth_pred).convert("RGB")
        depth_pred = depth_pred.resize(self.image_size) 
        
        #add first dims
        depth_pred = np.moveaxis(np.array(depth_pred, np.float32, copy=False), -1, 0)
        depth_pred = torch.from_numpy(depth_pred)[0].unsqueeze(0)
        
        with torch.no_grad():
            grads_pred = self.models[1](Xs[1].unsqueeze(0), depth_pred.unsqueeze(0))
            
        return (depth_pred, grads_pred)
    
    def __len__(self):
        return self.num_models