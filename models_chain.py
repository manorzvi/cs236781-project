import torch
import torch.nn as nn
import torchvision
import numpy as np

class ModelsChain(nn.Module):
    def __init__(self, models : list, image_size):
        super(ModelsChain, self).__init__()
        self.models = models
        self.num_models = len(self.models)
        self.image_size = image_size
        
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