import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import auto_fp16
from ..builder import FIELD


@FIELD.register_module()
class SampleField(nn.Module):
    def __init__(self, nb_layers=8, hid_dims=256, ori_emb_dims=63, 
            dir_emb_dims=27, n_samples=192):
        super().__init__()
        self.nb_layers = nb_layers
        self.hid_dims = hid_dims
        self.ori_emb_dims = ori_emb_dims
        self.dir_emb_dims = dir_emb_dims
        self.n_samples = n_samples
        self.input_dims = self.ori_emb_dims + self.dir_emb_dims
        self.skips = [nb_layers // 2]

        self.layers = nn.ModuleDict()
        self.layers.add_module('fc0', nn.Linear(self.input_dims, hid_dims))
        for i in range(1, nb_layers):
            if i in self.skips:
                self.layers.add_module(
                    'fc{}'.format(i), 
                    nn.Linear(hid_dims + self.input_dims, hid_dims)
                )
            else:
                self.layers.add_module(
                    'fc{}'.format(i), 
                    nn.Linear(hid_dims, hid_dims)
                )

        self.dist_out = nn.Linear(hid_dims, n_samples)
        self.fp16_enabled = False
    
    @auto_fp16()
    def forward(self, ori_embeds, dir_embeds):
        x = torch.cat([ori_embeds, dir_embeds], dim=1)
        cat_skip = x
        for i in range(self.nb_layers):
            key = 'fc{}'.format(i)
            layer = self.layers[key]
            if i in self.skips:
                x = torch.cat([x, cat_skip], dim=1)
            x = layer(x)
            x = F.relu(x, inplace=True)
        
        dists = self.dist_out(x)
        dists = torch.sigmoid(dists)
        return dists
