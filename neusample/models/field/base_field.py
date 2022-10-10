import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import auto_fp16, force_fp32
from ..builder import FIELD


@FIELD.register_module()
class BaseField(nn.Module):
    def __init__(self, nb_layers=8, hid_dims=256, xyz_emb_dims=63, dir_emb_dims=27, use_dirs=True):
        super().__init__()
        self.nb_layers = nb_layers
        self.hid_dims = hid_dims
        self.xyz_emb_dims = xyz_emb_dims
        self.dir_emb_dims = dir_emb_dims
        self.use_dirs = use_dirs
        self.skips = [nb_layers // 2]

        self.layers = nn.ModuleDict()
        self.layers.add_module('fc0', nn.Linear(xyz_emb_dims, hid_dims))
        for i in range(1, nb_layers):
            if i in self.skips:
                self.layers.add_module(
                    'fc{}'.format(i), 
                    nn.Linear(hid_dims + xyz_emb_dims, hid_dims)
                )
            else:
                self.layers.add_module(
                    'fc{}'.format(i), 
                    nn.Linear(hid_dims, hid_dims)
                )

        self.alpha_out = nn.Linear(hid_dims, 1)
        if use_dirs:
            self.feat_out = nn.Linear(hid_dims, hid_dims)
            self.color_fc = nn.Linear(hid_dims + dir_emb_dims, hid_dims // 2)
            self.color_out = nn.Linear(hid_dims // 2, 3)
        else:
            self.color_out = nn.Linear(hid_dims, 3)
        self.fp16_enabled = False
    
    @auto_fp16()
    def forward(self, xyz_embeds, dir_embeds=None):
        x = xyz_embeds
        for i in range(self.nb_layers):
            key = 'fc{}'.format(i)
            layer = self.layers[key]
            if i in self.skips:
                x = torch.cat([x, xyz_embeds], dim=1)
            x = layer(x)
            x = F.relu(x, inplace=True)
        
        alphas = self.alpha_out(x)
        if self.use_dirs:
            if dir_embeds is None:
                raise ValueError()
            
            feats = self.feat_out(x)
            # NOTE: there is no relu here in the official implementation
            x = torch.cat([feats, dir_embeds], dim=1)
            x = self.color_fc(x)
            x = F.relu(x, inplace=True)
            colors = self.color_out(x)
        else:
            colors = self.color_out(x)
        colors = torch.sigmoid(colors)
        return alphas, colors
