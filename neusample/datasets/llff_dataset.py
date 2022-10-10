import math
import os
import numpy as np
from PIL import Image
import torch
from torch._C import device

from neusample.utils import get_root_logger
from .synthetic_dataset import get_rays
from .builder import DATASETS
from .utils.llff_loader import load_llff_data


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    """Normalized device coordinate rays.
    Space such that the canvas is a cube with sides [-1, 1] in each axis.
    Args:
      H: int. Height in pixels.
      W: int. Width in pixels.
      focal: float. Focal length of pinhole camera.
      near: float or array of shape[batch_size]. Near depth bound for the scene.
      rays_o: array of shape [batch_size, 3]. Camera origin.
      rays_d: array of shape [batch_size, 3]. Ray direction.
    Returns:
      rays_o: array of shape [batch_size, 3]. Camera origin in NDC.
      rays_d: array of shape [batch_size, 3]. Ray direction in NDC.
    """
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1./(H/(2.*focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1./(W/(2.*focal)) * \
        (rays_d[..., 0]/rays_d[..., 2] - rays_o[..., 0]/rays_o[..., 2])
    d1 = -1./(H/(2.*focal)) * \
        (rays_d[..., 1]/rays_d[..., 2] - rays_o[..., 1]/rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d


@DATASETS.register_module()
class LLFFDataset:
    def __init__(self, 
                 datadir, 
                 factor, 
                 batch_size, 
                 split, 
                 spherify, 
                 no_ndc, 
                 holdout,
                 select_img=None,
                 batching=True,
                 to_cuda=False,
                 llff_data_param={}):
        self.logger = get_root_logger()
        self.batch_size = batch_size
        self.no_ndc = no_ndc
        datadir = os.path.expanduser(datadir)
        images, poses, bds, render_poses, i_test, imgfiles = load_llff_data(
            datadir, factor, recenter=True, bd_factor=.75, spherify=spherify, **llff_data_param)
        self.hwf = poses[0, :3, -1]
        self.h = int(self.hwf[0])
        self.w = int(self.hwf[1])
        self.focal = self.hwf[2]
        self.poses = poses[:, :3, :4]
        self.logger.info(f'Loaded llff images.shape {images.shape}, '
                    f'render_poses.shape {render_poses.shape}, '
                    f'hwf {self.hwf}, datadir {datadir}')
        if not isinstance(i_test, list):
            i_test = [i_test]

        if holdout > 0:
            self.logger.info(f'Hold out overwrite. Auto LLFF holdout: {holdout}')
            i_test = np.arange(images.shape[0])[::holdout]

        self.logger.info('DEFINING BOUNDS')
        if no_ndc:
            self.near = bds.min() * 0.9
            self.far = bds.max() * 1.0
        else:
            self.near = 0.
            self.far = 1.

        i_val = i_test
        if split == "train":
            all_idx = np.array([
                i for i in np.arange(int(images.shape[0])) 
                if (i not in i_test and i not in i_val)])
            # if not no_ndc:
            #     self.near -= 0.05*(self.far-self.near)
        elif split == 'val':
            all_idx = i_val
        elif split == 'all':
            all_idx = np.arange(images.shape[0])
        else:
            raise NotImplementedError
        if select_img is not None:
            all_idx = []
            img_name = []
            for img in select_img:
                for idx, f in enumerate(imgfiles):
                    if img in f:
                        img_name.append(f)
                        all_idx.append(idx)
            self.logger.info(f'split {split} selected names: {img_name}')
        self.logger.info(f'split {split} idx: {all_idx}')
        self.logger.info(f'NEAR {self.near} FAR {self.far}')

        self.imgs = torch.tensor(images[all_idx])
        self.poses = torch.tensor(self.poses[all_idx])
        self.render_poses = torch.tensor(render_poses)
        if to_cuda:
            self.imgs.cuda()
            self.poses.cuda()
            self.render_poses.cuda()
        self.batching = batching
        self.length = len(self.poses)
        if self.batching:
            all_rays_ori, all_rays_dir = [], []
            self.logger.info(f'creating all rays')
            for p in self.poses[:, :3, :4]:
                ro, rd = get_rays(self.h, self.w, self.focal, p) 
                all_rays_ori.append(ro)
                all_rays_dir.append(rd)
            self.logger.info(f'finish creating all rays')
            self.all_rays_ori = torch.stack(all_rays_ori, dim=0).view([-1,3])
            self.all_rays_dir = torch.stack(all_rays_dir, dim=0).view([-1,3])
            if not self.no_ndc:
                self.all_rays_ori, self.all_rays_dir = ndc_rays(
                    self.h, self.w, self.focal, 1., self.all_rays_ori, self.all_rays_dir)
            self.imgs = self.imgs.view(-1,3)
            self.length = self.imgs.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.batching:
            return {'rays_ori': self.all_rays_ori[idx,:], 
                    'rays_dir': self.all_rays_dir[idx,:], 
                    'rays_color': self.imgs[idx,:], 
                    'near': self.near, 'far': self.far}
        else:
            target = self.imgs[idx%len(self.imgs)]
            pose = self.poses[idx, :3,:4]
            rays_ori, rays_dir = get_rays(self.h, self.w, self.focal, pose)

        if not self.no_ndc:
            rays_ori, rays_dir = ndc_rays(self.h, self.w, self.focal, 1., rays_ori, rays_dir)

        if self.batch_size == -1:
            return {'rays_ori': rays_ori.view([-1,3]), 
                    'rays_dir': rays_dir.view([-1,3]), 
                    'rays_color': target.view([-1,3]), 
                    'near': self.near, 'far': self.far}

        select_idx = torch.randperm(self.h*self.w, device=rays_ori.device)[:self.batch_size]
        rays_ori = rays_ori.view([-1,3])[select_idx]  # (N, 3)
        rays_dir = rays_dir.view([-1,3])[select_idx]  # (N, 3)
        rays_color = target.view([-1,3])[select_idx]  # (N, 3)

        return {'rays_ori': rays_ori, 'rays_dir': rays_dir, 
                'rays_color': rays_color, 'near': self.near, 'far': self.far}


