import os
import os.path as osp
import random
import json
import numpy as np
import imageio
import cv2
import torch
from torch._C import device

from neusample.utils import get_root_logger
from .builder import DATASETS


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def get_rays(H, W, focal, c2w):
    device = c2w.device
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t().to(device)
    j = j.t().to(device)
    dirs = torch.stack([(i-W*.5)/focal, 
                        -(j-H*.5)/focal, 
                        -torch.ones_like(i, device=device)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, focal, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_dir = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_ori = np.broadcast_to(c2w[:3,-1], np.shape(rays_dir))
    return rays_ori, rays_dir


@DATASETS.register_module()
class SyntheticDataset(object):
    def __init__(self, 
                 base_dir, 
                 split, 
                 half_res,
                 batch_size,
                 white_bkgd=True,
                 precrop_frac=1,
                 testskip=8,):
        super().__init__()
        self.logger = get_root_logger()
        self.base_dir = os.path.expanduser(base_dir)
        self.split = split
        self.half_res = half_res
        self.batch_size = batch_size
        self.white_bkgd = white_bkgd
        self.precrop_frac = precrop_frac
        self.testskip = testskip

        self.__init_dataset()

    def __init_dataset(self):
        file = osp.join(self.base_dir, f'transforms_{self.split}.json')
        with open(file, 'r') as fp:
            meta = json.load(fp)

        imgs = []
        poses = []
        if self.split=='train' or self.testskip==0:
            skip = 1
        else:
            skip = self.testskip

        for frame in meta['frames'][::skip]:
            fname = osp.join(self.base_dir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        self.imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        self.poses = np.array(poses).astype(np.float32)

        self.h, self.w = self.imgs[0].shape[:2]
        camera_angle_x = float(meta['camera_angle_x'])
        self.focal = .5 * self.w / np.tan(.5 * camera_angle_x)
        self.render_poses = torch.stack(
            [pose_spherical(angle, -30.0, 4.0) 
            for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    
        self.h = int(self.h)
        self.w = int(self.w)

        self.near = 2.
        self.far = 6.

        if self.half_res:
            self.h = self.h//2
            self.w = self.w//2
            self.focal = self.focal/2.

            imgs_half_res = np.zeros((self.imgs.shape[0], self.h, self.w, 4))
            for i, img in enumerate(self.imgs):
                imgs_half_res[i] = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_AREA)
            self.imgs = imgs_half_res

        if self.white_bkgd:
            self.imgs = self.imgs[...,:3]*self.imgs[...,-1:] + (1.-self.imgs[...,-1:])
        else:
            self.imgs = self.imgs[...,:3]

        self.imgs = torch.tensor(self.imgs)
        self.poses = torch.tensor(self.poses)

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        target = self.imgs[idx]
        pose = self.poses[idx, :3,:4]
        rays_ori, rays_dir = get_rays(self.h, self.w, self.focal, pose)

        if self.batch_size == -1:
            rays_color = target.view([-1,3])  # (N, 3)
            return {'rays_ori': rays_ori.view([-1,3]), 
                    'rays_dir': rays_dir.view([-1,3]), 
                    'rays_color': rays_color, 
                    'near': self.near, 'far': self.far}

        if self.precrop_frac < 1:
            dH = int(self.h//2 * self.precrop_frac)
            dW = int(self.w//2 * self.precrop_frac)
            coords = torch.stack(
                torch.meshgrid(
                    torch.linspace(self.h//2 - dH, self.h//2 + dH - 1, 2*dH), 
                    torch.linspace(self.w//2 - dW, self.w//2 + dW - 1, 2*dW)
                ), -1)
        else:
            coords = torch.stack(torch.meshgrid(
                torch.linspace(0, self.h-1, self.h), 
                torch.linspace(0, self.w-1, self.w)), -1)

        coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
        select_inds = np.random.choice(coords.shape[0], size=[self.batch_size], replace=False)  # (N,)
        select_coords = coords[select_inds].long()  # (N, 2)
        rays_ori = rays_ori[select_coords[:, 0], select_coords[:, 1]]  # (N, 3)
        rays_dir = rays_dir[select_coords[:, 0], select_coords[:, 1]]  # (N, 3)
        rays_color = target[select_coords[:, 0], select_coords[:, 1]]  # (N, 3)

        return {'rays_ori': rays_ori, 'rays_dir': rays_dir, 
                'rays_color': rays_color, 'near': self.near, 'far': self.far}


