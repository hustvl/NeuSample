import os
import os.path as osp
import random
import subprocess
import copy
import cv2
import imageio as imio
import numpy as np

from neusample.utils import get_root_logger
from ..utils import collect_rays, center_poses
from ..builder import LOADERS

IMAGE_EXTENSIONS = ('png', 'PNG', 'jpg', 'JPG', 'jpeg', 'JPEG')


@LOADERS.register_module()
class LLFFLoader(object):
    def __init__(self, 
                 colmap_dir, 
                 im_dir, 
                 bound_factor=0.75, 
                 center_poses=True, 
                 spherify_poses=False, 
                 factor=8, 
                 ndc=True):
        super().__init__()
        self.logger = get_root_logger()
        self.colmap_dir = os.path.expanduser(colmap_dir)
        im_dir = im_dir.rstrip('/').rstrip('\\')
        self.im_dir = os.path.expanduser(im_dir)
        self.bound_factor = bound_factor
        self.center_poses = center_poses
        self.spherify_poses = spherify_poses
        self.factor = factor
        self.ndc = ndc
        # self.cross_im_sample = cross_im_sample  # 'use_batch' in the nerf implementation

        self.logger.info('Start to load data...')
        poses, bounds, im_array = self._load_data()
        self.poses = poses
        self.bounds = bounds
        self.im_array = im_array
        self.logger.info(f'im_array: {self.im_array.shape}')
        self.logger.info(f'poses: {self.poses.shape}')
        self.logger.info(f'bounds: {self.bounds.shape}')

        # self.im_array = torch.as_tensor(self.im_array, dtype=torch.float32)
        # self.poses = torch.as_tensor(self.poses, dtype=torch.float32)
        # self.bounds = torch.as_tensor(self.bounds, dtype=torch.float32)

        rays_ori, rays_dir, rays_color = self._collect_rays()
        self.rays_ori = rays_ori
        self.rays_dir = rays_dir
        self.rays_color = rays_color
        self.logger.info(f'rays_ori: {self.rays_ori.shape}')
        self.logger.info(f'rays_dir: {self.rays_dir.shape}')
        self.logger.info(f'rays_color: {self.rays_color.shape}')
        self.logger.info('Finish to load data...')

        h, w, focal = self.poses[0, :3, -1]
        self.h = int(h)
        self.w = int(w)
        self.focal = focal

        if ndc:
            self.near = 0.
            self.far = 1.
        else:
            self.near = 0.9 * np.min(self.bounds)
            self.far = 1.0 * np.max(self.bounds)

    def _load_data(self,):
        im_array = self._load_im_array()  # [#im, h, w, 3]

        # step1: loads poses and bounds
        poses_bounds = np.load(osp.join(self.colmap_dir, 'poses_bounds.npy'))  # [#im, 17] (17 = 3x5 + 2)
        poses = poses_bounds[:, :-2].reshape([-1, 3, 5])  # [#im, 3, 5]
        bounds = poses_bounds[:, -2:]  # [#im, 2]

        # sanity check
        assert poses.shape[0] == im_array.shape[0], \
            '# im != # pose ({}, {})'.format(im_array.shape, poses.shape)
        
        # step2: updates h, w, focal as im are resized
        im_shape = im_array.shape[1:]
        factor = poses[0, 0, 4] / im_shape[0]  # poses[0, 0, 4]: h of an unresized im
        assert factor == self.factor, 'factor != self.factor ({}, {})'.format(factor, self.factor)
        poses[:, 0, 4] = im_shape[0]  # updates h
        poses[:, 1, 4] = im_shape[1]  # updates w
        poses[:, 2, 4] = poses[:, 2, 4] / factor  # updates focal

        # step3: corrects rotation matrix
        poses = np.concatenate([
            poses[:, :, 1:2], 
            -poses[:, :, 0:1], 
            poses[:, :, 2:]], 2
        )

        # step4: rescales bounds so that near bounds are a bit more than 1
        if self.bound_factor is not None:
            scale = 1 / (self.bound_factor * bounds.min())
            poses[:, :3, 3] *= scale
            bounds *= scale
        
        # step5: centers and spherifies poses
        poses, hw_focal = np.split(poses, [4], axis=2)
        if self.center_poses:
            poses, _ = center_poses(poses)
        
        if self.spherify_poses:
            raise NotImplementedError()
        else:
            pass
        poses = np.concatenate([poses, hw_focal], axis=2)
        return poses, bounds, im_array

    def _load_im_array(self,):
        im_paths = [osp.join(self.im_dir, n) for n in sorted(os.listdir(self.im_dir))]
        im_paths = [p for p in im_paths if any([p.endswith(ext) for ext in IMAGE_EXTENSIONS])]
        _, im_ext = osp.splitext(im_paths[0])
        im_ext = im_ext[1:]  # removes the '.' symbol
        im_shape = cv2.imread(im_paths[0]).shape[:2]  # h, w
        
        dest_im_shape = self._dest_im_shape(im_shape)
        # dest_im_dir = self.im_dir if dest_im_shape == im_shape and im_ext in ('png', 'PNG') else \
        #     '{}_{:d}x{:d}'.format(self.im_dir, dest_im_shape[0], dest_im_shape[1])
        dest_im_dir = self.im_dir if self.factor == 1 and im_ext in ('png', 'PNG') else \
            '{}_{}'.format(self.im_dir, self.factor)
        if not osp.exists(dest_im_dir):
            # Resizes or reformats im
            os.makedirs(dest_im_dir)
            for path in im_paths:
                output = subprocess.check_output('cp {:s} {:s}'.format(path, dest_im_dir), shell=True)
            
            cur_dir = os.getcwd()
            os.chdir(dest_im_dir)
            # mogrify: w x h not h x w
            self.logger.info('Resize images')
            # cmd = 'mogrify -resize {:d}x{:d} *.{:s}'.format(dest_im_shape[1], dest_im_shape[0], im_ext)
            cmd = 'mogrify -resize {}% *.{:s}'.format(100 / self.factor, im_ext)
            output = subprocess.check_output(cmd, shell=True)
            if not im_ext in ('png', 'PNG'):
                self.logger.info('Reformat images\n')
                output = subprocess.check_output('mogrify -format PNG *.{:s}'.format(im_ext), shell=True)
                output = subprocess.check_output('rm {:s}/*.{:s}'.format(dest_im_dir, im_ext), shell=True)        
            os.chdir(cur_dir)

        def imread(path):
            if path.endswith('png') or path.endswith('PNG'):
                return imio.imread(path, ignoregamma=True)
            else:
                return imio.imread(path)
        
        dest_im_paths = [osp.join(dest_im_dir, n) for n in sorted(os.listdir(dest_im_dir))]
        im_array = [imread(p)[..., :3] / 255 for p in dest_im_paths]
        im_array = np.stack(im_array, 0)  # [#im, height, width, 3]
        return im_array

    def _dest_im_shape(self, im_shape):
        if self.factor is not None:
            dest_im_shape = (im_shape[0] / self.factor, im_shape[1] / self.factor)
        elif self.height is not None:
            dest_im_shape = (self.height, im_shape[1] / im_shape[0] * self.height)
        elif self.width is not None:
            dest_im_shape = (im_shape[0] / im_shape[1] * self.width, self.width)
        else:
            dest_im_shape = im_shape
        dest_im_shape = (int(dest_im_shape[0]), int(dest_im_shape[1]))
        return dest_im_shape
   
    def _collect_rays(self,):
        h, w, focal = self.poses[0, :3, -1]
        h = int(h)
        w = int(w)

        rays_ori, rays_dir = [], []
        for i, pose in enumerate(self.poses):
            rays_o, rays_d = collect_rays(h, w, focal, pose[:3, :4])
            rays_ori.append(rays_o)
            rays_dir.append(rays_d)
        
        # N: len(self.idx_train)
        rays_ori = np.stack(rays_ori, axis=0)  # [N, h, w, 3]
        rays_dir = np.stack(rays_dir, axis=0)  # [N, h, w, 3]
        rays_color = copy.deepcopy(self.im_array) # [#im, h, w, 3]
        # rays_color = np.stack([rays_color[i] for i in self.idx_train], axis=0)  # [N, h, w, 3]

        # if self.cross_im_sample:
        #     rays_ori = np.reshape(rays_ori, [-1, 3])
        #     rays_dir = np.reshape(rays_dir, [-1, 3])
        #     rays_color = np.reshape(rays_color, [-1, 3])
            
        #     indices = list(range(rays_ori.shape[0]))
        #     rays_color = rays_color[indices]
        return rays_ori, rays_dir, rays_color