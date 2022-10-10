import json
import os
import os.path as osp
import random
import subprocess
import copy
import cv2
import imageio as imio
import numpy as np

from neusample.utils import get_root_logger as get_root_logger
from ..utils import collect_rays, center_poses
from ..builder import LOADERS


@LOADERS.register_module()
class BlenderLoader(object):
    def __init__(self, root_dir, bound_factor=0.75, center_poses=True, spherify_poses=False, ndc=True):
        super().__init__()
        self.logger = get_root_logger()
        self.root_dir = os.path.expanduser(root_dir)
        self.bound_factor = bound_factor
        self.center_poses = center_poses
        self.spherify_poses = spherify_poses
        self.ndc = ndc

        self.logger.info('Start to load data...')

        cam_poses, bounds, im_array, focal, H, W = self._load_data()
        self.cam_poses = cam_poses
        self.poses = cam_poses  # an alias, for compatibility reason
        self.bounds = bounds
        self.im_array = im_array

        self.logger.info(f'im_array: {self.im_array.shape}')
        self.logger.info(f'cam poses: {self.cam_poses.shape}')
        self.logger.info(f'bounds: {self.bounds.shape}')

        rays_ori, rays_dir, rays_color = self._collect_rays(H, W, focal)
        self.rays_ori = rays_ori
        self.rays_dir = rays_dir
        self.rays_color = rays_color
        self.logger.info(f'rays_ori: {self.rays_ori.shape}')
        self.logger.info(f'rays_dir: {self.rays_dir.shape}')
        self.logger.info(f'rays_color: {self.rays_color.shape}')
        self.logger.info('Finish to load data...')

        self.H = H
        self.W = W
        self.focal = focal

        if ndc:
            self.near = 0.
            self.far = 1.
        else:
            self.near = 0.9 * np.min(self.bounds)
            self.far = 1.0 * np.max(self.bounds)

    def _load_data(self):
        im_array = self._load_im_array()  # [#im, H, W, 3]

        # step1: loads poses and bounds
        cam_poses, focal, H, W = self._load_cam_poses()  # [#im, 3, 4]
        head_locations, tail_locations = self._load_poses()
        bounds = self._pose_to_bounds(cam_poses, head_locations, tail_locations)  # [#im, 2]
        
        # sanity check
        assert cam_poses.shape[0] == bounds.shape[0] == im_array.shape[0], \
            '%d ? %d ? %d' % (cam_poses.shape[0], bounds.shape[0], im_array.shape[0])

         # step2: rescales bounds so that near bounds are a bit more than 1
        if self.bound_factor is not None:
            scale = 1 / (self.bound_factor * bounds.min())
            cam_poses[:, :3, 3] *= scale
            bounds *= scale
        
        # step3: centers and spherifies poses
        if self.center_poses:
            cam_poses, _ = center_poses(cam_poses)
        
        if self.spherify_poses:
            raise NotImplementedError()
        else:
            pass
        return cam_poses, bounds, im_array, focal, H, W

    def _load_cam_poses(self):
        cam_poses_path = osp.join(self.root_dir, 'camera_poses.json')
        cam_poses = json.load(open(cam_poses_path, 'r'))
        
        cam_keys = sorted(cam_poses.keys())
        poses = []
        for i, key in enumerate(cam_keys):
            pose = np.asarray(cam_poses[key]['matrix'])  # [4, 4]
            poses.append(pose[:3, ...])  # the last row is always [0, 0, 0, 1]
        poses = np.asarray(poses)

        cam_K_path = osp.join(self.root_dir, 'camera_intrinsic.json')
        K = json.load(open(cam_K_path, 'r'))
        assert K['focal_x'] == K['focal_y'], '%.2f != %.2f' % (K['focal_x'], K['focal_y'])
        focal = K['focal_x']
        H = K['v0'] * 2
        W = K['u0'] * 2
        return poses, focal, H, W

    def _load_poses(self):
        poses_path = osp.join(self.root_dir, 'poses_across_time.json')
        poses = json.load(open(poses_path, 'r'))

        pose = poses['1']
        head_locations = []
        tail_locations = []
        for name, head_tail_loc in pose.items():
            head_loc = head_tail_loc['head']
            tail_loc = head_tail_loc['tail']
            head_locations.append(head_loc)
            tail_locations.append(tail_loc)
        head_locations = np.asarray(head_locations)  # [#joints, 3]
        tail_locations = np.asarray(tail_locations)
        return head_locations, tail_locations
    
    def _pose_to_bounds(self, cam_poses, head_locations, tail_locations):
        nb_joints = head_locations.shape[0]
        # [#joints, 3] ->  [#joints, 4]
        head_locations = np.concatenate((head_locations, np.ones((nb_joints, 1))), axis=1)
        tail_locations = np.concatenate((tail_locations, np.ones((nb_joints, 1))), axis=1)
        
        bounds = []
        c2w = np.eye(4)
        for i, c2w_3x4 in enumerate(cam_poses):
            c2w[:3, ...] = c2w_3x4  # [3, 4] -> [4, 4]
            w2c = np.linalg.inv(c2w)
            cam_head_loc = w2c @ head_locations.transpose()  # [4, #joints]
            cam_tail_loc = w2c @ tail_locations.transpose()
            # A NEGATIVE siGn is important, depth has to be positive
            cam_z_vals = -np.concatenate((cam_head_loc[2, :], cam_tail_loc[2, :]))  # [2 * #joints]
            min_depth = np.min(cam_z_vals)  # np.percentile(cam_z_vals, 0.1), 
            max_depth = np.max(cam_z_vals)  # np.percentile(cam_z_vals, 99.9)
            bounds.append([min_depth, max_depth])
        bounds = np.asarray(bounds)
        return bounds

    def _load_im_array(self):
        im_names = [n for n in os.listdir(self.root_dir) if n.upper().endswith('PNG')]
        im_names = sorted(im_names)

        def imread(path):
            if path.upper().endswith('PNG'):
                return imio.imread(path, ignoregamma=True)
            else:
                return imio.imread(path)

        im_paths = [osp.join(self.root_dir, n) for n in im_names]
        im_array = [imread(p)[..., :3] / 255 for p in im_paths]
        im_array = np.stack(im_array, 0)  # [#im, height, width, 3]
        return im_array

    def _collect_rays(self, H, W, focal):
        rays_ori, rays_dir = [], []
        for i, pose in enumerate(self.poses):
            rays_o, rays_d = collect_rays(H, W, focal, pose[:3, :4])
            rays_ori.append(rays_o)
            rays_dir.append(rays_d)
        
        # N: len(self.idx_train)
        rays_ori = np.stack(rays_ori, axis=0)  # [N, H, W, 3]
        rays_dir = np.stack(rays_dir, axis=0)  # [N, H, W, 3]
        rays_color = copy.deepcopy(self.im_array) # [#im, H, W, 3]
        # rays_color = np.stack([rays_color[i] for i in self.idx_train], axis=0)  # [N, H, W, 3]

        # if self.cross_im_sample:
        #     rays_ori = np.reshape(rays_ori, [-1, 3])
        #     rays_dir = np.reshape(rays_dir, [-1, 3])
        #     rays_color = np.reshape(rays_color, [-1, 3])
            
        #     indices = list(range(rays_ori.shape[0]))
        #     rays_color = rays_color[indices]
        return rays_ori, rays_dir, rays_color


if __name__ == '__main__':
    root_dir = '/home/sliu/code/blender-data/blender-output/sit_lauGh_static_19cam_360x480_24x32_focal50.0_64spl_step1/'
    loader = BlenderLoader(root_dir)