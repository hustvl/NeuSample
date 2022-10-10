from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from mmcv.runner import auto_fp16
from neusample.core import im2mse, mse2psnr, raw2outputs
from ..builder import RENDERER, build_embedder, build_field


@RENDERER.register_module()
class NeuSample(nn.Module):
    def __init__(self, 
                ori_embedder,
                dir_ray_embedder,
                xyz_embedder, 
                sample_field,
                render_params,
                dir_embedder, 
                radiance_field,
                **kwargs):
        super().__init__()
        self.ori_embedder = build_embedder(ori_embedder)
        self.dir_ray_embedder = build_embedder(dir_ray_embedder)
        self.sample_field = build_field(sample_field)

        self.xyz_embedder = build_embedder(xyz_embedder)
        self.dir_embedder = build_embedder(dir_embedder)
        self.radiance_field = build_field(radiance_field)

        self.render_params = render_params
        self.fp16_enabled = False

    def _parse_losses(self, losses):
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if 'loss' not in loss_name:
                continue
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            elif isinstance(loss_value, dict):
                for name, value in loss_value.items():
                    log_vars[name] = value
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items())
        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def forward(self, rays, render_params=None):
        """        
        Args:
            rays (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: 
        """
        for k, v in rays.items():
            if len(v.shape) > 2:
                rays[k] = v.flatten(0, 1)

        if render_params is None:
            render_params = self.render_params
        outputs = self.forward_render(**rays, **render_params)

        im_loss_fine = im2mse(outputs['fine']['color_map'], rays['rays_color'])
        outputs['fine_loss'] = im_loss_fine

        return outputs

    def _parse_outputs(self, outputs):
        loss, log_vars = self._parse_losses(outputs)
        log_vars['psnr'] = mse2psnr(outputs['fine_loss']).item()
        outputs.update(dict(loss=loss, log_vars=log_vars))
        outputs['num_samples'] = 1
        return outputs

    @auto_fp16()
    def forward_render(self,
                       rays_ori, rays_dir, rays_color,
                       alpha_noise_std, inv_depth, # render param
                       max_rays_num, near=0.0, far=1.0, white_bkgd=False, **kwargs):

        if isinstance(near, torch.Tensor) and len(near.shape)>0:
            near = near[...,0].item()
            far = far[...,0].item()

        directions = F.normalize(rays_dir, p=2, dim=-1)

        sampled_dists = self.sample_batchified(
            rays_ori, directions, max_rays_num * 64 * 8 // self.sample_field.nb_layers)
        
        t_vals, sorted_idx = torch.sort(sampled_dists, dim=1)

        if not inv_depth:
            z_vals = near * (1 - t_vals) + far * t_vals
        else:
            z_vals = 1 / (1 / near * (1 - t_vals) + 1 / far * t_vals)

        points = rays_ori[..., None, :] + rays_dir[..., None, :] * \
            z_vals[..., :, None]

        densities, colors = self.forward_batchified(points, directions, 
                            max_rays_num=max_rays_num * 192 // points.shape[1])

        outputs = raw2outputs(densities,
                                colors,
                                z_vals,
                                rays_dir,
                                alpha_noise_std,
                                white_bkgd,)

        return {'fine': outputs}


    @auto_fp16()
    def sample_batchified(self, 
                        rays_ori, directions, max_rays_num):
        nb_rays = rays_ori.shape[0]
        if nb_rays <= max_rays_num or self.training:
            return self.sample_points(rays_ori, directions)
        else:
            outputs = []
            start = 0
            while start < nb_rays:
                end = min(start + max_rays_num, nb_rays)
                assert start < end, 'start >= end ({:d}, {:d})'.format(start, end)
                output = self.sample_points(rays_ori[start:end, ...], 
                                            directions[start:end, ...],)
                outputs.append(output)
                start += max_rays_num
            sampled_dists = torch.cat(outputs, dim=0)
            return sampled_dists


    @auto_fp16()
    def sample_points(self, rays_ori, directions):
        ori_embeds = self.ori_embedder(rays_ori)
        dir_embeds = self.dir_ray_embedder(directions)
        sampled_dists = self.sample_field(ori_embeds, dir_embeds)
        return sampled_dists


    @auto_fp16()
    def forward_batchified(self, 
                           points,
                           directions,
                           max_rays_num):
        nb_rays = points.shape[0]
        if nb_rays <= max_rays_num or self.training:
            return self.forward_points(points, directions)
        else:
            outputs = []
            start = 0
            while start < nb_rays:
                end = min(start+max_rays_num, nb_rays)
                assert start < end, 'start >= end ({:d}, {:d})'.format(start, end)
                output = self.forward_points(points[start:end, ...], 
                                            directions[start:end, ...],)
                outputs.append(output)
                start += max_rays_num
            
            densities_colors = []
            for out in zip(*outputs):
                if out[0] is not None:
                    out = torch.cat(out, dim=0)
                else:
                    out = None
                densities_colors.append(out)
            return densities_colors

    @auto_fp16(apply_to=('points',))
    def forward_points(self, points, directions):
        shape = tuple(points.shape[:-1])  # [B, n_points]
        # [B, 3] -> [B, n_points, 3]
        directions = directions[..., None, :].expand_as(points)
        directions = directions.reshape((-1, 3))    
        dir_embeds = self.dir_embedder(directions)

        points = points.reshape((-1, 3))
        xyz_embeds = self.xyz_embedder(points)
        densities, colors = self.radiance_field(xyz_embeds, dir_embeds)
        densities = densities.reshape(shape + (1,))
        colors = colors.reshape(shape + (3,))

        return densities, colors

    def train_step(self, data, optimizer, **kwargs):
        for k, v in data.items():
            if v.shape[0] == 1:
                data[k] = v[0] # batch size = 1
        outputs = self(data, **kwargs)
        outputs = self._parse_outputs(outputs)
        return outputs

    def val_step(self, data, optimizer, **kwargs):
        self.training = False
        collect_keys = kwargs.pop('collect_keys', None)
        outputs = self.train_step(data, optimizer, **kwargs)
        if collect_keys is None:
            return outputs
        new_out = {}
        for k in outputs.keys():
            if not isinstance(outputs[k], dict):
                new_out[k] = outputs[k]
                continue
            new_out[k] = {}
            for sub_k in outputs[k].keys():
                if sub_k in collect_keys:
                    new_out[k][sub_k] = outputs[k][sub_k]
        del outputs
        self.training = True
        return new_out


