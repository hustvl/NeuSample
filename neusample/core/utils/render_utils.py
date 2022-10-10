import torch


im2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10 * torch.log(x) / torch.log(torch.as_tensor([10], device=x.device))


def raw2outputs(densities, colors, z_vals, rays_dir, alpha_noise_std, white_bkgd):
    def process_alphas(densities, dists, act_fn=torch.nn.functional.relu): 
        return 1 - torch.exp(-act_fn(densities) * dists)

    # Computes distances
    dists = z_vals[..., 1:] - z_vals[..., :-1]

    # the distance that starts from the last point is infinity.
    dists = torch.cat([
        dists, 
        2e10 * torch.ones(dists[..., :1].shape, device=dists.device)
    ], dim=-1)  # [B, n_samples]

    # Multiplies each distance by the norm of its ray direction 
    # to convert it to real world distance (accounts for non-unit ray directions).
    dists = dists * torch.norm(rays_dir[..., None, :], dim=-1)

    # [B, n_points, 1] -> [B, n_points]
    densities = densities.squeeze(-1)

    # Adds noise to model's predictions for density. Can be used to 
    # regularize network (prevents floater artifacts).
    noise = 0
    if alpha_noise_std > 0:
        noise = torch.randn(densities.shape, device=densities.device) * alpha_noise_std

    # Predicts density of point. Higher values imply
    # higher likelihood of being absorbed at this point.
    alphas = process_alphas(densities + noise, dists)  # [B, n_points]

    # Compute weight for RGB of each sample.  A cumprod() is
    # used to express the idea of the ray not having reflected up to this
    # sample yet.
    # tf has an args: exclusive, but torch does not, so I have to do all these complicated staffs.
    # [B, n_points]    
    weights = alphas * torch.cumprod(
        torch.cat([torch.ones(tuple(alphas.shape[:-1]) + (1,), device=alphas.device), 
                   1 - alphas[..., :-1] + 1e-10], dim=-1),
        dim=-1
    )
    # Computed weighted color of each sample y
    color_map = torch.sum(weights[..., None] * colors, dim=-2)  # [B, 3]

    # Estimated depth map is expected distance.
    depth_map = torch.sum(weights * z_vals, dim=-1)  # [B]

    # Disparity map is inverse depth.
    disp_map = 1 / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, dim=-1))

    # Sum of weights along each ray. This value is in [0, 1] up to numerical error.
    acc_map = torch.sum(weights, dim=-1)

    # To composite onto a white background, use the accumulated alpha map
    if white_bkgd:
        color_map = color_map + (1 - acc_map[..., None]) 

    outputs = {
        'alphas': alphas, 
        'weights': weights, 
        'color_map': color_map, 
        'depth_map': depth_map, 
        'disp_map': disp_map, 
        'acc_map': acc_map
    }
    return outputs
