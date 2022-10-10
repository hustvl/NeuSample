import numpy as np


def collect_rays(h, w, focal, c2w):
    """
    Gets ray origins and ray directions.
    
    Args:
        h, w (int): height and width of images
        focal (float): focal length of the camera
        c2w (ndarray): [3, 4], transformation matrix from camera coordinate into world coordinate
    """
    x, y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32), 
        indexing='xy'
    )
    # ray directions in camera coordinate system
    dirs = np.stack([(x - w * .5) / focal, -(y - h * .5) / focal, -np.ones_like(x)], -1) # [h, w, 3]
    # ray directions in world coordinate system
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1) # [h, w, 1, 3] * [3, 3]
    # ray origins in world coordinate system
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d

# def convert_rays_to_ndc_rays(h, w, focal, near, rays_o, rays_d):
#     """
#     Inputs:
#         h, w, focal: image height, width and focal length
#         near: (N_rays) or float, the depths of the near plane
#         rays_o: (N_rays, 3), the origin of the rays in world coordinate
#         rays_d: (N_rays, 3), the direction of the rays in world coordinate
#     Outputs:
#         rays_o: (N_rays, 3), the origin of the rays in NDC
#         rays_d: (N_rays, 3), the direction of the rays in NDC
#     """
#     # Shift ray origins to near plane
#     t = -(near + rays_o[...,2]) / rays_d[...,2]
#     rays_o = rays_o + t[...,None] * rays_d

#     # Store some intermediate homogeneous results
#     ox_oz = rays_o[...,0] / rays_o[...,2]
#     oy_oz = rays_o[...,1] / rays_o[...,2]
    
#     # Projection
#     o0 = -1./(w/(2.*focal)) * ox_oz
#     o1 = -1./(h/(2.*focal)) * oy_oz
#     o2 = 1. + 2. * near / rays_o[...,2]

#     d0 = -1./(w/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - ox_oz)
#     d1 = -1./(h/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - oy_oz)
#     d2 = 1 - o2
    
#     rays_o = torch.stack([o0, o1, o2], -1) # (B, 3)
#     rays_d = torch.stack([d0, d1, d2], -1) # (B, 3)
#     return rays_o, rays_d


def convert_rays_to_ndc_rays(h, w, focal, near, rays_o, rays_d):
    """Normalized device coordinate rays.

    Space such that the canvas is a cube with sides [-1, 1] in each axis.

    Args:
      h: int. height in pixels.
      w: int. width in pixels.
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
    o0 = -1./(w/(2.*focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1./(h/(2.*focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1./(w/(2.*focal)) * \
        (rays_d[..., 0]/rays_d[..., 2] - rays_o[..., 0]/rays_o[..., 2])
    d1 = -1./(h/(2.*focal)) * \
        (rays_d[..., 1]/rays_d[..., 2] - rays_o[..., 1]/rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = np.stack([o0, o1, o2], -1)
    rays_d = np.stack([d0, d1, d2], -1)
    return rays_o, rays_d


if __name__ == '__main__':
    pass