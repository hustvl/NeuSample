# fp16 settings
fp16 = dict(loss_scale=512.)

# model settings
model = dict(
    type='NeuSample',
    ori_embedder=dict(
        type='BaseEmbedder',
        in_dims=3, 
        n_freqs=10, 
        include_input=True),
    dir_ray_embedder=dict(
        type='BaseEmbedder',
        in_dims=3, 
        n_freqs=10, 
        include_input=True),
    sample_field=dict(
        type='SampleField',
        nb_layers=8, 
        hid_dims=256, 
        ori_emb_dims=2*3*10+3,
        dir_emb_dims=2*3*10+3,
        n_samples=192),

    xyz_embedder=dict(
        type='BaseEmbedder',
        in_dims=3, 
        n_freqs=10, 
        include_input=True),
    dir_embedder=dict(
        type='BaseEmbedder',
        in_dims=3, 
        n_freqs=4, 
        include_input=True),
    radiance_field=dict(
        type='BaseField',
        nb_layers=8, 
        hid_dims=256, 
        xyz_emb_dims=2*3*10+3,
        dir_emb_dims=2*3*4+3,
        use_dirs=True),

    render_params=dict( # default render cfg; train cfg
        alpha_noise_std=1.0,
        inv_depth=False,
        max_rays_num=1024*4,))

# dataset settings
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        dataset=dict(
            type='SyntheticDataset',
            base_dir='./dataset/nerf_synthetic/lego', 
            half_res=False,
            batch_size=1024*4,
            white_bkgd=True,
            precrop_frac=0.5,
            testskip=8,
            split='train'),
        times=20),
    val=dict(
        type='SyntheticDataset',
        base_dir='./dataset/nerf_synthetic/lego', 
        half_res=False,
        batch_size=-1,
        white_bkgd=True,
        # precrop_frac=0.5,
        testskip=8,
        split='val'),
    test=dict(
        type='SyntheticDataset',
        base_dir='./dataset/nerf_synthetic/lego', 
        half_res=False,
        batch_size=-1,
        white_bkgd=True,
        # precrop_frac=0.5,
        testskip=1,
        split='test'),
    )

# optimizer
optimizer = dict(type='Adam', lr=5e-4, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='Poly', min_lr=5e-6, by_epoch=True)
runner = dict(type='EpochBasedRunner', max_epochs=200)
# misc settings
checkpoint_config = dict(interval=1, max_keep_ckpts=3)
log_config = dict(
    interval=200,
    hooks=[
        dict(type='TextLoggerHook', interval_exp_name=10000),
    ])
evaluation = dict(
    interval=1,
    render_params=dict(
        alpha_noise_std=0,
        inv_depth=False,
        white_bkgd=True,
        max_rays_num=1024*4,))
param_adjust_hooks = [
    dict(
        type='DatasetParamAdjustHook',
        param_name_adjust_iter_value = [
            ('precrop_frac', 1000, 1),],)]
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './data/out'
load_from = None
resume_from = None
workflow = [('train', 1)]
