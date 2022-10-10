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
        scale=32,
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
    samples_per_gpu=1024*4,
    workers_per_gpu=16,
    train=dict(        
        type='RepeatDataset',
        dataset=dict(
            type='LLFFDataset',
            datadir='./dataset/nerf_llff_data/fern', 
            factor=4, 
            batch_size=None,
            split='train',
            batching=True, 
            spherify=False, 
            no_ndc=False, 
            to_cuda=True,
            holdout=8),
        times=1),
    val=dict(
        type='LLFFDataset',
        datadir='./dataset/nerf_llff_data/fern', 
        factor=4, 
        batch_size=-1,
        split='val', 
        batching=False,
        spherify=False, 
        no_ndc=False, 
        to_cuda=True,
        holdout=8),)

# optimizer
optimizer = dict(type='Adam', lr=5e-4, betas=(0.9, 0.999))
# optimizer = dict(type='Adam', lr=1e-3, betas=(0.9, 0.999))
# optimizer = dict(type='AdamW', lr=1e-3)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='Poly', min_lr=5e-6, by_epoch=False)
runner = dict(type='EpochBasedRunner', max_epochs=100)
# misc settings
checkpoint_config = dict(interval=1, max_keep_ckpts=5)
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
        max_rays_num=1024*4,))
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './data/out'
load_from = None
resume_from = None
workflow = [('train', 1)]
