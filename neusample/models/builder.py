from mmcv.utils import Registry, build_from_cfg
from torch import nn

RENDERER = Registry('renderer')
EMBEDDER = Registry('embedder')
FIELD = Registry('field')


def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_embedder(cfg):
    if cfg is not None:
        return build(cfg, EMBEDDER)
    return None


def build_field(cfg):
    if cfg is not None:
        return build(cfg, FIELD)
    return None


def build_renderer(cfg, train_cfg=None, test_cfg=None):
    return build(cfg, RENDERER)