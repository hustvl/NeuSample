import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import EMBEDDER


@EMBEDDER.register_module()
class BaseEmbedder(object):
    def __init__(self, in_dims, n_freqs, scale=1, include_input=True):
        self.in_dims = in_dims
        self.n_freqs = n_freqs
        self.scale = scale
        self.include_input = include_input
        self.out_dims = (2 * in_dims * n_freqs + in_dims) \
            if include_input else (2 * in_dims * n_freqs)

        self.freqs = 2 ** torch.linspace(0, self.n_freqs-1, self.n_freqs)
        self.funcs = [torch.sin, torch.cos]

    def __call__(self, inputs):
        device = inputs.device
        embeds = [inputs/self.scale] if self.include_input else []
        for freq in self.freqs:
            freq = freq.unsqueeze(0).to(device)
            for func in self.funcs:
                embeds.append(func(inputs/self.scale * freq))
        embeds = torch.cat(embeds, dim=1)
        return embeds
