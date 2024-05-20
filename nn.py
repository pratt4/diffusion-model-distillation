import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import matplotlib.pyplot as plt
import numpy as np

'''
code adopted from https://github.com/hojonathanho/diffusion and https://github.com/ermongroup/ddim 
the blocks here are meant to be used in a DDPM_checkpoint_Model or DDIM_checkpoint_Model. 
was_pytorch argument for resnet_block and attn_block:
depending on the ordering of the variables in the checkpoints, layers must be initialized in different order to match the initial order.
as a result, the celeba model that was loaded from a pytorch file defines layers differently. Call method is kept the same.
'''

def get_timestep_embedding(timesteps, embedding_dim):
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1, 0, 0))
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb

class Downsample(nn.Module):
    def __init__(self, c, with_conv):
        super().__init__()
        if with_conv:
            self.down = nn.Conv2d(c, c, 3, stride=2, padding=1)
        else:
            self.down = nn.AvgPool2d(2)
    
    def forward(self, x, index):
        return self.down(x)

class Upsample(nn.Module):
    def __init__(self, c, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.up = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x, index):
        B, C, H, W = x.shape
        x = F.interpolate(x, size=[H * 2, W * 2], mode='nearest')
        if self.with_conv:
            x = self.up(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, c, was_pytorch, use_nin_shortcut=False, drop_rate=0.0):
        super().__init__()
        self.c = c
        self.drop_rate = drop_rate
        if was_pytorch:
            self.norm1 = nn.GroupNorm(32, c)
            self.conv1 = nn.Conv2d(c, c, 3, padding=1)
            self.temb_proj = nn.Linear(c, c)
            self.norm2 = nn.GroupNorm(32, c)
            self.conv2 = nn.Conv2d(c, c, 3, padding=1)
            if use_nin_shortcut:
                self.skip_conv = nn.Linear(c, c)
            else:
                self.skip_conv = None
        else:
            self.conv1 = nn.Conv2d(c, c, 3, padding=1)
            self.conv2 = nn.Conv2d(c, c, 3, padding=1)
            if use_nin_shortcut:
                self.skip_conv = nn.Linear(c, c)
            else:
                self.skip_conv = None

            self.norm1 = nn.GroupNorm(32, c)
            self.norm2 = nn.GroupNorm(32, c)
            self.temb_proj = nn.Linear(c, c)

    def forward(self, x, index):
        residual = x
        x = F.silu(self.norm1(x))
        x = self.conv1(x)
        
        x += self.temb_proj(F.silu(index))[:, :, None, None]
        x = F.silu(self.norm2(x))
        x = self.conv2(x)

        if self.skip_conv is not None:
            residual = self.skip_conv(residual)
        
        return x + residual

class AttnBlock(nn.Module):
    def __init__(self, c, was_pytorch):
        super().__init__()
        self.c = c
        if was_pytorch:
            self.norm = nn.GroupNorm(32, c)
            self.q = nn.Linear(c, c)
            self.k = nn.Linear(c, c)
            self.v = nn.Linear(c, c)
            self.proj_out = nn.Linear(c, c)
        else:
            self.k = nn.Linear(c, c)
            self.norm = nn.GroupNorm(32, c)
            self.proj_out = nn.Linear(c, c)
            self.q = nn.Linear(c, c)
            self.v = nn.Linear(c, c)
        
    def forward(self, x, index):
        B, C, H, W = x.shape
        residual = x
        x = self.norm(x)
        q, k, v = self.q(x), self.k(x), self.v(x)

        w = torch.einsum('bchw,bcHW->bhwHW', q, k) * (int(C) ** (-0.5))
        w = w.view(B, H, W, H * W)
        w = F.softmax(w, -1)
        w = w.view(B, H, W, H, W)
        x = torch.einsum('bhwHW,bcHW->bchw', w, v)

        x = self.proj_out(x)
        return x + residual
