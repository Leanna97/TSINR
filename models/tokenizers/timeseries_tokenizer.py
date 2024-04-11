import torch
import torch.nn as nn
import torch.nn.functional as F
import unfoldNd

from models import register


@register('timeseries_tokenizer')
class TimeseriesTokenizer(nn.Module):

    def __init__(self, input_size, patch_size, dim, padding=0, img_channels=2):
        super().__init__()
        
        self.patch_size = patch_size
        self.padding = padding
        self.prefc = nn.Linear(patch_size * img_channels, dim)
        n_patches = (input_size + padding * 2) // patch_size
        self.posemb = nn.Parameter(torch.randn(n_patches, dim))
        self.module = unfoldNd.UnfoldNd(self.patch_size, stride=self.patch_size, padding=self.padding) # (B, C * p * p, L)

    def forward(self, data):
        x = data['inp']
        p = self.patch_size
        x = self.module(x)
        x = x.permute(0, 2, 1).contiguous()
        x = self.prefc(x) + self.posemb.unsqueeze(0)
        return x
