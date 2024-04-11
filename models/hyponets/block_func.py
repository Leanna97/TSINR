import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .layers import batched_linear_mm

def seasonal_block(x, degree, params_s):
    coords_org = x.clone().detach()
    coords = coords_org
    device = coords.device
    argument = 2 * np.pi * torch.arange(1, degree + 1)
    sin_arg = argument.to(device) * coords
    monomial_cos = torch.cos(sin_arg)
    monomial_sin = torch.sin(sin_arg)
    full_monomial = torch.cat([monomial_cos, monomial_sin], dim=-1)
    output = batched_linear_mm(full_monomial, params_s[f'wb{0}'])
    return output

def trend_block(x, degree, params_t):
    coords_org = x.clone().detach()
    coords = coords_org

    monomial = torch.cat([coords ** n for n in range(1, degree + 1)], dim=-1)
    output = batched_linear_mm(monomial, params_t[f'wb{0}'])
    return output

def residual_block(x, depth, params_r, out_bias):
    coords_org = x.clone().detach()
    coords = coords_org

    # various input processing methods for different applications
    for i in range(depth):
        coords = batched_linear_mm(coords, params_r[f'wb{i}'])
        if i < depth - 1:
            coords = nn.ReLU()(coords)
        else:
            coords = coords + out_bias

    return coords


def residual_block_group(x, group, global_depth, group_depth, params_r, out_bias):
    coords_org = x.clone().detach()
    coords = coords_org

    for i in range(global_depth):
        coords = batched_linear_mm(coords, params_r[f'global_wb{i}'])
        coords = nn.ReLU()(coords)

    coords_all = []
    for j in range(group):
        coords_group = coords.clone()
        for i in range(group_depth):
            coords_group = batched_linear_mm(coords_group, params_r[f'group_wb{i}_{j}'])
            if i < group_depth - 1:
                coords_group = nn.ReLU()(coords_group)
            else:
                coords_group = coords_group + out_bias
        coords_all.append(coords_group)

    coords = torch.cat(coords_all, dim=-1)

    return coords
