import torch
import torch.nn as nn
import numpy as np

from models import register
from .block_func import seasonal_block, trend_block, residual_block_group


@register('TSINR_hypo')
class TSINRHypo(nn.Module):

    def __init__(self, group, global_depth, group_depth, in_dim, out_dim, global_hidden_dim, group_hidden_dim, use_pe, pe_dim, out_bias=0, pe_sigma=1024, fourier_coef=100):
        super().__init__()
        self.degree = fourier_coef
        self.use_pe = use_pe
        self.pe_dim = pe_dim
        self.pe_sigma = pe_sigma
        self.group = group
        self.global_depth = global_depth
        self.group_depth = group_depth

        self.param_shapes_r = dict()
        if use_pe:
            last_dim = in_dim * pe_dim
        else:
            last_dim = in_dim
        for i in range(global_depth):
            cur_dim = global_hidden_dim
            self.param_shapes_r[f'global_wb{i}'] = (last_dim + 1, cur_dim)
            last_dim = cur_dim

        global_dim_out = cur_dim
        for j in range(group):
            last_dim = global_dim_out
            if out_dim % group == 0:
                for i in range(group_depth):
                    cur_dim = group_hidden_dim if i < group_depth - 1 else (out_dim // group)
                    self.param_shapes_r[f'group_wb{i}_{j}'] = (last_dim + 1, cur_dim)
                    last_dim = cur_dim
            else:
                if out_dim - (group - 1) * ((out_dim // group) + 1) > 0:
                    a = (out_dim // group) + 1
                    b = out_dim - (group - 1) * ((out_dim // group) + 1)
                else:
                    a = out_dim // group
                    b = out_dim - (group - 1) * (out_dim // group)
                for i in range(group_depth):
                    if j != group - 1:
                        cur_dim = group_hidden_dim if i < group_depth - 1 else a
                        self.param_shapes_r[f'group_wb{i}_{j}'] = (last_dim + 1, cur_dim)
                    else:
                        cur_dim = group_hidden_dim if i < group_depth - 1 else b
                        self.param_shapes_r[f'group_wb{i}_{j}'] = (last_dim + 1, cur_dim)
                    last_dim = cur_dim

        self.param_shapes_s = dict()
        self.param_shapes_s[f'wb{0}'] = (2 * fourier_coef + 1, out_dim)

        self.param_shapes_t = dict()
        self.param_shapes_t[f'wb{0}'] = (2 + 1, out_dim)

        self.relu = nn.ReLU()
        self.params_t = None
        self.params_s = None
        self.params_r = None
        self.out_bias = out_bias

    def set_params(self, params_t, params_s, params_r):
        self.params_t = params_t
        self.params_s = params_s
        self.params_r = params_r

    def convert_posenc(self, x):
        w = torch.exp(torch.linspace(0, np.log(self.pe_sigma), self.pe_dim // 2, device=x.device))
        #print(x.dtype,w.dtype)
        x = torch.matmul(x.unsqueeze(-1), w.unsqueeze(0)).view(*x.shape[:-1], -1)
        x = torch.cat([torch.cos(np.pi * x), torch.sin(np.pi * x)], dim=-1)
        return x

    def forward(self, x):
        self.model_out_seas = seasonal_block(x, self.degree, self.params_s)
        self.model_out_trend = trend_block(x, degree=2, params_t=self.params_t)

        if self.use_pe:
            x = self.convert_posenc(x)
        self.model_out_res = residual_block_group(x, self.group, self.global_depth, self.group_depth, self.params_r, self.out_bias)
        self.model_output = self.model_out_res + self.model_out_trend + self.model_out_seas
        return self.model_output
