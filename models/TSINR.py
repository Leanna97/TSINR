import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

import models
from models import register
from einops import rearrange
from transformers import GPT2Model


def init_wb(shape):
    weight = torch.empty(shape[1], shape[0] - 1)
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

    bias = torch.empty(shape[1], 1)
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    nn.init.uniform_(bias, -bound, bound)

    return torch.cat([weight, bias], dim=1).t().detach()


@register('TSINR')
class TransInrTSINR(nn.Module):

    def __init__(self, use_llm, seg_num, llm_out, gpt_para, gpt_layer, mlp, tokenizer, hyponet, n_groups, transformer_encoder):
        super().__init__()
        self.use_llm = use_llm
        self.seg_num = seg_num
        self.gpt_para = gpt_para
        dim = transformer_encoder['args']['dim']
        self.tokenizer = models.make(tokenizer, args={'dim': dim})
        self.hyponet = models.make(hyponet)
        self.transformer_encoder = models.make(transformer_encoder)

        self.base_params_r = nn.ParameterDict()
        n_wtokens_r = 0
        self.wtoken_postfc_r = nn.ModuleDict()
        self.wtoken_rng_r = dict()
        for name, shape in self.hyponet.param_shapes_r.items():
            self.base_params_r[name] = nn.Parameter(init_wb(shape))
            g = min(n_groups, shape[1])
            assert shape[1] % g == 0
            self.wtoken_postfc_r[name] = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, shape[0] - 1),
            )
            self.wtoken_rng_r[name] = (n_wtokens_r, n_wtokens_r + g)
            n_wtokens_r += g

        self.base_params_s = nn.ParameterDict()
        n_wtokens_s = 0
        self.wtoken_postfc_s = nn.ModuleDict()
        self.wtoken_rng_s = dict()
        for name, shape in self.hyponet.param_shapes_s.items():
            self.base_params_s[name] = nn.Parameter(init_wb(shape))
            g = min(n_groups, shape[1])
            assert shape[1] % g == 0
            self.wtoken_postfc_s[name] = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, shape[0] - 1),
            )
            self.wtoken_rng_s[name] = (n_wtokens_s, n_wtokens_s + g)
            n_wtokens_s += g

        self.base_params_t = nn.ParameterDict()
        n_wtokens_t = 0
        self.wtoken_postfc_t = nn.ModuleDict()
        self.wtoken_rng_t = dict()
        for name, shape in self.hyponet.param_shapes_t.items():
            self.base_params_t[name] = nn.Parameter(init_wb(shape))
            g = min(n_groups, shape[1])
            assert shape[1] % g == 0
            self.wtoken_postfc_t[name] = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, shape[0] - 1),
            )
            self.wtoken_rng_t[name] = (n_wtokens_t, n_wtokens_t + g)
            n_wtokens_t += g

        self.n_wtokens_t = n_wtokens_t
        self.n_wtokens_s = n_wtokens_s

        n_wtokens = n_wtokens_t + n_wtokens_s + n_wtokens_r
        self.wtokens = nn.Parameter(torch.randn(n_wtokens, dim))

        # self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True,
        #                                       output_hidden_states=True)
        # self.gpt2.h = self.gpt2.h[:gpt_layer]

        # for i, (name, param) in enumerate(self.gpt2.named_parameters()):
        #     if 'ln' in name or 'wpe' in name:  # or 'mlp' in name:
        #         param.requires_grad = True
        #     elif 'mlp' in name and mlp == 1:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False

        self.out_layer1 = nn.Linear(llm_out, tokenizer['args']['img_channels'], bias=True)

    def forward(self, data, is_classification=False):
        x_ori = data['inp']
        if (self.use_llm == True):
            x_enc = data['inp'].permute(0, 2, 1) # 8, 100, 38
            B, L, M = x_enc.shape
            # Normalization from Non-stationary Transformer
            seg_num =self.seg_num
            x_enc = rearrange(x_enc, 'b (n s) m -> b n s m', s=seg_num)
            means = x_enc.mean(2, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=2, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
            x_enc = rearrange(x_enc, 'b n s m -> b (n s) m') # 8, 100, 38

            enc_out = torch.nn.functional.pad(x_enc, (0, 768 - x_enc.shape[-1])) # 8, 100, 768
            gpt_data = self.gpt2(inputs_embeds=enc_out).last_hidden_state
            #
            gpt_data = self.out_layer1(gpt_data)

            trans_out = rearrange(gpt_data, 'b (n s) m -> b n s m', s=seg_num)
            trans_out = trans_out * \
                      (stdev[:, :, 0, :].unsqueeze(2).repeat(
                          1, 1, seg_num, 1))
            trans_out = trans_out + \
                      (means[:, :, 0, :].unsqueeze(2).repeat(
                          1, 1, seg_num, 1))
            trans_out = rearrange(trans_out, 'b n s m -> b (n s) m')

            data['inp'] = trans_out.permute(0, 2, 1) #8,38,100

            dtokens = self.tokenizer(data)
        else:
            dtokens = self.tokenizer(data)

        B = dtokens.shape[0]
        wtokens = einops.repeat(self.wtokens, 'n d -> b n d', b=B)
        trans_out = self.transformer_encoder(torch.cat([dtokens, wtokens], dim=1))
        trans_out = trans_out[:, -len(self.wtokens):, :]

        params_t = dict()
        for name, shape in self.hyponet.param_shapes_t.items():
            # print('trans_inr_4')
            wb = einops.repeat(self.base_params_t[name], 'n m -> b n m', b=B)
            w, b = wb[:, :-1, :], wb[:, -1:, :]

            l, r = self.wtoken_rng_t[name]
            x = self.wtoken_postfc_t[name](trans_out[:, l: r, :])
            x = x.transpose(-1, -2)  # (B, shape[0] - 1, g)
            w = F.normalize(w * x.repeat(1, 1, w.shape[2] // x.shape[2]), dim=1)

            wb = torch.cat([w, b], dim=1)
            params_t[name] = wb

        params_s = dict()
        for name, shape in self.hyponet.param_shapes_s.items():
            #print('trans_inr_4')
            wb = einops.repeat(self.base_params_s                                                                                                                                                                                                                                                                                                                                                                                                      [name], 'n m -> b n m', b=B)
            w, b = wb[:, :-1, :], wb[:, -1:, :]

            l, r = self.wtoken_rng_s[name]
            x = self.wtoken_postfc_s[name](trans_out[:, l+self.n_wtokens_t: r+self.n_wtokens_t, :])
            x = x.transpose(-1, -2) # (B, shape[0] - 1, g)
            w = F.normalize(w * x.repeat(1, 1, w.shape[2] // x.shape[2]), dim=1)

            wb = torch.cat([w, b], dim=1)
            params_s[name] = wb

        params_r = dict()
        for name, shape in self.hyponet.param_shapes_r.items():
            # print('trans_inr_4')
            wb = einops.repeat(self.base_params_r[name], 'n m -> b n m', b=B)
            w, b = wb[:, :-1, :], wb[:, -1:, :]

            l, r = self.wtoken_rng_r[name]
            x = self.wtoken_postfc_r[name](trans_out[:, l+self.n_wtokens_t+self.n_wtokens_s: r+self.n_wtokens_t+self.n_wtokens_s, :])
            x = x.transpose(-1, -2)  # (B, shape[0] - 1, g)
            w = F.normalize(w * x.repeat(1, 1, w.shape[2] // x.shape[2]), dim=1)

            wb = torch.cat([w, b], dim=1)
            params_r[name] = wb

        self.hyponet.set_params(params_t, params_s, params_r)
        return self.hyponet
