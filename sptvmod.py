"""
SPTVMod (Series-Parallel Time-Variant Modulation)

Author: Yann Bourdin

2025/12/04
"""

from math import ceil, floor, prod
from typing import Sequence
import torch
from torch import nn
from torch.nn import functional as F
from rnn import RNN
from dp4_aa_taps import dp4_aa_taps

class CachedPadding1d(torch.nn.Module):
    """
    Cached Padding implementation, replace zero padding with the end of the previous tensor.
    https://github.com/acids-ircam/cached_conv/blob/master/cached_conv/convs.py
    """
    NoPadding = 0
    ZeroPadding = 1
    CachedPadding = 2

    def __init__(self, padding, max_batch_size=256):
        super().__init__()
        self.initialized = 0
        self.padding = padding
        self.max_batch_size = max_batch_size

    @torch.jit.unused
    @torch.no_grad()
    def init_cache(self, x):
        b, c, _ = x.shape
        self.register_buffer(
            "pad",
            torch.zeros(self.max_batch_size, c, self.padding).to(x),
            persistent=False
        )
        self.initialized += 1

    def forward(self, x, paddingmode=CachedPadding):
        if not self.initialized:
            self.init_cache(x)

        if self.padding:
            match paddingmode:
                case CachedPadding1d.ZeroPadding:
                    x = torch.nn.functional.pad(x, (self.padding, 0))
                case CachedPadding1d.CachedPadding:
                    x = torch.cat([self.pad[:x.shape[0]], x], -1)
                case CachedPadding1d.NoPadding:
                    pass
            self.pad[:x.shape[0]].copy_(x[..., -self.padding:].detach())

        return x

class CachedConv1d(nn.Module):
    """
    Implementation of a Conv1d **with stride 1** operation with cached padding (same).
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        assert kwargs.get("padding", 0) == 0 and kwargs.get("stride", 1) == 1
        self.conv = nn.Conv1d(*args, **kwargs)

        padding = (self.conv.kernel_size[0] - 1) * self.conv.dilation[0]

        self.cache = CachedPadding1d(padding)

    def forward(self, x, paddingmode=CachedPadding1d.CachedPadding):
        x = self.cache(x, paddingmode=paddingmode)
        return self.conv(
            x,
        )
    
class LinearInterp(torch.nn.Module):
    def __init__(self, channels, size_factor):
        super().__init__()
        self.convt = nn.ConvTranspose1d(channels, channels, 2 * size_factor, stride=size_factor, padding=size_factor, groups=channels, bias=False)
        self.convt.weight.data[:, :, :] = torch.concat(
            [
                torch.arange(0, size_factor) / size_factor,
                1 - torch.arange(0, size_factor) / size_factor
            ], 0).view(1, 1, 2 * size_factor)
        self.convt.weight.requires_grad = False

    def forward(self, *args, **kwargs):
        return self.convt(*args, **kwargs)

def film_op(x, ab, channels):
    if ab.ndim == 2 and x.ndim == 3:
        ab = ab.unsqueeze(-1)
    a, b = torch.split(ab, channels, dim=1)
    return a * x + b

def lookback_length(kernel_size_list, dilation_list, stride_list, pooling_list):
    n = 1
    for k, d, s, p in list(zip(kernel_size_list, dilation_list, stride_list, pooling_list, strict=True))[::-1]:
        n = 1 + d * (k - 1) + s * (p * n - 1)
    return n

class SPN1DFeatures(torch.nn.Module):
    """
    Sequence of `FeatBlocks' 
    """
    def __init__(self, in_channels, cond_dim, out_channels_list, kernel_size_list, dilation_list, stride_list, pooling_list, pooling_type, film_hidden_neurons_list):
        super().__init__()
        assert len(out_channels_list) == len(kernel_size_list) == len(dilation_list) == len(stride_list)

        self.in_channels = in_channels
        self.cond_dim = cond_dim
        self.film_hidden_neurons_list = film_hidden_neurons_list
        self.out_channels_list = out_channels_list
        self.kernel_size_list = kernel_size_list
        self.dilation_list = dilation_list
        self.stride_list = stride_list
        self.pooling_list = pooling_list
        assert pooling_type in ("max", "avg")
        self.pooling_type = pooling_type

        self.convs = nn.ModuleList([
            torch.nn.Conv1d(_i, _o, _k, stride=_s, dilation=_d) 
            for _i, _o, _k, _s, _d in zip(
                [in_channels] + out_channels_list[:-1],
                out_channels_list,
                kernel_size_list,
                stride_list,
                dilation_list
            )
        ])

        if cond_dim > 0:
            assert isinstance(film_hidden_neurons_list, Sequence)
            ## make a sequence of Linear -> PReLU -> Linear -> PReLU -> ... -> Linear
            self.films = []
            for c in out_channels_list:
                layers = []
                if len(film_hidden_neurons_list) == 0 or not film_hidden_neurons_list[0]:
                    h = cond_dim
                else:
                    for i, h in enumerate(film_hidden_neurons_list):
                        layers.append(nn.Sequential(
                            nn.Linear(cond_dim if i == 0 else film_hidden_neurons_list[i - 1], h),
                            nn.PReLU(h)
                        ))
                self.films.append(nn.Sequential(*layers, nn.Linear(h, 2 * c)))
            self.films = nn.ModuleList(self.films)
        else: 
            self.films = len(out_channels_list) * [None]
        self.acts = nn.ModuleList([
            torch.nn.PReLU(_o) for _o in out_channels_list
        ])
        self.pools = torch.nn.ModuleList([
            torch.nn.MaxPool1d(bk)
            if self.pooling_type == "max" else
            torch.nn.AvgPool1d(bk)
            for bk in self.pooling_list
        ])
    
    def get_needed_samples(self):
        n = 1
        for k, d, s, p in list(zip(self.kernel_size_list, self.dilation_list, self.stride_list, self.pooling_list, strict=True))[::-1]:
            n = 1 + d * (k - 1) + s * (p * n - 1)
        return n

    def forward(self, x, p):
        v = x

        ft_list = []
    
        for conv, act, film, pool, c in zip(self.convs, self.acts, self.films, self.pools, self.out_channels_list, strict=True):
            v = conv(v)
            v = act(v)
            if film is not None:
                ab = film(p)
                v = film_op(v, ab, c)
            v = pool(v)
            ft_list.append(v)

        return ft_list
    
class CachedTFiLM(torch.nn.Module):
    """
    Pooling -> FiLM (optional) -> RNN -> Upsample
    """
    def __init__(self, in_channels, block_size, cond_dim, pooling_type, rnn_cell, rnn_hidden_size, film_hidden_neurons_list):
        super().__init__()
        self.in_channels = in_channels
        self.block_size = block_size
        self.rnn_cell = rnn_cell
        self.rnn_hidden_size = rnn_hidden_size
        self.cond_dim = cond_dim
        self.film_hidden_neurons_list = film_hidden_neurons_list
        assert pooling_type in ("max", "avg")
        self.pooling_type = pooling_type
        self.maxpool = nn.MaxPool1d(block_size) if pooling_type == "max" else nn.AvgPool1d(block_size)
        self.rnn = RNN(in_channels, rnn_hidden_size, rnn_cell)
        self.state = None
        self.upsample = LinearInterp(rnn_hidden_size, block_size)
        self.cache = CachedPadding1d(1)

        self.film_nn = None

        if cond_dim > 0:
            if isinstance(film_hidden_neurons_list, Sequence):
                self.film_nn = nn.ModuleList()
                if len(film_hidden_neurons_list) == 0 or not film_hidden_neurons_list[0]:
                    h = cond_dim
                else:
                    for i, h in enumerate(film_hidden_neurons_list):
                        self.film_nn.append(nn.Sequential(
                            nn.Linear(cond_dim if i == 0 else film_hidden_neurons_list[i - 1], h),
                            nn.PReLU(h)
                        ))

                self.film_nn.append(nn.Linear(h, 2 * self.in_channels))
                self.film_nn = nn.Sequential(*self.film_nn)
    
    def forward(self, z, p, state=None, paddingmode=0):
        """ z [B, C, L] p [B, P] """
        assert z.shape[2] % self.block_size == 0

        if state is None:
            if self.state is None:
                self.state = torch.zeros((z.size(0), (2 if self.rnn_cell == "LSTM" else 1) * self.rnn_hidden_size), device=z.device, requires_grad=True).detach()
        else:
            self.state = state

        z2 = self.maxpool(z)
        if self.film_nn is not None:
            ab = self.film_nn(p)
            z2 = film_op(z2, ab, self.in_channels)
        z2, self.state = self.rnn(z2, self.state)
        z2 = self.cache(z2, paddingmode=paddingmode)
        z2 = self.upsample(z2)
        return z2

def get_act(name, channels=None):
    match name:
        case None | "identity":
            act = nn.Identity()
        case "prelu":
            act = nn.PReLU(channels)
        case _:
            activations_lc = [str(a).lower() for a in nn.modules.activation.__all__]
            if (name := str(name).lower()) in activations_lc:
                # match actual name from lower-case list, return function/factory
                idx = activations_lc.index(name)
                act_name2 = nn.modules.activation.__all__[idx]
                act = getattr(nn.modules.activation, act_name2)()
            else:
                raise ValueError(f"Cannot find activation function for string <{act}>")
    return act
    
class Conv1dSequence(torch.nn.Module):
    """
    Sequence of Conv1d layers
    """
    def __init__(self, in_channels, conv_channels_list, conv_kernel_size_list, conv_dilation_list, conv_groups_list):
        super().__init__()
        self.in_channels = in_channels
        self.conv_channels_list = conv_channels_list
        self.conv_kernel_size_list = conv_kernel_size_list
        self.conv_dilation_list = conv_dilation_list
        self.conv_groups_list = conv_groups_list

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels if i == 0 else conv_channels_list[i - 1], conv_channels, conv_kernel_size, dilation=conv_dilation, groups=conv_groups)
            for i, (conv_channels, conv_kernel_size, conv_dilation, conv_groups) 
            in enumerate(zip(conv_channels_list, conv_kernel_size_list, conv_dilation_list, conv_groups_list, strict=True))
        ])

        self.needed_samples = sum(d * (k - 1) for d, k in zip(self.conv_dilation_list, self.conv_kernel_size_list, strict=True))
        self.pad = CachedPadding1d(self.needed_samples)

    @property
    def out_channels(self):
        return self.conv_channels_list[-1]
    
    def forward(self, x, paddingmode=CachedPadding1d.NoPadding):
        x = self.pad(x, paddingmode=paddingmode)
        for conv in self.convs:
            x = conv(x)
        return x

class Model(nn.Module):
    def __init__(
            self, /, dp4_antialias, s_input_kind, n_params,
            f_conv_channels_listlist, f_conv_kernel_size_listlist, f_conv_dilation_listlist, f_conv_groups_listlist,
            f_activation_list, b_kernel_size,
            s_channels_list, s_kernel_size_list, s_dilation_list, tfilm_block_size, tfilm_pooling_type,
            film_hidden_neurons_list, rnn_cell, rnn_hidden_size,
            spn, spn_use_disc_features, init_state_method, spn_bef_state_transf, spn_nn_concat_params, spn_nn_hidden_list,
            spn_out_channels_list, spn_kernel_size_list, spn_dilation_list, spn_stride_list, spn_pooling_list, spn_pooling_type, spn_film_hidden_neurons_list,
            disc_out_channels_list, disc_kernel_size_list, disc_dilation_list, disc_stride_list, disc_pooling_list, disc_pooling_type, disc_film_hidden_neurons_list,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.n_params = n_params
        self.tfilm_block_size = tfilm_block_size
        self.s_input_kind = s_input_kind
        self.init_state_method = init_state_method
        self.spn = spn
        self.spn_use_disc_features = spn_use_disc_features

        self.n_blocks = len(f_conv_channels_listlist or [])

        self.s_convs = nn.ModuleList(
            CachedConv1d(1 if i==0 else s_channels_list[i-1], c, k, dilation=d, padding=0)
            for i, (c, k, d) in enumerate(zip(s_channels_list, s_kernel_size_list, s_dilation_list))
        )

        self.s_conv11m = nn.ModuleList(
            nn.Conv1d(rnn_hidden_size, 2 * c, 1)
            for c in s_channels_list
        )

        self.f_convs = nn.ModuleList(
            Conv1dSequence(
                in_channels=1 if i == 0 else f_conv_channels_listlist[i - 1][-1], 
                conv_channels_list=conv_channels_list,
                conv_kernel_size_list=conv_kernel_size_list,
                conv_dilation_list=conv_dilation_list,
                conv_groups_list=conv_groups_list
            )
            for i, (conv_channels_list, conv_kernel_size_list, conv_dilation_list, conv_groups_list)
            in enumerate(zip(f_conv_channels_listlist, f_conv_kernel_size_listlist, f_conv_dilation_listlist, f_conv_groups_listlist, strict=True))
        )

        self.s_conv11a = nn.ModuleList(
            nn.Conv1d(rnn_hidden_size, 2 * conv.out_channels, 1)
            for conv in self.f_convs
        )

        self.f_skips = nn.ModuleList(
            nn.Conv1d(block.in_channels, block.out_channels, 1)
            for block in self.f_convs
        )

        self.f_acts = nn.ModuleList(get_act(act_name, channels=block.out_channels) for act_name, block in zip(f_activation_list, self.f_convs, strict=True))

        self.b_conv = CachedConv1d(self.f_convs[-1].out_channels, 1, b_kernel_size) if b_kernel_size else None

        cond_dim = n_params

        self.s_tfilms = nn.ModuleList(
            CachedTFiLM(c, tfilm_block_size, cond_dim, tfilm_pooling_type, rnn_cell, rnn_hidden_size, film_hidden_neurons_list)
            for c in s_channels_list
        )

        self.s_acts = nn.ModuleList(
            nn.PReLU(conv.conv.out_channels) for conv in self.s_convs
        )

        self.state_size_d = {}
        self.state_size_d["lstm"] = [len(s_channels_list), (2 if rnn_cell == "LSTM" else 1) * rnn_hidden_size]
        # if more states were to be predicted, their shape should be added to state_size_d
        self.state_size_sections_d = { k: prod(v) for k, v in self.state_size_d.items() }
        self.state_size_total = sum(self.state_size_sections_d.values())

        if spn and spn_out_channels_list and spn_use_disc_features: 
            print("WARNING: SPN is used and its HPs are defined but spn_use_disc_features is True")
        
        self.init_state_noise_size = 2 if init_state_method == "angle" else 32
        self.init_state_transf = nn.Linear(self.init_state_noise_size, self.state_size_total)

        _channels_list = disc_out_channels_list if spn_use_disc_features else spn_out_channels_list
        self.spn_ft_11 = nn.ModuleList([
            nn.Conv1d(_o, _channels_list[-1], 1) for _o in _channels_list[:-1]
        ]) if spn else None

        if spn:
            _spn_nn_in_size = _channels_list[-1] + (n_params if spn_nn_concat_params else 0)
            _spn_nn_out_size = self.init_state_noise_size if spn_bef_state_transf else self.state_size_total
            _modules = nn.ModuleList()
            for i, h in enumerate(spn_nn_hidden_list):
                _modules.append(nn.Sequential(
                    nn.Linear(_spn_nn_in_size if i == 0 else h, h),
                    nn.PReLU(h)
                ))
            _modules.append(nn.Linear(_spn_nn_in_size if len(spn_nn_hidden_list) == 0 else h, _spn_nn_out_size))
            self.spn_nn = nn.Sequential(*_modules)
        else:
            self.spn_nn = None
        
        self.disc_ft_11 = nn.ModuleList([
            nn.Conv1d(_o, disc_out_channels_list[-1], 1) for _o in disc_out_channels_list[:-1]
        ])

        self.spn_features = SPN1DFeatures(
            in_channels=2, 
            cond_dim=cond_dim,
            out_channels_list=spn_out_channels_list,
            kernel_size_list=spn_kernel_size_list,
            dilation_list=spn_dilation_list,
            stride_list=spn_stride_list,
            pooling_list=spn_pooling_list,
            pooling_type=spn_pooling_type,
            film_hidden_neurons_list=spn_film_hidden_neurons_list,
        ) if spn and not spn_use_disc_features else None

        self.disc_features = SPN1DFeatures(
            in_channels=2, 
            cond_dim=cond_dim,
            out_channels_list=disc_out_channels_list,
            kernel_size_list=disc_kernel_size_list,
            dilation_list=disc_dilation_list,
            stride_list=disc_stride_list,
            pooling_list=disc_pooling_list,
            pooling_type=disc_pooling_type,
            film_hidden_neurons_list=disc_film_hidden_neurons_list,
        )

        self.disc_nn = nn.Sequential(
            nn.PReLU(disc_out_channels_list[-1]),
            nn.Linear(disc_out_channels_list[-1], disc_out_channels_list[-1]),
            nn.PReLU(disc_out_channels_list[-1]),
            nn.Linear(disc_out_channels_list[-1], 1)
        )

        if dp4_antialias:
            self.dp4_aa_conv = CachedConv1d(1, 1, len(dp4_aa_taps), bias=False)
            self.dp4_aa_conv.conv.weight.data[:, :, :] = torch.tensor(dp4_aa_taps).flip(-1).view(1, 1, -1)
            self.dp4_aa_conv.conv.weight.requires_grad = False
        else:
            self.dp4_aa_conv = None

        self._first_pass = True
    
    def disc_from_features(self, ft_list):
        _disc_v = ft_list[-1].mean(-1)
        for ft, conv11 in zip(ft_list[:-1], self.disc_ft_11, strict=True):
            _disc_v += conv11(ft.mean(2, keepdims=True)).squeeze(-1)
        _disc_v = self.disc_nn(_disc_v)
        return _disc_v
    
    def reset_caches(self):
        for module in self.modules():
            if isinstance(module, CachedPadding1d):
                module.initialized = False

    def reset_states(self):
        for tfilm in self.s_tfilms:
            tfilm.state = None
    
    def is_state_initialized(self):
        if self.s_tfilms is not None:
            for tfilm in self.s_tfilms:
                return tfilm.state is not None

    def detach_states(self):
        for tfilm in self.s_tfilms:
            tfilm.state = tfilm.state.detach()
    
    def calc_indices(self, target_length):
        """
        see https://github.com/ybourdin/sptmod?tab=readme-ov-file#computing-the-slicing-indices-and-the-cropping-sizes for details about this function
        returns (
            expected input length,
            starting index for modulation path,
            starting index for audio path,
            cropping size list for modulation path,
            cropping size list for audio path
        )
        """
        L = target_length
        N = len(self.s_convs)
        P = self.tfilm_block_size
        sa = [
            conv.needed_samples
            for conv in self.f_convs
        ]
        sm = [
            (conv.conv.kernel_size[0] - 1) * conv.conv.dilation[0]
            for conv in self.s_convs
        ]
        km = [0 for _ in range(N)]
        cmm = [0 for _ in range(N)]
        cma = [0 for _ in range(N)]
        km[-1] = 1 + ceil(L / P)
        cmm[-1] = - L + (km[-1] - 1) * P
        cma[-1] = - L + (km[-1] - 1) * P
        for j in range(N - 2, -1, -1):
            sum_sm = sum(sm[j + 1 :])
            sum_sa = sum(sa[j + 1 :])
            sum_cmm = sum(cmm[j + 1 :])
            km[j] = max(N - j + ceil((L + sum_sm + sum_cmm) / P), 1 + ceil((L + sum_sa) / P))
            cmm[j] = - L + (km[j] - (N - j)) * P - sum_sm - sum_cmm
            cma[j] = - L - sum_sa + (km[j] - 1) * P
        tm0 = target_length - sm[0] - km[0] * P
        ta0 = - sum(sa)
        input_length = max(target_length - tm0, target_length - ta0)
        i0m = input_length - (target_length - tm0)
        i0a = input_length - (target_length - ta0)
        return input_length, i0m, i0a, cmm, cma
    
    def set_target_length(self, target_length):
        self.target_length = target_length
        target_for_calc_indices = target_length + self.b_conv.conv.kernel_size[0] + (len(dp4_aa_taps) if self.dp4_aa_conv is not None else 0)
        # to re-use the same calc_indices function from SPTMod we need to include the last convolution (1x1 in the paper) and the antialiasing filter
        self.input_length, self.i0m, self.i0a, self.cmm, self.cma = self.calc_indices(target_for_calc_indices)

    def forward(self, x, p, y_true=None, *, use_spn=False, paddingmode=CachedPadding1d.NoPadding):
        """
        x: tensor of shape [B, C, Lin] (batch size, input_channels, input length)
        p: tensor of shape [B, P] (batch size, number of parameters)
        y_true: tensor of shape [B, C, Lin] (batch size, input_channels, input length)

        output:
            y: tensor of shape [B, C, Lout]
        """
        
        lbstates = {}
        if not self.is_state_initialized():
            if use_spn and self.spn:
                if self.spn_use_disc_features:
                    disc_input_true = torch.concat((
                                x[:, :, - self.target_length :],
                                y_true[:, :, - self.target_length :]
                        ), 
                        dim=1
                    )
                    ft_true_list = self.disc_features(disc_input_true, p)
                else:
                    spn_input = torch.concat((
                            x[:, :, - self.spn_features.expected_input_length :],
                            y_true[:, :, - self.spn_features.expected_input_length :]
                        ), 
                        dim=1
                    )
                    ft_true_list = self.spn_features(spn_input, p)
                if self._first_pass:
                    print(f"[SPN] Shapes in ft_true_list: {[_v.shape for _v in ft_true_list]}")

                _spn_v = ft_true_list[-1].mean(dim=-1)
                for ft, conv11 in zip(ft_true_list[:-1], self.spn_ft_11, strict=True):  
                    _spn_v += torch.mean(conv11(ft), dim=2)

                v = self.spn_nn(_spn_v)

            else:
                match self.init_state_method:
                    case "zero":
                        v = torch.zeros((x.size(0), self.init_state_noise_size), device=x.device, requires_grad=False)
                    case "noise":
                        v = torch.randn((x.size(0), self.init_state_noise_size), device=x.device, requires_grad=False)
                    case "angle":
                        angle = 2 * torch.pi * torch.rand((x.size(0), 1), device=x.device, requires_grad=False)
                        v = torch.concat([torch.cos(angle), torch.sin(angle)], dim=1)
                    case _:
                        raise ValueError(f"{self.hparams.init_state_method=}, maybe you need to enable SPN?")
                v = self.init_state_transf(v)

            _states_l = torch.split(v, list(self.state_size_sections_d.values()), dim=1)
            lbstates = {
                k: s.reshape(x.size(0), *size) for (k, size), s in zip(self.state_size_d.items(), _states_l, strict=True)
            }

        match self.s_input_kind:
            case None | "zero":
                m = torch.zeros_like(x)
            case "input":
                m = x
            case _:
                raise ValueError

        if paddingmode == CachedPadding1d.NoPadding:
            # select the time ranges calculated with self.calc_indices()
            vm = m[:, :, self.i0m:]
            va = x[:, :, self.i0a:]
        else:
            vm = m
            va = x
        # vm (resp. va) is the intermediary tensor of the operations in the modulation (resp. audio) path

        for j in range(self.n_blocks):
            vm = self.s_convs[j](vm, paddingmode=paddingmode)
            vm = self.s_acts[j](vm)
            if not lbstates:
                initial_state = None
            else:
                initial_state = lbstates["lstm"][:, j]
            tfilm = self.s_tfilms[j](vm, p, state=initial_state, paddingmode=paddingmode)
            if paddingmode == CachedPadding1d.NoPadding:
                # cropping
                tfilm_a = tfilm[:, :, self.cma[j]:]
            else:
                tfilm_a = tfilm
            mu_j = self.s_conv11a[j](tfilm_a)

            va0 = va
            va = self.f_convs[j](va, paddingmode=paddingmode)
            va = film_op(va, mu_j, self.f_convs[j].out_channels)
            va = self.f_acts[j](va)
            va_skip = self.f_skips[j](va0[..., -va.size(-1):])
            va = va_skip + va

            if j < self.n_blocks - 1 :
                tfilm_m = self.s_conv11m[j](tfilm)
                if paddingmode == CachedPadding1d.NoPadding:
                    # TFiLM consumes `tfilm_block_size` samples
                    # crop to allow the following FiLM operation between vm and tfilm_m
                    vm = vm[:, :, self.tfilm_block_size:]
                vm = film_op(vm, tfilm_m, self.s_convs[j].conv.out_channels)
                if paddingmode == CachedPadding1d.NoPadding:
                    # cropping
                    vm = vm[:, :, self.cmm[j]:]
            
        y = va
        y = self.b_conv(va, paddingmode=paddingmode)
        if self.dp4_aa_conv is not None:
            y = self.dp4_aa_conv(y, paddingmode=paddingmode)
        if paddingmode == CachedPadding1d.NoPadding:
            y = y[:, :, -self.target_length:]

        return y