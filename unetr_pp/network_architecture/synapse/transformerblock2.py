
import torch.nn as nn
import torch
from unetr_pp.network_architecture.dynunet_block import UnetResBlock
import math
from functools import partial
import torch.nn.functional as F
import math
from timm.models.layers import trunc_normal_tf_
from timm.models.helpers import named_apply


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

# Other types of layers can go here (e.g., nn.Linear, etc.)
def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # efficientnet like
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer
class Cheap3DLocalMixer(nn.Module):
    """
    Cheap replacement for conv51+conv52+conv8.
    Depthwise 3D conv for local mixing + pointwise 1x1 for channel mixing.
    """
    def __init__(self, channels: int, drop: float = 0.1, norm="batch"):
        super().__init__()
        # depthwise conv: very cheap vs full conv
        self.dw = nn.Conv3d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)

        if norm == "batch":
            self.norm = nn.BatchNorm3d(channels)
        elif norm == "instance":
            self.norm = nn.InstanceNorm3d(channels, affine=True)
        else:
            raise ValueError("norm must be 'batch' or 'instance'")

        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        # pointwise channel mix (equivalent to your conv8 1x1)
        self.pw = nn.Conv3d(channels, channels, kernel_size=1, bias=True)
        self.drop = nn.Dropout3d(drop, inplace=False)

    def forward(self, x):
        # x: [B,C,H,W,D]
        y = self.dw(x)
        y = self.norm(y)
        y = self.act(y)
        y = self.drop(y)
        y = self.pw(y)
        return y

class TransformerBlock(nn.Module):
    """
    A transformer block, based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            proj_size: int,
            num_heads: int,
            dropout_rate: float = 0.0,
            pos_embed=False,
    ) -> None:
        """
        Args:
            input_size: the size of the input for each stage.
            hidden_size: dimension of hidden layer.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            pos_embed: bool argument to determine if positional embedding is used.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            print("Hidden size is ", hidden_size)
            print("Num heads is ", num_heads)
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.norm = nn.LayerNorm(hidden_size)
        self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)
        self.epa_block = EPA(input_size=input_size, hidden_size=hidden_size, proj_size=proj_size, num_heads=num_heads,
                             channel_attn_drop=dropout_rate,spatial_attn_drop=dropout_rate)
        self.conv51 = UnetResBlock(3, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="batch")
        self.conv52 = UnetResBlock(3, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="batch")
        self.conv8 = nn.Sequential(nn.Dropout3d(0.1, False), nn.Conv3d(hidden_size, hidden_size, 1))
        self.local_mixer = Cheap3DLocalMixer(hidden_size, drop=0.1, norm="batch")

        self.pos_embed = None
        if pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size))

    def forward(self, x):
        B, C, H, W, D = x.shape

        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        attn = x + self.gamma * self.epa_block(self.norm(x))

        attn_skip = attn.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)  # (B, C, H, W, D)
        attn = self.conv51(attn_skip)
        # attn = self.conv52(attn)
        x = attn_skip + self.conv8(attn)
        #x = attn_skip + self.local_mixer(attn_skip)

        return x


# class CAB(nn.Module):
#     def __init__(self, in_channels, out_channels=None, ratio=16, activation='relu'):
#         super(CAB, self).__init__()

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         if self.in_channels < ratio:
#             ratio = self.in_channels
#         self.reduced_channels = self.in_channels // ratio
#         if self.out_channels == None:
#             self.out_channels = in_channels

#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.activation = act_layer(activation, inplace=True)
#         self.fc1 = nn.Conv2d(self.in_channels, self.reduced_channels, 1, bias=False)
#         self.fc2 = nn.Conv2d(self.reduced_channels, self.out_channels, 1, bias=False)
        
#         self.sigmoid = nn.Sigmoid()

#         self.init_weights('normal')
    
#     def init_weights(self, scheme=''):
#         named_apply(partial(_init_weights, scheme=scheme), self)

#     def forward(self, x):
#         avg_pool_out = self.avg_pool(x) 
#         avg_out = self.fc2(self.activation(self.fc1(avg_pool_out)))

#         max_pool_out= self.max_pool(x) 
#         max_out = self.fc2(self.activation(self.fc1(max_pool_out)))

#         out = avg_out + max_out
#         return self.sigmoid(out) 
    


class MSDC(nn.Module):
    def __init__(self, in_channels, kernel_sizes, stride, activation='relu6', dw_parallel=True):
        super(MSDC, self).__init__()

        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes
        self.activation = activation
        self.dw_parallel = dw_parallel

        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_channels, self.in_channels, kernel_size, stride, kernel_size // 2, groups=self.in_channels, bias=False),
                nn.BatchNorm2d(self.in_channels),
                act_layer(self.activation, inplace=True)
            )
            for kernel_size in self.kernel_sizes
        ])

        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        # Apply the convolution layers in a loop
        outputs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            outputs.append(dw_out)
            if self.dw_parallel == False:
                x = x+dw_out
        # You can return outputs based on what you intend to do with them
        return outputs
def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups    
    # reshape
    x = x.view(batchsize, groups, 
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

class MSCB(nn.Module):
    """
    Multi-scale convolution block (MSCB) 
    """
    def __init__(self, in_channels, out_channels, stride, kernel_sizes=[1,3,5], expansion_factor=2, dw_parallel=True, add=True, activation='relu6'):
        super(MSCB, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_sizes = kernel_sizes
        self.expansion_factor = expansion_factor
        self.dw_parallel = dw_parallel
        self.add = add
        self.activation = activation
        self.n_scales = len(self.kernel_sizes)
        # check stride value
        assert self.stride in [1, 2]
        # Skip connection if stride is 1
        self.use_skip_connection = True if self.stride == 1 else False

        # expansion factor
        self.ex_channels = int(self.in_channels * self.expansion_factor)
        self.pconv1 = nn.Sequential(
            # pointwise convolution
            nn.Conv2d(self.in_channels, self.ex_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ex_channels),
            act_layer(self.activation, inplace=True)
        )
        self.msdc = MSDC(self.ex_channels, self.kernel_sizes, self.stride, self.activation, dw_parallel=self.dw_parallel)
        if self.add == True:
            self.combined_channels = self.ex_channels*1
        else:
            self.combined_channels = self.ex_channels*self.n_scales
        self.pconv2 = nn.Sequential(
            # pointwise convolution
            nn.Conv2d(self.combined_channels, self.out_channels, 1, 1, 0, bias=False), 
            nn.BatchNorm2d(self.out_channels),
        )
        if self.use_skip_connection and (self.in_channels != self.out_channels):
            self.conv1x1 = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0, bias=False)
        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        pout1 = self.pconv1(x)
        msdc_outs = self.msdc(pout1)
        if self.add == True:
            dout = 0
            for dwout in msdc_outs:
                dout = dout + dwout
        else:
            dout = torch.cat(msdc_outs, dim=1)
        dout = channel_shuffle(dout, gcd(self.combined_channels,self.out_channels))
        out = self.pconv2(dout)
        if self.use_skip_connection:
            if self.in_channels != self.out_channels:
                x = self.conv1x1(x)
            return x + out
        else:
            return out

        

class EPA(nn.Module):
    def __init__(self, input_size, hidden_size, proj_size, mscb_kernel_sizes=[1, 3, 5],
                 num_heads=4, qkv_bias=False, channel_attn_drop=0.1, spatial_attn_drop=0.1):
        super().__init__()
        self.epa = LightCBAM(input_size, hidden_size, proj_size, num_heads,
                                 qkv_bias, channel_attn_drop, spatial_attn_drop)

        self.hidden_size = hidden_size
        self.reshape_required = True  # controls whether we go to [B, C, H, W]

        self.mscb = MSCB(
            in_channels=hidden_size,
            out_channels=hidden_size,
            stride=1,
            kernel_sizes=mscb_kernel_sizes,
            expansion_factor=2,
            dw_parallel=True,
            add=True,
            activation='relu6'
        )

    def forward(self, x):
        # x: [B, N, C]
        x = self.epa(x)  # still [B, N, C]
        

        B, N, C = x.shape
        H = int(math.sqrt(N))
        W = math.ceil(N / H)
        pad_len = H * W - N

        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))  # pad sequence dim

        x_reshaped = x.permute(0, 2, 1).reshape(B, C, H, W)  # [B, C, H, W]
        x_mscb = self.mscb(x_reshaped)  # [B, C, H, W]

        x_out = x_mscb.view(B, C, H * W).permute(0, 2, 1)  # [B, N_pad, C]
        if pad_len > 0:
            x_out = x_out[:, :-pad_len, :]

       
        return x_out
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature', 'temperature2'}

# import math
# from typing import Iterable
# import torch
# from torch import nn, Tensor
# import torch.nn.functional as F

# # --- multi-kernel depthwise conv over a sequence [B, N, C] ---
# class _DepthwiseMK1d(nn.Module):
#     def __init__(self, channels: int, kernel_sizes: Iterable[int] = (3, 5, 7),
#                  combine: str = "concat", bias: bool = True, device=None, dtype=None):
#         super().__init__()
#         assert combine in ("sum", "concat")
#         self.combine = combine
#         ks = [max(1, int(k)) for k in kernel_sizes]
#         self.convs = nn.ModuleList([
#             nn.Conv1d(channels, channels, k, padding=k // 2, groups=channels,
#                       bias=bias, device=device, dtype=dtype)
#             for k in ks
#         ])
#         self.proj = (nn.Conv1d(channels * len(ks), channels, 1, bias=bias,
#                                device=device, dtype=dtype)
#                      if combine == "concat" else nn.Identity())

#     def forward(self, x_bnC: Tensor) -> Tensor:
#         x = x_bnC.transpose(1, 2)                                 # [B,C,N]
#         feats = [conv(x) for conv in self.convs]
#         y = torch.cat(feats, dim=1) if self.combine == "concat" else torch.stack(feats, 0).sum(0)
#         y = self.proj(y)                                          # [B,C,N]
#         return y.transpose(1, 2)                                  # [B,N,C]

# # --- SimAM + MK Bi-Mamba spatial attention over images [B,C,H,W] ---
# class simam_spatial_mk_mamba(nn.Module):
#     def __init__(
#         self, c: int,
#         *, e_lambda: float = 1e-4, d_state: int = 16,
#         bidirectional: bool = True,
#         angle_mode: str = "learned", angle_init_deg: float = 0.0,
#         angle_cycle_degrees: Iterable[float] = (45.0, 90.0, 180.0),
#         kernel_sizes: Iterable[int] = (3, 5, 7), kernel_combine: str = "concat",
#         kernel_bias: bool = True, use_layernorm: bool = True, norm_epsilon: float = 1e-5,
#         device=None, dtype=None,
#     ):
#         super().__init__()
#         self.C = int(c)
#         self.e_lambda = e_lambda
#         self.bidirectional = bidirectional
#         fk = dict(device=device, dtype=dtype)
#         self.norm = nn.LayerNorm(self.C, eps=norm_epsilon, **fk) if use_layernorm else nn.Identity()
#         self.mk = _DepthwiseMK1d(self.C, kernel_sizes, kernel_combine, kernel_bias, **fk)
#         # Mamba on [B,N,C]
#         self.ssm_f = _instantiate_mamba_compat(self.C, d_state=d_state, layer_idx=None, **fk)
#         self.ssm_b = _instantiate_mamba_compat(self.C, d_state=d_state, layer_idx=None, **fk) if bidirectional else nn.Identity()
#         # angle (scalar)
#         self.angle_mode = angle_mode.lower()
#         init_rad = math.radians(angle_init_deg)
#         if self.angle_mode == "learned":
#             self.mix_angle_s = nn.Parameter(torch.tensor(init_rad, **{k: v for k, v in fk.items() if v is not None}))
#         else:
#             self.register_buffer("mix_angle_s", torch.tensor(init_rad), persistent=False)
#         self.angle_cycle_degrees = list(angle_cycle_degrees)
#         self._angle_idx = 0

#     def _simam_mask(self, x: Tensor) -> Tensor:
#         # x: [B,C,H,W] -> [B,1,H,W]
#         b, c, h, w = x.size()
#         n = c - 1 if c > 1 else 1
#         x_mean = x.mean(dim=1, keepdim=True)
#         x_msq  = (x - x_mean).pow(2)
#         var = x_msq.sum(dim=1, keepdim=True) / n + self.e_lambda
#         y = x_msq.sum(dim=1, keepdim=True) / (4 * var) + 0.5
#         return torch.sigmoid(y)

#     def _angle_for_forward(self, like: Tensor) -> Tensor:
#         if not self.bidirectional:
#             return torch.zeros((), device=like.device, dtype=like.dtype)
#         if self.angle_mode == "learned":
#             return self.mix_angle_s.to(device=like.device, dtype=like.dtype)
#         if self.angle_mode == "fixed":
#             return (self.mix_angle_s if isinstance(self.mix_angle_s, torch.Tensor)
#                     else torch.tensor(self.mix_angle_s, device=like.device, dtype=like.dtype))
#         deg = self.angle_cycle_degrees[self._angle_idx % max(len(self.angle_cycle_degrees), 1)]
#         s = torch.tensor(math.radians(deg), device=like.device, dtype=like.dtype)
#         self._angle_idx = (self._angle_idx + 1) % max(len(self.angle_cycle_degrees), 1)
#         return s

#     def forward(self, x: Tensor) -> Tensor:
#         B, C, H, W = x.shape
#         N = H * W
#         simam_mask = self._simam_mask(x)                          # [B,1,H,W]
#         seq = x.permute(0, 2, 3, 1).reshape(B, N, C)              # [B,N,C]
#         seq = self.norm(seq)
#         seq = self.mk(seq)
#         y_f = self.ssm_f(seq)
#         y = y_f
#         if self.bidirectional:
#             y_b = self.ssm_b(seq.flip(1)).flip(1)
#             s = self._angle_for_forward(seq)
#             y = torch.cos(s) * y_f + torch.sin(s) * y_b
#         y_map = y.view(B, H, W, C).permute(0, 3, 1, 2)            # [B,C,H,W]
#         mamba_mask = torch.sigmoid(y_map)
#         return x * (simam_mask * mamba_mask)                      # modulated features

# # === Fully modified LightCBAM (same signature) ===============================
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math

# # SimAM for channel attention
# class simam_module(nn.Module):
#     def __init__(self, channels=None, e_lambda=1e-4):
#         super(simam_module, self).__init__()
#         self.activation = nn.Sigmoid()
#         self.e_lambda = e_lambda

#     def forward(self, x):
#         b, c, h, w = x.size()
#         n = w * h - 1
#         x_mean = x.mean(dim=[2, 3], keepdim=True)
#         x_minus_mu_square = (x - x_mean).pow(2)
#         var = x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda
#         y = x_minus_mu_square / (4 * var) + 0.5
#         return x * self.activation(y)

# # SimAM-style spatial attention (over channel dimension)
# class simam_spatial(nn.Module):
#     def __init__(self, e_lambda=1e-4):
#         super(simam_spatial, self).__init__()
#         self.activation = nn.Sigmoid()
#         self.e_lambda = e_lambda

#     def forward(self, x):
#         # x: [B, C, H, W]
#         b, c, h, w = x.size()
#         n = c - 1
#         x_mean = x.mean(dim=1, keepdim=True)
#         x_minus_mu_square = (x - x_mean).pow(2)
#         var = x_minus_mu_square.sum(dim=1, keepdim=True) / n + self.e_lambda
#         y = x_minus_mu_square.sum(dim=1, keepdim=True) / (4 * var) + 0.5
#         return x * self.activation(y)
# import torch
# import torch.nn as nn
# from torch import Tensor

# # pip install mamba-ssm  (or the repo you're using)
# from mamba_ssm import Mamba


# class SimAMGate(nn.Module):
#     """
#     SimAM-style spatial attention (channel-wise statistics -> 1 map per location).
#     Returns a gate in [0,1] shaped [B,1,H,W] (no multiplication done here).
#     """
#     def __init__(self, e_lambda: float = 1e-4):
#         super().__init__()
#         self.sigma = nn.Sigmoid()
#         self.e_lambda = e_lambda

#     def forward(self, x: Tensor) -> Tensor:
#         # x: [B, C, H, W]
#         b, c, h, w = x.size()
#         n = max(c - 1, 1)  # avoid /0 when C=1
#         mu = x.mean(dim=1, keepdim=True)                          # [B,1,H,W]
#         xm2 = (x - mu).pow(2)                                     # [B,C,H,W]
#         var = xm2.sum(dim=1, keepdim=True) / n + self.e_lambda    # [B,1,H,W]
#         y = xm2.sum(dim=1, keepdim=True) / (4 * var) + 0.5        # [B,1,H,W]
#         gate = self.sigma(y)                                      # [B,1,H,W]
#         return gate


# class SimAMVisionMamba(nn.Module):
#     """
#     SimAM -> (bi)Mamba over spatial tokens -> proj -> residual.
#     Intended as a lightweight Vision Mamba encoder block.

#     Args:
#         channels: feature dimension C
#         bidirectional: if True, run Mamba on forward + reversed sequences and average
#         dropout: dropout after output projection
#         use_ln: apply LayerNorm on token features before Mamba (recommended)
#     """
#     def __init__(
#         self,
#         channels: int,
#         bidirectional: bool = True,
#         dropout: float = 0.0,
#         use_ln: bool = True,
#         e_lambda: float = 1e-4,
#     ):
#         super().__init__()
#         self.gate = SimAMGate(e_lambda=e_lambda)
#         self.use_ln = use_ln
#         self.bidirectional = bidirectional

#         if use_ln:
#             self.ln = nn.LayerNorm(channels)

#         # 1×1 pre/post projections are optional; keep identity by default
#         self.in_proj  = nn.Identity()
#         self.out_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

#         # Core SSM
#         self.mamba = Mamba(d_model=channels)  # expects [B, L, C]
#         self.dropout = nn.Dropout(dropout)

#     def _run_mamba(self, tokens: Tensor) -> Tensor:
#         """
#         tokens: [B, L, C]
#         """
#         y_f = self.mamba(tokens)  # forward
#         if not self.bidirectional:
#             return y_f
#         # backward pass: flip L, run, flip back
#         y_b = torch.flip(self.mamba(torch.flip(tokens, dims=[1])), dims=[1])
#         return 0.5 * (y_f + y_b)

#     def forward(self, x: Tensor) -> Tensor:
#         """
#         x: [B, C, H, W]
#         """
#         b, c, h, w = x.shape
#         residual = x

        # # SimAM spatial gate -> modulate features
        # g = self.gate(x)                 # [B,1,H,W]
        # xg = x * g                       # [B,C,H,W]

        # # (Optional) in-proj, then flatten to tokens
        # xg = self.in_proj(xg)
        # tokens = xg.flatten(2).transpose(1, 2)   # [B, L=H*W, C]

        # if self.use_ln:
        #     tokens = self.ln(tokens)

        # # Vision Mamba over spatial sequence
        # y = self._run_mamba(tokens)              # [B, L, C]

        # # Back to image layout
        # y = y.transpose(1, 2).reshape(b, c, h, w)
        # y = self.out_proj(y)
        # y = self.dropout(y)

        # # Residual connection (pre-norm style)
        # return y
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch import Tensor
# from mamba_ssm import Mamba

# class SimAMVisionMamba(nn.Module):
#     """
#     Same signature as your original class, but uses bidirectional multi-axis scans:
#       - bi-scan along W for each H-row
#       - bi-scan along H for each W-column
#     Then mixes the two with a learned weight and projects back.

#     Input:  [B, C, H, W]
#     Output: [B, C, H, W]
#     """
#     def __init__(
#         self,
#         channels: int,
#         bidirectional: bool = True,
#         dropout: float = 0.0,
#         use_ln: bool = True,
#         e_lambda: float = 1e-4,
#     ):
#         super().__init__()
#         self.gate = SimAMGate(e_lambda=e_lambda)
#         self.use_ln = use_ln
#         self.bidirectional = bidirectional

#         self.ln = nn.LayerNorm(channels) if use_ln else nn.Identity()

#         # keep the same style as you had
#         self.in_proj  = nn.Identity()
#         self.out_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
#         self.dropout = nn.Dropout(dropout)

#         # one shared Mamba (cheaper + regularizes). still same signature.
#         self.mamba = Mamba(d_model=channels)  # expects [B, L, C]

#         # learned mix between W-scan and H-scan (starts equal)
#         self.mix = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))

#         # optional LayerScale for stability (near-free)
#         self.gamma = nn.Parameter(torch.zeros(1))

#     def _run_seq(self, tokens: Tensor) -> Tensor:
#         """
#         tokens: [B, L, C]
#         """
#         y_f = self.mamba(tokens)
#         if not self.bidirectional:
#             return y_f
#         y_b = torch.flip(self.mamba(torch.flip(tokens, dims=[1])), dims=[1])
#         return 0.5 * (y_f + y_b)

#     def _scan_w(self, x: Tensor) -> Tensor:
#         """
#         Scan along W for each row.
#         x: [B,C,H,W] -> seq: [B*H, W, C] -> back to [B,C,H,W]
#         """
#         B, C, H, W = x.shape
#         seq = x.permute(0, 2, 3, 1).contiguous().view(B * H, W, C)  # [B*H, W, C]
#         seq = self.ln(seq) if self.use_ln else seq
#         y = self._run_seq(seq)                                      # [B*H, W, C]
#         y = y.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()     # [B,C,H,W]
#         return y

#     def _scan_h(self, x: Tensor) -> Tensor:
#         """
#         Scan along H for each column.
#         x: [B,C,H,W] -> seq: [B*W, H, C] -> back to [B,C,H,W]
#         """
#         B, C, H, W = x.shape
#         seq = x.permute(0, 3, 2, 1).contiguous().view(B * W, H, C)  # [B*W, H, C]
#         seq = self.ln(seq) if self.use_ln else seq
#         y = self._run_seq(seq)                                      # [B*W, H, C]
#         y = y.view(B, W, H, C).permute(0, 3, 2, 1).contiguous()     # [B,C,H,W]
#         return y

#     def forward(self, x: Tensor) -> Tensor:
#         """
#         x: [B, C, H, W]
#         """
#         residual = x

#         # SimAM gate
#         g = self.gate(x)           # [B,1,H,W]
#         xg = self.in_proj(x * g)   # [B,C,H,W]

#         # two scan orders
#         yw = self._scan_w(xg)
#         yh = self._scan_h(xg)

#         # mix
#         w = torch.softmax(self.mix, dim=0)
#         y = w[0] * yw + w[1] * yh

#         # proj + dropout
#         y = self.out_proj(y)
#         y = self.dropout(y)

#         # Residual (recommended). If you truly want no residual, return y.
#         return residual + self.gamma * y

# Fully modified LightCBAM
class LightCBAM(nn.Module):
    """
    Lightweight CBAM with SimAM for both channel and spatial attention.
    Input: [B, N, C] -> Output: [B, N, C]
    """
    def __init__(self, input_size, hidden_size, proj_size,
                 num_heads=4, qkv_bias=False,
                 channel_attn_drop=0.1, spatial_attn_drop=0.1):
        super().__init__()
        self.hidden_size = hidden_size

        # SimAM replaces channel attention
        self.simam_channel = simam_module()
        self.ca_drop = nn.Dropout(channel_attn_drop)

        # SimAM-style spatial attention
        self.simam_spatial = simam_spatial()
        self.sa_drop = nn.Dropout(spatial_attn_drop)
        self.block = SimAMVisionMamba(channels=hidden_size, bidirectional=True, dropout=0.1)

    def forward(self, x):
        B, N, C = x.shape

        # Reshape [B, N, C] -> [B, C, H, W]
        H = int(math.sqrt(N))
        W = math.ceil(N / H)
        pad_len = H * W - N
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))  # pad N dimension

        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        
        # Channel Attention via SimAM
        x = self.ca_drop(self.simam_channel(x))

        # Spatial Attention via SimAM
        x2 = self.block(x)

        x = self.sa_drop(self.simam_spatial(x2))#+2

        # Reshape back to [B, N, C]
        x = x.view(B, C, H * W).permute(0, 2, 1)
        if pad_len > 0:
            x = x[:, :-pad_len, :]

        return x
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba  # pip install mamba-ssm


# ----------------------------
# SimAM blocks (kept lightweight)
# ----------------------------
class simam_module(nn.Module):
    """
    SimAM: parameter-free attention. Works on [B,C,H,W].
    """
    def __init__(self, e_lambda: float = 1e-4):
        super().__init__()
        self.e_lambda = e_lambda
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B,C,H,W]
        mu = x.mean(dim=(2, 3), keepdim=True)
        x2 = (x - mu) ** 2
        denom = x2.sum(dim=(2, 3), keepdim=True) / (x.shape[2] * x.shape[3] - 1 + 1e-6) + self.e_lambda
        attn = x2 / (4 * denom) + 0.5
        return x * self.sigmoid(attn)


class simam_spatial(nn.Module):
    """
    SimAM-like spatial emphasis: compute SimAM across channels by treating C as "spatial" axis.
    Lightweight and parameter-free.
    """
    def __init__(self, e_lambda: float = 1e-4):
        super().__init__()
        self.e_lambda = e_lambda
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B,C,H,W]
        mu = x.mean(dim=1, keepdim=True)              # [B,1,H,W]
        x2 = (x - mu) ** 2
        denom = x2.mean(dim=1, keepdim=True) + self.e_lambda
        attn = x2 / (4 * denom) + 0.5
        return x * self.sigmoid(attn)


# ----------------------------
# Mamba vision mixer for [B,C,H,W] (no window partition)
# ----------------------------
class SimAMVisionMamba(nn.Module):
    """
    Takes [B,C,H,W] -> returns [B,C,H,W]
    Flatten tokens HW and run Mamba over sequence.
    Includes optional bidirectional pass.
    """
    def __init__(
        self,
        channels: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.channels = int(channels)
        self.bidirectional = bool(bidirectional)

        self.ln = nn.LayerNorm(self.channels)
        self.drop = nn.Dropout(dropout)

        self.m_f = Mamba(d_model=self.channels, d_state=d_state, d_conv=d_conv, expand=expand)
        self.m_b = Mamba(d_model=self.channels, d_state=d_state, d_conv=d_conv, expand=expand) if self.bidirectional else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,H,W] -> tokens: [B, L, C]
        B, C, H, W = x.shape
        tok = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)  # [B,L,C]
        t = self.ln(tok)

        yf = self.m_f(t)
        if self.bidirectional:
            yb = torch.flip(self.m_b(torch.flip(t, dims=[1])), dims=[1])
            y = 0.5 * (yf + yb)
        else:
            y = yf

        y = self.drop(y)
        tok = tok + y  # residual

        out = tok.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return out


# ============================================================
# LightCBAM (signature preserved) + Mamba inside (as requested)
# Input: [B,N,C] -> Output: [B,N,C]
# ============================================================
class LightCBAM(nn.Module):
    """
    Lightweight CBAM-style block using:
      - SimAM for "channel attention"
      - Mamba mixer in the middle (token mixing over HW)
      - SimAM-style spatial attention

    Preserves signature: [B, N, C] -> [B, N, C]
    """
    def __init__(
        self,
        input_size,
        hidden_size,
        proj_size,
        num_heads=4,
        qkv_bias=False,
        channel_attn_drop=0.1,
        spatial_attn_drop=0.1,
        # Mamba knobs (safe defaults)
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_bidirectional: bool = True,
        mamba_dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)

        # SimAM replaces channel attention
        self.simam_channel = simam_module()
        self.ca_drop = nn.Dropout(channel_attn_drop)

        # # Mamba block (token mixer)
        # self.block = SimAMVisionMamba(
        #     channels=self.hidden_size,
        #     d_state=mamba_d_state,
        #     d_conv=mamba_d_conv,
        #     expand=mamba_expand,
        #     bidirectional=mamba_bidirectional,
        #     dropout=mamba_dropout,
        # )

        # SimAM-style spatial attention
        self.simam_spatial = simam_spatial()
        self.sa_drop = nn.Dropout(spatial_attn_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, C]
        """
        B, N, C = x.shape
        if C != self.hidden_size:
            raise ValueError(f"Expected C==hidden_size ({self.hidden_size}), got {C}")

        # Infer H,W from N (pad to rectangle)
        H = int(math.sqrt(N))
        W = int(math.ceil(N / H))
        pad_len = H * W - N
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))  # pad N dimension at end

        # [B,N,C] -> [B,C,H,W]
        x = x.permute(0, 2, 1).contiguous().view(B, C, H, W)

        # Channel attention (SimAM)
        x = self.ca_drop(self.simam_channel(x))

        # Mamba token mixing (no window partition)
        #x = self.block(x)

        # Spatial attention (SimAM-like)
        x = self.sa_drop(self.simam_spatial(x))

        # Back to [B,N,C]
        x = x.view(B, C, H * W).permute(0, 2, 1).contiguous()
        if pad_len > 0:
            x = x[:, :-pad_len, :]
        return x
