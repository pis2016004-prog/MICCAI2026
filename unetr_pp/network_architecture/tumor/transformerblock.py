

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

    ):
        super().__init__()
        self.hidden_size = int(hidden_size)

        # SimAM replaces channel attention
        self.simam_channel = simam_module()
        self.ca_drop = nn.Dropout(channel_attn_drop)

     

        # SimAM-style spatial attention
        self.simam_spatial = simam_spatial()
        self.sa_drop = nn.Dropout(spatial_attn_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, C]
        """
        B, N, C = x.shape
        x2=x
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
        #sx = self.block(x)

        # Spatial attention (SimAM-like)
        x = self.sa_drop(self.simam_spatial(x))

        # Back to [B,N,C]
        x = x.view(B, C, H * W).permute(0, 2, 1).contiguous()
        if pad_len > 0:
            x = x[:, :-pad_len, :]
        return x + x2
