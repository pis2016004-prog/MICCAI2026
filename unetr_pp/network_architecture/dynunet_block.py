from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.utils import get_act_layer, get_norm_layer
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveAvgMaxPool3d(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool3d(output_size)
        self.maxpool = nn.AdaptiveMaxPool3d(output_size)
    def forward(self, x):
        return torch.cat([self.avgpool(x), self.maxpool(x)], dim=1)


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba  # pip install mamba-ssm

class Conv2p5D(nn.Module):
    """
    Conv2.5D using ONLY Mamba (no Conv3d, no pointwise/depthwise).
    - Bidirectional Mamba over time (T)
    - Multi-angle bidirectional Mamba over the spatial plane (H,W) via rotations
    Signature preserved:
        Conv2p5D(in_channels, out_channels, kernel_size=3, padding=None)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=None,
                 angles_deg=(0.0, 45.0, 90.0, 135.0),
                 d_state_time=16, d_state_space=8, d_conv=4, expand=2, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.angles_deg = angles_deg

        # channel projections (Linear instead of 1x1 convs)
        self.proj_in  = nn.Identity() if in_channels == out_channels else nn.Linear(in_channels, out_channels, bias=False)
        self.proj_res = nn.Identity() if in_channels == out_channels else nn.Linear(in_channels, out_channels, bias=False)
        self.proj_out = nn.Linear(out_channels, out_channels, bias=False)

        # layer norms for sequences
        self.ln_t   = nn.LayerNorm(out_channels)
        self.ln_w   = nn.LayerNorm(out_channels)
        self.ln_out = nn.LayerNorm(out_channels)

        # gate (per-voxel) with Linear -> SiLU -> Sigmoid (no convs)
        self.gate_in = nn.Linear(out_channels, out_channels, bias=True)
        self.gate_act = nn.SiLU(inplace=True)

        # Mamba streams
        self.mamba_tf = Mamba(d_model=out_channels, d_state=d_state_time,  d_conv=d_conv, expand=expand)
        self.mamba_tb = Mamba(d_model=out_channels, d_state=d_state_time,  d_conv=d_conv, expand=expand)

        self.mamba_xf = nn.ModuleList([Mamba(d_model=out_channels, d_state=d_state_space, d_conv=d_conv, expand=expand)
                                       for _ in self.angles_deg])
        self.mamba_xb = nn.ModuleList([Mamba(d_model=out_channels, d_state=d_state_space, d_conv=d_conv, expand=expand)
                                       for _ in self.angles_deg])

        # learned mixture across angles
        self.angle_logits = nn.Parameter(torch.zeros(len(self.angles_deg)))
        self.drop = nn.Dropout(dropout)

    # ---------- helpers ----------
    @staticmethod
    def _to_time_seq(x):  # (B,C,T,H,W) -> (B*H*W, T, C)
        B, C, T, H, W = x.shape
        y = x.permute(0, 3, 4, 2, 1).contiguous().view(B*H*W, T, C)
        return y, (B, C, T, H, W)

    @staticmethod
    def _from_time_seq(y, shp):  # (B*H*W,T,C) -> (B,C,T,H,W)
        B, C, T, H, W = shp
        return y.view(B, H, W, T, C).permute(0, 4, 3, 1, 2).contiguous()

    @staticmethod
    def _to_width_seq(x):  # (B,C,T,H,W) -> (B*T*H, W, C)
        B, C, T, H, W = x.shape
        y = x.permute(0, 2, 3, 4, 1).contiguous().view(B*T*H, W, C)
        return y, (B, C, T, H, W)

    @staticmethod
    def _from_width_seq(y, shp):  # (B*T*H,W,C) -> (B,C,T,H,W)
        B, C, T, H, W = shp
        return y.view(B, T, H, W, C).permute(0, 4, 1, 2, 3).contiguous()

    @staticmethod
    def _apply_linear_channels(x, linear):  # apply nn.Linear across channel dim at every voxel
        if isinstance(linear, nn.Identity):
            return x
        B, C, T, H, W = x.shape
        y = x.permute(0, 2, 3, 4, 1).contiguous().view(B*T*H*W, C)  # (..., C)
        y = linear(y)
        C2 = y.shape[-1]
        y = y.view(B, T, H, W, C2).permute(0, 4, 1, 2, 3).contiguous()
        return y

    @staticmethod
    def _rotate_hw(x, deg):
        """Rotate (H,W) for every time step; x: (B,C,T,H,W)"""
        if abs(deg) % 360 < 1e-4:
            return x
        B, C, T, H, W = x.shape
        xt = x.permute(0, 2, 1, 3, 4).contiguous().view(B*T, C, H, W)  # (BT,C,H,W)
        theta = math.radians(deg)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        A = x.new_tensor([[cos_t, -sin_t, 0.0],
                          [sin_t,  cos_t, 0.0]])  # (2,3)
        A = A.unsqueeze(0).repeat(B*T, 1, 1)      # (BT,2,3)
        grid = F.affine_grid(A, size=xt.size(), align_corners=False)
        xr = F.grid_sample(xt, grid, mode='bilinear',
                           padding_mode='zeros', align_corners=False)
        return xr.view(B, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()

    # ---------- mamba runners ----------
    def _mamba_over_time(self, x, mod):
        s, shp = self._to_time_seq(x)      # (BHW,T,C)
        s = self.ln_t(s)
        y = mod(s)                         # (BHW,T,C)
        return self._from_time_seq(y, shp)

    def _mamba_over_width(self, x, mod):
        s, shp = self._to_width_seq(x)     # (BTH,W,C)
        s = self.ln_w(s)
        y = mod(s)                         # (BTH,W,C)
        return self._from_width_seq(y, shp)

    # ---------- forward ----------
    def forward(self, x):  # x: (B,C,T,H,W)
        # residual path
        res = self._apply_linear_channels(x, self.proj_res)

        # input projection
        y = self._apply_linear_channels(x, self.proj_in)

        # gate z in [0,1] per voxel (no convs)
        z = self._apply_linear_channels(y, self.gate_in)
        z = self.gate_act(z)
        z = torch.sigmoid(z)

        # ===== Temporal streams (bidirectional) =====
        yt_f = self._mamba_over_time(y, self.mamba_tf)
        yt_b = torch.flip(self._mamba_over_time(torch.flip(y, dims=[2]), self.mamba_tb), dims=[2])
        t_out = 0.5 * (yt_f + yt_b)

        # ===== Multi-angle spatial streams over width (bidirectional) =====
        angle_w = torch.softmax(self.angle_logits, dim=0)
        s_out = 0.0
        for i, deg in enumerate(self.angles_deg):
            yr = self._rotate_hw(y,  deg)
            yr_f = self._mamba_over_width(yr, self.mamba_xf[i])
            yr_b = torch.flip(self._mamba_over_width(torch.flip(yr, dims=[4]), self.mamba_xb[i]), dims=[4])
            yr   = 0.5 * (yr_f + yr_b)
            yr   = self._rotate_hw(yr, -deg)  # undo rotation
            s_out = s_out + angle_w[i] * yr

        # gated fusion
        out = (t_out * z) + (s_out * z)

        # output norm + linear + dropout + residual
        out = out.permute(0, 2, 3, 4, 1)     # (B,T,H,W,C)
        out = self.ln_out(out)
        out = self.proj_out(out)
        out = self.drop(out)
        out = out.permute(0, 4, 1, 2, 3).contiguous()  # (B,C,T,H,W)

        return F.relu(out + res, inplace=True)

# Your Inception3D keeps the same signature and usage
class Inception3D(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4, stride=1):
        super().__init__()
        self.p1_1 = Conv2p5D(in_channels, c1, kernel_size=1)
        self.p2_1 = Conv2p5D(in_channels, c2[0], kernel_size=1)
        self.p2_2 = Conv2p5D(c2[0], c2[1], kernel_size=3)
        self.p3_1 = Conv2p5D(in_channels, c3[0], kernel_size=1)
        self.p3_2 = Conv2p5D(c3[0], c3[1], kernel_size=5, padding=2)
        self.p4_1 = nn.MaxPool3d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = Conv2p5D(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        return torch.cat([p1, p2, p3, p4], dim=1)
import torch
import torch.nn as nn
import torch.nn.functional as F

# assumes your Mamba-only Conv2p5D is already defined/imported

# class MambaFusion3D(nn.Module):
#     """
#     Drop-in replacement for Inception3D without branches.
#     - Same signature: (in_channels, c1, c2, c3, c4, stride=1)
#     - Output channels = c1 + c2[1] + c3[1] + c4 (matches Inception3D concat)
#     - Uses a single Conv2p5D (Mamba-only) trunk, optional spatial downsample when stride>1.
#     """
#     def __init__(self, in_channels, c1, c2, c3, c4, stride=1, kernel_size=3, padding=None):
#         super().__init__()
#         # resolve target output channels from the original Inception args
#         def last(o):  # handle int or (in,out)
#             if isinstance(o, (tuple, list)):
#                 return int(o[-1])
#             return int(o)
#         out_channels = int(c1) + last(c2) + last(c3) + int(c4)

#         if padding is None:
#             padding = kernel_size // 2

#         self.stride = int(stride)
#         # single Mamba trunk to produce the same channel budget as Inception concat
#         self.trunk = Conv2p5D(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
#         # lightweight norm/activation (no convs)
#         self.norm = nn.LayerNorm(out_channels)
#         self.act = nn.ReLU(inplace=True)

#         # residual projection (linear via Conv2p5D’s internals already handle proj;
#         # here we simply remember in/out for identity check in forward)
#         self.in_channels = in_channels
#         self.out_channels = out_channels

#     def _downsample_if_needed(self, x):
#         if self.stride == 1:
#             return x
#         # spatial-only downsample; keep time T intact
#         return F.avg_pool3d(x, kernel_size=(1, self.stride, self.stride),
#                                stride=(1, self.stride, self.stride), padding=0)

#     def forward(self, x):
#         x_ds = self._downsample_if_needed(x)
#         y = self.trunk(x_ds)                       # (B, out_channels, T, H/stride, W/stride)
#         # LayerNorm expects channels last; permute, norm, permute back (no convs)
#         y = y.permute(0, 2, 3, 4, 1)
#         y = self.act(self.norm(y))
#         y = y.permute(0, 4, 1, 2, 3).contiguous()
#         return y

# class UnetResBlock(nn.Module):
#     """
#     Inception-style residual block with downsampling support.
#     """

#     def __init__(
#         self,
#         spatial_dims: int,
#         in_channels: int,
#         out_channels: int,
#         kernel_size: Union[Sequence[int], int],
#         stride: Union[Sequence[int], int],
#         norm_name: Union[Tuple, str],
#         act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
#         dropout: Optional[Union[Tuple, str, float]] = None,
#     ):
#         super().__init__()

#         assert spatial_dims == 3, "This Inception version only supports 3D."

#         self.downsample = in_channels != out_channels
#         stride_np = np.atleast_1d(stride)
#         if not np.all(stride_np == 1):
#             self.downsample = True

#         # Inception splits: total channels must sum to out_channels
#         c1 = out_channels // 4
#         c2 = (out_channels // 8, out_channels // 4)
#         c3 = (out_channels // 8, out_channels // 4)
#         c4 = out_channels // 4

#         self.inception = MambaFusion3D(in_channels, c1, c2, c3, c4, stride=1)
#         self.norm1 = get_norm_layer(name=norm_name, spatial_dims=3, channels=out_channels)
#         self.act = get_act_layer(name=act_name)

#         # Optional second Inception (as second conv) with stride=1
#         self.inception2 = get_conv_layer(spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=1, dropout=dropout, conv_only=True)
#         self.norm2 = get_norm_layer(name=norm_name, spatial_dims=3, channels=out_channels)

#         # Residual path
#         if self.downsample:
#             self.res_inception = MambaFusion3D(in_channels, c1, c2, c3, c4, stride=stride)
#             self.norm_res = get_norm_layer(name=norm_name, spatial_dims=3, channels=out_channels)

#     def forward(self, x):
#         residual = x

#         out = self.inception(x)
#         out = self.norm1(out)
#         out = self.act(out)

#         out = self.inception2(out)
#         out = self.norm2(out)

#         if self.downsample:
#             residual = self.res_inception(residual)
#             residual = self.norm_res(residual)

#         out += residual
#         out = self.act(out)
#         return out
class UnetBasicBlock(nn.Module):
    """
    A CNN module module that can be used for DynUNet, based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.

    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: Optional[Union[Tuple, str, float]] = None,
    ):
        super().__init__()
        self.conv1 = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dropout=dropout,
            conv_only=True,
        )
        self.conv2 = get_conv_layer(
            spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=1, dropout=dropout, conv_only=True
        )
        self.lrelu = get_act_layer(name=act_name)
        self.norm1 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.norm2 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)

    def forward(self, inp):
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.lrelu(out)
        return out


class UnetUpBlock(nn.Module):
    """
    An upsampling module that can be used for DynUNet, based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        upsample_kernel_size: convolution kernel size for transposed convolution layers.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.
        trans_bias: transposed convolution bias.

    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: Optional[Union[Tuple, str, float]] = None,
        trans_bias: bool = False,
    ):
        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            dropout=dropout,
            bias=trans_bias,
            conv_only=True,
            is_transposed=True,
        )
        self.conv_block = UnetBasicBlock(
            spatial_dims,
            out_channels + out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout=dropout,
            norm_name=norm_name,
            act_name=act_name,
        )

    def forward(self, inp, skip):
        # number of channels for skip should equals to out_channels
        out = self.transp_conv(inp)
        out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)
        return out


class UnetOutBlock(nn.Module):
    def __init__(
        self, spatial_dims: int, in_channels: int, out_channels: int, dropout: Optional[Union[Tuple, str, float]] = None
    ):
        super().__init__()
        self.conv = get_conv_layer(
            spatial_dims, in_channels, out_channels, kernel_size=1, stride=1, dropout=dropout, bias=True, conv_only=True
        )

    def forward(self, inp):
        return self.conv(inp)


def get_conv_layer(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    kernel_size: Union[Sequence[int], int] = 3,
    stride: Union[Sequence[int], int] = 1,
    act: Optional[Union[Tuple, str]] = Act.PRELU,
    norm: Union[Tuple, str] = Norm.INSTANCE,
    dropout: Optional[Union[Tuple, str, float]] = None,
    bias: bool = False,
    conv_only: bool = True,
    is_transposed: bool = False,
):
    padding = get_padding(kernel_size, stride)
    output_padding = None
    if is_transposed:
        output_padding = get_output_padding(kernel_size, stride, padding)
    return Convolution(
        spatial_dims,
        in_channels,
        out_channels,
        strides=stride,
        kernel_size=kernel_size,
        act=act,
        norm=norm,
        dropout=dropout,
        bias=bias,
        conv_only=conv_only,
        is_transposed=is_transposed,
        padding=padding,
        output_padding=output_padding,
    )


def get_padding(
    kernel_size: Union[Sequence[int], int], stride: Union[Sequence[int], int]
) -> Union[Tuple[int, ...], int]:

    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = (kernel_size_np - stride_np + 1) / 2
    if np.min(padding_np) < 0:
        raise AssertionError("padding value should not be negative, please change the kernel size and/or stride.")
    padding = tuple(int(p) for p in padding_np)

    return padding if len(padding) > 1 else padding[0]


def get_output_padding(
    kernel_size: Union[Sequence[int], int], stride: Union[Sequence[int], int], padding: Union[Sequence[int], int]
) -> Union[Tuple[int, ...], int]:
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = np.atleast_1d(padding)

    out_padding_np = 2 * padding_np + stride_np - kernel_size_np
    if np.min(out_padding_np) < 0:
        raise AssertionError("out_padding value should not be negative, please change the kernel size and/or stride.")
    out_padding = tuple(int(p) for p in out_padding_np)

    return out_padding if len(out_padding) > 1 else out_padding[0]
class UnetResBlock22(nn.Module):
    """
    A skip-connection based module that can be used for DynUNet, based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.

    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: Optional[Union[Tuple, str, float]] = None,
    ):
        super().__init__()
        self.conv1 = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dropout=dropout,
            conv_only=True,
        )
        self.conv2 = get_conv_layer(
            spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=1, dropout=dropout, conv_only=True
        )
        self.lrelu = get_act_layer(name=act_name)
        self.norm1 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.norm2 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.downsample = in_channels != out_channels
        stride_np = np.atleast_1d(stride)
        if not np.all(stride_np == 1):
            self.downsample = True
        if self.downsample:
            self.conv3 = get_conv_layer(
                spatial_dims, in_channels, out_channels, kernel_size=1, stride=stride, dropout=dropout, conv_only=True
            )
            self.norm3 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)

    def forward(self, inp):
        residual = inp
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if hasattr(self, "conv3"):
            residual = self.conv3(residual)
        if hasattr(self, "norm3"):
            residual = self.norm3(residual)
        out += residual
        out = self.lrelu(out)
        return out
class UnetBasicBlock(nn.Module):
    """
    A CNN module module that can be used for DynUNet, based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.

    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: Optional[Union[Tuple, str, float]] = None,
    ):
        super().__init__()
        self.conv1 = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dropout=dropout,
            conv_only=True,
        )
        self.conv2 = get_conv_layer(
            spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=1, dropout=dropout, conv_only=True
        )
        self.lrelu = get_act_layer(name=act_name)
        self.norm1 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.norm2 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)

    def forward(self, inp):
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.lrelu(out)
        return out


class UnetUpBlock2(nn.Module):
    """
    An upsampling module that can be used for DynUNet, based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        upsample_kernel_size: convolution kernel size for transposed convolution layers.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.
        trans_bias: transposed convolution bias.

    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: Optional[Union[Tuple, str, float]] = None,
        trans_bias: bool = False,
    ):
        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            dropout=dropout,
            bias=trans_bias,
            conv_only=True,
            is_transposed=True,
        )
        self.conv_block = UnetBasicBlock(
            spatial_dims,
            out_channels + out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout=dropout,
            norm_name=norm_name,
            act_name=act_name,
        )

    def forward(self, inp, skip):
        # number of channels for skip should equals to out_channels
        out = self.transp_conv(inp)
        out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)
        return out
from typing import Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- helper: DW-separable conv3d/conv2d ----
class DWSeparableConvNd(nn.Module):
    def __init__(self, spatial_dims: int, in_ch: int, out_ch: int, k: int, stride=1, dropout: Optional[float]=None):
        super().__init__()
        pad = k // 2
        if spatial_dims == 3:
            self.dw = nn.Conv3d(in_ch, in_ch, k, stride=stride, padding=pad, groups=in_ch, bias=False)
            self.pw = nn.Conv3d(in_ch, out_ch, 1, bias=False)
            self.do = nn.Dropout3d(dropout) if (isinstance(dropout, float) and dropout > 0) else nn.Identity()
        else:
            self.dw = nn.Conv2d(in_ch, in_ch, k, stride=stride, padding=pad, groups=in_ch, bias=False)
            self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
            self.do = nn.Dropout2d(dropout) if (isinstance(dropout, float) and dropout > 0) else nn.Identity()

    def forward(self, x):
        return self.pw(self.do(self.dw(x)))

# ---- cheap bidirectional mamba mixer for 3D feature maps ----
class CheapBiMamba3D(nn.Module):
    """
    Cheap mixer: project C->Cr, run Mamba over flattened spatial tokens per-slice (D),
    then project back. Uses aggressive token downsample to keep cost low.
    """
    def __init__(self, channels: int, reduction: int = 12, token_stride: int = 4,
                 d_state: int = 16, d_conv: int = 4, expand: int = 2, bidirectional: bool = True):
        super().__init__()
        from mamba_ssm import Mamba

        self.channels = int(channels)
        self.reduction = max(1, int(reduction))
        self.token_stride = max(1, int(token_stride))
        self.bidirectional = bool(bidirectional)

        cr = max(8, self.channels // self.reduction)
        self.cr = cr

        self.in_proj = nn.Conv3d(self.channels, cr, 1, bias=False)
        self.out_proj = nn.Conv3d(cr, self.channels, 1, bias=False)
        self.ln = nn.LayerNorm(cr)

        self.m_f = Mamba(d_model=cr, d_state=d_state, d_conv=d_conv, expand=expand)
        self.m_b = Mamba(d_model=cr, d_state=d_state, d_conv=d_conv, expand=expand) if bidirectional else None

    def _run(self, tok):
        tok = self.ln(tok)
        yf = self.m_f(tok)
        if not self.bidirectional:
            return yf
        yb = torch.flip(self.m_b(torch.flip(tok, dims=[1])), dims=[1])
        return 0.5 * (yf + yb)

    def forward(self, x):
        # x: [B,C,D,H,W]
        B, C, D, H, W = x.shape
        y = self.in_proj(x)  # [B,Cr,D,H,W]

        s = self.token_stride
        outs = []
        for d in range(D):
            z = y[:, :, d]          # [B,Cr,H,W]
            z = z[:, :, ::s, ::s]   # downsample tokens
            Bb, Cr, Hs, Ws = z.shape
            tok = z.flatten(2).transpose(1, 2).contiguous()  # [B,L,Cr]
            tok = self._run(tok)
            feat = tok.transpose(1, 2).contiguous().view(Bb, Cr, Hs, Ws)
            feat = F.interpolate(feat, size=(H, W), mode="nearest")
            outs.append(feat)

        out = torch.stack(outs, dim=2)  # [B,Cr,D,H,W]
        return self.out_proj(out)

# ============================================================
# Efficient UnetResBlock (same signature)
# ============================================================
class UnetResBlock(nn.Module):
    """
    Same signature as original.
    You still call: UnetResBlock(spatial_dims, in_ch, out_ch, kernel_size, stride, norm_name, act_name, dropout)
    """
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: Optional[Union[Tuple, str, float]] = None,
    ):
        super().__init__()

        # ---- original layers (kept) ----
        self.conv1 = get_conv_layer(spatial_dims, in_channels, out_channels, kernel_size=kernel_size,
                                    stride=stride, dropout=dropout, conv_only=True)
        self.conv2 = get_conv_layer(spatial_dims, out_channels, out_channels, kernel_size=kernel_size,
                                    stride=1, dropout=dropout, conv_only=True)

        self.lrelu = get_act_layer(name=act_name)
        self.norm1 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.norm2 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)

        # ---- downsample skip if needed ----
        self.downsample = (in_channels != out_channels)
        stride_np = np.atleast_1d(stride)
        if not np.all(stride_np == 1):
            self.downsample = True
        if self.downsample:
            self.conv3 = get_conv_layer(spatial_dims, in_channels, out_channels, kernel_size=1,
                                        stride=stride, dropout=dropout, conv_only=True)
            self.norm3 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)

        # ---- efficient alternatives (created once) ----
        k = kernel_size[0] if isinstance(kernel_size, (tuple, list)) else int(kernel_size)
        drop_p = dropout if isinstance(dropout, float) else None

        self.dw1 = DWSeparableConvNd(spatial_dims, in_channels, out_channels, k=k, stride=stride, dropout=drop_p)
        self.dw2 = DWSeparableConvNd(spatial_dims, out_channels, out_channels, k=k, stride=1, dropout=drop_p)

        self.mamba = CheapBiMamba3D(out_channels, reduction=12, token_stride=4, bidirectional=True) if spatial_dims == 3 else None

        # ---- runtime toggles (default: off = baseline) ----
        self._use_dwsep = False
        self._use_mamba = False
        self._mamba_scale = 0.05

        # guard: only apply mamba when volume is small (prevents MAC blow-up)
        self._mamba_max_tokens = 8 * 8 * 8  # apply only when D*H*W <= 512

    # --- user-facing toggles ---
    def enable_dwsep(self, enabled: bool = True):
        self._use_dwsep = bool(enabled)

    def enable_mamba(self, enabled: bool = True, bidirectional: Optional[bool] = None, scale: float = 0.05):
        self._use_mamba = bool(enabled)
        self._mamba_scale = float(scale)
        if (bidirectional is not None) and (self.mamba is not None):
            self.mamba.bidirectional = bool(bidirectional)

    def forward(self, inp):
        residual = inp

        # conv1
        if self._use_dwsep:
            out = self.dw1(inp)
        else:
            out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)

        # conv2 OR (cheap) mamba mix
        if self._use_mamba and (self.mamba is not None):
            # only apply on small volumes
            B, C, D, H, W = out.shape
            if D * H * W <= self._mamba_max_tokens:
                out = out + self._mamba_scale * self.mamba(out)
            else:
                # fallback to cheap conv if too big
                out = self.dw2(out) if self._use_dwsep else self.conv2(out)
        else:
            out = self.dw2(out) if self._use_dwsep else self.conv2(out)

        out = self.norm2(out)

        if hasattr(self, "conv3"):
            residual = self.conv3(residual)
        if hasattr(self, "norm3"):
            residual = self.norm3(residual)

        out = out + residual
        out = self.lrelu(out)
        return out 
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.utils import get_act_layer, get_norm_layer
import numpy as np

from mamba_ssm import Mamba  # pip install mamba-ssm


# ----------------------------
# MONAI conv helper (your version)
# ----------------------------
def get_padding(kernel_size: Union[Sequence[int], int], stride: Union[Sequence[int], int]) -> Union[Tuple[int, ...], int]:
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = (kernel_size_np - stride_np + 1) / 2
    if np.min(padding_np) < 0:
        raise AssertionError("padding value should not be negative, please change kernel size and/or stride.")
    padding = tuple(int(p) for p in padding_np)
    return padding if len(padding) > 1 else padding[0]


def get_output_padding(
    kernel_size: Union[Sequence[int], int], stride: Union[Sequence[int], int], padding: Union[Sequence[int], int]
) -> Union[Tuple[int, ...], int]:
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = np.atleast_1d(padding)
    out_padding_np = 2 * padding_np + stride_np - kernel_size_np
    if np.min(out_padding_np) < 0:
        raise AssertionError("out_padding value should not be negative, please change kernel size and/or stride.")
    out_padding = tuple(int(p) for p in out_padding_np)
    return out_padding if len(out_padding) > 1 else out_padding[0]


def get_conv_layer(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    kernel_size: Union[Sequence[int], int] = 3,
    stride: Union[Sequence[int], int] = 1,
    act: Optional[Union[Tuple, str]] = Act.PRELU,
    norm: Union[Tuple, str] = Norm.INSTANCE,
    dropout: Optional[Union[Tuple, str, float]] = None,
    bias: bool = False,
    conv_only: bool = True,
    is_transposed: bool = False,
):
    padding = get_padding(kernel_size, stride)
    output_padding = None
    if is_transposed:
        output_padding = get_output_padding(kernel_size, stride, padding)
    return Convolution(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        strides=stride,
        kernel_size=kernel_size,
        act=act,
        norm=norm,
        dropout=dropout,
        bias=bias,
        conv_only=conv_only,
        is_transposed=is_transposed,
        padding=padding,
        output_padding=output_padding,
    )


# ============================================================
# Mamba mixer that matches your diagram (NO window partition)
#   Linear -> Conv1D -> σ -> SSM, plus parallel z-branch
# Implemented via mamba_ssm.Mamba which internally does the
# Conv1D + SSM. We keep explicit gating like the diagram.
# ============================================================

class MambaTokenMixer(nn.Module):
    """
    Token mixer for N-D feature maps (2D/3D) without window partition.
    Operates on flattened tokens, optionally on a downsampled token grid for cheaper compute.

    x: [B, C, ...]  ->  tokens: [B, L, C]  -> Mamba -> [B, L, C] -> reshape back
    """

    def __init__(
        self,
        channels: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        bidirectional: bool = False,
        token_stride: int = 1,     # 1 = full-res tokens; >1 = downsample tokens then upsample
        dropout: float = 0.0,
    ):
        super().__init__()
        self.channels = int(channels)
        self.bidirectional = bool(bidirectional)
        self.token_stride = max(1, int(token_stride))

        # diagram top/bottom "Linear"
        self.in_proj = nn.Linear(channels, channels, bias=False)
        self.out_proj = nn.Linear(channels, channels, bias=False)

        # gating branch (σ)
        self.gate = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(inplace=True),
            nn.Sigmoid(),
        )

        self.ln = nn.LayerNorm(channels)
        self.drop = nn.Dropout(dropout)

        self.m_f = Mamba(d_model=channels, d_state=d_state, d_conv=d_conv, expand=expand)
        self.m_b = Mamba(d_model=channels, d_state=d_state, d_conv=d_conv, expand=expand) if self.bidirectional else None

    def _flatten_tokens(self, x: torch.Tensor):
        # x: [B,C,H,W] or [B,C,D,H,W] -> tokens [B,L,C] and shape meta
        if x.dim() == 4:
            B, C, H, W = x.shape
            tokens = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
            meta = ("2d", H, W)
            return tokens, meta
        if x.dim() == 5:
            B, C, D, H, W = x.shape
            tokens = x.permute(0, 2, 3, 4, 1).contiguous().view(B, D * H * W, C)
            meta = ("3d", D, H, W)
            return tokens, meta
        raise ValueError(f"Expected 4D/5D input, got shape {tuple(x.shape)}")

    def _unflatten_tokens(self, tokens: torch.Tensor, meta):
        # tokens [B,L,C] -> x [B,C,...]
        kind = meta[0]
        if kind == "2d":
            _, H, W = meta
            B, L, C = tokens.shape
            assert L == H * W
            x = tokens.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            return x
        if kind == "3d":
            _, D, H, W = meta
            B, L, C = tokens.shape
            assert L == D * H * W
            x = tokens.view(B, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
            return x
        raise ValueError(f"Unknown meta kind: {kind}")

    def _token_downsample(self, x: torch.Tensor) -> torch.Tensor:
        # Downsample only spatial (and D for 3D) token grid to reduce L.
        s = self.token_stride
        if s == 1:
            return x
        if x.dim() == 4:
            return x[:, :, ::s, ::s]
        # 3D
        return x[:, :, ::s, ::s, ::s]

    def _token_upsample(self, x: torch.Tensor, ref_shape) -> torch.Tensor:
        s = self.token_stride
        if s == 1:
            return x
        if len(ref_shape) == 4:
            _, _, H, W = ref_shape
            return F.interpolate(x, size=(H, W), mode="nearest")
        _, _, D, H, W = ref_shape
        return F.interpolate(x, size=(D, H, W), mode="nearest")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ref_shape = x.shape

        # optional token downsample to make it cheap
        x_ds = self._token_downsample(x)

        # flatten to tokens
        tokens, meta = self._flatten_tokens(x_ds)  # [B,L,C]

        # Linear
        tokens = self.in_proj(tokens)

        # σ gate (diagram right branch)
        z = self.gate(tokens)  # [B,L,C]

        # SSM path (diagram left branch)
        t = self.ln(tokens)
        y_f = self.m_f(t)
        if self.bidirectional:
            y_b = torch.flip(self.m_b(torch.flip(t, dims=[1])), dims=[1])
            y = 0.5 * (y_f + y_b)
        else:
            y = y_f

        # gated fusion and output Linear
        out = self.out_proj(self.drop(y * z))

        # residual in token space
        out = tokens + out

        # unflatten to feature map
        x_out = self._unflatten_tokens(out, meta)

        # upsample back if token_stride > 1
        x_out = self._token_upsample(x_out, ref_shape)

        return x_out + ref_shape


# ============================================================
# UNet Up Block using ONLY the Mamba-style mixer (NO window partition)
# Same signature as MONAI/DynUNet UnetUpBlock.
# ============================================================

class UnetUpBlock(nn.Module):
    """
    upsample -> concat skip -> 1x1 fuse -> MambaTokenMixer (no window partition)

    Same signature as your original UnetUpBlock.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: Optional[Union[Tuple, str, float]] = None,
        trans_bias: bool = False,
    ):
        super().__init__()

        # --- upsample (kept exactly) ---
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            dropout=dropout,
            bias=trans_bias,
            conv_only=True,
            is_transposed=True,
        )

        # --- concat fuse (cheap) ---
        # keep convs allowed (this is a 1x1 conv; if you truly want "no convs" except transposed,
        # replace with Linear per-voxel, but MONAI pipelines usually accept this.
        self.fuse = get_conv_layer(
            spatial_dims=spatial_dims,
            in_channels=2 * out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            dropout=None,
            bias=False,
            conv_only=True,
            is_transposed=True,
        )
        self.fuse_norm = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.fuse_act = get_act_layer(name=act_name)

        # --- Mamba mixer (NO window partition) ---
        # Defaults are conservative; tune token_stride to keep it cheap at high-res decoder stages.
        self.mixer = MambaTokenMixer(
            channels=out_channels,
            d_state=16,
            d_conv=4,
            expand=2,
            bidirectional=False,
            token_stride=1,   # set to 2 or 4 for cheaper compute at large D/H/W
            dropout=float(dropout) if isinstance(dropout, float) else 0.0,
        )
        self.post_norm = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)

    @staticmethod
    def _center_crop_or_pad_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """
        Match x spatial size to ref for concatenation (handles odd-size mismatches).
        Works for 2D (B,C,H,W) and 3D (B,C,D,H,W).
        """
        if x.shape[2:] == ref.shape[2:]:
            return x

        # crop
        xs = list(x.shape[2:])
        rs = list(ref.shape[2:])
        slices = []
        for i, (a, b) in enumerate(zip(xs, rs)):
            if a > b:
                d = a - b
                start = d // 2
                end = start + b
                slices.append(slice(start, end))
            else:
                slices.append(slice(0, a))
        if x.dim() == 4:
            x = x[:, :, slices[0], slices[1]]
        else:
            x = x[:, :, slices[0], slices[1], slices[2]]

        # pad
        xs = list(x.shape[2:])
        pad = []
        for a, b in zip(reversed(xs), reversed(rs)):
            if a < b:
                d = b - a
                pad.extend([d // 2, d - d // 2])
            else:
                pad.extend([0, 0])
        if any(pad):
            x = F.pad(x, pad)
        return x

    def forward(self, inp: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # upsample
        out = self.transp_conv(inp)

        # ensure concat shapes match
        out = self._center_crop_or_pad_to(out, skip)

        # concat
        out2 = torch.cat((out, skip), dim=1)

        # fuse
        out = self.fuse(out2)
        out = self.fuse_norm(out)
        out = self.fuse_act(out)

        # mamba mix (no window partition)
        out = out + self.mixer(out)
        out = self.post_norm(out)
        return out + out2


# ----------------------------
# Usage tip (keeps signature)
# ----------------------------
# After constructing the network, you can set per-stage compute:
#   up_block.mixer.token_stride = 2   # cheaper at high-res decoder stages
#   up_block.mixer.bidirectional = True  # more compute, sometimes better accuracy
