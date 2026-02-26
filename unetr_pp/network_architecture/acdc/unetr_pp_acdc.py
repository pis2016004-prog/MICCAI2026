from typing import Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from unetr_pp.network_architecture.neural_network import SegmentationNetwork
from unetr_pp.network_architecture.dynunet_block import UnetOutBlock, UnetResBlock
from unetr_pp.network_architecture.acdc.model_components import UnetrPPEncoder, UnetrUpBlock


# =========================
# Ultra-cheap expressive LGAG
# =========================
class LGAG3D_FastExpressive(nn.Module):
    """
    Very low-cost gated attention for 3D features.
    - Two grouped 3x3x3 conv branches (one dilated) + 1x1 projection
    - Optional tiny channel calibration (ECA-like) using 1D conv over channels
    """
    def __init__(
        self,
        channels: int,
        groups: int = 8,
        dilation: int = 2,
        eca_k: int = 3,
        use_eca: bool = True,
    ):
        super().__init__()
        C = int(channels)
        g = max(1, min(int(groups), C))
        while C % g != 0 and g > 1:
            g -= 1

        self.b1 = nn.Sequential(
            nn.Conv3d(C, C, 3, 1, 1, groups=g, bias=False),
            nn.BatchNorm3d(C),
            nn.SiLU(inplace=True),
        )
        self.b2 = nn.Sequential(
            nn.Conv3d(C, C, 3, 1, dilation, dilation=dilation, groups=g, bias=False),
            nn.BatchNorm3d(C),
            nn.SiLU(inplace=True),
        )

        self.use_eca = bool(use_eca)
        if self.use_eca:
            self.avg = nn.AdaptiveAvgPool3d(1)
            self.eca = nn.Conv1d(1, 1, kernel_size=eca_k, padding=eca_k // 2, bias=False)

        self.proj = nn.Sequential(
            nn.Conv3d(C, C, 1, 1, 0, bias=False),
            nn.BatchNorm3d(C),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.b1(x) + self.b2(x)  # [B,C,D,H,W]

        if self.use_eca:
            # ECA gate from pooled features
            y = self.avg(s).view(s.size(0), 1, s.size(1))             # [B,1,C]
            y = torch.sigmoid(self.eca(y)).view(s.size(0), s.size(1), 1, 1, 1)
            s = s * y

        gate = self.sigmoid(self.proj(s))
        return x * gate


# ==========================================
# Fast + expressive disagreement skip gating
# ==========================================
class DisagreeFiLMGate3D(nn.Module):
    """
    More expressive than plain MLP gate but still cheap.

    Inputs:
      enc_skip: [B, Cenc, D, H, W]
      dec_feat: [B, Cdec, D',H',W']

    Outputs:
      gated_skip: [B, Cenc, D,H,W]
      u_c:        [B, Cenc]   mean abs disagreement per channel

    Components:
      - 1x1 dec->enc projection
      - channel stats: mean(enc), mean(dec), mean(|enc-dec|)
      - tiny MLP -> FiLM gamma+beta per channel (initialized to identity)
      - optional spatial gate from single-channel disagreement map:
          DW 3x3x3 + 1x1 (very cheap)
      - residual gating for stability
    """
    def __init__(
        self,
        enc_ch: int,
        dec_ch: int,
        r: int = 16,
        beta: float = 0.5,
        interp_mode: str = "nearest",     # fastest; use "trilinear" if you prefer
        spatial: bool = True,
        spatial_dw_kernel: int = 3,
    ):
        super().__init__()
        self.dec_to_enc = nn.Conv3d(dec_ch, enc_ch, kernel_size=1, bias=False)
        hidden = max(enc_ch // int(r), 16)

        # 3C -> 2C (gamma,bias)
        self.mlp = nn.Sequential(
            nn.Linear(3 * enc_ch, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, 2 * enc_ch),
        )

        # init close to identity (gamma≈0 -> sigmoid(0)=0.5, we apply residual form)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        self.spatial = bool(spatial)
        if self.spatial:
            k = int(spatial_dw_kernel)
            p = k // 2
            self.spatial_net = nn.Sequential(
                nn.Conv3d(1, 1, kernel_size=k, padding=p, groups=1, bias=False),
                nn.BatchNorm3d(1),
                nn.SiLU(inplace=True),
                nn.Conv3d(1, 1, kernel_size=1, bias=True),
                nn.Sigmoid(),
            )

        self.beta = float(beta)
        self.interp_mode = str(interp_mode)

    def forward(self, enc_skip: torch.Tensor, dec_feat: torch.Tensor):
        B, C = enc_skip.shape[:2]

        dec_proj = self.dec_to_enc(dec_feat)
        if dec_proj.shape[2:] != enc_skip.shape[2:]:
            if self.interp_mode == "nearest":
                dec_proj = F.interpolate(dec_proj, size=enc_skip.shape[2:], mode="nearest")
            else:
                dec_proj = F.interpolate(dec_proj, size=enc_skip.shape[2:], mode="trilinear", align_corners=False)

        # disagreement summary
        diff = (enc_skip - dec_proj).abs()
        u_c = diff.mean(dim=(2, 3, 4))                    # [B,C]
        m_enc = enc_skip.mean(dim=(2, 3, 4))              # [B,C]
        m_dec = dec_proj.mean(dim=(2, 3, 4))              # [B,C]

        # FiLM params
        params = self.mlp(torch.cat([m_enc, m_dec, u_c], dim=1))  # [B,2C]
        gamma, bias = params[:, :C], params[:, C:]

        # residual channel modulation
        gamma = torch.sigmoid(gamma).view(B, C, 1, 1, 1)  # [B,C,1,1,1]
        bias = bias.view(B, C, 1, 1, 1)
        x = enc_skip * (1.0 + self.beta * gamma) + (self.beta * bias)

        # optional spatial gate (single channel)
        if self.spatial:
            dmap = diff.mean(dim=1, keepdim=True)         # [B,1,D,H,W]
            sgate = self.spatial_net(dmap)                # [B,1,D,H,W]
            x = x * (1.0 + self.beta * sgate)

        return x, u_c


# ======================
# UNETR++ (ACDC) upgraded
# ======================
class UNETR_PP(SegmentationNetwork):
    """
    ACDC UNETR++ with:
      - LGAG3D_FastExpressive bridge
      - DisagreeFiLMGate3D skip gating (FiLM)
      - Robust to spatial mismatch (interpolation)
      - Optional unc outputs per scale
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        feature_size: int = 16,
        hidden_size: int = 256,
        num_heads: int = 4,
        pos_embed: str = "perceptron",
        norm_name: Union[Tuple, str] = "instance",
        dropout_rate: float = 0.0,
        depths=None,
        dims=None,
        conv_op=nn.Conv3d,
        do_ds: bool = True,

        # gate knobs
        gate_r: int = 8,
        gate_beta: float = 0.5,
        interp_mode: str = "trilinear",   # "nearest" fastest
        spatial_gates: bool = False,      # enable only if you want spatial gating in deep skips

        # LGAG knobs
        lgag_groups: int = 8,
        lgag_dilation: int = 2,
        lgag_use_eca: bool = True,

        return_unc: bool = True,
    ) -> None:
        super().__init__()
        if depths is None:
            depths = [3, 3, 3, 3]

        self.do_ds = bool(do_ds)
        self.conv_op = conv_op
        self.num_classes = out_channels
        self.return_unc = bool(return_unc)

        if not (0.0 <= dropout_rate <= 1.0):
            raise AssertionError("dropout_rate should be between 0 and 1.")
        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        # ACDC-specific feat_size (keep your original)
        self.feat_size = (2, 5, 5)
        self.hidden_size = int(hidden_size)

        # encoder
        self.unetr_pp_encoder = UnetrPPEncoder(dims=dims, depths=depths, num_heads=num_heads)

        # stem
        self.encoder1 = UnetResBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )

        # bridge LGAG (upgraded)
        self.lgag_bridge = LGAG3D_FastExpressive(
            channels=self.hidden_size,
            groups=lgag_groups,
            dilation=lgag_dilation,
            use_eca=lgag_use_eca,
        )

        # decoders (keep your ACDC sizes)
        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 16,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=4 * 10 * 10,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=8 * 20 * 20,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=16 * 40 * 40,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=(1, 4, 4),
            norm_name=norm_name,
            out_size=16 * 160 * 160,
            conv_decoder=True,
        )

        # ---- upgraded FiLM disagreement gates ----
        # IMPORTANT: these enc_ch assumptions must match your encoder outputs.
        # If shapes mismatch, print enc1/enc2/enc3 channels and adjust.
        self.gate_enc3 = DisagreeFiLMGate3D(
            enc_ch=feature_size * 8,
            dec_ch=self.hidden_size,
            r=gate_r,
            beta=gate_beta,
            interp_mode=interp_mode,
            spatial=bool(spatial_gates),
        )
        self.gate_enc2 = DisagreeFiLMGate3D(
            enc_ch=feature_size * 4,
            dec_ch=feature_size * 8,
            r=gate_r,
            beta=gate_beta,
            interp_mode=interp_mode,
            spatial=bool(spatial_gates),
        )
        self.gate_enc1 = DisagreeFiLMGate3D(
            enc_ch=feature_size * 2,
            dec_ch=feature_size * 4,
            r=gate_r,
            beta=gate_beta,
            interp_mode=interp_mode,
            spatial=bool(spatial_gates),
        )
        # full-res: spatial OFF recommended (cost)
        self.gate_conv = DisagreeFiLMGate3D(
            enc_ch=feature_size,
            dec_ch=feature_size * 2,
            r=gate_r,
            beta=gate_beta,
            interp_mode=interp_mode,
            spatial=False,
        )

        # heads
        self.out1 = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)
        if self.do_ds:
            self.out2 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 2, out_channels=out_channels)
            self.out3 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 4, out_channels=out_channels)

    def proj_feat(self, x: torch.Tensor, hidden_size: int, feat_size: Tuple[int, int, int]) -> torch.Tensor:
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in: torch.Tensor):
        _, hidden_states = self.unetr_pp_encoder(x_in)
        convBlock = self.encoder1(x_in)

        enc1, enc2, enc3, enc4 = hidden_states[0], hidden_states[1], hidden_states[2], hidden_states[3]

        # bridge
        dec4 = self.proj_feat(enc4, self.hidden_size, self.feat_size)  # [B, hidden, 2,5,5]
        dec4 = self.lgag_bridge(dec4)

        # gated skips + decode
        enc3_g, u3 = self.gate_enc3(enc3, dec4)
        dec3 = self.decoder5(dec4, enc3_g)

        enc2_g, u2 = self.gate_enc2(enc2, dec3)
        dec2 = self.decoder4(dec3, enc2_g)

        enc1_g, u1 = self.gate_enc1(enc1, dec2)
        dec1 = self.decoder3(dec2, enc1_g)

        conv_g, u0 = self.gate_conv(convBlock, dec1)
        out = self.decoder2(dec1, conv_g)

        # heads
        if self.do_ds:
            logits = [self.out1(out), self.out2(dec1), self.out3(dec2)]
        else:
            logits = self.out1(out)

        # dummy loss (kept to match your trainer expectation)
        loss = (logits[0].sum() * 0.0) if isinstance(logits, (list, tuple)) else (logits.sum() * 0.0)

        if self.return_unc:
            unc = {"u0": u0, "u1": u1, "u2": u2, "u3": u3}
            return logits, loss, unc

        return logits, loss
