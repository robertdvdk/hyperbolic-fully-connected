import torch
import torch.nn as nn
import torch.nn.functional as F
from .lorentz import Lorentz
from .LConv import LorentzConv2d
from .LBatchNorm import LorentzBatchNorm2d

class LorentzResBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        kernel_size,
        stride,
        padding,
        manifold,
        init_method: str = "kaiming",
        skip_bn2: bool = False,  # Skip bn2 for last block (prevents gamma explosion with MLR)
        fix_gamma: bool = False,  # Fix gamma=1 in all BatchNorms (forces linear layers to handle scaling)
        clamp_scale: bool = False,  # Clamp BN scale to [0.5, 2.0]
        normalize_variance: bool = True,  # If False, mean-only normalization (no variance scaling)
        use_weight_norm: bool = False,
        fc_variant: str = "ours",
    ):

        super().__init__()
        self.manifold = manifold or Lorentz(k=1.0)
        self.skip_bn2 = skip_bn2

        self.layer1 = LorentzConv2d(
            in_channels=input_dim, out_channels=output_dim,
            kernel_size=kernel_size, stride=1, padding=padding,
            manifold=manifold, activation=nn.Identity(), init_method=init_method,
            use_weight_norm=use_weight_norm,
            fc_variant=fc_variant,
        )
        self.bn1 = LorentzBatchNorm2d(
            num_features=output_dim,
            manifold=manifold,
            fix_gamma=fix_gamma,
            clamp_scale=clamp_scale,
            normalize_variance=normalize_variance,
        )
        self.layer2 = LorentzConv2d(
            in_channels=output_dim, out_channels=output_dim,
            kernel_size=kernel_size, stride=stride, padding=padding,
            manifold=manifold, activation=nn.Identity(), init_method=init_method,
            use_weight_norm=use_weight_norm,
            fc_variant=fc_variant,
        )
        self.bn2 = None if skip_bn2 else LorentzBatchNorm2d(
            num_features=output_dim,
            manifold=manifold,
            fix_gamma=fix_gamma,
            clamp_scale=clamp_scale,
            normalize_variance=normalize_variance,
        )
        if input_dim != output_dim:
            proj_layers = [LorentzConv2d(
                in_channels=input_dim, out_channels=output_dim,
                kernel_size=1, stride=stride, padding=0,
                manifold=manifold, activation=nn.Identity(), init_method=init_method,
                use_weight_norm=use_weight_norm,
                fc_variant=fc_variant,
            )]
            proj_layers.append(LorentzBatchNorm2d(
                num_features=output_dim,
                manifold=manifold,
                fix_gamma=fix_gamma,
                clamp_scale=clamp_scale,
                normalize_variance=normalize_variance,
            ))
            self.proj = nn.Sequential(*proj_layers)
        else:
            self.proj = nn.Identity()

    
    def forward(self, x):
        x2 = self.layer1(x)
        x2 = self.bn1(x2)
        x2 = self.manifold.relu(x2, manifold_dim=1)
        x2 = self.layer2(x2)
        if self.bn2 is not None:
            x2 = self.bn2(x2)

        x = self.proj(x)
        out_space = x[:, 1:, :, :] + x2[:, 1:, :, :]
        out = self.manifold.projection_space_orthogonal(out_space, manifold_dim=1)
        out = self.manifold.relu(out, manifold_dim=1)

        return out