import torch
import torch.nn as nn
from .LConv import LorentzConv2d
from .LBatchNormTheirs import LorentzBatchNorm2d
from .lorentz import Lorentz

class EuclideanToLorentzConv(nn.Module):
    """Project Euclidean image onto Lorentz manifold via 1x1 conv."""

    def __init__(
        self,
        in_channels,
        out_channels,
        manifold: Lorentz,
        proj_type: str = "conv_bn_relu",
        fix_gamma: bool = False,
        clamp_scale: bool = False,
        normalize_variance: bool = True,
        init_method: str = "lorentz_kaiming",
    ):
        """
        Args:
            in_channels: Euclidean channels (e.g., 3 for RGB)
            out_channels: Lorentz channels INCLUDING time (e.g., 16 means 15 space + 1 time)
            fix_gamma: If True, fix gamma=1 in BatchNorm (not learnable)
            clamp_scale: If True, clamp BN scale to [0.5, 2.0]
            normalize_variance: If False, use mean-only normalization (no variance scaling)
            init_method: Initialization method for LorentzConv2d
        """
        super().__init__()
        self.manifold = manifold
        self.proj = nn.Sequential(
            LorentzConv2d(
                in_channels=in_channels + 1,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                stride=1,
                manifold=manifold,
                activation=nn.Identity(),
                init_method=init_method,
            ),
            LorentzBatchNorm2d(
                num_features=out_channels,
                manifold=manifold,
                fix_gamma=fix_gamma,
                clamp_scale=clamp_scale,
                normalize_variance=normalize_variance,
            ),
        )
    
    def forward(self, x):
        """
        Args:
            x: [batch, in_channels, H, W] Euclidean image
        Returns:
            [batch, out_channels, H, W] on Lorentz manifold (each pixel is a Lorentz point)
        """
        x = self.manifold.projection_space_orthogonal(x, manifold_dim=1)
        x = self.proj(x)
        return self.manifold.relu(x, manifold_dim=1)



        # space = self.proj(x)  # [batch, out_channels - 1, H, W])
        
        # Compute time component for each pixel
        # time = sqrt(||space||^2 + 1/k)
        # time = torch.sqrt((space ** 2).sum(dim=1, keepdim=True) + 1.0 / self.manifold.k())
        
        # return torch.cat([time, space], dim=1)  # [batch, out_channels, H, W]