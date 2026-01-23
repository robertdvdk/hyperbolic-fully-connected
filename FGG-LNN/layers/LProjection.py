import torch
import torch.nn as nn
from .LConv import LorentzConv2d
from LBatchNorm import LorentzBatchNorm2d
from .lorentz import Lorentz

class EuclideanToLorentzConv(nn.Module):
    """Project Euclidean image onto Lorentz manifold via 1x1 conv."""
    
    def __init__(self, in_channels, out_channels, manifold: Lorentz, proj_type: str = "conv_bn_relu"):
        """
        Args:
            in_channels: Euclidean channels (e.g., 3 for RGB)
            out_channels: Lorentz channels INCLUDING time (e.g., 16 means 15 space + 1 time)
        """
        super().__init__()
        self.manifold = manifold
        # if proj_type == "conv":
        #     self.proj = nn.Conv2d(in_channels, out_channels - 1, kernel_size=3)
        # elif proj_type == "conv_bn_relu":
        #     self.proj = nn.Sequential(
        #         nn.Conv2d(in_channels, out_channels - 1, kernel_size=3),
        #         nn.BatchNorm2d(out_channels - 1),
        #         nn.ReLU()
        #     )
        # else:
        #     raise ValueError(f"Unknown proj_type: {proj_type}")

        self.proj = nn.Sequential(
            LorentzConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, manifold=manifold, activation=nn.Identity()),
            LorentzBatchNorm2d(num_features=out_channels, manifold=manifold),

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