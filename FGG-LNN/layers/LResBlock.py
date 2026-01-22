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
    ):

        super().__init__()
        self.manifold = manifold or Lorentz(k=1.0)

        self.layer1 = LorentzConv2d(
            in_channels=input_dim, out_channels=output_dim,
            kernel_size=kernel_size, stride=1, padding=padding,
            manifold=manifold, activation=nn.Identity(), init_method=init_method
        )
        self.bn1 = LorentzBatchNorm2d(num_features=output_dim, manifold=manifold)
        self.layer2 = LorentzConv2d(
            in_channels=output_dim, out_channels=output_dim,
            kernel_size=kernel_size, stride=stride, padding=padding,
            manifold=manifold, activation=nn.Identity(), init_method=init_method
        )
        self.bn2 = LorentzBatchNorm2d(num_features=output_dim, manifold=manifold)
        if input_dim != output_dim:
            proj_layers = [LorentzConv2d(
                in_channels=input_dim, out_channels=output_dim,
                kernel_size=1, stride=stride, padding=0,
                manifold=manifold, activation=nn.Identity(), init_method=init_method
            )]
            proj_layers.append(LorentzBatchNorm2d(num_features=output_dim, manifold=manifold))
            self.proj = nn.Sequential(*proj_layers)
        else:
            self.proj = nn.Identity()

    
    def forward(self, x):
        x2 = self.layer1(x)
        x2 = self.bn1(x2)
        x2 = self.manifold.relu(x2, manifold_dim=1)
        x2 = self.layer2(x2)
        x2 = self.bn2(x2)
        
        x = self.proj(x)
        out_space = x[:, 1:, :, :] + x2[:, 1:, :, :]
        out = self.manifold.projection_space_orthogonal(out_space, manifold_dim=1)
        out = self.manifold.relu(out, manifold_dim=1)

        return out