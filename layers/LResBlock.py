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
        activation,
        init_method: str = "kaiming",
        proj_bn: bool = True,
        residual_mode: str = "midpoint",
        midpoint_relu: bool = False,
    ):

        super().__init__()
        self.manifold = manifold or Lorentz(k=1.0)

        self.layer1 = LorentzConv2d(
            in_channels=input_dim, out_channels=output_dim,
            kernel_size=kernel_size, stride=1, padding=padding,
            manifold=manifold, activation=activation, init_method=init_method
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
            if proj_bn:
                proj_layers.append(LorentzBatchNorm2d(num_features=output_dim, manifold=manifold))
            self.proj = nn.Sequential(*proj_layers)
        else:
            self.proj = nn.Identity()

        self.alpha_logit = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5
        self.residual_mode = residual_mode
        self.midpoint_relu = midpoint_relu

    
    def forward(self, x):
        x2 = self.layer1(x)
        x2 = self.bn1(x2)
        x2 = self.layer2(x2)
        x2 = self.bn2(x2)
        x = self.proj(x)

        x = x.permute(0, 2, 3, 1)
        x2 = x2.permute(0, 2, 3, 1)

        if self.residual_mode == "midpoint":
            stacked = torch.stack([x, x2], dim=-2)
            alpha = torch.sigmoid(self.alpha_logit)
            weights = torch.stack([1 - alpha, alpha])
            x = self.manifold.lorentz_midpoint(stacked, weights)

            if self.midpoint_relu:
                x_space = F.relu(x[..., 1:])
                x = self.manifold.projection_space_orthogonal(x_space)
        elif self.residual_mode == "add":
            x_space = x[..., 1:]
            x2_space = x2[..., 1:]
            summed_space = x_space + x2_space
            if self.midpoint_relu:
                summed_space = F.relu(summed_space)
            x = self.manifold.projection_space_orthogonal(summed_space)
        else:
            raise ValueError(f"Unknown residual_mode: {self.residual_mode}")

        x = x.permute(0, 3, 1, 2)
        return x