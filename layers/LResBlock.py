import torch
import torch.nn as nn
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
            self.proj = LorentzConv2d(
                in_channels=input_dim, out_channels=output_dim,
                kernel_size=1, stride=stride, padding=0,
                manifold=manifold, activation=nn.Identity(), init_method=init_method
            )
        else:
            self.proj = nn.Identity()

        self.alpha_logit = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5

    
    def forward(self, x):
        x2 = self.layer1(x)
        x2 = self.bn1(x2)
        x2 = self.layer2(x2)
        x2 = self.bn2(x2)
        x = self.proj(x)


        x = x.permute(0, 2, 3, 1)
        x2 = x2.permute(0, 2, 3, 1)
        stacked = torch.stack([x, x2], dim=-2)
        alpha = torch.sigmoid(self.alpha_logit)
        weights = torch.stack([1-alpha, alpha])
        x = self.manifold.lorentz_midpoint(stacked, weights)

        x = x.permute(0, 3, 1, 2)

        return x