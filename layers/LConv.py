import torch
import torch.nn as nn
import torch.nn.functional as F
from .lorentz import Lorentz
from .LLinear import LorentzFullyConnected

class LorentzConv2d(nn.Module):
    """
    Lorentz Conv2d using direct concatenation + existing Lorentz FC.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int],
        padding: int | tuple[int, int],
        manifold: Lorentz,
        activation,
        init_method: str = "kaiming",
        backbone_std_mult = 1.0,
        mlr_std_mult = 1.0,
    ):
        super().__init__()
        self.manifold = manifold or Lorentz(k=1.0)

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # After concatenating k*k Lorentz points:
        # concat_dim = 1 + (in_channels - 1) * k * k
        concat_dim = 1 + (in_channels - 1) * kernel_size[0] * kernel_size[1]

        # Reuse existing Lorentz FC
        self.fc = LorentzFullyConnected(
            in_features=concat_dim,
            out_features=out_channels,
            manifold=self.manifold,
            activation=activation,
            reset_params=init_method,
            backbone_std_mult=backbone_std_mult,
            mlr_std_mult=mlr_std_mult
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, in_channels, H, W]
        Returns:
            [batch, out_channels, H', W']
        """
        batch, C, H, W = x.shape
        kH, kW = self.kernel_size
        sH, sW = self.stride
        pH, pW = self.padding
        
        # Pad with origin points
        if pH > 0 or pW > 0:
            sqrt_k_inv = (1.0 / self.manifold.k()).sqrt()
            x = F.pad(x, (pW, pW, pH, pH), mode='constant', value=0.0)
            _, _, H_pad, W_pad = x.shape
            
            # Fix time component in padded regions
            mask = torch.ones(1, 1, H_pad, W_pad, device=x.device, dtype=x.dtype)
            mask[:, :, pH:pH+H, pW:pW+W] = 0
            x[:, 0:1] = x[:, 0:1] * (1 - mask) + sqrt_k_inv * mask
        
        # Unfold: [batch, C * kH * kW, num_patches]
        patches = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride)
        
        # Reshape to [batch, num_patches, kH * kW, C]
        num_patches = patches.shape[-1]
        patches = patches.view(batch, C, kH * kW, num_patches)
        patches = patches.permute(0, 3, 2, 1)  # [batch, num_patches, k*k, C]
        
        # Direct concat: [batch * num_patches, k*k, C] -> [batch * num_patches, concat_dim]
        patches_flat = patches.reshape(batch * num_patches, kH * kW, C)
        concat_points = self.manifold.direct_concat(patches_flat)
        
        # Apply Lorentz FC: [batch * num_patches, concat_dim] -> [batch * num_patches, out_channels]
        out = self.fc(concat_points)
        
        # Reshape to spatial: [batch, out_channels, H', W']
        H_out = (H + 2 * pH - kH) // sH + 1
        W_out = (W + 2 * pW - kW) // sW + 1
        out = out.view(batch, H_out, W_out, -1).permute(0, 3, 1, 2)
        
        return out