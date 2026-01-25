import torch
import torch.nn as nn
import torch.nn.functional as F
from .lorentz import Lorentz
from .LLinear import resolve_lorentz_fc_class


class LorentzConv1d(nn.Module):
    """
    Lorentz Conv1d using direct concatenation + Lorentz FC.
    Analogous to LorentzConv2d but for 1D sequences (e.g., genomic data).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        manifold: Lorentz = None,
        activation=F.relu,
        init_method: str = "kaiming",
        use_weight_norm: bool = False,
        fc_variant: str = "ours",
    ):
        super().__init__()
        self.manifold = manifold or Lorentz(k_value=1.0)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # After concatenating k Lorentz points (space dims only, shared time):
        # concat_dim = 1 + (in_channels - 1) * kernel_size
        concat_dim = 1 + (in_channels - 1) * kernel_size

        # Reuse selected Lorentz FC implementation
        fc_cls = resolve_lorentz_fc_class(fc_variant)
        self.fc = fc_cls(
            in_features=concat_dim,
            out_features=out_channels,
            manifold=self.manifold,
            activation=activation,
            reset_params=init_method,
            use_weight_norm=use_weight_norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, length] where channels includes time component
               i.e., channels = in_features (time + space dims)
        Returns:
            [batch, out_channels, length']
        """
        batch, C, L = x.shape
        k = self.kernel_size
        s = self.stride
        p = self.padding

        # Pad with origin points if needed
        if p > 0:
            sqrt_k_inv = (1.0 / self.manifold.k()).sqrt()
            x = F.pad(x, (p, p), mode='constant', value=0.0)
            _, _, L_pad = x.shape

            # Fix time component in padded regions
            mask = torch.ones(1, 1, L_pad, device=x.device, dtype=x.dtype)
            mask[:, :, p:p+L] = 0
            x[:, 0:1] = x[:, 0:1] * (1 - mask) + sqrt_k_inv * mask

        # Unfold: extract sliding windows
        # [batch, C, L_pad] -> [batch, C * k, num_patches]
        patches = x.unfold(dimension=2, size=k, step=s)  # [batch, C, num_patches, k]
        num_patches = patches.shape[2]

        # Reshape to [batch, num_patches, k, C]
        patches = patches.permute(0, 2, 3, 1)  # [batch, num_patches, k, C]

        # Direct concat: [batch * num_patches, k, C] -> [batch * num_patches, concat_dim]
        patches_flat = patches.reshape(batch * num_patches, k, C)
        concat_points = self.manifold.direct_concat(patches_flat)

        # Apply Lorentz FC: [batch * num_patches, concat_dim] -> [batch * num_patches, out_channels]
        out = self.fc(concat_points)

        # Reshape to [batch, out_channels, L_out]
        L_out = (L + 2 * p - k) // s + 1
        out = out.view(batch, num_patches, -1).permute(0, 2, 1)  # [batch, out_channels, L_out]

        return out


class LorentzReLU(nn.Module):
    """ReLU activation for Lorentz manifold - applies ReLU to space dimensions only."""

    def __init__(self, manifold: Lorentz = None):
        super().__init__()
        self.manifold = manifold or Lorentz(k_value=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, length] or [batch, length, channels]
        """
        if x.dim() == 3:
            # Assume [batch, channels, length] format
            manifold_dim = 1
        else:
            # [batch, channels] format
            manifold_dim = -1
        return self.manifold.relu(x, manifold_dim=manifold_dim)
