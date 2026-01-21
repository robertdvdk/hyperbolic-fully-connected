import torch
import torch.nn as nn

class EuclideanToLorentzConv(nn.Module):
    """Project Euclidean image onto Lorentz manifold via 1x1 conv."""
    
    def __init__(self, in_channels, out_channels, manifold, proj_type: str = "conv_bn_relu"):
        """
        Args:
            in_channels: Euclidean channels (e.g., 3 for RGB)
            out_channels: Lorentz channels INCLUDING time (e.g., 16 means 15 space + 1 time)
        """
        super().__init__()
        self.manifold = manifold
        if proj_type == "conv":
            self.proj = nn.Conv2d(in_channels, out_channels - 1, kernel_size=1)
        elif proj_type == "conv_bn_relu":
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels - 1, kernel_size=1),
                nn.BatchNorm2d(out_channels - 1),
                nn.ReLU()
            )
        else:
            raise ValueError(f"Unknown proj_type: {proj_type}")

        conv = self.proj[0] if isinstance(self.proj, nn.Sequential) else self.proj
        nn.init.xavier_uniform_(conv.weight, gain=0.1)
        if conv.bias is not None:
            nn.init.zeros_(conv.bias)
    
    def forward(self, x):
        """
        Args:
            x: [batch, in_channels, H, W] Euclidean image
        Returns:
            [batch, out_channels, H, W] on Lorentz manifold (each pixel is a Lorentz point)
        """
        space = self.proj(x)  # [batch, out_channels - 1, H, W])
        
        # Compute time component for each pixel
        # time = sqrt(||space||^2 + 1/k)
        time = torch.sqrt((space ** 2).sum(dim=1, keepdim=True) + 1.0 / self.manifold.k())
        
        return torch.cat([time, space], dim=1)  # [batch, out_channels, H, W]