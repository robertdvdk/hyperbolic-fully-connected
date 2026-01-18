import torch
import torch.nn as nn

class EuclideanToLorentzConv(nn.Module):
    """Project Euclidean image onto Lorentz manifold via 1x1 conv."""
    
    def __init__(self, in_channels, out_channels, manifold):
        """
        Args:
            in_channels: Euclidean channels (e.g., 3 for RGB)
            out_channels: Lorentz channels INCLUDING time (e.g., 16 means 15 space + 1 time)
        """
        super().__init__()
        self.manifold = manifold
        self.conv = nn.Conv2d(in_channels, out_channels - 1, kernel_size=1)
        
        nn.init.xavier_uniform_(self.conv.weight, gain=0.1)
        nn.init.zeros_(self.conv.bias)
    
    def forward(self, x):
        """
        Args:
            x: [batch, in_channels, H, W] Euclidean image
        Returns:
            [batch, out_channels, H, W] on Lorentz manifold (each pixel is a Lorentz point)
        """
        space = self.conv(x)  # [batch, out_channels - 1, H, W]
        
        # Compute time component for each pixel
        # time = sqrt(||space||^2 + 1/k)
        time = torch.sqrt((space ** 2).sum(dim=1, keepdim=True) + 1.0 / self.manifold.k())
        
        return torch.cat([time, space], dim=1)  # [batch, out_channels, H, W]