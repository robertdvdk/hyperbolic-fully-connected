import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
from layers import Lorentz_fully_connected
from layers import Lorentz


class EuclideanToLorentz(nn.Module):
    """Project Euclidean features onto Lorentz manifold."""
    
    def __init__(self, in_features, out_features, manifold):
        """
        Args:
            in_features: Euclidean input dimension (e.g., 784 for MNIST)
            out_features: Lorentz output dimension INCLUDING time (e.g., 100 means 99 space + 1 time)
        """
        super().__init__()
        self.manifold = manifold
        self.linear = nn.Linear(in_features, out_features - 1)  # Output space components only
        
        # Small init to keep points near origin initially (more stable)
        nn.init.xavier_uniform_(self.linear.weight, gain=0.1)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x):
        """
        Args:
            x: [batch, in_features] Euclidean vectors
        Returns:
            [batch, out_features] points on Lorentz manifold
        """
        space = self.linear(x)  # [batch, out_features - 1]
        return self.manifold.projection_space_orthogonal(space)  # [batch, out_features]
    
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

class LorentzResidualMidpoint(nn.Module):
    """Residual via weighted Lorentz midpoint."""
    
    def __init__(self, dim, manifold, activation):
        super().__init__()
        self.manifold = manifold
        self.fc = Lorentz_fully_connected(
            in_features=dim,
            out_features=dim,
            manifold=manifold,
            reset_params="kaiming",
            activation=activation
        )
        # Learnable weight (0.5 = equal weighting)
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5
    
    def forward(self, x):
        out = self.fc(x)
        
        # Weighted midpoint on manifold
        alpha = torch.sigmoid(self.alpha_logit)
        # Stack for centroid computation: [batch, 2, dim]
        stacked = torch.stack([x, out], dim=-2)
        
        # Weights: [1, 2] -> broadcast to [batch, 2]
        weights = torch.tensor([[1 - alpha, alpha]], device=x.device)
        return self.manifold.lorentz_midpoint(stacked, weights)
    

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
        self.fc = Lorentz_fully_connected(
            in_features=concat_dim,
            out_features=out_channels,
            manifold=self.manifold,
            activation=activation,
            reset_params="orthogonal"
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
    
class LorentzMLPWithResidual(nn.Module):
    def __init__(
        self, 
        input_dim,
        hidden_dim,
        num_classes,
        num_layers=3,
        manifold=None,
        activation=F.relu
    ):
    
        super().__init__()
        self.manifold = manifold or Lorentz(k=1.0)
        
        # Input projection
        self.input_proj = EuclideanToLorentz(input_dim, hidden_dim, self.manifold)
        
        # Select residual block type
        block_cls = LorentzResidualMidpoint
        
        # Hidden layers with residuals
        self.layers = nn.ModuleList([
            block_cls(hidden_dim, self.manifold, activation=activation)
            for _ in range(num_layers - 1)
        ])
        
        # Classifier
        self.classifier = Lorentz_fully_connected(
            in_features=hidden_dim,
            out_features=num_classes + 1,
            manifold=self.manifold,
            reset_params="kaiming",
            do_mlr=True
        )
    
    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        x = self.input_proj(x)
        
        for layer in self.layers:
            x = layer(x)
        
        return self.classifier(x)
    

class LorentzConvNet(nn.Module):
    def __init__(
        self, 
        input_dim,
        hidden_dim,
        num_classes,
        num_layers=3,
        manifold=None,
        activation=F.relu
    ):
    
        super().__init__()
        self.manifold = manifold or Lorentz(k=1.0)
        
        # Input projection
        self.input_proj = EuclideanToLorentzConv(input_dim, hidden_dim, self.manifold)
        
        # Select residual block type
        self.layer1 = LorentzConv2d(hidden_dim, hidden_dim, 3, 1, 0, manifold, activation)
        
        
        # Classifier
        self.classifier = Lorentz_fully_connected(
            in_features=hidden_dim,
            out_features=num_classes + 1,
            manifold=self.manifold,
            reset_params="kaiming",
            do_mlr=True
        )
    
    def forward(self, x):
        print(x.shape)
        x = self.input_proj(x)
        print(x.shape)
        
        x = self.layer1(x)
        print(x.shape)
        
        return self.classifier(x)