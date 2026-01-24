import torch
import torch.nn as nn
from layers import Lorentz
from geoopt import ManifoldParameter

class LorentzBatchNorm2d(nn.Module):
    """
    Lorentz Batch Normalization following Bdeir et al.
    Simplified to use manifold primitives.
    """
    
    def __init__(
        self,
        num_features: int,
        manifold: Lorentz = None,
        momentum: float = 0.1,
        eps: float = 1e-5,
        fix_gamma: bool = False,
        clamp_scale: bool = False,
        normalize_variance: bool = True,
    ):
        super().__init__()
        self.manifold = manifold or Lorentz(k=1.0)
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.fix_gamma = fix_gamma
        self.clamp_scale = clamp_scale
        self.normalize_variance = normalize_variance
        
        # Learnable scale (positive real)
        if fix_gamma:
            # Fixed gamma=1, not learnable (forces linear layers to handle scaling)
            self.register_buffer('gamma', torch.ones((1,)))
        else:
            self.gamma = torch.nn.Parameter(torch.ones((1,)))
        
        # Learnable shift (space components, will be projected to manifold)
        self.beta = ManifoldParameter(self.manifold.origin(num_features), manifold=self.manifold)
        
        # Running statistics (store space components of centroid)
        self.register_buffer('running_mean', self.manifold.origin(dim=num_features))
        self.register_buffer('running_var', torch.ones(1))
    
    def _to_flat(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, H, W] -> [B*H*W, C]"""
        return x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])
    
    def _to_spatial(self, x_flat: torch.Tensor, batch: int, H: int, W: int) -> torch.Tensor:
        """[B*H*W, C] -> [B, C, H, W]"""
        return x_flat.view(batch, H, W, -1).permute(0, 3, 1, 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, C, H, W = x.shape
        x_flat = self._to_flat(x)  # [N, C]

        if self.training:
            # Calculate mean and variance
            mean = self.manifold.centroid(x_flat).unsqueeze(0)
            if self.normalize_variance:
                var = (self.manifold.dist(x_flat, mean, keepdim=False) ** 2).mean()
                div_factor = torch.sqrt(var + self.eps)
            else:
                div_factor = 1.0

            # Get tangent vectors, and transport them to the origin (analogous to subtracting the mean in Euclidean space)
            xT_at_mu = self.manifold.logmap(mean, x_flat)  # [N, C]
            xT_at_origin = self.manifold.transp_to_origin(mean, xT_at_mu)

            # Divide tangent vectors by std and multiply by learned std
            scale = self.gamma / div_factor
            if self.clamp_scale and self.normalize_variance:
                scale = torch.clamp(scale, min=0.5, max=2.0)
            xT_at_origin = xT_at_origin * scale

            with torch.no_grad():
                means = torch.cat([self.running_mean.unsqueeze(0), mean.detach()])
                self.running_mean.copy_(self.manifold.centroid(
                            means,
                            weights=torch.tensor(((1 - self.momentum), self.momentum), device=means.device),
                        ))
                if self.normalize_variance:
                    self.running_var.copy_((1 - self.momentum)*self.running_var + self.momentum*var.detach())
        else:
            # Get tangent vectors at running mean, and transport them to the origin (analogous to subtracting the running mean in Euclidean space)
            xT_at_running_mu = self.manifold.logmap(self.running_mean, x_flat)
            xT_at_origin = self.manifold.transp_to_origin(self.running_mean, xT_at_running_mu)

            # Divide tangent vectors by running std and multiply by learned std
            if self.normalize_variance:
                div_factor = torch.sqrt(self.running_var + self.eps)
            else:
                div_factor = 1.0
            scale = self.gamma / div_factor
            if self.clamp_scale and self.normalize_variance:
                scale = torch.clamp(scale, min=0.5, max=2.0)
            xT_at_origin = xT_at_origin * scale

        # Transport tangent vectors to beta, and expmap at beta (analogous to adding the mean in Euclidean space)
        xT_at_beta = self.manifold.transp_from_origin(xT_at_origin, self.beta)
        output = self.manifold.expmap(self.beta, xT_at_beta)
        output = self._to_spatial(output, batch, H, W)


        return output
    