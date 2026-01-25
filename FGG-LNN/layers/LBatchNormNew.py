import torch
import torch.nn as nn
from .lorentz import Lorentz
from geoopt import ManifoldParameter


class LorentzBatchNormBase(nn.Module):
    """
    Base class for Lorentz Batch Normalization following Bdeir et al.

    The normalization happens in tangent space:
    1. Compute batch centroid (Fréchet mean) on the manifold
    2. Log-map points to tangent space at centroid
    3. Transport tangent vectors to origin
    4. Scale by gamma / std
    5. Transport to learned center beta
    6. Exp-map back to manifold
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
        self.manifold = manifold or Lorentz(k_value=1.0)
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.fix_gamma = fix_gamma
        self.clamp_scale = clamp_scale
        self.normalize_variance = normalize_variance

        # Learnable scale (positive real)
        if fix_gamma:
            self.register_buffer('gamma', torch.ones((1,)))
        else:
            self.gamma = nn.Parameter(torch.ones((1,)))

        # Learnable center on manifold
        self.beta = ManifoldParameter(
            self.manifold.origin(num_features),
            manifold=self.manifold
        )

        # Running statistics
        self.register_buffer('running_mean', self.manifold.origin(dim=num_features))
        self.register_buffer('running_var', torch.ones(1))

    def _to_flat(self, x: torch.Tensor) -> torch.Tensor:
        """Flatten spatial dimensions. Override in subclasses."""
        raise NotImplementedError

    def _to_spatial(self, x_flat: torch.Tensor, *shape_info) -> torch.Tensor:
        """Restore spatial dimensions. Override in subclasses."""
        raise NotImplementedError

    def _forward_flat(self, x_flat: torch.Tensor) -> torch.Tensor:
        """
        Core batch norm logic on flattened input.

        Args:
            x_flat: [N, C] where N = batch * spatial_dims, C = num_features

        Returns:
            Normalized output [N, C]
        """
        if self.training:
            # Compute batch centroid
            mean = self.manifold.centroid(x_flat).unsqueeze(0)

            # Compute variance if needed
            if self.normalize_variance:
                var = (self.manifold.dist(x_flat, mean, keepdim=False) ** 2).mean()
                div_factor = torch.sqrt(var + self.eps)
            else:
                var = None
                div_factor = 1.0

            # Log-map to tangent space at mean, then transport to origin
            xT_at_mu = self.manifold.logmap(mean, x_flat)
            xT_at_origin = self.manifold.transp_to_origin(mean, xT_at_mu)

            # Scale tangent vectors
            scale = self.gamma / div_factor
            if self.clamp_scale and self.normalize_variance:
                scale = torch.clamp(scale, min=0.5, max=2.0)
            xT_at_origin = xT_at_origin * scale

            # Update running statistics
            with torch.no_grad():
                means = torch.cat([self.running_mean.unsqueeze(0), mean.detach()])
                self.running_mean.copy_(
                    self.manifold.centroid(
                        means,
                        weights=torch.tensor(
                            (1 - self.momentum, self.momentum),
                            device=means.device
                        ),
                    )
                )
                if self.normalize_variance and var is not None:
                    self.running_var.copy_(
                        (1 - self.momentum) * self.running_var + self.momentum * var.detach()
                    )
        else:
            # Use running statistics
            xT_at_running_mu = self.manifold.logmap(self.running_mean, x_flat)
            xT_at_origin = self.manifold.transp_to_origin(self.running_mean, xT_at_running_mu)

            if self.normalize_variance:
                div_factor = torch.sqrt(self.running_var + self.eps)
            else:
                div_factor = 1.0

            scale = self.gamma / div_factor
            if self.clamp_scale and self.normalize_variance:
                scale = torch.clamp(scale, min=0.5, max=2.0)
            xT_at_origin = xT_at_origin * scale

        # Transport to beta and exp-map back to manifold
        xT_at_beta = self.manifold.transp_from_origin(xT_at_origin, self.beta)
        output = self.manifold.expmap(self.beta, xT_at_beta)

        return output


class LorentzBatchNorm1d(LorentzBatchNormBase):
    """
    1D Lorentz Batch Normalization for sequence data.

    Input shape: [B, C, L] where C = num_features (including time component)
    """

    def _to_flat(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, L] -> [B*L, C]"""
        return x.permute(0, 2, 1).reshape(-1, x.shape[1])

    def _to_spatial(self, x_flat: torch.Tensor, batch: int, L: int) -> torch.Tensor:
        """[B*L, C] -> [B, C, L]"""
        return x_flat.view(batch, L, -1).permute(0, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, C, L = x.shape
        x_flat = self._to_flat(x)
        output_flat = self._forward_flat(x_flat)
        return self._to_spatial(output_flat, batch, L)


class LorentzBatchNorm2d(LorentzBatchNormBase):
    """
    2D Lorentz Batch Normalization for image data.

    Input shape: [B, C, H, W] where C = num_features (including time component)
    """

    def _to_flat(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, H, W] -> [B*H*W, C]"""
        return x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])

    def _to_spatial(self, x_flat: torch.Tensor, batch: int, H: int, W: int) -> torch.Tensor:
        """[B*H*W, C] -> [B, C, H, W]"""
        return x_flat.view(batch, H, W, -1).permute(0, 3, 1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, C, H, W = x.shape
        x_flat = self._to_flat(x)
        output_flat = self._forward_flat(x_flat)
        return self._to_spatial(output_flat, batch, H, W)
