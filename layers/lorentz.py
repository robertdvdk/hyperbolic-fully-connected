import torch
import torch.nn as nn
import torch.nn.functional as F


class Lorentz(nn.Module):
    """Lorentz model for hyperbolic geometry.
    The Lorentz model is defined as the hyperboloid in Minkowski space.
    The manifold is defined by the equation:
        -x_0^2 + x_1^2 + ... + x_n^2 = -1/k"""

    def __init__(
        self, k: float = 0.1, requires_grad=False, constraining_strategy=nn.Identity()
    ):
        super().__init__()
        k_value = torch.log(torch.exp(torch.tensor(k)) - 1)
        self.c_softplus_inv = nn.Parameter(k_value, requires_grad=requires_grad)
        self.constraining_strategy = constraining_strategy

    def k(self):
        """Returns the negative curvature of the Lorentz model."""
        return F.softplus(self.c_softplus_inv)

    def forward(self, x):
        return self.expmap0(x)

    def normL(self, x, metric=None):
        if metric is None:
            metric = torch.ones(x.size(-1), device=x.device, dtype=x.dtype)
        metric[0] = -1

        return (x * x * metric).sum(dim=-1, keepdim=True).sqrt()

    def expmap0(self, x):
        """
        Maps tangent vectors from the origin of the tangent space T_0 H^n_k
        to the Lorentz hyperboloid H^n_k.
        Handles the case where the input vector norm is zero.
        """
        sqrt_k = self.k() ** 0.5
        norm_x = torch.norm(x, dim=-1, keepdim=True)

        eps = 1e-9
        is_zero = norm_x < eps

        sqrt_k_norm_x = norm_x * sqrt_k
        time = 1.0 / sqrt_k * torch.cosh(sqrt_k_norm_x)

        factor = torch.where(
            is_zero, torch.ones_like(norm_x), torch.sinh(sqrt_k_norm_x) / sqrt_k_norm_x
        )
        space = factor * x
        return torch.cat([time, space], dim=-1)

    def logmap0(self, y):
        """
        Logarithmic map from the origin for the Lorentz model.

        Args:
            y: Point on the hyperboloid

        Returns:
            Tangent vector at the origin that maps to y under expmap0
        """
        k = self.k()
        sqrt_k = k**0.5

        y_time = y[..., :1]  # First component (time)
        y_space = y[..., 1:]  # Remaining components (space)

        # A small epsilon to avoid instability when y is close to the origin.
        eps = 1e-9

        # Calculate the factor based on the formula
        # arccosh(sqrt(k) * y_time) / sqrt((sqrt(k) * y_time)^2 - 1)
        # The argument to sqrt can be negative due to floating point errors, so clamp at 0.
        norm_y_space_sq = torch.sum(y_space * y_space, dim=-1, keepdim=True)
        denominator_sqrt = torch.sqrt(torch.clamp(k * norm_y_space_sq, min=eps))

        factor = torch.acosh(sqrt_k * y_time) / denominator_sqrt

        # Compute the tangent vector (0, y_space) scaled by the factor
        return factor * y_space

    def projection_space_orthogonal(self, x):
        """
        Projects a point onto the Lorentz model orthogonally from the space dimensions.

        Args:
            x: Point in the Lorentz model dim [batch_size, dim]
        Returns:
            Projected point dim [batch_size, dim]
        """
        return torch.cat(
            [torch.sqrt(1 / self.k() + x.pow(2).sum(-1, keepdim=True)), x], dim=-1
        )

    def poincare_to_lorentz(self, x_poincare: torch.Tensor):
        """
        Converts points from the Poincaré ball model to the Lorentz hyperboloid model.
        The conversion assumes both models share the same curvature parameter k > 0.
        """
        k = self.k()
        sqrt_k = k.sqrt()

        # Calculate the squared Euclidean norm of the Poincaré points
        x_norm_sq = torch.sum(x_poincare * x_poincare, dim=-1, keepdim=True)

        # Denominator for the conversion formula
        # Add epsilon for numerical stability
        denom = 1 - k * x_norm_sq + 1e-9

        # Time component of the Lorentz point
        time_component = (1 / sqrt_k) * (1 + k * x_norm_sq) / denom

        # Space components of the Lorentz point
        space_components = (2 * x_poincare) / denom

        # Concatenate time and space to form the Lorentz point
        return torch.cat([time_component, space_components], dim=-1)
    
    def direct_concat(self, xs: torch.Tensor):
        """
        Perform Lorentz direct concatenation as described by Qu, E., et al. (https://openreview.net/forum?id=NQi9U0YLW3)

        Args:
            xs: list of tensors to concatenate. Assumes the last dimension is the dimension along which the tensor lies on the manifold, and the second last dimension is the dimension to concatenate over.

        
        """
        # Input check
        time_sq = xs.narrow(dim=-1, start=0, length=1) ** 2
        space_sq = xs.narrow(dim=-1, start=1, length=xs.size(-1)-1) ** 2
        lorentz_norm_sq = -time_sq + space_sq.sum(dim=-1, keepdim=True)
        target_norm = -1.0 / self.k()
        assert torch.allclose(lorentz_norm_sq, target_norm, atol=1e-5), f"Input tensors do not lie on the Lorentz manifold. Mean deviation: {torch.abs(lorentz_norm_sq - target_norm).mean().item()}"

        time = torch.sqrt(((xs[..., 0]) ** 2).sum(dim=-1, keepdim=True) - (xs.shape[-2] - 1) / self.k())
        space = xs[..., 1:].reshape(*xs.shape[:-2], -1)
        out = torch.cat([time, space], dim=-1)
        assert torch.allclose(-out[..., 0]**2 + (out[..., 1:]**2).sum(dim=-1), -1.0 / self.k(), atol=1e-5), "Output tensor does not lie on the Lorentz manifold."
        return out
    