import torch
import torch.nn as nn
from .lorentz import Lorentz
from geoopt import ManifoldParameter

class LorentzBatchNorm(nn.Module):
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
    ):
        super().__init__()
        self.manifold = manifold or Lorentz(k=1.0)
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        
        # Learnable scale (positive real)
        self.beta = ManifoldParameter(self.manifold.origin(num_features), manifold=self.manifold)
        self.gamma = nn.Parameter(torch.ones((1,)))
        
        # Running statistics (store space components of centroid)
        self.register_buffer('running_mean', self.manifold.origin(num_features))
        self.register_buffer('running_var', torch.ones(1,))

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        if self.training:
            mean = self.manifold.centroid(x)
            if len(x.shape) == 3:
                mean = self.manifold.centroid(mean)
            origin = self.manifold.origin(C, device=x.device, dtype=x.dtype)
            x_T = self.manifold.logmap(mean, x)
            x_T = self.manifold.parallel_transport(mean, x_T, origin)

            # Compute Fréchet variance
            if len(x.shape) == 3:
                var = torch.mean(torch.norm(x_T, dim=-1), dim=(0,1))
            else:
                var = torch.mean(torch.norm(x_T, dim=-1), dim=0)
            # Rescale batch
            x_T = x_T*(self.gamma/(var+self.eps))


            # Transport batch to learned mean
            x_T = self.manifold.parallel_transport(origin, x_T, self.beta)
            output = self.manifold.expmap(self.beta, x_T)

            with torch.no_grad():
                means = torch.concat((self.running_mean.unsqueeze(0), mean.detach().unsqueeze(0)), dim=0)
                self.running_mean.copy_(self.manifold.centroid(means, weights=torch.tensor(((1 - self.momentum), self.momentum), device=means.device)))
                self.running_var.copy_((1 - self.momentum)*self.running_var + self.momentum*var.detach())
        else:
            # Transport batch to origin (center batch)
            origin = self.manifold.origin(C, device=x.device, dtype=x.dtype)
            x_T = self.manifold.logmap(self.running_mean, x)
            x_T = self.manifold.parallel_transport(self.running_mean, x_T, origin)

            # Rescale batch
            x_T = x_T*(self.gamma/(self.running_var+self.eps))

            # Transport batch to learned mean
            x_T = self.manifold.parallel_transport(origin, x_T, self.beta)
            output = self.manifold.expmap(self.beta, x_T)


        return output
    
class LorentzBatchNorm2d(LorentzBatchNorm):
    """ 2D Lorentz Batch Normalization with Centroid and Fréchet variance
    """
    def __init__(self, manifold: Lorentz, num_features: int):
        super(LorentzBatchNorm2d, self).__init__(num_features, manifold)

    def forward(self, x):
        """ x has to be in channel last representation -> Shape = bs x H x W x C """
        bs, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(bs, -1, c)
        x = super(LorentzBatchNorm2d, self).forward(x)
        x = x.reshape(bs, h, w, c).permute(0, 3, 1, 2)
        return x


# class LorentzBatchNorm2d(nn.Module):
#     """
#     Lorentz Batch Normalization following Bdeir et al.
#     Simplified to use manifold primitives.
#     """
    
#     def __init__(
#         self,
#         num_features: int,
#         manifold: Lorentz = None,
#         momentum: float = 0.1,
#         eps: float = 1e-5,
#     ):
#         super().__init__()
#         self.manifold = manifold or Lorentz(k=1.0)
#         self.num_features = num_features
#         self.momentum = momentum
#         self.eps = eps
        
#         # Learnable scale (positive real)
#         self.gamma = nn.Parameter(torch.ones(1, num_features - 1, 1, 1))
        
#         # Learnable shift (space components, will be projected to manifold)
#         # self.beta_space = nn.Parameter(torch.zeros(1, num_features - 1, 1, 1))
#         self.beta = ManifoldParameter(self.manifold.origin(num_features), manifold=self.manifold)
        
#         # Running statistics (store space components of centroid)
#         self.register_buffer('running_mean_space', torch.zeros(1, num_features - 1, 1, 1))
#         self.register_buffer('running_var', torch.ones(1))
    
#     def _to_flat(self, x: torch.Tensor) -> torch.Tensor:
#         """[B, C, H, W] -> [B*H*W, C]"""
#         return x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])
    
#     def _to_spatial(self, x_flat: torch.Tensor, batch: int, H: int, W: int) -> torch.Tensor:
#         """[B*H*W, C] -> [B, C, H, W]"""
#         return x_flat.view(batch, H, W, -1).permute(0, 3, 1, 2)
    
#     def _compute_centroid(self, x: torch.Tensor) -> torch.Tensor:
#         """Compute Lorentz centroid using manifold method."""
#         batch, C, H, W = x.shape
#         x_flat = self._to_flat(x)  # [N, C]
        
#         # lorentz_midpoint expects [..., num_points, dim]
#         # We want centroid over all N points, so reshape to [1, N, C]
#         centroid = self.manifold.lorentz_midpoint(x_flat.unsqueeze(0))  # [1, C]
        
#         return centroid.view(1, C, 1, 1)
    
#     def _compute_variance(self, x: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
#         """Compute Fréchet variance (mean squared geodesic distance)."""
#         batch, C, H, W = x.shape
#         x_flat = self._to_flat(x)  # [N, C]
#         mean_flat = mean.view(1, C).expand(x_flat.shape[0], -1)  # [N, C]
        
#         # Use manifold distance
#         dist_sq = self.manifold.dist(x_flat, mean_flat, keepdim=False) ** 2
#         return dist_sq.mean()
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         batch, C, H, W = x.shape
        
#         if self.training:
#             mean = self._compute_centroid(x)
#             var = self._compute_variance(x, mean)
            
#             with torch.no_grad():
#                 self.running_mean_space.mul_(1 - self.momentum).add_(
#                     mean[:, 1:, :, :] * self.momentum
#                 )
#                 self.running_var.mul_(1 - self.momentum).add_(var * self.momentum)
#         else:
#             # Reconstruct mean from running space components
#             mean_space_flat = self.running_mean_space.view(1, -1)  # [1, C-1]
#             mean_flat = self.manifold.projection_space_orthogonal(mean_space_flat)  # [1, C]
#             mean = mean_flat.view(1, C, 1, 1)
#             var = self.running_var
        
#         # Flatten for manifold operations
#         x_flat = self._to_flat(x)  # [N, C]
#         mean_flat = mean.view(1, C).expand(x_flat.shape[0], -1)
        
#         # Origin point
#         origin = self.manifold.origin(C).unsqueeze(0).expand(x_flat.shape[0], -1)
        
#         # 1. Log map: get tangent vector at mean pointing to x
#         # logmap expects [batch, dim] for base and [batch, m, dim] for target
#         v_at_mean = self.manifold.logmap(mean_flat, x_flat.unsqueeze(1)).squeeze(1)  # [N, C]
        
#         # 2. Parallel transport tangent vector from mean to origin
#         v_at_origin = self.manifold.parallel_transport(mean_flat, v_at_mean.unsqueeze(1), origin).squeeze(1)
        
#         # 3. Scale in tangent space (only space components, time should stay ~0)
#         # Tangent vectors at origin have time ≈ 0
#         v_space = v_at_origin[:, 1:]  # [N, C-1]
#         gamma_flat = self.gamma.view(1, -1)  # [1, C-1]
#         v_scaled_space = gamma_flat * v_space / (var.sqrt() + self.eps)
        
#         # 4. Add shift (beta)
#         print(v_scaled_space.shape)
#         print(self.beta.shape)
#         print(origin.shape)
#         v_transported = self.manifold.parallel_transport(origin, v_scaled_space, self.beta)
#         output = self.manifold.expmap(self.beta, v_transported)
#         return output

#         # v_shifted_space = v_scaled_space + self.beta
#         # beta_flat = self.beta_space.view(1, -1)  # [1, C-1]
#         # v_shifted_space = v_scaled_space + beta_flat
        
#         # 5. Project back to manifold from space components
#         # x_out_flat = self.manifold.projection_space_orthogonal(v_shifted_space)
        
#         # return self._to_spatial(x_out_flat, batch, H, W)