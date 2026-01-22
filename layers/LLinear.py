import torch
import torch.nn as nn
from .lorentz import Lorentz
from einops import rearrange

class LorentzFullyConnected(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        manifold: Lorentz = Lorentz(0.1),
        reset_params="eye",
        a_default=0.0,
        activation=nn.functional.relu,
        do_mlr = False,
        mlr_init: str | None = None,
    ):
        super().__init__()
        self.manifold = manifold
        in_features = in_features - 1
        out_features = out_features - 1
        self.U = nn.Parameter(torch.randn(in_features, out_features))
        self.a = nn.Parameter(torch.zeros(1, out_features))  # -b
        self.V_auxiliary = nn.Parameter(torch.randn(in_features + 1, out_features))
        
        self.activation = activation
        self.do_mlr = do_mlr
        if do_mlr:
            reset_params = mlr_init if mlr_init is not None else "mlr"
        self.reset_parameters(reset_params=reset_params, a_default=a_default)

    def reset_parameters(self, reset_params, a_default):
        in_features, out_features = self.U.shape
        if reset_params == "eye":
            if in_features <= out_features:
                with torch.no_grad():
                    self.U.data.copy_(0.5 * torch.eye(in_features, out_features))
            else:
                with torch.no_grad():
                    self.U.data.copy_(0.5 * torch.eye(in_features, out_features))
            
        elif reset_params == "lorentz_kaiming":
            # For Lorentz models: divide std by 0.5 to account for time coordinate
            std = (1.0 / in_features) ** 0.5
            with torch.no_grad():
                self.U.data.normal_(0, std)
            self.a.data.fill_(a_default)

        elif reset_params == "mlr":
            std = (5.0 / in_features) ** 0.5
            with torch.no_grad():
                self.U.data.normal_(0, std)
            self.a.data.fill_(a_default)

        elif reset_params == "mlr_eye":
            if in_features <= out_features:
                with torch.no_grad():
                    self.U.data.copy_(0.5 * torch.eye(in_features, out_features))
            else:
                with torch.no_grad():
                    self.U.data.copy_(0.5 * torch.eye(in_features, out_features))
            self.a.data.fill_(a_default)
        

        else:
            raise KeyError(f"Unknown reset_params value: {reset_params}")

    def create_spacelike_vector(self):
        U_norm = self.U.norm(dim=0, keepdim=True)
        U_norm_sqrt_k_b = self.manifold.k().sqrt() * self.a * U_norm  # ADD ABLATION HERE
        time = -U_norm * torch.sinh(U_norm_sqrt_k_b)
        space = torch.cosh(U_norm_sqrt_k_b) * self.U
        return torch.cat([time, space], dim=0)

    def signed_dist2hyperplanes_scaled_angle(self, x):
        """Scale the distances by scaling the angle (implicitly)"""
        V = self.create_spacelike_vector()
        sqrt_k = self.manifold.k().sqrt()
        return 1 / sqrt_k * torch.asinh(sqrt_k * x @ V)

    def signed_dist2hyperplanes_scaled_dist(self, x):
        """Scale the distances by scaling the total distance (explicitly)"""
        V = self.create_spacelike_vector()
        V_norm = self.manifold.normL(V.transpose(0, 1)).transpose(0, 1)
        sqrt_k = self.manifold.k().sqrt()
        return V_norm / sqrt_k * torch.asinh(sqrt_k * x @ (V / V_norm))

    def compute_output_space(self, x):
        V = self.create_spacelike_vector()
        return self.activation(x @ V)

    def forward(self, x):
        if self.do_mlr:
            return self.mlr(x)
        output_space = self.compute_output_space(x)
        return self.manifold.projection_space_orthogonal(output_space)

    def forward_cache(self, x):
        output_space = self.activation(x @ self.V_auxiliary)
        return self.manifold.projection_space_orthogonal(output_space)

    def mlr(self, x):
        return self.signed_dist2hyperplanes_scaled_angle(x)
    
    def compute_V_auxiliary(self):
        self.V_auxiliary = torch.nn.Parameter(self.create_spacelike_vector())