import torch
import torch.nn as nn
import torch.nn.functional as F
from .lorentz import Lorentz
from einops import rearrange
import math

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
        use_weight_norm: bool = False,
    ):
        super().__init__()
        self.manifold = manifold
        self.use_weight_norm = use_weight_norm
        in_features = in_features - 1
        out_features = out_features - 1
        if self.use_weight_norm:
            self.v = nn.Parameter(torch.randn(in_features, out_features))
            self.g = nn.Parameter(torch.ones(out_features))
            self.U = None
        else:
            self.U = nn.Parameter(torch.randn(in_features, out_features))
            self.v = None
            self.g = None
        self.a = nn.Parameter(torch.zeros(1, out_features))  # -b
        self.V_auxiliary = nn.Parameter(torch.randn(in_features + 1, out_features))
        
        self.activation = activation
        self.do_mlr = do_mlr
        if do_mlr:
            reset_params = mlr_init if mlr_init is not None else "mlr"
        self.reset_parameters(reset_params=reset_params, a_default=a_default)

    def get_U(self):
        if not self.use_weight_norm:
            return self.U
        v_norm = self.v.norm(dim=0, keepdim=True).clamp(min=1e-8)
        g_pos = F.softplus(self.g)
        return g_pos.unsqueeze(0) * self.v / v_norm

    def reset_parameters(self, reset_params, a_default):
        if self.use_weight_norm:
            in_features, _ = self.v.shape
            # Initialize direction randomly
            nn.init.kaiming_normal_(self.v)
            # Initialize magnitude based on desired init scheme
            if reset_params == "lorentz_kaiming":
                self.g.data.fill_((1.0 / in_features) ** 0.5)
            else:
                self.g.data.fill_(1.0)
            self.a.data.fill_(a_default)
            return

        in_features, out_features = self.U.shape
        if reset_params == "eye":
            if in_features <= out_features:
                with torch.no_grad():
                    self.U.data.copy_(0.5 * torch.eye(in_features, out_features))
            else:
                with torch.no_grad():
                    self.U.data.copy_(0.5 * torch.eye(in_features, out_features))

        elif reset_params == "xavier":
            std = (1.0 / (in_features + out_features)) ** 0.5
            with torch.no_grad():
                self.U.data.normal_(0, std)
        elif reset_params == "lorentz_kaiming":
            std = (1.0 / in_features) ** 0.5
            with torch.no_grad():
                self.U.data.normal_(0, std)

        elif reset_params == "kaiming":
            # For Lorentz models: divide std by 0.5 to account for time coordinate
            std = (2.0 / in_features) ** 0.5
            with torch.no_grad():
                self.U.data.normal_(0, std)

        elif reset_params == "mlr":
            std = (5.0 / in_features) ** 0.5
            with torch.no_grad():
                self.U.data.normal_(0, std)

        elif reset_params == "mlr_eye":
            if in_features <= out_features:
                with torch.no_grad():
                    self.U.data.copy_(0.5 * torch.eye(in_features, out_features))
            else:
                with torch.no_grad():
                    self.U.data.copy_(0.5 * torch.eye(in_features, out_features))

        else:
            raise KeyError(f"Unknown reset_params value: {reset_params}")

        self.a.data.fill_(a_default)

    def create_spacelike_vector(self):
        U = self.get_U()
        U_norm = U.norm(dim=0, keepdim=True)
        U_norm_sqrt_k_b = self.manifold.k().sqrt() * self.a / U_norm
        time = -U_norm * torch.sinh(U_norm_sqrt_k_b)
        space = torch.cosh(U_norm_sqrt_k_b) * U
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


class LorentzMLR(nn.Module):
    """ Multinomial logistic regression (MLR) in the Lorentz model
    """
    def __init__(
            self, 
            manifold: Lorentz, 
            num_features: int, 
            num_classes: int
        ):
        super(LorentzMLR, self).__init__()

        self.manifold = manifold

        self.a = torch.nn.Parameter(torch.zeros(num_classes + 1,))
        self.z = torch.nn.Parameter(F.pad(torch.zeros(num_classes + 1, num_features-2), pad=(1,0), value=1)) # z should not be (0,0)

        self.init_weights()

    def forward(self, x):
        # Hyperplane
        sqrt_mK = 1/self.manifold.k().sqrt()
        norm_z = torch.norm(self.z, dim=-1)
        w_t = (torch.sinh(sqrt_mK*self.a)*norm_z)
        w_s = torch.cosh(sqrt_mK*self.a.view(-1,1))*self.z
        beta = torch.sqrt(-w_t**2+torch.norm(w_s, dim=-1)**2)
        alpha = -w_t*x.narrow(-1, 0, 1) + (torch.cosh(sqrt_mK*self.a)*torch.inner(x.narrow(-1, 1, x.shape[-1]-1), self.z))

        d = self.manifold.k().sqrt()*torch.abs(torch.asinh(sqrt_mK*alpha/beta))  # Distance to hyperplane
        logits = torch.sign(alpha)*beta*d

        return logits
        
    def init_weights(self):
        stdv = 1. / math.sqrt(self.z.size(1))
        nn.init.uniform_(self.z, -stdv, stdv)
        nn.init.uniform_(self.a, -stdv, stdv)