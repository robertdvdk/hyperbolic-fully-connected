import torch.nn as nn

from layers.lorentz import Lorentz
from layers.LLinear import LorentzFullyConnected
from layers.bdeir import BdeirLorentzMLR
from layers.poincare import Poincare, Poincare_distance2hyperplanes


class Baseline_Euclidean(nn.Module):
    """Baseline Euclidean model for comparison."""

    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        return self.linear(x)


class Lorentz_fully_connected_forward(nn.Module):
    """A fully connected layer in the Lorentz model. with identity activation."""

    # the activation is the identity
    def __init__(
        self,
        in_features,
        out_features,
        k=0.1,
        reset_params="kaiming",
        activation=nn.Identity(),
    ):
        super().__init__()
        self.manifold = Lorentz(k)
        self.linear = LorentzFullyConnected(
            in_features, out_features, self.manifold, reset_params, activation
        )

    def forward(self, x):
        x = self.manifold.expmap0(x)
        return self.linear.compute_output_space(x)


class Lorentz_fully_connected_mlr_angle(nn.Module):
    """A fully connected layer in the Lorentz model. with identity activation."""

    # the activation is the identity
    def __init__(
        self,
        in_features,
        out_features,
        k=0.1,
        reset_params="kaiming",
        activation=nn.Identity(),
    ):
        super().__init__()
        self.manifold = Lorentz(k)
        self.linear = LorentzFullyConnected(
            in_features, out_features, self.manifold, reset_params, activation
        )

    def forward(self, x):
        x = self.manifold.expmap0(x)
        return self.linear.signed_dist2hyperplanes_scaled_angle(x)


class Lorentz_fully_connected_mlr_dist(nn.Module):
    """A fully connected layer in the Lorentz model. with identity activation."""

    # the activation is the identity
    def __init__(
        self,
        in_features,
        out_features,
        k=0.1,
        reset_params="kaiming",
        activation=nn.Identity(),
    ):
        super().__init__()
        self.manifold = Lorentz(k)
        self.linear = LorentzFullyConnected(
            in_features, out_features, self.manifold, reset_params, activation
        )

    def forward(self, x):
        x = self.manifold.expmap0(x)
        return self.linear.signed_dist2hyperplanes_scaled_dist(x)


class BdeirLorentzMLR_model(nn.Module):
    """A multinomial logistic regression layer in the Lorentz model."""

    def __init__(self, in_features: int, out_features: int, k=0.1):
        super().__init__()
        self.manifold = Lorentz(k)
        self.mlr = BdeirLorentzMLR(in_features + 1, out_features, self.manifold)

    def forward(self, x):
        x = self.manifold.expmap0(x)
        return self.mlr(x)


class Poincare_MLR_Shimizu_van_Spengler(nn.Module):
    """A multinomial logistic regression layer in the Poincaré model."""

    def __init__(self, in_features: int, out_features: int, k=0.1):
        super().__init__()
        self.manifold = Poincare(k)
        self.mlr = Poincare_distance2hyperplanes(
            in_features, out_features, self.manifold
        )

    def forward(self, x):
        x = self.manifold.expmap0(x)
        return self.mlr(x)


# Lorentz_fully_connected(10, 5, manifold=Lorentz(0.1)) # accepts input of shape [batch_size, 10+1] and outputs [batch_size, 5+1]
# BdeirLorentzMLR(num_features=10, num_classes=5, manifold=Lorentz(0.1)) # accepts input of shape [batch_size, 10] and outputs [batch_size, 5]
# ChenLinear(manifold=Lorentz(0.1), in_features=10, out_features=5) # accepts input of shape [batch_size, 10] and outputs [batch_size, 5]
# Poincare_linear(10, 5, manifold=Poincare(0.1)) # accepts input of shape [batch_size, 10] and outputs [batch_size, 5]
