"""
Hyperbolic CNN model for genomic sequence classification.
Based on HGE's HyperbolicCNN but using FGG-LNN's Lorentz layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys

# Add parent directory to path for layers import
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from layers.lorentz import Lorentz
from layers.LLinear import LorentzFullyConnected, LorentzMLR
from layers.LConv1d import LorentzConv1d, LorentzReLU
from layers.LBatchNormNew import LorentzBatchNorm1d


def get_hyperbolic_conv_block(
    manifold: Lorentz,
    in_channels: int,
    out_channels: int,
    kernel_size: int = 9,
    padding: int = 4,
    use_bn: bool = True,
    fix_gamma: bool = False,
    clamp_scale: bool = False,
    normalize_variance: bool = True,
    use_weight_norm: bool = False,
):
    """Creates a hyperbolic convolution block: Conv -> BN -> ReLU -> Conv -> BN"""
    layers = [
        LorentzConv1d(
            manifold=manifold,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            use_weight_norm=use_weight_norm,
        ),
    ]
    if use_bn:
        layers.append(
            LorentzBatchNorm1d(
                manifold=manifold,
                num_features=out_channels,
                fix_gamma=fix_gamma,
                clamp_scale=clamp_scale,
                normalize_variance=normalize_variance,
            )
        )
    layers.append(LorentzReLU(manifold=manifold))
    layers.append(
        LorentzConv1d(
            manifold=manifold,
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            use_weight_norm=use_weight_norm,
        )
    )
    if use_bn:
        layers.append(
            LorentzBatchNorm1d(
                manifold=manifold,
                num_features=out_channels,
                fix_gamma=fix_gamma,
                clamp_scale=clamp_scale,
                normalize_variance=normalize_variance,
            )
        )

    return nn.Sequential(*layers)


class GenomeHyperbolicCNN(nn.Module):
    """
    Hyperbolic CNN for genomic sequence classification.

    Architecture:
    - DNA one-hot (5 channels) -> pad to 6 -> project to Lorentz manifold
    - Multiple conv blocks with residual connections
    - Lorentz flatten (direct concatenation)
    - Fully connected layer
    - MLR classifier
    """

    def __init__(
        self,
        num_classes: int,
        length: int,
        model_dim: int = 32,
        fc_dim: int = 528,
        num_layers: int = 3,
        kernel_size: int = 9,
        learnable_k: bool = False,
        k: float = 1.0,
        use_bn: bool = True,
        fix_gamma: bool = False,
        clamp_scale: bool = False,
        normalize_variance: bool = True,
        mlr_type: str = "fc_mlr",  # "lorentz_mlr" or "fc_mlr"
        use_weight_norm: bool = False,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.output_length = length
        self.mlr_type = mlr_type

        # Create manifold
        self.manifold = Lorentz(k_value=k, requires_grad=learnable_k)

        # Padding for convolution (same padding)
        padding = kernel_size // 2

        # Initial conv block: 6 channels (5 DNA + 1 time) -> model_dim
        initial_in_channels = 6  # 5 DNA bases + 1 time component
        self.conv_layers = nn.ModuleList([
            get_hyperbolic_conv_block(
                manifold=self.manifold,
                in_channels=initial_in_channels,
                out_channels=model_dim,
                kernel_size=kernel_size,
                padding=padding,
                use_bn=use_bn,
                fix_gamma=fix_gamma,
                clamp_scale=clamp_scale,
                normalize_variance=normalize_variance,
                use_weight_norm=use_weight_norm,
            )
        ])

        # Subsequent conv blocks: model_dim -> model_dim
        for _ in range(num_layers - 1):
            self.conv_layers.append(
                get_hyperbolic_conv_block(
                    manifold=self.manifold,
                    in_channels=model_dim,
                    out_channels=model_dim,
                    kernel_size=kernel_size,
                    padding=padding,
                    use_bn=use_bn,
                    fix_gamma=fix_gamma,
                    clamp_scale=clamp_scale,
                    normalize_variance=normalize_variance,
                    use_weight_norm=use_weight_norm,
                )
            )

        # Activation for residual connections
        self.activations = nn.ModuleList([
            LorentzReLU(manifold=self.manifold) for _ in range(num_layers)
        ])

        # FC layer after flattening
        # After flattening: concat_dim = 1 + (model_dim - 1) * length
        flatten_dim = 1 + (model_dim - 1) * length
        self.fc_layer = LorentzFullyConnected(
            manifold=self.manifold,
            in_features=flatten_dim,
            out_features=fc_dim,
            use_weight_norm=use_weight_norm,
        )
        self.fc_activation = LorentzReLU(manifold=self.manifold)

        # Classifier head
        if mlr_type == "lorentz_mlr":
            self.classifier = LorentzMLR(
                manifold=self.manifold,
                num_features=fc_dim,
                num_classes=num_classes,
            )
        else:  # fc_mlr
            self.classifier = LorentzFullyConnected(
                manifold=self.manifold,
                in_features=fc_dim,
                out_features=num_classes + 1,  # +1 for Lorentz format
                do_mlr=True,
                use_weight_norm=use_weight_norm,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, 5, length] one-hot encoded DNA sequences (5 = ACGTN)

        Returns:
            logits: [batch, num_classes]
        """
        # Permute to [batch, length, 5] for manifold operations
        x = x.permute(0, 2, 1)

        # Pad with zero to add time dimension: [batch, length, 6]
        x = F.pad(x, pad=(1, 0))

        # Project to Lorentz manifold by computing time component
        x = self.manifold.projx(x, dim=-1)

        # Permute back to [batch, 6, length] for conv layers
        x = x.permute(0, 2, 1)

        # Apply conv layers with residual connections
        res = None
        for i in range(self.num_layers):
            out = self.conv_layers[i](x)

            if i > 0 and res is not None:
                # Residual connection on space dimensions only
                out_space = out[:, 1:, :] + res[:, 1:, :]
                # Recompute time from space
                out = self.manifold.projection_space_orthogonal(out_space, manifold_dim=1)

            x = self.activations[i](out)
            res = x

        # Flatten: [batch, channels, length] -> [batch, 1 + (channels-1) * length]
        # Use direct_concat to properly combine all positions on the manifold
        batch, C, L = x.shape

        # Permute to [batch, length, channels] for direct_concat
        x = x.permute(0, 2, 1)  # [batch, length, channels]

        # direct_concat expects [batch, num_points, dim]
        # It will combine L points of dim C into 1 point of dim 1 + (C-1)*L
        x = self.manifold.direct_concat(x)  # [batch, 1 + (C-1)*L]

        # FC layer
        x = self.fc_layer(x)
        x = self.fc_activation(x)

        # Classifier
        logits = self.classifier(x)

        # Handle LorentzMLR output format
        if self.mlr_type == "lorentz_mlr":
            # LorentzMLR returns [batch, num_classes + 1], drop last dim
            logits = logits[:, :-1]

        return logits


class EuclideanCNN(nn.Module):
    """
    Euclidean baseline CNN for genomic sequence classification.
    """

    def __init__(
        self,
        num_classes: int,
        length: int,
        model_dim: int = 32,
        fc_dim: int = 528,
        num_layers: int = 3,
        kernel_size: int = 9,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.output_length = length

        padding = kernel_size // 2

        # Initial conv block
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(5, model_dim, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm1d(model_dim),
                nn.ReLU(inplace=True),
                nn.Conv1d(model_dim, model_dim, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm1d(model_dim),
            )
        ])

        for _ in range(num_layers - 1):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(model_dim, model_dim, kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm1d(model_dim),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(model_dim, model_dim, kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm1d(model_dim),
                )
            )

        self.activations = nn.ModuleList([
            nn.ReLU(inplace=True) for _ in range(num_layers + 1)
        ])

        self.fc_layer = nn.Linear(length * model_dim, fc_dim)
        self.classifier = nn.Linear(fc_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, 5, length] one-hot encoded DNA sequences

        Returns:
            logits: [batch, num_classes]
        """
        res = None
        for i in range(self.num_layers):
            out = self.conv_layers[i](x)

            if i > 0 and res is not None:
                out = out + res

            x = self.activations[i](out)
            res = x

        # Flatten
        x = x.view(x.shape[0], -1)

        # FC layers
        x = self.fc_layer(x)
        x = self.activations[-1](x)
        x = self.classifier(x)

        return x
