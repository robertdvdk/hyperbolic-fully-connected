from turtle import back
import torch
import torch.nn as nn
from typing import Optional, Type
from .lorentz import Lorentz
from .LLinear import LorentzFullyConnected, LorentzMLR
from .LProjection import EuclideanToLorentzConv
from .LResBlock import LorentzResBlock


class LorentzResNet(nn.Module):
    """ResNet architecture in Lorentz space."""

    def __init__(
        self,
        input_dim: int = 3,
        base_dim: int = 64,
        num_classes: int = 100,
        layers: list[int] = [2, 2, 2, 2],  # ResNet18
        manifold: Optional[Lorentz] = None,
        init_method: str = "kaiming",
        input_proj_type: str = "conv_bn_relu",
        mlr_init: str = "mlr",
        normalisation_mode: str = "normal",  # "normal", "fix_gamma", "skip_final_bn2", "clamp_scale", or "mean_only"
    ):
        """
        Args:
            normalisation_mode: Controls normalization behavior to prevent gamma explosion with MLR.
                - "normal": Default behavior (learnable gamma in all BatchNorms)
                - "fix_gamma": Fix gamma=1 everywhere (forces linear layers to handle scaling)
                - "skip_final_bn2": Skip bn2 in the last ResBlock before classifier
                - "clamp_scale": Clamp BN scale to [0.5, 2.0] without fixing gamma
                - "mean_only": Mean-only normalization (no variance scaling)
        """
        super().__init__()
        self.manifold = manifold or Lorentz(k_value=1.0)
        self.base_dim = base_dim
        self.init_method = init_method
        self.normalisation_mode = normalisation_mode

        # Determine BatchNorm settings based on mode
        self.fix_gamma = (normalisation_mode == "fix_gamma")
        self.skip_final_bn2 = (normalisation_mode == "skip_final_bn2")
        self.clamp_scale = (normalisation_mode == "clamp_scale")
        self.normalize_variance = (normalisation_mode != "mean_only")

        # Initial projection: 3 -> 64
        self.input_proj = EuclideanToLorentzConv(
            input_dim,
            base_dim + 1,
            self.manifold,
            proj_type=input_proj_type,
            fix_gamma=self.fix_gamma,
            clamp_scale=self.clamp_scale,
            normalize_variance=self.normalize_variance,
        )

        # ResNet stages
        self.stage1 = self._make_stage(
            base_dim + 1,
            base_dim + 1,
            layers[0],
            stride=1,
        )
        self.stage2 = self._make_stage(
            base_dim + 1,
            base_dim * 2 + 1,
            layers[1],
            stride=2,
        )
        self.stage3 = self._make_stage(
            base_dim * 2 + 1,
            base_dim * 4 + 1,
            layers[2],
            stride=2,
        )
        self.stage4 = self._make_stage(
            base_dim * 4 + 1,
            base_dim * 8 + 1,
            layers[3],
            stride=2,
            is_final_stage=True,  # Might skip bn2 in last block
        )

        # Classifier
        # self.classifier = LorentzFullyConnected(
        #     in_features=base_dim * 8 + 1,
        #     out_features=num_classes + 1,
        #     manifold=self.manifold,
        #     reset_params=mlr_init,
        #     do_mlr=True,
        #     mlr_init=mlr_init,
        # )

        self.classifier = LorentzMLR(
            manifold = self.manifold,
            num_features = base_dim * 8 + 1,
            num_classes = num_classes,
        )
    
    def _make_stage(
        self,
        in_dim: int,
        out_dim: int,
        num_blocks: int,
        stride: int,
        is_final_stage: bool = False,
    ) -> nn.Sequential:
        """Create a stage with multiple residual blocks."""
        blocks = []

        # First block handles dimension change and downsampling
        is_last_block = (num_blocks == 1 and is_final_stage)
        blocks.append(
            LorentzResBlock(
                input_dim=in_dim,
                output_dim=out_dim,
                kernel_size=3,
                stride=stride,
                padding=1,
                manifold=self.manifold,
                init_method=self.init_method,
                skip_bn2=(is_last_block and self.skip_final_bn2),
                fix_gamma=self.fix_gamma,
                clamp_scale=self.clamp_scale,
                normalize_variance=self.normalize_variance,
            )
        )

        # Remaining blocks maintain dimensions
        for i in range(1, num_blocks):
            is_last_block = (i == num_blocks - 1) and is_final_stage
            blocks.append(
                LorentzResBlock(
                    input_dim=out_dim,
                    output_dim=out_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    manifold=self.manifold,
                    init_method=self.init_method,
                    skip_bn2=(is_last_block and self.skip_final_bn2),
                    fix_gamma=self.fix_gamma,
                    clamp_scale=self.clamp_scale,
                    normalize_variance=self.normalize_variance,
                )
            )

        return nn.Sequential(*blocks)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input projection
        x = self.input_proj(x)
        
        # ResNet stages
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        # Global pooling
        x = self._global_pool(x)
        
        # Classification
        return self.classifier(x)
    
    def _global_pool(self, x: torch.Tensor) -> torch.Tensor:
        """Apply global average pooling in Lorentz space."""
        # Spatial dimensions should be 1x1 after all downsampling
        x = x.squeeze(-1).squeeze(-1)
        
        # Handle case where spatial dims aren't 1x1
        if len(x.shape) == 2:
            return x
        else:
            # Reshape and compute Lorentz midpoint across spatial locations
            x = x.view(x.shape[0], x.shape[1], -1)  # [B, C, H*W]
            x = x.permute(0, 2, 1)  # [B, H*W, C]
            x = self.manifold.lorentz_midpoint(x)  # [B, C]
            return x
        
    def compute_V_auxiliary(self):
        def _maybe_compute(module: nn.Module) -> None:
            if module is self:
                return
            method = getattr(module, "compute_V_auxiliary", None)
            if callable(method):
                method()

        self.apply(_maybe_compute)


# Factory functions for common architectures
def lorentz_resnet18(num_classes: int = 100, base_dim: int = 64, **kwargs) -> LorentzResNet:
    """ResNet-18 in Lorentz space."""
    return LorentzResNet(
        num_classes=num_classes,
        base_dim=base_dim,
        layers=[2, 2, 2, 2],
        **kwargs
    )