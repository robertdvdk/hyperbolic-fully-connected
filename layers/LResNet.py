import torch
import torch.nn as nn
from typing import Optional, Type
from .lorentz import Lorentz
from .LLinear import LorentzMLR, resolve_lorentz_fc_class
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
        normalisation_mode: str = "normal",  # "normal", "fix_gamma", "skip_final_bn2", "clamp_scale", "mean_only", or "centering_only"
        mlr_type: str = "lorentz_mlr",  # "lorentz_mlr" or "fc_mlr"
        use_weight_norm: bool = False,
        fc_variant: str = "ours",
        embedding_dim: Optional[int] = None,
    ):
        """
        Args:
            normalisation_mode: Controls normalization behavior to prevent gamma explosion with MLR.
                - "normal": Default behavior (learnable gamma in all BatchNorms)
                - "fix_gamma": Fix gamma=1 everywhere (forces linear layers to handle scaling)
                - "skip_final_bn2": Skip bn2 in the last ResBlock before classifier
                - "clamp_scale": Clamp BN scale to [0.5, 2.0] without fixing gamma
                - "mean_only": Mean-only normalization (no variance scaling)
                - "centering_only": Mean-only normalization with fixed gamma=1
            mlr_type: Choose classifier head implementation.
                - "lorentz_mlr": Use LorentzMLR
                - "fc_mlr": Use LorentzFullyConnected with do_mlr=True
        """
        super().__init__()
        self.manifold = manifold or Lorentz(k_value=1.0)
        self.base_dim = base_dim
        self.init_method = init_method
        self.normalisation_mode = normalisation_mode
        self.use_weight_norm = use_weight_norm
        self.fc_variant = fc_variant
        self.embedding_dim = embedding_dim
        if self.embedding_dim is not None and self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive when set")

        # Determine BatchNorm settings based on mode
        self.fix_gamma = (normalisation_mode in {"fix_gamma", "centering_only"})
        self.skip_final_bn2 = (normalisation_mode == "skip_final_bn2")
        self.clamp_scale = (normalisation_mode == "clamp_scale")
        self.normalize_variance = (normalisation_mode not in {"mean_only", "centering_only"})

        # Initial projection: 3 -> 64
        self.input_proj = EuclideanToLorentzConv(
            input_dim,
            base_dim + 1,
            self.manifold,
            proj_type=input_proj_type,
            fix_gamma=self.fix_gamma,
            clamp_scale=self.clamp_scale,
            normalize_variance=self.normalize_variance,
            init_method=self.init_method,
            use_weight_norm=self.use_weight_norm,
            fc_variant=self.fc_variant,
        )

        # ResNet stages
        self.stage1 = self._make_stage(
            base_dim + 1,
            base_dim + 1,
            layers[0],
            stride=1,
            use_weight_norm=self.use_weight_norm,
            fc_variant=self.fc_variant,
        )
        self.stage2 = self._make_stage(
            base_dim + 1,
            base_dim * 2 + 1,
            layers[1],
            stride=2,
            use_weight_norm=self.use_weight_norm,
            fc_variant=self.fc_variant,
        )
        self.stage3 = self._make_stage(
            base_dim * 2 + 1,
            base_dim * 4 + 1,
            layers[2],
            stride=2,
            use_weight_norm=self.use_weight_norm,
            fc_variant=self.fc_variant,
        )
        final_dim = self.embedding_dim if self.embedding_dim is not None else base_dim * 8
        self.stage4 = self._make_stage(
            base_dim * 4 + 1,
            final_dim + 1,
            layers[3],
            stride=2,
            is_final_stage=True,  # Might skip bn2 in last block
            use_weight_norm=self.use_weight_norm,
            fc_variant=self.fc_variant,
        )

        # Classifier
        if mlr_type == "fc_mlr":
            fc_cls = resolve_lorentz_fc_class(self.fc_variant)
            self.classifier = fc_cls(
                in_features=final_dim + 1,
                out_features=num_classes + 1,
                manifold=self.manifold,
                reset_params=mlr_init,
                do_mlr=True,
                mlr_init=mlr_init,
            )
        elif mlr_type == "lorentz_mlr":
            self.classifier = LorentzMLR(
                manifold=self.manifold,
                num_features=final_dim + 1,
                num_classes=num_classes,
            )
        else:
            raise ValueError(f"Unknown mlr_type: {mlr_type}")
    
    def _make_stage(
        self,
        in_dim: int,
        out_dim: int,
        num_blocks: int,
        stride: int,
        is_final_stage: bool = False,
        use_weight_norm: bool = False,
        fc_variant: str = "ours",
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
                use_weight_norm=use_weight_norm,
                fc_variant=fc_variant,
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
                    use_weight_norm=use_weight_norm,
                    fc_variant=fc_variant,
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
        x = x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, x.shape[1])
        x = self.classifier(x)
        x = x.mean(dim=1)
        
        # Classification
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