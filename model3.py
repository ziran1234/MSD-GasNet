"""
Lightweight timm-backed models for gas recognition.

This module adds three efficient backbones:

1. GhostNetV3-1.0
2. StarNet-S2
3. FastViT-T8

They are exposed through small task-specific builders so model_train_test.py can
reuse the same API style as the existing local models.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


GHOSTNETV3_100_ALIASES = (
    "ghostnetv3_100",
    "ghostnetv3_100.in1k",
)
STARNET_S2_ALIASES = (
    "starnet_s2",
    "starnet_s2.in1k",
)
FASTVIT_T8_ALIASES = (
    "fastvit_t8",
    "fastvit_t8.apple_in1k",
)


def _import_timm():
    """Import timm lazily so syntax checks do not require the package."""

    try:
        import timm
    except ImportError as exc:
        raise ImportError(
            "model3.py requires the 'timm' package. Install it in the active "
            "Python environment before building GhostNetV3-1.0, StarNet-S2, "
            "or FastViT-T8."
        ) from exc
    return timm


def _create_backbone(
    candidate_names: Sequence[str],
    in_channels: int,
    use_pretrained: bool,
) -> nn.Module:
    """
    Create one timm backbone and remove its classifier head.

    The candidate list makes the code more tolerant to timm version differences
    in model naming and pretrained tag exposure.
    """

    timm = _import_timm()
    last_error = None
    for candidate_name in candidate_names:
        try:
            backbone = timm.create_model(
                candidate_name,
                pretrained=use_pretrained,
                in_chans=in_channels,
                num_classes=0,
                global_pool="avg",
            )
            backbone.resolved_model_name = candidate_name
            return backbone
        except Exception as exc:
            last_error = exc
    raise ValueError(
        "None of the timm model names could be created: {}. Last error: {}".format(
            list(candidate_names),
            repr(last_error),
        )
    )


def _forward_backbone_features(backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run one backbone forward pass and normalize the output into [B, C]."""

    features = backbone(x)
    if isinstance(features, (tuple, list)):
        if not features:
            raise ValueError("The backbone returned an empty feature collection.")
        features = features[-1]
    if features.ndim == 1:
        features = features.unsqueeze(0)
    elif features.ndim > 2:
        features = torch.flatten(features, start_dim=1)
    return features


def _infer_backbone_feature_dim(
    backbone: nn.Module,
    in_channels: int,
    image_size: int = 64,
) -> int:
    """Infer the actual backbone output dimension with one dummy forward pass."""

    was_training = backbone.training
    backbone.eval()
    with torch.no_grad():
        dummy_input = torch.zeros(1, in_channels, image_size, image_size)
        features = _forward_backbone_features(backbone, dummy_input)
    if was_training:
        backbone.train()
    if features.ndim != 2:
        raise ValueError("Expected a 2D feature tensor after normalization.")
    return int(features.shape[1])


class TimmGasClassificationNet(nn.Module):
    """Generic classification wrapper around a timm backbone."""

    def __init__(
        self,
        candidate_names: Sequence[str],
        num_classes: int = 5,
        in_channels: int = 3,
        use_pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        """Initialize this module and its sublayers."""

        super().__init__()
        self.backbone = _create_backbone(
            candidate_names=candidate_names,
            in_channels=in_channels,
            use_pretrained=use_pretrained,
        )
        feature_dim = _infer_backbone_feature_dim(
            backbone=self.backbone,
            in_channels=in_channels,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass and return classification logits."""

        features = _forward_backbone_features(self.backbone, x)
        return self.classifier(features)


class TimmGasRegressionNet(nn.Module):
    """Generic regression wrapper around a timm backbone."""

    def __init__(
        self,
        candidate_names: Sequence[str],
        in_channels: int = 3,
        use_pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        """Initialize this module and its sublayers."""

        super().__init__()
        self.backbone = _create_backbone(
            candidate_names=candidate_names,
            in_channels=in_channels,
            use_pretrained=use_pretrained,
        )
        feature_dim = _infer_backbone_feature_dim(
            backbone=self.backbone,
            in_channels=in_channels,
        )
        hidden_dim = max(feature_dim // 2, 64)
        self.regressor = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass and return normalized regression outputs."""

        features = _forward_backbone_features(self.backbone, x)
        return self.regressor(features).squeeze(dim=1)


def build_ghostnetv3_100_classification_model(
    num_classes: int = 5,
    in_channels: int = 3,
    use_pretrained: bool = True,
) -> TimmGasClassificationNet:
    """Factory for GhostNetV3-1.0 gas classification."""

    return TimmGasClassificationNet(
        candidate_names=GHOSTNETV3_100_ALIASES,
        num_classes=num_classes,
        in_channels=in_channels,
        use_pretrained=use_pretrained,
    )


def build_starnet_s2_classification_model(
    num_classes: int = 5,
    in_channels: int = 3,
    use_pretrained: bool = True,
) -> TimmGasClassificationNet:
    """Factory for StarNet-S2 gas classification."""

    return TimmGasClassificationNet(
        candidate_names=STARNET_S2_ALIASES,
        num_classes=num_classes,
        in_channels=in_channels,
        use_pretrained=use_pretrained,
    )


def build_fastvit_t8_classification_model(
    num_classes: int = 5,
    in_channels: int = 3,
    use_pretrained: bool = True,
) -> TimmGasClassificationNet:
    """Factory for FastViT-T8 gas classification."""

    return TimmGasClassificationNet(
        candidate_names=FASTVIT_T8_ALIASES,
        num_classes=num_classes,
        in_channels=in_channels,
        use_pretrained=use_pretrained,
    )


def build_ghostnetv3_100_regression_model(
    in_channels: int = 3,
    use_pretrained: bool = True,
) -> TimmGasRegressionNet:
    """Factory for GhostNetV3-1.0 gas concentration regression."""

    return TimmGasRegressionNet(
        candidate_names=GHOSTNETV3_100_ALIASES,
        in_channels=in_channels,
        use_pretrained=use_pretrained,
    )


def build_starnet_s2_regression_model(
    in_channels: int = 3,
    use_pretrained: bool = True,
) -> TimmGasRegressionNet:
    """Factory for StarNet-S2 gas concentration regression."""

    return TimmGasRegressionNet(
        candidate_names=STARNET_S2_ALIASES,
        in_channels=in_channels,
        use_pretrained=use_pretrained,
    )


def build_fastvit_t8_regression_model(
    in_channels: int = 3,
    use_pretrained: bool = True,
) -> TimmGasRegressionNet:
    """Factory for FastViT-T8 gas concentration regression."""

    return TimmGasRegressionNet(
        candidate_names=FASTVIT_T8_ALIASES,
        in_channels=in_channels,
        use_pretrained=use_pretrained,
    )
