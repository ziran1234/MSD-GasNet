"""
Compact VGG19-style models for gas classification and concentration regression.

The public API is compatible with model_train_test.py:

1. build_classification_model(num_classes=5) returns logits [B, num_classes].
2. build_regression_model() returns normalized concentration predictions [B].

This version intentionally moves closer to VGG19 than the previous ultra-light
version. It keeps the VGG19 2-2-4-4-4 convolution layout, uses real 3x3
convolutions, but greatly reduces channel width so the parameter count remains
small compared with VGG19.

Innovation points:
- residual VGG blocks with learnable layer scale for more stable deep training;
- lightweight channel-spatial attention after each stage;
- multi-stage avg/max/std/GeM descriptor fusion for concentration regression.

Input tensors are expected to be normalized RGB images with shape [B, 3, 64, 64].
"""

from __future__ import annotations

import torch
from torch import nn


class ConvBNAct(nn.Sequential):
    """3x3/1x1 convolution + BatchNorm + GELU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        activation: bool = True,
    ) -> None:
        """Initialize this module and its sublayers."""

        padding = kernel_size // 2
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if activation:
            layers.append(nn.GELU())
        super().__init__(*layers)


class ChannelSpatialAttention(nn.Module):
    """
    Lightweight channel-spatial attention.

    The channel branch selects useful response channels, while the spatial
    branch highlights discriminative regions in GASF/MTF/RP gas images.
    """

    def __init__(self, channels: int, reduction: int = 8) -> None:
        """Initialize this module and its sublayers."""

        super().__init__()
        hidden_channels = max(channels // reduction, 8)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass and return the module output."""

        x = x * self.channel_gate(x)
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map = torch.max(x, dim=1, keepdim=True)[0]
        spatial_weight = self.spatial_gate(torch.cat([avg_map, max_map], dim=1))
        return x * spatial_weight


class ResidualVGGBlock(nn.Module):
    """
    VGG-style 3x3 convolution block with residual projection.

    It is still a VGG block structurally, but the residual path and layer scale
    make the 16-convolution stack easier to optimize from scratch.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ) -> None:
        """Initialize this module and its sublayers."""

        super().__init__()
        self.conv = nn.Sequential(
            ConvBNAct(in_channels, out_channels, kernel_size=3),
            nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity(),
        )
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = ConvBNAct(
                in_channels,
                out_channels,
                kernel_size=1,
                activation=False,
            )
        self.layer_scale = nn.Parameter(torch.ones(1, out_channels, 1, 1) * 0.1)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass and return the module output."""

        return self.activation(self.shortcut(x) + self.layer_scale * self.conv(x))


class CompactVGGStage(nn.Module):
    """One VGG stage followed by attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        dropout: float = 0.0,
    ) -> None:
        """Initialize this module and its sublayers."""

        super().__init__()
        blocks = []
        for block_idx in range(depth):
            blocks.append(
                ResidualVGGBlock(
                    in_channels if block_idx == 0 else out_channels,
                    out_channels,
                    dropout=dropout,
                )
            )
        self.blocks = nn.Sequential(*blocks)
        self.attention = ChannelSpatialAttention(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass and return the module output."""

        return self.attention(self.blocks(x))


class GeneralizedMeanPooling(nn.Module):
    """GeM pooling keeps high-response regions useful for ppm regression."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        """Initialize this module and its sublayers."""

        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass and return the module output."""

        p = self.p.clamp(min=1.0, max=6.0)
        x = x.clamp(min=self.eps).pow(p)
        x = torch.mean(x, dim=(2, 3), keepdim=True)
        return x.pow(1.0 / p)


class StageDescriptor(nn.Module):
    """avg/max/std/GeM descriptor for one feature stage."""

    def __init__(
        self,
        in_channels: int,
        descriptor_channels: int = 64,
    ) -> None:
        """Initialize this module and its sublayers."""

        super().__init__()
        self.project = ConvBNAct(in_channels, descriptor_channels, kernel_size=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.gem_pool = GeneralizedMeanPooling()
        self.flatten = nn.Flatten()
        self.out_features = descriptor_channels * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass and return the module output."""

        x = self.project(x)
        avg_features = self.avg_pool(x)
        max_features = self.max_pool(x)
        std_features = torch.sqrt(
            torch.mean((x - avg_features).pow(2), dim=(2, 3), keepdim=True) + 1e-6
        )
        gem_features = self.gem_pool(x)
        return self.flatten(
            torch.cat(
                [avg_features, max_features, std_features, gem_features],
                dim=1,
            )
        )


class CompactVGG19Backbone(nn.Module):
    """
    Width-reduced VGG19-like backbone.

    Input:  [B, 3, 64, 64]
    Output: [B, feature_dim]
    """

    def __init__(
        self,
        in_channels: int = 3,
        feature_dim: int = 512,
        descriptor_channels: int = 72,
        projection_dropout: float = 0.08,
    ) -> None:
        """Initialize this module and its sublayers."""

        super().__init__()
        channels = [32, 56, 96, 160, 256]
        depths = [2, 2, 4, 4, 4]

        self.stage1 = CompactVGGStage(in_channels, channels[0], depths[0])
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stage2 = CompactVGGStage(channels[0], channels[1], depths[1])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stage3 = CompactVGGStage(channels[1], channels[2], depths[2], dropout=0.03)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stage4 = CompactVGGStage(channels[2], channels[3], depths[3], dropout=0.04)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stage5 = CompactVGGStage(channels[3], channels[4], depths[4], dropout=0.05)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.desc2 = StageDescriptor(channels[1], descriptor_channels)
        self.desc3 = StageDescriptor(channels[2], descriptor_channels)
        self.desc4 = StageDescriptor(channels[3], descriptor_channels)
        self.desc5 = StageDescriptor(channels[4], descriptor_channels)
        self.final_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        fused_features = (
            self.desc2.out_features
            + self.desc3.out_features
            + self.desc4.out_features
            + self.desc5.out_features
            + channels[4]
        )
        self.projection = nn.Sequential(
            nn.Linear(fused_features, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(p=projection_dropout),
        )
        self.out_features = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass and return the module output."""

        x = self.stage1(x)
        x = self.pool1(x)
        x = self.stage2(x)
        descriptor2 = self.desc2(x)
        x = self.pool2(x)
        x = self.stage3(x)
        descriptor3 = self.desc3(x)
        x = self.pool3(x)
        x = self.stage4(x)
        descriptor4 = self.desc4(x)
        x = self.pool4(x)
        x = self.stage5(x)
        descriptor5 = self.desc5(x)
        x = self.pool5(x)
        final_descriptor = self.flatten(self.final_pool(x))
        descriptors = torch.cat(
            [descriptor2, descriptor3, descriptor4, descriptor5, final_descriptor],
            dim=1,
        )
        return self.projection(descriptors)


class GasClassificationNet(nn.Module):
    """Five-class gas classification model."""

    def __init__(
        self,
        num_classes: int = 5,
        in_channels: int = 3,
        feature_dim: int = 512,
        dropout: float = 0.25,
    ) -> None:
        """Initialize this module and its sublayers."""

        super().__init__()
        self.backbone = CompactVGG19Backbone(
            in_channels=in_channels,
            feature_dim=feature_dim,
            projection_dropout=0.08,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.backbone.out_features, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )
        initialize_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass and return the module output."""

        return self.classifier(self.backbone(x))


class GasRegressionNet(nn.Module):
    """
    Gas concentration regression model.

    The final Sigmoid matches normalized 0-1 ppm labels in model_train_test.py.
    """

    def __init__(
        self,
        in_channels: int = 3,
        feature_dim: int = 512,
        dropout: float = 0.08,
    ) -> None:
        """Initialize this module and its sublayers."""

        super().__init__()
        self.backbone = CompactVGG19Backbone(
            in_channels=in_channels,
            feature_dim=feature_dim,
            projection_dropout=0.05,
        )
        self.regressor = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.backbone.out_features, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        initialize_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass and return the module output."""

        return self.regressor(self.backbone(x)).squeeze(dim=1)


def initialize_weights(model: nn.Module) -> None:
    """Initialize modules for stable training from scratch."""

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def build_classification_model(num_classes: int = 5) -> GasClassificationNet:
    """Factory function for the gas classification model."""

    return GasClassificationNet(num_classes=num_classes)


def build_regression_model() -> GasRegressionNet:
    """Factory function for the gas concentration regression model."""

    return GasRegressionNet()


if __name__ == "__main__":
    dummy_input = torch.randn(4, 3, 64, 64)

    cls_model = build_classification_model(num_classes=5)
    reg_model = build_regression_model()

    cls_params = sum(p.numel() for p in cls_model.parameters())
    reg_params = sum(p.numel() for p in reg_model.parameters())
    cls_output = cls_model(dummy_input)
    reg_output = reg_model(dummy_input)

    print("classification output shape:", cls_output.shape)
    print("regression output shape:", reg_output.shape)
    print("classification params:", cls_params)
    print("regression params:", reg_params)
