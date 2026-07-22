"""
CNN models for gas image recognition and concentration estimation.

The datasets in Dataset.py return normalized RGB tensors with shape
``[batch_size, 3, 64, 64]``.  This file provides two task-specific models:

1. GasClassificationNet: five-class gas identification.
2. GasRegressionNet: gas concentration regression.

Both models share the same attention-enhanced convolutional backbone.
"""

from __future__ import annotations

import torch
from torch import nn


class ConvBNAct(nn.Sequential):
    """Convolution + BatchNorm + SiLU activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        """Initialize this module and its sublayers."""

        padding = kernel_size // 2
        super().__init__(
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
            nn.SiLU(inplace=True),
        )


class ChannelAttention(nn.Module):
    """
    Channel attention branch from CBAM.

    It learns which feature channels are more useful for distinguishing gas
    response images, using both average-pooled and max-pooled global context.
    """

    def __init__(self, channels: int, reduction: int = 8) -> None:
        """Initialize this module and its sublayers."""

        super().__init__()
        hidden_channels = max(channels // reduction, 4)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass and return the module output."""

        avg_context = torch.mean(x, dim=(2, 3), keepdim=True)
        max_context = torch.amax(x, dim=(2, 3), keepdim=True)
        attention = self.shared_mlp(avg_context) + self.shared_mlp(max_context)
        return x * self.sigmoid(attention)


class SpatialAttention(nn.Module):
    """
    Spatial attention branch from CBAM.

    It highlights local image regions that carry discriminative response-pattern
    information, such as texture bands or transition areas in GASF/MTF/RP images.
    """

    def __init__(self, kernel_size: int = 7) -> None:
        """Initialize this module and its sublayers."""

        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass and return the module output."""

        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map = torch.amax(x, dim=1, keepdim=True)
        attention = self.conv(torch.cat([avg_map, max_map], dim=1))
        return x * self.sigmoid(attention)


class CBAM(nn.Module):
    """Convolutional Block Attention Module: channel attention + spatial attention."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        """Initialize this module and its sublayers."""

        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction=reduction)
        self.spatial_attention = SpatialAttention(kernel_size=7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass and return the module output."""

        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class MultiScaleDepthwiseConv(nn.Module):
    """
    Lightweight multi-scale convolution.

    Two depthwise branches with 3x3 and 5x5 kernels observe local texture and
    broader response-pattern structures, then a 1x1 projection fuses them.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        """Initialize this module and its sublayers."""

        super().__init__()
        self.pre_conv = ConvBNAct(in_channels, out_channels, kernel_size=1)
        self.branch_3x3 = ConvBNAct(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            groups=out_channels,
        )
        self.branch_5x5 = ConvBNAct(
            out_channels,
            out_channels,
            kernel_size=5,
            stride=stride,
            groups=out_channels,
        )
        self.fuse = ConvBNAct(out_channels * 2, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass and return the module output."""

        x = self.pre_conv(x)
        x_3 = self.branch_3x3(x)
        x_5 = self.branch_5x5(x)
        return self.fuse(torch.cat([x_3, x_5], dim=1))


class AttentionResidualBlock(nn.Module):
    """
    CNN block with multi-scale convolution and CBAM attention.

    Residual shortcuts are optional because the ablation results showed the
    no-residual variant can be stronger for this gas classification task.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_residual: bool = False,
    ) -> None:
        """Initialize this module and its sublayers."""

        super().__init__()
        self.main = nn.Sequential(
            MultiScaleDepthwiseConv(in_channels, out_channels, stride=stride),
            ConvBNAct(out_channels, out_channels, kernel_size=3),
            CBAM(out_channels),
        )
        self.use_residual = use_residual

        if use_residual and (stride != 1 or in_channels != out_channels):
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        elif use_residual:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = None

        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass and return the module output."""

        x_main = self.main(x)
        if self.use_residual:
            x_main = x_main + self.shortcut(x)
        return self.activation(x_main)


class SingleScaleDepthwiseConv(nn.Module):
    """Single-branch replacement used for the no-multiscale ablation."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        """Initialize this module and its sublayers."""

        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(in_channels, out_channels, kernel_size=1),
            ConvBNAct(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                groups=out_channels,
            ),
            ConvBNAct(out_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass and return the module output."""

        return self.block(x)


class AblationResidualBlock(nn.Module):
    """
    Configurable block for classification ablation studies.

    It can independently disable multi-scale convolution, CBAM attention and
    residual shortcuts while keeping the same stage widths.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_multiscale: bool = True,
        use_attention: bool = True,
        use_residual: bool = True,
    ) -> None:
        """Initialize this module and its sublayers."""

        super().__init__()
        conv = (
            MultiScaleDepthwiseConv(in_channels, out_channels, stride=stride)
            if use_multiscale
            else SingleScaleDepthwiseConv(in_channels, out_channels, stride=stride)
        )
        layers = [
            conv,
            ConvBNAct(out_channels, out_channels, kernel_size=3),
        ]
        if use_attention:
            layers.append(CBAM(out_channels))
        self.main = nn.Sequential(*layers)
        self.use_residual = use_residual

        if use_residual:
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                )
            else:
                self.shortcut = nn.Identity()
        else:
            self.shortcut = None

        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass and return the module output."""

        x_main = self.main(x)
        if self.use_residual:
            x_main = x_main + self.shortcut(x)
        return self.activation(x_main)


class GasFeatureBackbone(nn.Module):
    """
    Shared feature extractor for gas image tasks.

    Input:  [B, 3, 64, 64]
    Output: [B, feature_dim]
    """

    def __init__(self, in_channels: int = 3, feature_dim: int = 256) -> None:
        """Initialize this module and its sublayers."""

        super().__init__()
        self.features = nn.Sequential(
            ConvBNAct(in_channels, 32, kernel_size=3, stride=1),
            AblationResidualBlock(
                32,
                48,
                stride=2,
                use_multiscale=True,
                use_attention=False,
                use_residual=False,
            ),  # 64 -> 32
            AblationResidualBlock(
                48,
                64,
                stride=2,
                use_multiscale=True,
                use_attention=False,
                use_residual=False,
            ),  # 32 -> 16
            AblationResidualBlock(
                64,
                128,
                stride=2,
                use_multiscale=True,
                use_attention=False,
                use_residual=False,
            ),  # 16 -> 8
            AblationResidualBlock(
                128,
                192,
                stride=2,
                use_multiscale=True,
                use_attention=False,
                use_residual=False,
            ),  # 8 -> 4
            ConvBNAct(192, feature_dim, kernel_size=1),
        )
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass and return the module output."""

        x = self.features(x)
        x = self.pool(x)
        return self.flatten(x)


class GasAblationBackbone(nn.Module):
    """
    Backbone used by classification ablation models.

    It follows GasFeatureBackbone's stage widths and output dimension, while
    toggling individual innovation components.
    """

    def __init__(
        self,
        in_channels: int = 3,
        feature_dim: int = 256,
        use_multiscale: bool = True,
        use_attention: bool = True,
        use_residual: bool = True,
    ) -> None:
        """Initialize this module and its sublayers."""

        super().__init__()
        self.features = nn.Sequential(
            ConvBNAct(in_channels, 32, kernel_size=3, stride=1),
            AblationResidualBlock(
                32,
                48,
                stride=2,
                use_multiscale=use_multiscale,
                use_attention=use_attention,
                use_residual=use_residual,
            ),
            AblationResidualBlock(
                48,
                64,
                stride=2,
                use_multiscale=use_multiscale,
                use_attention=use_attention,
                use_residual=use_residual,
            ),
            AblationResidualBlock(
                64,
                128,
                stride=2,
                use_multiscale=use_multiscale,
                use_attention=use_attention,
                use_residual=use_residual,
            ),
            AblationResidualBlock(
                128,
                192,
                stride=2,
                use_multiscale=use_multiscale,
                use_attention=use_attention,
                use_residual=use_residual,
            ),
            ConvBNAct(192, feature_dim, kernel_size=1),
        )
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass and return the module output."""

        x = self.features(x)
        x = self.pool(x)
        return self.flatten(x)


class GasClassificationNet(nn.Module):
    """
    Five-class gas classification model.

    Forward output is raw logits with shape ``[B, num_classes]``. During
    training, use ``nn.CrossEntropyLoss`` directly; do not apply softmax before
    the loss.
    """

    def __init__(
        self,
        num_classes: int = 5,
        in_channels: int = 3,
        feature_dim: int = 256,
        dropout: float = 0.25,
    ) -> None:
        """Initialize this module and its sublayers."""

        super().__init__()
        self.backbone = GasFeatureBackbone(
            in_channels=in_channels,
            feature_dim=feature_dim,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass and return the module output."""

        features = self.backbone(x)
        return self.classifier(features)


class GasAblationClassificationNet(nn.Module):
    """Classification model with selected innovation components disabled."""

    def __init__(
        self,
        num_classes: int = 5,
        in_channels: int = 3,
        feature_dim: int = 256,
        dropout: float = 0.25,
        use_multiscale: bool = True,
        use_attention: bool = True,
        use_residual: bool = True,
    ) -> None:
        """Initialize this module and its sublayers."""

        super().__init__()
        self.backbone = GasAblationBackbone(
            in_channels=in_channels,
            feature_dim=feature_dim,
            use_multiscale=use_multiscale,
            use_attention=use_attention,
            use_residual=use_residual,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass and return the module output."""

        features = self.backbone(x)
        return self.classifier(features)


class PlainCNNBackbone(nn.Module):
    """Plain standard-convolution CNN baseline without depthwise design."""

    def __init__(self, in_channels: int = 3, feature_dim: int = 256) -> None:
        """Initialize this module and its sublayers."""

        super().__init__()
        self.features = nn.Sequential(
            ConvBNAct(in_channels, 32, kernel_size=3, stride=1),
            ConvBNAct(32, 48, kernel_size=3, stride=2),
            ConvBNAct(48, 64, kernel_size=3, stride=2),
            ConvBNAct(64, 128, kernel_size=3, stride=2),
            ConvBNAct(128, 192, kernel_size=3, stride=2),
            ConvBNAct(192, feature_dim, kernel_size=1),
        )
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass and return the module output."""

        x = self.features(x)
        x = self.pool(x)
        return self.flatten(x)


class PlainCNNClassificationNet(nn.Module):
    """Plain CNN classification baseline."""

    def __init__(
        self,
        num_classes: int = 5,
        in_channels: int = 3,
        feature_dim: int = 256,
        dropout: float = 0.65,
    ) -> None:
        """Initialize this module and its sublayers."""

        super().__init__()
        self.backbone = PlainCNNBackbone(
            in_channels=in_channels,
            feature_dim=feature_dim,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass and return the module output."""

        features = self.backbone(x)
        return self.classifier(features)


class AlexNetMultiScaleBackbone(nn.Module):
    """
    AlexNet-inspired backbone with multi-scale convolution blocks.

    It keeps the stronger early downsampling rhythm and wider stages that often
    help on image classification, while replacing AlexNet's standard mid-stage
    convolutions with the project's multi-scale depthwise fusion blocks.
    """

    def __init__(self, in_channels: int = 3, feature_dim: int = 256) -> None:
        """Initialize this module and its sublayers."""

        super().__init__()
        self.features = nn.Sequential(
            ConvBNAct(in_channels, 64, kernel_size=11, stride=2),  # 64 -> 32
            nn.MaxPool2d(kernel_size=3, stride=2),  # 32 -> 15
            ConvBNAct(64, 96, kernel_size=1),
            MultiScaleDepthwiseConv(96, 192, stride=1),
            ConvBNAct(192, 192, kernel_size=3),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 15 -> 7
            MultiScaleDepthwiseConv(192, 384, stride=1),
            MultiScaleDepthwiseConv(384, 256, stride=1),
            MultiScaleDepthwiseConv(256, 256, stride=1),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 7 -> 3
            ConvBNAct(256, feature_dim, kernel_size=1),
        )
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass and return the module output."""

        x = self.features(x)
        x = self.pool(x)
        return self.flatten(x)


class AlexNetMultiScaleClassificationNet(nn.Module):
    """Classification model that combines AlexNet-style staging with multi-scale blocks."""

    def __init__(
        self,
        num_classes: int = 5,
        in_channels: int = 3,
        feature_dim: int = 256,
        dropout: float = 0.5,
    ) -> None:
        """Initialize this module and its sublayers."""

        super().__init__()
        self.backbone = AlexNetMultiScaleBackbone(
            in_channels=in_channels,
            feature_dim=feature_dim,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, 512),
            nn.SiLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.SiLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass and return the module output."""

        features = self.backbone(x)
        return self.classifier(features)


class StableAlexNetMultiScaleBackbone(nn.Module):
    """
    More stable AlexNet-style backbone for gas image classification.

    Compared with the first AlexNet-multiscale variant, this version keeps
    AlexNet's stronger standard-convolution stem for early feature extraction
    and only injects multi-scale blocks after the representation is already
    reasonably compact. That usually makes optimization less noisy on small
    datasets while still preserving the multi-scale design intent.
    """

    def __init__(self, in_channels: int = 3, feature_dim: int = 256) -> None:
        """Initialize this module and its sublayers."""

        super().__init__()
        self.features = nn.Sequential(
            ConvBNAct(in_channels, 64, kernel_size=7, stride=2),  # 64 -> 32
            nn.MaxPool2d(kernel_size=3, stride=2),  # 32 -> 15
            ConvBNAct(64, 128, kernel_size=5, stride=1),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 15 -> 7
            ConvBNAct(128, 192, kernel_size=3, stride=1),
            MultiScaleDepthwiseConv(192, 256, stride=1),
            ConvBNAct(256, 256, kernel_size=3, stride=1),
            MultiScaleDepthwiseConv(256, 256, stride=1),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 7 -> 3
            ConvBNAct(256, feature_dim, kernel_size=1),
        )
        self.pool = nn.AdaptiveAvgPool2d(output_size=(2, 2))
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass and return the module output."""

        x = self.features(x)
        x = self.pool(x)
        return self.flatten(x)


class StableAlexNetMultiScaleClassificationNet(nn.Module):
    """AlexNet-inspired multiscale classifier tuned for smoother optimization."""

    def __init__(
        self,
        num_classes: int = 5,
        in_channels: int = 3,
        feature_dim: int = 256,
        dropout: float = 0.05,
    ) -> None:
        """Initialize this module and its sublayers."""

        super().__init__()
        self.backbone = StableAlexNetMultiScaleBackbone(
            in_channels=in_channels,
            feature_dim=feature_dim,
        )
        classifier_input_dim = feature_dim * 4
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(classifier_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass and return the module output."""

        features = self.backbone(x)
        return self.classifier(features)


class HybridAlexNetMultiScaleBlock(nn.Module):
    """
    Hybrid block that keeps a standard AlexNet-style conv path and adds a multi-scale path.

    This is deliberately conservative: AlexNet remains the main inductive bias,
    and the multi-scale branch acts as a complementary enhancement instead of a
    full replacement.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize this module and its sublayers."""

        super().__init__()
        self.standard_path = ConvBNAct(in_channels, out_channels, kernel_size=3, stride=1)
        self.multiscale_path = MultiScaleDepthwiseConv(in_channels, out_channels, stride=1)
        self.fuse = ConvBNAct(out_channels * 2, out_channels, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass and return the module output."""

        standard_features = self.standard_path(x)
        multiscale_features = self.multiscale_path(x)
        return self.fuse(torch.cat([standard_features, multiscale_features], dim=1))


class HybridAlexNetMultiScaleBackbone(nn.Module):
    """
    AlexNet-like backbone with late-stage multi-scale enhancement.

    The early and middle stages stay close to AlexNet because they already work
    well on this task. Multi-scale fusion is introduced only in the later
    representation stages, where it can refine texture and structural patterns
    without destabilizing the whole optimization process.
    """

    def __init__(self, in_channels: int = 3, feature_dim: int = 256) -> None:
        """Initialize this module and its sublayers."""

        super().__init__()
        self.features = nn.Sequential(
            ConvBNAct(in_channels, 64, kernel_size=7, stride=2),  # 64 -> 32
            nn.MaxPool2d(kernel_size=3, stride=2),  # 32 -> 15
            ConvBNAct(64, 192, kernel_size=5, stride=1),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 15 -> 7
            ConvBNAct(192, 384, kernel_size=3, stride=1),
            HybridAlexNetMultiScaleBlock(384, 256),
            HybridAlexNetMultiScaleBlock(256, 256),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 7 -> 3
            ConvBNAct(256, feature_dim, kernel_size=1, stride=1),
        )
        self.pool = nn.AdaptiveAvgPool2d(output_size=(2, 2))
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass and return the module output."""

        x = self.features(x)
        x = self.pool(x)
        return self.flatten(x)


class HybridAlexNetMultiScaleClassificationNet(nn.Module):
    """AlexNet-leaning classifier with conservative late-stage multi-scale fusion."""

    def __init__(
        self,
        num_classes: int = 5,
        in_channels: int = 3,
        feature_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        """Initialize this module and its sublayers."""

        super().__init__()
        self.backbone = HybridAlexNetMultiScaleBackbone(
            in_channels=in_channels,
            feature_dim=feature_dim,
        )
        classifier_input_dim = feature_dim * 4
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(classifier_input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.SiLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass and return the module output."""

        features = self.backbone(x)
        return self.classifier(features)


class GasAblationRegressionNet(nn.Module):
    """Regression model with selected innovation components disabled."""

    def __init__(
        self,
        in_channels: int = 3,
        feature_dim: int = 256,
        dropout: float = 0.2,
        use_multiscale: bool = True,
        use_attention: bool = True,
        use_residual: bool = True,
    ) -> None:
        """Initialize this module and its sublayers."""

        super().__init__()
        self.backbone = GasAblationBackbone(
            in_channels=in_channels,
            feature_dim=feature_dim,
            use_multiscale=use_multiscale,
            use_attention=use_attention,
            use_residual=use_residual,
        )
        self.regressor = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, 128),
            nn.SiLU(inplace=True),
            nn.Linear(128, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass and return the module output."""

        features = self.backbone(x)
        return self.regressor(features).squeeze(dim=1)


class GasRegressionNet(nn.Module):
    """
    Gas concentration regression model.

    Forward output is a one-dimensional tensor with shape ``[B]``. It is suited
    for ``nn.MSELoss``, ``nn.SmoothL1Loss`` or ``nn.L1Loss``.
    """

    def __init__(
        self,
        in_channels: int = 3,
        feature_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        """Initialize this module and its sublayers."""

        super().__init__()
        self.backbone = GasFeatureBackbone(
            in_channels=in_channels,
            feature_dim=feature_dim,
        )
        self.regressor = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, 128),
            nn.SiLU(inplace=True),
            nn.Linear(128, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass and return the module output."""

        features = self.backbone(x)
        return self.regressor(features).squeeze(dim=1)


def build_classification_model(
    num_classes: int = 5,
    in_channels: int = 3,
) -> GasClassificationNet:
    """Factory function for the five-class classification model."""

    return GasClassificationNet(num_classes=num_classes, in_channels=in_channels)


def build_classification_ablation_model(
    num_classes: int = 5,
    in_channels: int = 3,
    use_multiscale: bool = True,
    use_attention: bool = False,
    use_residual: bool = False,
) -> GasAblationClassificationNet:
    """Factory for systematic classification ablation combinations."""

    return GasAblationClassificationNet(
        num_classes=num_classes,
        in_channels=in_channels,
        use_multiscale=use_multiscale,
        use_attention=use_attention,
        use_residual=use_residual,
    )


def build_classification_no_attention_model(
    num_classes: int = 5,
    in_channels: int = 3,
) -> GasAblationClassificationNet:
    """Backward-compatible alias for the current no-attention main model."""

    return build_classification_model(num_classes=num_classes, in_channels=in_channels)


def build_classification_with_attention_model(
    num_classes: int = 5,
    in_channels: int = 3,
) -> GasAblationClassificationNet:
    """Ablation from the new main model: add CBAM attention back."""

    return GasAblationClassificationNet(
        num_classes=num_classes,
        in_channels=in_channels,
        use_multiscale=True,
        use_attention=True,
        use_residual=False,
    )


def build_classification_no_multiscale_model(
    num_classes: int = 5,
    in_channels: int = 3,
) -> GasAblationClassificationNet:
    """Ablation from the no-attention main model: use a single 3x3 branch."""

    return GasAblationClassificationNet(
        num_classes=num_classes,
        in_channels=in_channels,
        use_multiscale=False,
        use_attention=False,
        use_residual=False,
    )


def build_classification_no_residual_model(
    num_classes: int = 5,
    in_channels: int = 3,
) -> GasAblationClassificationNet:
    """Backward-compatible alias for the current no-attention/no-residual model."""

    return build_classification_model(num_classes=num_classes, in_channels=in_channels)


def build_classification_with_residual_model(
    num_classes: int = 5,
    in_channels: int = 3,
) -> GasAblationClassificationNet:
    """Ablation from the new main model: add residual shortcuts back."""

    return GasAblationClassificationNet(
        num_classes=num_classes,
        in_channels=in_channels,
        use_multiscale=True,
        use_attention=False,
        use_residual=True,
    )


def build_classification_plain_cnn_model(
    num_classes: int = 5,
    in_channels: int = 3,
) -> PlainCNNClassificationNet:
    """Plain standard-convolution CNN baseline."""

    return PlainCNNClassificationNet(num_classes=num_classes, in_channels=in_channels)


def build_classification_alexnet_multiscale_model(
    num_classes: int = 5,
    in_channels: int = 3,
) -> AlexNetMultiScaleClassificationNet:
    """AlexNet-inspired classification baseline upgraded with multi-scale blocks."""

    return AlexNetMultiScaleClassificationNet(num_classes=num_classes, in_channels=in_channels)


def build_classification_stable_alexnet_multiscale_model(
    num_classes: int = 5,
    in_channels: int = 3,
) -> StableAlexNetMultiScaleClassificationNet:
    """More stable AlexNet-style classifier with later multi-scale fusion."""

    return StableAlexNetMultiScaleClassificationNet(num_classes=num_classes, in_channels=in_channels)


def build_classification_hybrid_alexnet_multiscale_model(
    num_classes: int = 5,
    in_channels: int = 3,
) -> HybridAlexNetMultiScaleClassificationNet:
    """AlexNet-leaning classifier with conservative late-stage multi-scale fusion."""

    return HybridAlexNetMultiScaleClassificationNet(num_classes=num_classes, in_channels=in_channels)


def build_regression_model(in_channels: int = 3) -> GasRegressionNet:
    """Factory function for the concentration regression model."""

    return GasRegressionNet(in_channels=in_channels)


def build_regression_ablation_model(
    in_channels: int = 3,
    use_multiscale: bool = True,
    use_attention: bool = False,
    use_residual: bool = False,
) -> GasAblationRegressionNet:
    """Factory for systematic regression ablation combinations."""

    return GasAblationRegressionNet(
        in_channels=in_channels,
        use_multiscale=use_multiscale,
        use_attention=use_attention,
        use_residual=use_residual,
    )


def build_regression_no_attention_model(in_channels: int = 3) -> GasAblationRegressionNet:
    """Ablation: disable CBAM attention only."""

    return GasAblationRegressionNet(
        in_channels=in_channels,
        use_multiscale=True,
        use_attention=False,
        use_residual=True,
    )


def build_regression_no_multiscale_model(in_channels: int = 3) -> GasAblationRegressionNet:
    """Ablation: replace multi-scale 3x3/5x5 branch with a single 3x3 branch."""

    return GasAblationRegressionNet(
        in_channels=in_channels,
        use_multiscale=False,
        use_attention=True,
        use_residual=True,
    )


def build_regression_no_residual_model(in_channels: int = 3) -> GasAblationRegressionNet:
    """Ablation: disable residual shortcuts only."""

    return GasAblationRegressionNet(
        in_channels=in_channels,
        use_multiscale=True,
        use_attention=True,
        use_residual=False,
    )


def build_regression_plain_cnn_model(in_channels: int = 3) -> GasAblationRegressionNet:
    """Ablation baseline: disable multi-scale convolution, CBAM and residuals."""

    return GasAblationRegressionNet(
        in_channels=in_channels,
        use_multiscale=False,
        use_attention=False,
        use_residual=False,
    )


if __name__ == "__main__":
    dummy_input = torch.randn(4, 3, 64, 64)

    cls_model = build_classification_model(num_classes=5)
    reg_model = build_regression_model()

    cls_output = cls_model(dummy_input)
    reg_output = reg_model(dummy_input)

    print("classification output shape:", cls_output.shape)
    print("regression output shape:", reg_output.shape)
