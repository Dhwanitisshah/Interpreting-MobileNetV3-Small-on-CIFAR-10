from functools import partial
from typing import List

import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from torchvision.models.mobilenetv3 import InvertedResidualConfig, MobileNetV3

VARIANTS = ("vanilla", "no_se", "small_kernel")

# (in_ch, kernel, expanded_ch, out_ch, use_se, activation, stride, dilation)
_BLOCK_SETTING = [
    (16, 3, 16, 16, True, "RE", 2, 1),
    (16, 3, 72, 24, False, "RE", 2, 1),
    (24, 3, 88, 24, False, "RE", 1, 1),
    (24, 5, 96, 40, True, "HS", 2, 1),
    (40, 5, 240, 40, True, "HS", 1, 1),
    (40, 5, 240, 40, True, "HS", 1, 1),
    (40, 5, 120, 48, True, "HS", 1, 1),
    (48, 5, 144, 48, True, "HS", 1, 1),
    (48, 5, 288, 96, True, "HS", 2, 1),
    (96, 5, 576, 96, True, "HS", 1, 1),
    (96, 5, 576, 96, True, "HS", 1, 1),
]

_LAST_CHANNEL = 1024


def _build_inverted_residual_setting(
    variant: str, width_mult: float
) -> List[InvertedResidualConfig]:
    bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)

    setting = []
    for in_ch, kernel, expanded_ch, out_ch, use_se, activation, stride, dilation in _BLOCK_SETTING:
        if variant == "no_se":
            use_se = False
        elif variant == "small_kernel" and kernel == 5:
            kernel = 3
        setting.append(
            bneck_conf(in_ch, kernel, expanded_ch, out_ch, use_se, activation, stride, dilation)
        )
    return setting


def build_mobilenetv3_small(
    variant: str = "vanilla",
    num_classes: int = 10,
    pretrained: bool = False,
    width_mult: float = 1.0,
    dropout: float = 0.2,
) -> nn.Module:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Expected one of {VARIANTS}.")

    if pretrained and variant != "vanilla":
        raise ValueError(
            f"pretrained=True is only supported for variant='vanilla'. The '{variant}' "
            "variant changes the network topology, so ImageNet weights cannot be mapped "
            "onto it. Build it with pretrained=False and train from scratch."
        )

    if variant == "vanilla" and pretrained:
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    inverted_residual_setting = _build_inverted_residual_setting(variant, width_mult)
    last_channel = InvertedResidualConfig.adjust_channels(_LAST_CHANNEL, width_mult)

    model = MobileNetV3(
        inverted_residual_setting=inverted_residual_setting,
        last_channel=last_channel,
        num_classes=num_classes,
        dropout=dropout,
    )
    return model


def get_gradcam_target_layer(model: nn.Module) -> nn.Module:
    return model.features[-1]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
