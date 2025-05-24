import torch
import torch.nn as nn
from torch.nn.modules.conv import Conv2d

def modify_in_conv(module: nn.Module) -> nn.Module:
    old_conv = module.transformer.transformer.patch_embed.proj

    assert isinstance(old_conv, Conv2d), f"Expected Conv2d, got {type(old_conv)}"
    assert old_conv.in_channels == 32, f"Expected in_channels=32, got {old_conv.in_channels}"
    
    new_conv = Conv2d(
        in_channels=48,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=old_conv.groups,
        bias=old_conv.bias is not None,
        padding_mode=old_conv.padding_mode,
    )
    
    # change the pretrained model's in_channels to 48
    with torch.no_grad():
        new_conv.weight[:, :old_conv.in_channels, :, :] = old_conv.weight.clone()
        if old_conv.bias is not None:
            new_conv.bias.data = old_conv.bias.clone()
        new_conv.weight[:, old_conv.in_channels:, :, :] = 0.0
    module.transformer.transformer.patch_embed.proj = new_conv
    return module


def modify_in_linear(module: nn.Module) -> nn.Module:
    old_linear = module.transformer.transformer.patch_embed.proj
    assert isinstance(old_linear, nn.Linear), f"Expected Linear, got {type(old_linear)}"
    assert old_linear.in_features == 256, f"Expected in_features=32, got {old_linear.in_features}"
    
    new_linear = nn.Linear(
        in_features=384,
        out_features=old_linear.out_features,
        bias=old_linear.bias is not None,
    )
    
    # change the pretrained model's in_features to 384
    with torch.no_grad():
        new_linear.weight[:, :old_linear.in_features] = old_linear.weight.clone()
        if old_linear.bias is not None:
            new_linear.bias.data = old_linear.bias.clone()
        new_linear.weight[:, old_linear.in_features:] = 0.0
    module.transformer.transformer.patch_embed.proj = new_linear
    return module