"""
model.py — U-Net with an EfficientNet encoder via segmentation-models-pytorch.

We use encoder_weights=None because ImageNet weights expect 3 input channels and we
have 6 (S2 B2,B3,B4,B8 + S1 VV,VH). Inflating pretrained weights to N channels is a
known trick but we keep v1 simple and train the encoder from scratch.
"""
from __future__ import annotations

import segmentation_models_pytorch as smp

import config as C


def build_model(encoder=C.ENCODER, in_channels=C.IN_CHANNELS, classes=C.CLASSES):
    """U-Net returning raw logits (use with BCEWithLogitsLoss)."""
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights=C.ENCODER_WEIGHTS,   # None -> train from scratch (6-ch input)
        in_channels=in_channels,
        classes=classes,
        activation=None,                     # logits
    )


if __name__ == "__main__":
    import torch
    m = build_model()
    n_params = sum(p.numel() for p in m.parameters()) / 1e6
    x = torch.zeros(2, C.IN_CHANNELS, C.IMG_SIZE, C.IMG_SIZE)
    with torch.no_grad():
        y = m(x)
    print(f"{C.ENCODER} U-Net | params: {n_params:.1f}M | in {tuple(x.shape)} -> out {tuple(y.shape)}")
