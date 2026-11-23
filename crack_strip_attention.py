import torch
import torch.nn as nn


class CrackStripAttention(nn.Module):
    """
    Crack Strip Attention (CSA/CDSA)
    A lightweight, channel-preserving module for long and thin crack features.
    It combines horizontal/vertical strip spatial attention with ECA-style channel attention.

    Args:
        k: strip kernel size for horizontal/vertical spatial attention. Use odd numbers such as 7, 11, 15.
        eca_k: kernel size of ECA-style channel attention. Usually 3 or 5.
    """

    def __init__(self, k=15, eca_k=3):
        super().__init__()
        assert k % 2 == 1, "k must be an odd number, e.g. 7, 11, 15"
        assert eca_k % 2 == 1, "eca_k must be an odd number, e.g. 3 or 5"

        # Spatial strip attention: channel-agnostic, so it does not need c1/c2 from parse_model.
        self.h_conv = nn.Conv2d(2, 1, kernel_size=(1, k), padding=(0, k // 2), bias=False)
        self.v_conv = nn.Conv2d(2, 1, kernel_size=(k, 1), padding=(k // 2, 0), bias=False)

        # ECA-style channel attention: does not require explicit channel number.
        self.eca = nn.Conv1d(1, 1, kernel_size=eca_k, padding=eca_k // 2, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        # Spatial attention from average and maximum responses.
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        s = torch.cat([avg_map, max_map], dim=1)
        a_spatial = self.act(self.h_conv(s) + self.v_conv(s))  # [B, 1, H, W]

        # Channel attention from global descriptor.
        y = torch.mean(x, dim=(2, 3), keepdim=False).unsqueeze(1)  # [B, 1, C]
        a_channel = self.act(self.eca(y)).squeeze(1).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]

        # Residual re-weighting, preserves tensor shape and channels.
        return x * (1.0 + a_spatial * a_channel)
