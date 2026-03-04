"""Utility helpers for TimeVQVAE components."""

from __future__ import annotations

from typing import Union

import numpy as np
import torch
import torch.jit as jit
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


def freeze(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def unfreeze(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


def time_to_timefreq(x: torch.Tensor, n_fft: int, channels: int, norm: bool = True) -> torch.Tensor:
    """
    Convert time series to time-frequency domain using STFT.

    Args:
        x: (B, C, L)
        n_fft: FFT size
        channels: Number of channels in input
        norm: Whether to normalize STFT

    Returns:
        (B, C*2, H, W) real/imag stacked
    """
    x = rearrange(x, "b c l -> (b c) l")
    window = torch.hann_window(window_length=n_fft, device=x.device)
    xf = torch.stft(x, n_fft, normalized=norm, return_complex=True, window=window)
    xf = torch.view_as_real(xf)  # (B, N, T, 2)
    xf = rearrange(xf, "(b c) n t z -> b (c z) n t", c=channels)
    return xf.float()


def timefreq_to_time(x: torch.Tensor, n_fft: int, channels: int, norm: bool = True) -> torch.Tensor:
    """
    Convert time-frequency representation back to time series using iSTFT.
    """
    x = rearrange(x, "b (c z) n t -> (b c) n t z", c=channels).contiguous()
    x = torch.view_as_complex(x)
    window = torch.hann_window(window_length=n_fft, device=x.device)
    xt = torch.istft(x, n_fft, normalized=norm, window=window)
    xt = rearrange(xt, "(b c) l -> b c l", c=channels)
    return xt.float()


def quantize(
    z: torch.Tensor,
    vq_model: nn.Module,
    transpose_channel_length_axes: bool = False,
    svq_temp: Union[float, None] = None,
):
    input_dim = len(z.shape) - 2
    if input_dim == 2:
        h, w = z.shape[2:]
        z_flat = rearrange(z, "b c h w -> b (h w) c")
        z_q, indices, vq_loss, perplexity = vq_model(z_flat, svq_temp)
        z_q = rearrange(z_q, "b (h w) c -> b c h w", h=h, w=w)
    elif input_dim == 1:
        z_flat = z
        if transpose_channel_length_axes:
            z_flat = rearrange(z_flat, "b c l -> b l c")
        z_q, indices, vq_loss, perplexity = vq_model(z_flat, svq_temp)
        if transpose_channel_length_axes:
            z_q = rearrange(z_q, "b l c -> b c l")
    else:
        raise ValueError("Unsupported input dimension for quantize")
    return z_q, indices, vq_loss, perplexity


def zero_pad_high_freq(xf: torch.Tensor, copy: bool = False) -> torch.Tensor:
    """
    Zero out high-frequency components.

    Args:
        xf: (B, C, H, W)
        copy: If True, copy LF to all bands
    """
    if not copy:
        xf_l = torch.zeros_like(xf)
        xf_l[:, :, 0, :] = xf[:, :, 0, :]
    else:
        xf_l = xf[:, :, [0], :]
        xf_l = repeat(xf_l, "b c 1 w -> b c h w", h=xf.shape[2]).float()
    return xf_l


def zero_pad_low_freq(xf: torch.Tensor, copy: bool = False) -> torch.Tensor:
    """
    Zero out low-frequency components.

    Args:
        xf: (B, C, H, W)
        copy: If True, copy first HF component into LF band
    """
    if not copy:
        xf_h = torch.zeros_like(xf)
        xf_h[:, :, 1:, :] = xf[:, :, 1:, :]
    else:
        xf_h = xf[:, :, 1:, :]
        xf_h = torch.cat((xf_h[:, :, [0], :], xf_h), dim=2).float()
    return xf_h


def compute_downsample_rate(input_length: int, n_fft: int, downsampled_width: int) -> int:
    if input_length >= downsampled_width:
        return round(input_length / (np.log2(n_fft) - 1) / downsampled_width)
    return 1


class SnakeActivation(jit.ScriptModule):
    """
    Snake activation with learnable frequency per channel.
    """

    def __init__(self, num_features: int, dim: int, a_base: float = 0.2, learnable: bool = True, a_max: float = 0.5):
        super().__init__()
        assert dim in [1, 2], "SnakeActivation supports 1D or 2D inputs."

        if learnable:
            if dim == 1:
                a = np.random.uniform(a_base, a_max, size=(1, num_features, 1))
                self.a = nn.Parameter(torch.tensor(a, dtype=torch.float32))
            else:
                a = np.random.uniform(a_base, a_max, size=(1, num_features, 1, 1))
                self.a = nn.Parameter(torch.tensor(a, dtype=torch.float32))
        else:
            self.register_buffer("a", torch.tensor(a_base, dtype=torch.float32))

    @jit.script_method
    def forward(self, x):
        return x + (1 / self.a) * torch.sin(self.a * x) ** 2
