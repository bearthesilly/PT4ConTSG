"""VQ-VAE encoder/decoder blocks for TimeVQVAE."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from contsg.models.timevqvae_modules.utils import time_to_timefreq, timefreq_to_time, SnakeActivation


def _get_group_norm(num_channels: int) -> nn.GroupNorm:
    for num_groups in [num_channels, 32, 16, 8, 4, 2, 1]:
        if num_channels % num_groups == 0 and num_groups <= num_channels:
            return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
    return nn.GroupNorm(num_groups=1, num_channels=num_channels)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, frequency_indepence: bool, mid_channels: int | None = None, dropout: float = 0.0):
        super().__init__()

        if mid_channels is None:
            mid_channels = out_channels

        kernel_size = (1, 3) if frequency_indepence else (3, 3)
        padding = (0, 1) if frequency_indepence else (1, 1)

        layers = [
            SnakeActivation(in_channels, 2),
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=(1, 1),
                padding=padding,
                groups=in_channels,
                bias=False,
            ),
            _get_group_norm(in_channels),
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            _get_group_norm(mid_channels),
            SnakeActivation(out_channels, 2),
            nn.Conv2d(
                mid_channels,
                mid_channels,
                kernel_size=kernel_size,
                stride=(1, 1),
                padding=padding,
                groups=mid_channels,
                bias=False,
            ),
            _get_group_norm(mid_channels),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            _get_group_norm(out_channels),
            nn.Dropout(dropout),
        ]
        self.convs = nn.Sequential(*layers)
        self.proj = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x) + self.convs(x)


class VQVAEEncBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, frequency_indepence: bool, dropout: float = 0.0):
        super().__init__()

        kernel_size = (1, 4) if frequency_indepence else (3, 4)
        padding = (0, 1) if frequency_indepence else (1, 1)

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=(1, 2),
                padding=padding,
                padding_mode="replicate",
                groups=in_channels,
                bias=False,
            ),
            _get_group_norm(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            _get_group_norm(out_channels),
            SnakeActivation(out_channels, 2),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class VQVAEDecBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, frequency_indepence: bool, dropout: float = 0.0):
        super().__init__()

        kernel_size = (1, 4) if frequency_indepence else (3, 4)
        padding = (0, 1) if frequency_indepence else (1, 1)

        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=(1, 2),
                padding=padding,
                groups=in_channels,
                bias=False,
            ),
            _get_group_norm(in_channels),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, bias=False),
            _get_group_norm(out_channels),
            SnakeActivation(out_channels, 2),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class VQVAEEncoder(nn.Module):
    """VQ-VAE encoder for LF/HF branches."""

    def __init__(
        self,
        init_dim: int,
        hid_dim: int,
        num_channels: int,
        downsample_rate: int,
        n_resnet_blocks: int,
        kind: str,
        n_fft: int,
        frequency_indepence: bool,
        dropout: float = 0.3,
        **kwargs,
    ):
        super().__init__()
        self.kind = kind
        self.n_fft = n_fft

        d = init_dim
        enc_layers = [VQVAEEncBlock(num_channels, d, frequency_indepence)]
        d *= 2
        for _ in range(int(round(np.log2(downsample_rate))) - 1):
            enc_layers.append(VQVAEEncBlock(d // 2, d, frequency_indepence))
            for _ in range(n_resnet_blocks):
                enc_layers.append(ResBlock(d, d, frequency_indepence, dropout=dropout))
            d *= 2
        enc_layers.append(ResBlock(d // 2, hid_dim, frequency_indepence, dropout=dropout))
        self.encoder = nn.Sequential(*enc_layers)

        self.is_num_tokens_updated = False
        self.register_buffer("num_tokens", torch.tensor(0))
        self.register_buffer("H_prime", torch.tensor(0))
        self.register_buffer("W_prime", torch.tensor(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (b, c, l). Returns: (b, c, h, w)."""
        in_channels = x.shape[1]
        x = time_to_timefreq(x, self.n_fft, in_channels)

        if self.kind == "lf":
            x = x[:, :, [0], :]
        elif self.kind == "hf":
            x = x[:, :, 1:, :]

        out = self.encoder(x)
        out = F.normalize(out, dim=1)
        if not self.is_num_tokens_updated:
            self.H_prime = torch.tensor(out.shape[2])
            self.W_prime = torch.tensor(out.shape[3])
            self.num_tokens = self.H_prime * self.W_prime
            self.is_num_tokens_updated = True
        return out


class VQVAEDecoder(nn.Module):
    """VQ-VAE decoder for LF/HF branches."""

    def __init__(
        self,
        init_dim: int,
        hid_dim: int,
        num_channels: int,
        downsample_rate: int,
        n_resnet_blocks: int,
        input_length: int,
        kind: str,
        n_fft: int,
        x_channels: int,
        frequency_indepence: bool,
        dropout: float = 0.3,
        **kwargs,
    ):
        super().__init__()
        self.kind = kind
        self.n_fft = n_fft
        self.x_channels = x_channels

        kernel_size = (1, 4) if frequency_indepence else (3, 4)
        padding = (0, 1) if frequency_indepence else (1, 1)

        d = int(init_dim * 2 ** (int(round(np.log2(downsample_rate))) - 1))
        if round(np.log2(downsample_rate)) == 0:
            d = int(init_dim * 2 ** (int(round(np.log2(downsample_rate)))))
        dec_layers = [ResBlock(hid_dim, d, frequency_indepence, dropout=dropout)]
        for _ in range(int(round(np.log2(downsample_rate))) - 1):
            for _ in range(n_resnet_blocks):
                dec_layers.append(ResBlock(d, d, frequency_indepence, dropout=dropout))
            d //= 2
            dec_layers.append(VQVAEDecBlock(2 * d, d, frequency_indepence))
        dec_layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    d,
                    d,
                    kernel_size=kernel_size,
                    stride=(1, 2),
                    padding=padding,
                    groups=d,
                    bias=False,
                ),
                _get_group_norm(d),
                nn.ConvTranspose2d(d, num_channels, kernel_size=1, bias=False),
                _get_group_norm(num_channels),
            )
        )
        dec_layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    num_channels,
                    num_channels,
                    kernel_size=kernel_size,
                    stride=(1, 2),
                    padding=padding,
                    bias=False,
                ),
                _get_group_norm(num_channels),
                nn.ConvTranspose2d(num_channels, num_channels, kernel_size=1, bias=False),
                _get_group_norm(num_channels),
            )
        )
        self.decoder = nn.Sequential(*dec_layers)

        self.interp = nn.Upsample(input_length, mode="linear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.decoder(x)

        if self.kind == "lf":
            zeros = torch.zeros(
                (out.shape[0], out.shape[1], self.n_fft // 2 + 1, out.shape[-1]),
                dtype=out.dtype,
                device=out.device,
            )
            zeros[:, :, [0], :] = out
            out = zeros
        elif self.kind == "hf":
            zeros = torch.zeros(
                (out.shape[0], out.shape[1], self.n_fft // 2 + 1, out.shape[-1]),
                dtype=out.dtype,
                device=out.device,
            )
            zeros[:, :, 1:, :] = out
            out = zeros

        out = timefreq_to_time(out, self.n_fft, self.x_channels)
        out = self.interp(out)
        return out
