"""TimeVQVAE implementation (LF/HF VQ-VAE)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from contsg.models.timevqvae_modules.encoder_decoders.vq_vae_encdec import VQVAEEncoder, VQVAEDecoder
from contsg.models.timevqvae_modules.utils import (
    compute_downsample_rate,
    quantize,
    time_to_timefreq,
    timefreq_to_time,
    zero_pad_high_freq,
    zero_pad_low_freq,
)
from contsg.models.timevqvae_modules.vector_quantization import VectorQuantize


class TimeVQVAE(nn.Module):
    def __init__(self, configs: dict):
        super().__init__()
        self.input_length = configs["timepoint"]
        self.in_channels = configs["variable_num"]
        self.configs = configs

        self.n_fft = configs["VQ-VAE"]["n_fft"]
        init_dim = configs["encoder"]["init_dim"]
        hid_dim = configs["encoder"]["hid_dim"]
        downsampled_width_l = configs["encoder"]["downsampled_width"]["lf"]
        downsampled_width_h = configs["encoder"]["downsampled_width"]["hf"]
        downsample_rate_l = compute_downsample_rate(self.input_length, self.n_fft, downsampled_width_l)
        downsample_rate_h = compute_downsample_rate(self.input_length, self.n_fft, downsampled_width_h)

        self.encoder_l = VQVAEEncoder(
            init_dim,
            hid_dim,
            2 * self.in_channels,
            downsample_rate_l,
            configs["encoder"]["n_resnet_blocks"],
            "lf",
            self.n_fft,
            frequency_indepence=True,
        )
        self.encoder_h = VQVAEEncoder(
            init_dim,
            hid_dim,
            2 * self.in_channels,
            downsample_rate_h,
            configs["encoder"]["n_resnet_blocks"],
            "hf",
            self.n_fft,
            frequency_indepence=False,
        )

        self.vq_model_l = VectorQuantize(hid_dim, configs["VQ-VAE"]["codebook_sizes"]["lf"], **configs["VQ-VAE"])
        self.vq_model_h = VectorQuantize(hid_dim, configs["VQ-VAE"]["codebook_sizes"]["hf"], **configs["VQ-VAE"])

        self.decoder_l = VQVAEDecoder(
            init_dim,
            hid_dim,
            2 * self.in_channels,
            downsample_rate_l,
            configs["decoder"]["n_resnet_blocks"],
            self.input_length,
            "lf",
            self.n_fft,
            self.in_channels,
            frequency_indepence=True,
        )
        self.decoder_h = VQVAEDecoder(
            init_dim,
            hid_dim,
            2 * self.in_channels,
            downsample_rate_h,
            configs["decoder"]["n_resnet_blocks"],
            self.input_length,
            "hf",
            self.n_fft,
            self.in_channels,
            frequency_indepence=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        recons_loss = {"LF.time": 0.0, "HF.time": 0.0}
        vq_losses = {"LF": None, "HF": None}
        perplexities = {"LF": 0.0, "HF": 0.0}

        in_channels = x.shape[1]
        xf = time_to_timefreq(x, self.n_fft, in_channels)
        u_l = zero_pad_high_freq(xf)
        x_l = F.interpolate(
            timefreq_to_time(u_l, self.n_fft, in_channels),
            self.input_length,
            mode="linear",
        )
        u_h = zero_pad_low_freq(xf)
        x_h = F.interpolate(
            timefreq_to_time(u_h, self.n_fft, in_channels),
            self.input_length,
            mode="linear",
        )

        z_l = self.encoder_l(x)
        z_q_l, s_l, vq_loss_l, perplexity_l = quantize(z_l, self.vq_model_l)
        xhat_l = self.decoder_l(z_q_l)

        z_h = self.encoder_h(x)
        z_q_h, s_h, vq_loss_h, perplexity_h = quantize(z_h, self.vq_model_h)
        xhat_h = self.decoder_h(z_q_h)

        recons_loss["LF.time"] = F.mse_loss(x_l, xhat_l)
        perplexities["LF"] = perplexity_l
        vq_losses["LF"] = vq_loss_l

        recons_loss["HF.time"] = F.l1_loss(x_h, xhat_h)
        perplexities["HF"] = perplexity_h
        vq_losses["HF"] = vq_loss_h

        loss = (
            recons_loss["LF.time"]
            + recons_loss["HF.time"]
            + vq_losses["LF"]["loss"]
            + vq_losses["HF"]["loss"]
        )

        return loss

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct time series without computing loss.

        Args:
            x: Input time series (B, C, L)

        Returns:
            Reconstructed time series (B, C, L)
        """
        z_l = self.encoder_l(x)
        z_q_l, _, _, _ = quantize(z_l, self.vq_model_l)
        xhat_l = self.decoder_l(z_q_l)

        z_h = self.encoder_h(x)
        z_q_h, _, _, _ = quantize(z_h, self.vq_model_h)
        xhat_h = self.decoder_h(z_q_h)

        return xhat_l + xhat_h
