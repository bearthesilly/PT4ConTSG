import math
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from contsg.models.wavestitch_modules.s4 import S4Layer


def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


def calc_diffusion_step_embedding(
    diffusion_steps: Tensor, diffusion_step_embed_dim_in: int
) -> Tensor:
    if diffusion_step_embed_dim_in % 2 != 0:
        raise ValueError("diffusion_step_embed_dim_in must be even")

    half_dim = diffusion_step_embed_dim_in // 2
    device = diffusion_steps.device
    scale = math.log(10000.0) / (half_dim - 1)
    freq = torch.exp(torch.arange(half_dim, device=device) * -scale)
    emb = diffusion_steps.float() * freq
    return torch.cat((torch.sin(emb), torch.cos(emb)), dim=1)


class Conv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=padding)
        self.conv = nn.utils.parametrizations.weight_norm(conv)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class ZeroConv1d(nn.Module):
    def __init__(self, in_channel: int, out_channel: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=1, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        res_channels: int,
        skip_channels: int,
        diffusion_step_embed_dim_out: int,
        in_channels: int,
        s4_lmax: int,
        s4_d_state: int,
        s4_dropout: float,
        s4_bidirectional: bool,
        s4_layernorm: bool,
        cond_channels: int = 0,
    ):
        super().__init__()
        self.res_channels = res_channels
        self.fc_t = nn.Linear(diffusion_step_embed_dim_out, res_channels)
        self.cond_channels = int(cond_channels)

        self.s4_1 = S4Layer(
            features=2 * res_channels,
            lmax=s4_lmax,
            N=s4_d_state,
            dropout=s4_dropout,
            bidirectional=s4_bidirectional,
            layer_norm=s4_layernorm,
        )
        self.s4_2 = S4Layer(
            features=2 * res_channels,
            lmax=s4_lmax,
            N=s4_d_state,
            dropout=s4_dropout,
            bidirectional=s4_bidirectional,
            layer_norm=s4_layernorm,
        )

        self.conv_layer = Conv(res_channels, 2 * res_channels, kernel_size=3)

        self.cond_conv = None
        if self.cond_channels > 0:
            self.cond_conv = Conv(self.cond_channels, 2 * res_channels, kernel_size=1)

        self.res_conv = nn.Conv1d(res_channels, res_channels, kernel_size=1)
        self.res_conv = nn.utils.parametrizations.weight_norm(self.res_conv)
        nn.init.kaiming_normal_(self.res_conv.weight)

        self.skip_conv = nn.Conv1d(res_channels, skip_channels, kernel_size=1)
        self.skip_conv = nn.utils.parametrizations.weight_norm(self.skip_conv)
        nn.init.kaiming_normal_(self.skip_conv.weight)

    def forward(
        self,
        x: Tensor,
        diffusion_step_embed: Tensor,
        cond: Tensor | None = None,
    ) -> Tuple[Tensor, Tensor]:
        h = x
        batch_size, channels, _ = x.shape
        if channels != self.res_channels:
            raise ValueError("Input channels must match res_channels")

        step = self.fc_t(diffusion_step_embed).view(batch_size, self.res_channels, 1)
        h = h + step

        h = self.conv_layer(h)
        h = self.s4_1(h.permute(2, 0, 1)).permute(1, 2, 0)
        if self.cond_conv is not None:
            if cond is None:
                raise ValueError("Conditional input is required when cond_channels > 0")
            h = h + self.cond_conv(cond)
        h = self.s4_2(h.permute(2, 0, 1)).permute(1, 2, 0)

        out = torch.tanh(h[:, : self.res_channels, :]) * torch.sigmoid(
            h[:, self.res_channels :, :]
        )

        res = self.res_conv(out)
        skip = self.skip_conv(out)
        return (x + res) * math.sqrt(0.5), skip


class ResidualGroup(nn.Module):
    def __init__(
        self,
        res_channels: int,
        skip_channels: int,
        num_res_layers: int,
        diffusion_step_embed_dim_in: int,
        diffusion_step_embed_dim_mid: int,
        diffusion_step_embed_dim_out: int,
        in_channels: int,
        s4_lmax: int,
        s4_d_state: int,
        s4_dropout: float,
        s4_bidirectional: bool,
        s4_layernorm: bool,
        cond_channels: int = 0,
    ):
        super().__init__()
        self.num_res_layers = num_res_layers
        self.diffusion_step_embed_dim_in = diffusion_step_embed_dim_in
        self.fc_t1 = nn.Linear(diffusion_step_embed_dim_in, diffusion_step_embed_dim_mid)
        self.fc_t2 = nn.Linear(diffusion_step_embed_dim_mid, diffusion_step_embed_dim_out)

        self.residual_blocks = nn.ModuleList(
            [
                ResidualBlock(
                    res_channels=res_channels,
                    skip_channels=skip_channels,
                    diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                    in_channels=in_channels,
                    cond_channels=cond_channels,
                    s4_lmax=s4_lmax,
                    s4_d_state=s4_d_state,
                    s4_dropout=s4_dropout,
                    s4_bidirectional=s4_bidirectional,
                    s4_layernorm=s4_layernorm,
                )
                for _ in range(num_res_layers)
            ]
        )

    def forward(
        self,
        noise: Tensor,
        diffusion_steps: Tensor,
        cond: Tensor | None = None,
    ) -> Tensor:
        diffusion_step_embed = calc_diffusion_step_embedding(
            diffusion_steps, self.diffusion_step_embed_dim_in
        )
        diffusion_step_embed = swish(self.fc_t1(diffusion_step_embed))
        diffusion_step_embed = swish(self.fc_t2(diffusion_step_embed))

        h = noise
        skip = 0.0
        for block in self.residual_blocks:
            h, skip_n = block(h, diffusion_step_embed, cond)
            skip = skip + skip_n

        return skip * math.sqrt(1.0 / self.num_res_layers)


class SSSDS4Imputer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        res_channels: int,
        skip_channels: int,
        out_channels: int,
        num_res_layers: int,
        diffusion_step_embed_dim_in: int,
        diffusion_step_embed_dim_mid: int,
        diffusion_step_embed_dim_out: int,
        s4_lmax: int,
        s4_d_state: int,
        s4_dropout: float,
        s4_bidirectional: bool,
        s4_layernorm: bool,
        cond_channels: int = 0,
    ):
        super().__init__()

        self.cond_channels = int(cond_channels)
        self.init_conv = nn.Sequential(Conv(in_channels, res_channels, kernel_size=1), nn.ReLU())
        self.residual_layer = ResidualGroup(
            res_channels=res_channels,
            skip_channels=skip_channels,
            num_res_layers=num_res_layers,
            diffusion_step_embed_dim_in=diffusion_step_embed_dim_in,
            diffusion_step_embed_dim_mid=diffusion_step_embed_dim_mid,
            diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
            in_channels=in_channels,
            cond_channels=self.cond_channels,
            s4_lmax=s4_lmax,
            s4_d_state=s4_d_state,
            s4_dropout=s4_dropout,
            s4_bidirectional=s4_bidirectional,
            s4_layernorm=s4_layernorm,
        )
        self.final_conv = nn.Sequential(
            Conv(skip_channels, skip_channels, kernel_size=1),
            nn.ReLU(),
            ZeroConv1d(skip_channels, out_channels),
        )

    def forward(
        self,
        input_data: Tensor,
        timesteps: Tensor,
        cond: Tensor | None = None,
    ) -> Tensor:
        noised_data = input_data.permute((0, 2, 1))
        cond_data = None
        if self.cond_channels > 0:
            if cond is None:
                raise ValueError("Conditional input is required when cond_channels > 0")
            cond_data = cond.permute((0, 2, 1))
        x = self.init_conv(noised_data)
        x = self.residual_layer(x, timesteps, cond_data)
        return self.final_conv(x)
