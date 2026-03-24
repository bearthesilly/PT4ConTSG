"""Models package for ConTSG."""

from contsg.models.base import BaseGeneratorModule, BaseGANModule, DiffusionMixin

# Import example models to trigger registration
from contsg.models import examples

# Import all migrated models to trigger registration
from contsg.models import verbalts  # Multi-view noise estimation with adaLN
from contsg.models import timeweaver  # Heterogeneous attribute conditioning
from contsg.models import bridge  # Domain-agnostic diffusion with prototyper
from contsg.models import t2s  # Flow matching with Diffusion Transformer
from contsg.models import tedit  # Multi-scale patch diffusion
from contsg.models import diffusets  # Latent diffusion with VAE (two-stage)
from contsg.models import cttp  # Contrastive text-to-time series pretraining (eval)
from contsg.models import timevqvae  # VQ-VAE + MaskGIT (label-conditioned)
from contsg.models import wavestitch  # WaveStitch core diffusion (SSSDS4Imputer)
from contsg.models import text2motion  # Text-to-Motion (CVPR'22) backbone adapted for time series
from contsg.models import ttscgan  # TTS-CGAN: Transformer GAN (label-conditioned)
from contsg.models import pt_factor_generator_model
__all__ = ["BaseGeneratorModule", "BaseGANModule", "DiffusionMixin"]
