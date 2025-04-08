# -*- coding: utf-8 -*-


from typing import Any
import torch.nn as nn

from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin
from diffusers.loaders.single_file_model import FromOriginalModelMixin

from cogkit.finetune.base import BaseComponents


class TrainableModel(ModelMixin):
    def __init__(self, transformer: nn.Module,
                 trajectory_encoder: nn.Module,
                 trajectory_fuser: nn.Module):
        super().__init__()
        self.transformer = transformer
        self.trajectory_encoder = trajectory_encoder
        self.trajectory_fuser = trajectory_fuser

    def forward(self, **kwargs):
        output = self.transformer(**kwargs)
        return output

    def encode_trajectory(self, trajectory):
        output = self.trajectory_encoder(trajectory)
        return output

    def fuse_trajectory(self, prompt_embedding, traj_embedding):
        output = self.trajectory_fuser(prompt_embedding, traj_embedding)
        return output


class DiffusionComponents(BaseComponents):
    # Tokenizers
    tokenizer: Any = None
    tokenizer_2: Any = None
    tokenizer_3: Any = None

    # Text encoders
    text_encoder: Any = None
    text_encoder_2: Any = None
    text_encoder_3: Any = None

    # Autoencoder
    vae: Any = None

    # Add additional Denoiser type
    unet: Any = None

    # Scheduler
    scheduler: Any = None
