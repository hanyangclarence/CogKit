# -*- coding: utf-8 -*-


from typing import Any

from cogkit.finetune.base import BaseComponents


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

    # Trajectory encoder
    trajectory_encoder: Any = None

    # Trajectory fuser
    trajectory_fuser: Any = None
