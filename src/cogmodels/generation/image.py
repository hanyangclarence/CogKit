# -*- coding: utf-8 -*-


import os
from pathlib import Path

import torch
from diffusers import DiffusionPipeline

from cogmodels.generation.util import before_generation, guess_image_resolution
from cogmodels.logging import get_logger
from cogmodels.utils import mkdir, rand_generator, resolve_path

_logger = get_logger(__name__)


def generate_image(
    prompt: str,
    model_id_or_path: str,
    save_file: str | Path,
    # * params for model loading
    dtype: torch.dtype = torch.bfloat16,
    # * params for generated images
    height: int | None = None,
    width: int | None = None,
    # * params for the generation process
    num_images_per_prompt: int = 1,
    num_inference_steps: int = 50,
    guidance_scale: float = 3.5,
    seed: int | None = 42,
):
    pipeline = DiffusionPipeline.from_pretrained(model_id_or_path, torch_dtype=dtype)

    height, width = guess_image_resolution(pipeline, height, width)

    before_generation(pipeline)

    batch_image = pipeline(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images_per_prompt,
        generator=rand_generator(seed),
    ).images

    save_file = resolve_path(save_file)
    mkdir(save_file.parent)
    _logger.info("Saving the generated image to path '%s'.", os.fspath(save_file))
    batch_image[0].save(save_file)
