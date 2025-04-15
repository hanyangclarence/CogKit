# -*- coding: utf-8 -*-


from diffusers.loaders import CogVideoXLoraLoaderMixin


def load_lora_checkpoint(
    pipeline: CogVideoXLoraLoaderMixin,
    lora_model_id_or_path: str,
    lora_scale: float = 1.0,
) -> None:
    pipeline.load_lora_weights(lora_model_id_or_path)
    pipeline.fuse_lora(components=["transformer"], lora_scale=lora_scale)


def unload_lora_checkpoint(
    pipeline: CogVideoXLoraLoaderMixin,
) -> None:
    pipeline.unload_lora_weights()
