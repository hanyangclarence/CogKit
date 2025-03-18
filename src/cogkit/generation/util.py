# -*- coding: utf-8 -*-


from diffusers import (
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXPipeline,
    CogVideoXVideoToVideoPipeline,
    CogView4Pipeline,
)

TVideoPipeline = CogVideoXPipeline | CogVideoXImageToVideoPipeline | CogVideoXVideoToVideoPipeline
TPipeline = CogView4Pipeline | TVideoPipeline


def _guess_cogview_resolution(
    pipeline: CogView4Pipeline, height: int | None = None, width: int | None = None
) -> tuple[int, int]:
    default_height = pipeline.transformer.config.sample_size * pipeline.vae_scale_factor
    default_width = pipeline.transformer.config.sample_size * pipeline.vae_scale_factor
    if height is None and width is None:
        return default_height, default_width

    if height is None:
        height = int(width * default_height / default_width)

    if width is None:
        width = int(height * default_width / default_height)
    # FIXME: checks if `(height, width)` is reasonable. If not, warn users and return the default/recommend resolution when required.
    return height, width


def _guess_cogvideox_resolution(
    pipeline: TVideoPipeline, height: int | None, width: int | None = None
) -> tuple[int, int]:
    default_height = pipeline.transformer.config.sample_height * pipeline.vae_scale_factor_spatial
    default_width = pipeline.transformer.config.sample_width * pipeline.vae_scale_factor_spatial

    if height is None and width is None:
        return default_height, default_width

    if height is None:
        height = int(width * default_height / default_width)

    if width is None:
        width = int(height * default_width / default_height)

    # FIXME: checks if `(height, width)` is reasonable. If not, warn users and return the default/recommend resolution when required.
    return height, width


def guess_resolution(
    pipeline: TPipeline,
    height: int | None = None,
    width: int | None = None,
) -> tuple[int, int]:
    if isinstance(pipeline, CogView4Pipeline):
        return _guess_cogview_resolution(pipeline, height=height, width=width)
    if isinstance(pipeline, TVideoPipeline):
        return _guess_cogvideox_resolution(pipeline, height=height, width=width)

    err_msg = f"The pipeline '{pipeline.__class__.__name__}' is not supported."
    raise ValueError(err_msg)


def before_generation(pipeline: TPipeline) -> None:
    if isinstance(pipeline, TVideoPipeline):
        pipeline.scheduler = CogVideoXDPMScheduler.from_config(
            pipeline.scheduler.config, timestep_spacing="trailing"
        )

    # * enables CPU offload for the model.
    # turns off if you have multiple GPUs or enough GPU memory(such as H100) and it will cost less time in inference
    # and enable to("cuda")
    # pipe.to("cuda")

    # pipeline.to("cuda")
    pipeline.enable_model_cpu_offload()
    # pipe.enable_sequential_cpu_offload()
    if hasattr(pipeline, "vae"):
        pipeline.vae.enable_slicing()
        pipeline.vae.enable_tiling()
