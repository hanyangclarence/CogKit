import torch

from diffusers import CogVideoXDPMScheduler
from diffusers.pipelines.cogvideo.pipeline_cogvideox_image2video import *


def call(
    model,
    image: PipelineImageInput,
    prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_frames: int = 49,
    num_inference_steps: int = 50,
    timesteps: Optional[List[int]] = None,
    guidance_scale: float = 6,
    use_dynamic_cfg: bool = False,
    num_videos_per_prompt: int = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: str = "pil",
    return_dict: bool = True,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end: Optional[
        Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
    ] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 226,
) -> Union[CogVideoXPipelineOutput, Tuple]:

    if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
        callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

    height = height or model.transformer.config.sample_height * model.vae_scale_factor_spatial
    width = width or model.transformer.config.sample_width * model.vae_scale_factor_spatial
    num_frames = num_frames or model.transformer.config.sample_frames

    num_videos_per_prompt = 1

    # 1. Check inputs. Raise error if not correct
    model.check_inputs(
        image=image,
        prompt=prompt,
        height=height,
        width=width,
        negative_prompt=negative_prompt,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        latents=latents,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
    )
    model._guidance_scale = guidance_scale
    model._current_timestep = None
    model._attention_kwargs = attention_kwargs
    model._interrupt = False

    # 2. Default call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = model._execution_device

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0

    # 3. Encode input prompt
    prompt_embeds, negative_prompt_embeds = model.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=do_classifier_free_guidance,
        num_videos_per_prompt=num_videos_per_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        max_sequence_length=max_sequence_length,
        device=device,
    )
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

    # 4. Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(model.scheduler, num_inference_steps, device, timesteps)
    model._num_timesteps = len(timesteps)

    # 5. Prepare latents
    latent_frames = (num_frames - 1) // model.vae_scale_factor_temporal + 1

    # For CogVideoX 1.5, the latent frames should be padded to make it divisible by patch_size_t
    patch_size_t = model.transformer.config.patch_size_t
    additional_frames = 0
    if patch_size_t is not None and latent_frames % patch_size_t != 0:
        additional_frames = patch_size_t - latent_frames % patch_size_t
        num_frames += additional_frames * model.vae_scale_factor_temporal

    image = model.video_processor.preprocess(image, height=height, width=width).to(
        device, dtype=prompt_embeds.dtype
    )

    latent_channels = model.transformer.config.in_channels // 2
    latents, image_latents = model.prepare_latents(
        image,
        batch_size * num_videos_per_prompt,
        latent_channels,
        num_frames,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = model.prepare_extra_step_kwargs(generator, eta)

    # 7. Create rotary embeds if required
    image_rotary_emb = (
        model._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
        if model.transformer.config.use_rotary_positional_embeddings
        else None
    )

    # 8. Create ofs embeds if required
    ofs_emb = None if model.transformer.config.ofs_embed_dim is None else latents.new_full((1,), fill_value=2.0)

    # 8. Denoising loop
    num_warmup_steps = max(len(timesteps) - num_inference_steps * model.scheduler.order, 0)

    with model.progress_bar(total=num_inference_steps) as progress_bar:
        # for DPM-solver++
        old_pred_original_sample = None
        for i, t in enumerate(timesteps):
            if model.interrupt:
                continue

            model._current_timestep = t
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = model.scheduler.scale_model_input(latent_model_input, t)

            latent_image_input = torch.cat([image_latents] * 2) if do_classifier_free_guidance else image_latents
            latent_model_input = torch.cat([latent_model_input, latent_image_input], dim=2)

            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latent_model_input.shape[0])

            # predict noise model_output
            noise_pred = model.transformer(
                hidden_states=latent_model_input,
                encoder_hidden_states=prompt_embeds,
                timestep=timestep,
                ofs=ofs_emb,
                image_rotary_emb=image_rotary_emb,
                attention_kwargs=attention_kwargs,
                return_dict=False,
            )[0]
            noise_pred = noise_pred.float()

            # perform guidance
            if use_dynamic_cfg:
                model._guidance_scale = 1 + guidance_scale * (
                    (1 - math.cos(math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2
                )
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + model.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            if not isinstance(model.scheduler, CogVideoXDPMScheduler):
                latents = model.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
            else:
                latents, old_pred_original_sample = model.scheduler.step(
                    noise_pred,
                    old_pred_original_sample,
                    t,
                    timesteps[i - 1] if i > 0 else None,
                    latents,
                    **extra_step_kwargs,
                    return_dict=False,
                )
            latents = latents.to(prompt_embeds.dtype)

            # call the callback, if provided
            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(model, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % model.scheduler.order == 0):
                progress_bar.update()

            if XLA_AVAILABLE:
                xm.mark_step()

    model._current_timestep = None

    if not output_type == "latent":
        # move the latent and vae to cpu
        latents = latents.to(dtype=torch.float32, device="cpu")
        model.vae = model.vae.to(dtype=torch.float32, device="cpu")
        
        # Discard any padding frames that were added for CogVideoX 1.5
        latents = latents[:, additional_frames:]
        video = model.decode_latents(latents)
        video = model.video_processor.postprocess_video(video=video, output_type=output_type)
        
        # move back
        model.vae.to(dtype=torch.float16, device=model._execution_device)
    else:
        video = latents

    # Offload all models
    model.maybe_free_model_hooks()

    if not return_dict:
        return (video,)

    return CogVideoXPipelineOutput(frames=video)


def call_with_depth(
    model,
    image: PipelineImageInput,
    depth: PipelineImageInput,
    prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_frames: int = 49,
    num_inference_steps: int = 50,
    timesteps: Optional[List[int]] = None,
    guidance_scale: float = 6,
    use_dynamic_cfg: bool = False,
    num_videos_per_prompt: int = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: str = "pil",
    return_dict: bool = True,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end: Optional[
        Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
    ] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 226,
) -> Union[CogVideoXPipelineOutput, Tuple]:

    if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
        callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

    height = height or model.transformer.config.sample_height * model.vae_scale_factor_spatial
    width = width or model.transformer.config.sample_width * model.vae_scale_factor_spatial
    num_frames = num_frames or model.transformer.config.sample_frames

    num_videos_per_prompt = 1

    # 1. Check inputs. Raise error if not correct
    model.check_inputs(
        image=image,
        prompt=prompt,
        height=height,
        width=width,
        negative_prompt=negative_prompt,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        latents=latents,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
    )
    model.check_inputs(
        image=depth,
        prompt=prompt,
        height=height,
        width=width,
        negative_prompt=negative_prompt,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        latents=latents,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
    )
    model._guidance_scale = guidance_scale
    model._current_timestep = None
    model._attention_kwargs = attention_kwargs
    model._interrupt = False

    # 2. Default call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = model._execution_device

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0

    # 3. Encode input prompt
    prompt_embeds, negative_prompt_embeds = model.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=do_classifier_free_guidance,
        num_videos_per_prompt=num_videos_per_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        max_sequence_length=max_sequence_length,
        device=device,
    )
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

    # 4. Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(model.scheduler, num_inference_steps, device, timesteps)
    model._num_timesteps = len(timesteps)

    # 5. Prepare latents
    latent_frames = (num_frames - 1) // model.vae_scale_factor_temporal + 1

    # For CogVideoX 1.5, the latent frames should be padded to make it divisible by patch_size_t
    patch_size_t = model.transformer.config.patch_size_t
    additional_frames = 0
    if patch_size_t is not None and latent_frames % patch_size_t != 0:
        additional_frames = patch_size_t - latent_frames % patch_size_t
        num_frames += additional_frames * model.vae_scale_factor_temporal

    image = model.video_processor.preprocess(image, height=height, width=width).to(
        device, dtype=prompt_embeds.dtype
    )
    depth = model.video_processor.preprocess(depth, height=height, width=width).to(
        device, dtype=prompt_embeds.dtype
    )
    
    latent_channels = model.transformer.config.in_channels // 2
    latents, image_latents = model.prepare_latents(
        image,
        batch_size * num_videos_per_prompt,
        latent_channels,
        num_frames,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )
    _, depth_latents = model.prepare_latents(
        depth,
        batch_size * num_videos_per_prompt,
        latent_channels,
        num_frames,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
    )

    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = model.prepare_extra_step_kwargs(generator, eta)

    # 7. Create rotary embeds if required
    image_rotary_emb = (
        model._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
        if model.transformer.config.use_rotary_positional_embeddings
        else None
    )

    # 8. Create ofs embeds if required
    ofs_emb = None if model.transformer.config.ofs_embed_dim is None else latents.new_full((1,), fill_value=2.0)

    # 8. Denoising loop
    num_warmup_steps = max(len(timesteps) - num_inference_steps * model.scheduler.order, 0)

    with model.progress_bar(total=num_inference_steps) as progress_bar:
        # for DPM-solver++
        old_pred_original_sample = None
        for i, t in enumerate(timesteps):
            if model.interrupt:
                continue

            model._current_timestep = t
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = model.scheduler.scale_model_input(latent_model_input, t)

            latent_image_input = torch.cat([image_latents] * 2) if do_classifier_free_guidance else image_latents
            latent_depth_input = torch.cat([depth_latents] * 2) if do_classifier_free_guidance else depth_latents
            latent_model_input = torch.cat([latent_model_input, latent_image_input, latent_depth_input], dim=2)

            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latent_model_input.shape[0])

            # predict noise model_output
            noise_pred = model.transformer(
                hidden_states=latent_model_input,
                encoder_hidden_states=prompt_embeds,
                timestep=timestep,
                ofs=ofs_emb,
                image_rotary_emb=image_rotary_emb,
                attention_kwargs=attention_kwargs,
                return_dict=False,
            )[0]
            noise_pred = noise_pred.float()

            # perform guidance
            if use_dynamic_cfg:
                model._guidance_scale = 1 + guidance_scale * (
                    (1 - math.cos(math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2
                )
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + model.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            if not isinstance(model.scheduler, CogVideoXDPMScheduler):
                latents = model.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
            else:
                latents, old_pred_original_sample = model.scheduler.step(
                    noise_pred,
                    old_pred_original_sample,
                    t,
                    timesteps[i - 1] if i > 0 else None,
                    latents,
                    **extra_step_kwargs,
                    return_dict=False,
                )
            latents = latents.to(prompt_embeds.dtype)

            # call the callback, if provided
            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(model, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % model.scheduler.order == 0):
                progress_bar.update()

            if XLA_AVAILABLE:
                xm.mark_step()

    model._current_timestep = None

    if not output_type == "latent":
        # move the latent and vae to cpu
        latents = latents.to(dtype=torch.float32, device="cpu")
        model.vae = model.vae.to(dtype=torch.float32, device="cpu")
        
        # Discard any padding frames that were added for CogVideoX 1.5
        latents = latents[:, additional_frames:]
        video = model.decode_latents(latents)
        video = model.video_processor.postprocess_video(video=video, output_type=output_type)
        
        # move back
        model.vae.to(dtype=torch.float16, device=model._execution_device)
    else:
        video = latents

    # Offload all models
    model.maybe_free_model_hooks()

    if not return_dict:
        return (video,)

    return CogVideoXPipelineOutput(frames=video)

