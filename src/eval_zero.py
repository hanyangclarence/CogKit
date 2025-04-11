import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from pathlib import Path
import typing as tp
import json
import pickle
import os

from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXTransformer3DModel,
)
from diffusers.utils.export_utils import export_to_video
from diffusers.pipelines.cogvideo.pipeline_cogvideox_image2video import *
from transformers import AutoTokenizer, T5EncoderModel
from torchvision import transforms
from decord import VideoReader
from decord import cpu

from cogkit.finetune.diffusion.schemas import TrainableModel
from cogkit.utils.utils import instantiate_from_config, get_obj_from_str
from cogkit.datasets.utils import preprocess_image_with_resize
from cogkit.finetune.utils.rlbench_utils import interpolate_joint_trajectory
from cogkit.finetune.diffusion.schemas.components import DiffusionComponents, TrainableModel
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint


def call(
    self,
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

    height = height or self.transformer.config.sample_height * self.vae_scale_factor_spatial
    width = width or self.transformer.config.sample_width * self.vae_scale_factor_spatial
    num_frames = num_frames or self.transformer.config.sample_frames

    num_videos_per_prompt = 1

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
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
    self._guidance_scale = guidance_scale
    self._current_timestep = None
    self._attention_kwargs = attention_kwargs
    self._interrupt = False

    # 2. Default call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0

    # 3. Encode input prompt
    prompt_embeds, negative_prompt_embeds = self.encode_prompt(
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
    timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
    self._num_timesteps = len(timesteps)

    # 5. Prepare latents
    latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1

    # For CogVideoX 1.5, the latent frames should be padded to make it divisible by patch_size_t
    patch_size_t = self.transformer.config.patch_size_t
    additional_frames = 0
    if patch_size_t is not None and latent_frames % patch_size_t != 0:
        additional_frames = patch_size_t - latent_frames % patch_size_t
        num_frames += additional_frames * self.vae_scale_factor_temporal

    image = self.video_processor.preprocess(image, height=height, width=width).to(
        device, dtype=prompt_embeds.dtype
    )

    latent_channels = self.transformer.config.in_channels // 2
    latents, image_latents = self.prepare_latents(
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
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    # 7. Create rotary embeds if required
    image_rotary_emb = (
        self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
        if self.transformer.config.use_rotary_positional_embeddings
        else None
    )

    # 8. Create ofs embeds if required
    ofs_emb = None if self.transformer.config.ofs_embed_dim is None else latents.new_full((1,), fill_value=2.0)

    # 8. Denoising loop
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

    with self.progress_bar(total=num_inference_steps) as progress_bar:
        # for DPM-solver++
        old_pred_original_sample = None
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            self._current_timestep = t
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            latent_image_input = torch.cat([image_latents] * 2) if do_classifier_free_guidance else image_latents
            latent_model_input = torch.cat([latent_model_input, latent_image_input], dim=2)

            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latent_model_input.shape[0])

            # predict noise model_output
            noise_pred = self.transformer(
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
                self._guidance_scale = 1 + guidance_scale * (
                    (1 - math.cos(math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2
                )
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
            else:
                latents, old_pred_original_sample = self.scheduler.step(
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
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()

            if XLA_AVAILABLE:
                xm.mark_step()

    self._current_timestep = None

    if not output_type == "latent":
        # move the latent and vae to cpu
        latents = latents.to(dtype=torch.float32, device="cpu")
        self.vae = self.vae.to(dtype=torch.float32, device="cpu")
        
        # Discard any padding frames that were added for CogVideoX 1.5
        latents = latents[:, additional_frames:]
        video = self.decode_latents(latents)
        video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        
        # move back
        self.vae.to(dtype=torch.float16, device=self._execution_device)
    else:
        video = latents

    # Offload all models
    self.maybe_free_model_hooks()

    if not return_dict:
        return (video,)

    return CogVideoXPipelineOutput(frames=video)



class LoraEvaluator:
    def __init__(
            self, model_name: str, ckpt_dir: str, additional_cfg: str,
            resolution: tp.Tuple[int, int, int] = (17, 480, 720),
            device: str = "cuda"
    ):
        self.device = device
        self.resolution = resolution
        additional_cfg = OmegaConf.load(additional_cfg)

        self.pipeline_cls = CogVideoXImageToVideoPipeline

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, subfolder="tokenizer")

        self.text_encoder = T5EncoderModel.from_pretrained(model_name, subfolder="text_encoder")

        self.transformer = TrainableModel(
            transformer=CogVideoXTransformer3DModel.from_pretrained(
                model_name, subfolder="transformer"
            ),
            trajectory_encoder=instantiate_from_config(additional_cfg["trajectory_encoder"]),
            trajectory_fuser=instantiate_from_config(additional_cfg["trajectory_fuser"]),
        )

        self.vae = AutoencoderKLCogVideoX.from_pretrained(model_name, subfolder="vae")

        self.scheduler = CogVideoXDPMScheduler.from_pretrained(model_name, subfolder="scheduler")

        # load weights
        load_state_dict_from_zero_checkpoint(self.transformer, ckpt_dir)
        print(f"Loaded checkpoint from {ckpt_dir}")

        self.pipe = self.pipeline_cls(
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            vae=self.vae,
            transformer=self.transformer.transformer,
            scheduler=self.scheduler,
        )

        self.pipe.to(torch.float16)
        self.pipe.to(self.device)
        self.transformer.trajectory_encoder.to(self.device)
        self.transformer.trajectory_fuser.to(self.device)
        self.transformer.trajectory_encoder.half()
        self.transformer.trajectory_fuser.half()

    @torch.no_grad()
    def encode_text(self, prompt: str) -> torch.Tensor:
        prompt_token_ids = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.transformer.transformer.config.max_text_seq_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        prompt_token_ids = prompt_token_ids.input_ids
        prompt_embedding = self.text_encoder(
            prompt_token_ids.to(self.device)
        ).last_hidden_state[0]

        # shape of prompt_embedding: [seq_len, hidden_size]
        assert prompt_embedding.ndim == 2
        return prompt_embedding

    @torch.no_grad()
    def generate(self, prompt: str, image: Image.Image, trajectory: np.ndarray, save_dir: str):
        prompt_embedding = self.encode_text(prompt)  # (seq_len, hidden_size)

        image = preprocess_image_with_resize(image, self.resolution[1], self.resolution[2], torch.device(self.device))
        image = transforms.Compose(
            [transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)]
        )(image)  # (C, H, W)

        trajectory = torch.tensor(trajectory).float()

        prompt_embedding = prompt_embedding[None, ...].to(self.device)  # (1, seq_len, hidden_size)
        trajectory = trajectory[None, ...].to(self.device)  # (1, T, D)

        traj_embedding = self.transformer.encode_trajectory(trajectory)
        prompt_embedding = self.transformer.fuse_trajectory(prompt_embedding, traj_embedding)

        generator = torch.Generator(device=self.device)
        video_generate = call(
            self.pipe,
            num_frames=self.resolution[0],
            height=self.resolution[1],
            width=self.resolution[2],
            prompt_embeds=prompt_embedding,
            image=image,
            generator=generator,
        ).frames

        for idx, videos in enumerate(video_generate):
            filename = f"{save_dir}/video_{idx}.mp4"
            print(f"Saving video to {filename}")

            # resize the video to 480 x 480
            videos = [img.resize((480, 480)) for img in videos]

            export_to_video(videos, filename, fps=5)


if __name__ == "__main__":
    ckpt = "training_logs/checkpoint-3000/"
    config_path = "config/v1.yaml"
    model_name = "THUDM/CogVideoX-5B-I2V"
    data_dir = "/gpfs/u/scratch/LMCG/LMCGhazh/enroot/rlbench_data/root/RACER-DataGen/racer_datagen/rlbench_videos/test"
    save_dir = "generated_videos"
    
    os.makedirs(save_dir, exist_ok=True)

    config = OmegaConf.load(config_path)
    evaluator = LoraEvaluator(
        model_name=model_name,
        ckpt_dir=ckpt,
        additional_cfg=config_path,
        resolution=(17, 480, 720),
        device="cuda",
    )
    
    with open(f"{data_dir}/metadata.jsonl", "r") as f:
        test_data = [json.loads(line) for line in f]
    with open(f"{data_dir}/videos/metadata.jsonl", "r") as f:
        video_path_data = [json.loads(line) for line in f]
    video_path_data = {data["id"]: data["file_name"] for data in video_path_data}
    target_traj_length = config["trajectory_encoder"]["params"]["target_traj_len"]
    get_traj_fn = get_obj_from_str(config["dataset"]["get_traj_fn_name"])
    for data in test_data:
        idx = data["id"]
        prompt = data["prompt"]
        
        traj_path = data["trajectory"]
        demo = pickle.load(open(f"{data_dir}/trajectories/{traj_path}", "rb"))
        trajectory = get_traj_fn(demo, target_traj_length)
        
        video_path = f"{data_dir}/videos/{video_path_data[idx]}"
        vr = VideoReader(video_path, ctx=cpu(0))
        first_frame = Image.fromarray(vr[0].asnumpy())
        
        os.makedirs(f"{save_dir}/{idx}", exist_ok=True)
        evaluator.generate(
            prompt=prompt,
            image=first_frame,
            trajectory=trajectory,
            save_dir=f"{save_dir}/{idx}",
        )
        print(f"Generated video for {idx}: {prompt}")

