# -*- coding: utf-8 -*-


from typing import Any

import torch
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXTransformer3DModel,
)
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from PIL import Image
from transformers import AutoTokenizer, T5EncoderModel
from typing_extensions import override
from omegaconf import OmegaConf

from cogkit.finetune import register
from cogkit.finetune.diffusion.schemas import DiffusionComponents, TrainableModel
from cogkit.finetune.diffusion.trainer import DiffusionTrainer
from cogkit.finetune.utils import unwrap_model
from cogkit.utils.utils import instantiate_from_config, get_obj_from_str


class CogVideoXI2VLoraTrainer(DiffusionTrainer):
    UNLOAD_LIST = ["text_encoder"]

    @override
    def load_components(self) -> DiffusionComponents:
        components = DiffusionComponents()
        model_path = str(self.args.model_path)
        self.additional_configs = OmegaConf.load(self.args.config)

        components.pipeline_cls = CogVideoXImageToVideoPipeline

        components.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")

        components.text_encoder = T5EncoderModel.from_pretrained(
            model_path, subfolder="text_encoder"
        )

        components.transformer = TrainableModel(
            transformer=CogVideoXTransformer3DModel.from_pretrained(
                model_path, subfolder="transformer"
            ),
            trajectory_encoder=instantiate_from_config(self.additional_configs["trajectory_encoder"]),
            trajectory_fuser=instantiate_from_config(self.additional_configs["trajectory_fuser"]),
        )

        components.vae = AutoencoderKLCogVideoX.from_pretrained(model_path, subfolder="vae")

        components.scheduler = CogVideoXDPMScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )
        
        # if needed, modify the parameters of the pretrained model
        modify_model_config = self.additional_configs.get("modify_model", None)
        if modify_model_config is not None:
            modify_fn = get_obj_from_str(modify_model_config["target"])
            components = modify_fn(components)

        return components

    @override
    def initialize_pipeline(self) -> CogVideoXImageToVideoPipeline:
        pipe = CogVideoXImageToVideoPipeline(
            tokenizer=self.components.tokenizer,
            text_encoder=self.components.text_encoder,
            vae=self.components.vae,
            transformer=unwrap_model(self.accelerator, self.components.transformer).transformer,
            scheduler=self.components.scheduler,
        )
        return pipe

    @override
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        # shape of input video: [B, C, F, H, W]
        vae = self.components.vae
        video = video.to(vae.device, dtype=vae.dtype)
        latent_dist = vae.encode(video).latent_dist
        latent = latent_dist.sample() * vae.config.scaling_factor
        return latent

    @override
    def encode_text(self, prompt: str) -> torch.Tensor:
        prompt_token_ids = self.components.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.state.transformer_config.max_text_seq_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        prompt_token_ids = prompt_token_ids.input_ids
        prompt_embedding = self.components.text_encoder(
            prompt_token_ids.to(self.accelerator.device)
        ).last_hidden_state[0]

        # shape of prompt_embedding: [seq_len, hidden_size]
        assert prompt_embedding.ndim == 2
        return prompt_embedding

    @override
    def collate_fn(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        ret = {
            "prompt": [],
            "prompt_embedding": [],
            "image": [],
            "image_preprocessed": [],
            "encoded_videos": [],
            "trajectory": [],
            "depth": [],
        }

        for sample in samples:
            prompt = sample["prompt"]
            prompt_embedding = sample["prompt_embedding"]
            image = sample["image"]
            image_preprocessed = sample["image_preprocessed"]
            encoded_video = sample.get("encoded_video", None)
            trajectory = sample["trajectory"]
            depth = sample["depth"]

            ret["prompt"].append(prompt)
            ret["prompt_embedding"].append(prompt_embedding)
            ret["image"].append(image)
            ret["image_preprocessed"].append(image_preprocessed)
            if encoded_video is not None:
                ret["encoded_videos"].append(encoded_video)
            ret["trajectory"].append(trajectory)
            ret["depth"].append(depth)

        ret["prompt_embedding"] = torch.stack(ret["prompt_embedding"])
        ret["image_preprocessed"] = torch.stack(ret["image_preprocessed"])
        ret["encoded_videos"] = (
            torch.stack(ret["encoded_videos"]) if ret["encoded_videos"] else None
        )
        ret["trajectory"] = torch.stack(ret["trajectory"])
        ret["depth"] = torch.stack(ret["depth"])

        return ret

    @override
    def compute_loss(self, batch) -> torch.Tensor:
        prompt_embedding = batch["prompt_embedding"]
        latent = batch["encoded_videos"]
        images = batch["image_preprocessed"]
        depths = batch["depth"]
        trajectory = batch["trajectory"]

        # Shape of prompt_embedding: [B, seq_len, hidden_size]
        # Shape of latent: [B, C, F, H, W]
        # Shape of images: [B, C, H, W]
        # Shape of depth: [B, C, H, W]
        # Shape of trajectory: [B, seq_len, 8]
        
        
        print(f"debug: line 159, latent: {latent.dtype}")
        

        patch_size_t = self.state.transformer_config.patch_size_t
        if patch_size_t is not None:
            ncopy = latent.shape[2] % patch_size_t
            # Copy the first frame ncopy times to match patch_size_t
            first_frame = latent[:, :, :1, :, :]  # Get first frame [B, C, 1, H, W]
            latent = torch.cat([first_frame.repeat(1, 1, ncopy, 1, 1), latent], dim=2)
            assert latent.shape[2] % patch_size_t == 0

        batch_size, num_channels, num_frames, height, width = latent.shape

        # Get prompt embeddings
        _, seq_len, _ = prompt_embedding.shape
        prompt_embedding = prompt_embedding.view(batch_size, seq_len, -1).to(dtype=latent.dtype)

        # embed trajectory
        traj_embedding = self.components.transformer.encode_trajectory(trajectory)
        # fuse trajectory with text prompt embedding
        prompt_embedding = self.components.transformer.fuse_trajectory(prompt_embedding, traj_embedding)

        # Add frame dimension to images [B,C,H,W] -> [B,C,F,H,W]
        images = images.unsqueeze(2)
        # Add noise to images
        image_noise_sigma = torch.normal(
            mean=-3.0, std=0.5, size=(1,), device=self.accelerator.device
        )
        image_noise_sigma = torch.exp(image_noise_sigma).to(dtype=images.dtype)
        noisy_images = (
            images + torch.randn_like(images) * image_noise_sigma[:, None, None, None, None]
        )
        image_latent_dist = self.components.vae.encode(
            noisy_images.to(dtype=self.components.vae.dtype)
        ).latent_dist
        image_latents = image_latent_dist.sample() * self.components.vae.config.scaling_factor

        # Sample a random timestep for each sample
        timesteps = torch.randint(
            0,
            self.components.scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.accelerator.device,
        )
        timesteps = timesteps.long()

        # from [B, C, F, H, W] to [B, F, C, H, W]
        latent = latent.permute(0, 2, 1, 3, 4)
        image_latents = image_latents.permute(0, 2, 1, 3, 4)
        assert (latent.shape[0], *latent.shape[2:]) == (
            image_latents.shape[0],
            *image_latents.shape[2:],
        )

        # Padding image_latents to the same frame number as latent
        padding_shape = (
            latent.shape[0],
            latent.shape[1] - 1,
            *latent.shape[2:],
        )
        latent_padding = image_latents.new_zeros(padding_shape)
        image_latents = torch.cat([image_latents, latent_padding], dim=1)
        
        # Do the same for depth, just without adding noise
        depths = depths.unsqueeze(2)
        depth_noise_sigma = torch.normal(
            mean=-3.0, std=0.5, size=(1,), device=self.accelerator.device
        )
        depth_noise_sigma = torch.exp(depth_noise_sigma).to(dtype=depths.dtype)
        noisy_depths = (
            depths + torch.randn_like(depths) * depth_noise_sigma[:, None, None, None, None]
        )
        depth_latent_dist = self.components.vae.encode(
            noisy_depths.to(dtype=self.components.vae.dtype)
        ).latent_dist
        depth_latents = depth_latent_dist.sample() * self.components.vae.config.scaling_factor
        depth_latents = depth_latents.permute(0, 2, 1, 3, 4)
        assert (latent.shape[0], *latent.shape[2:]) == (
            depth_latents.shape[0], *depth_latents.shape[2:]
        )
        # Padding depth_latents to the same frame number as latent
        latent_padding = depth_latents.new_zeros(padding_shape)
        depth_latents = torch.cat([depth_latents, latent_padding], dim=1)

        # Add noise to latent
        noise = torch.randn_like(latent)
        latent_noisy = self.components.scheduler.add_noise(latent, noise, timesteps)

        # Concatenate latent image_latents, and depth_latents in the channel dimension
        latent_img_noisy = torch.cat([latent_noisy, image_latents, depth_latents], dim=2)

        # Prepare rotary embeds
        vae_scale_factor_spatial = 2 ** (len(self.components.vae.config.block_out_channels) - 1)
        transformer_config = self.state.transformer_config
        rotary_emb = (
            self.prepare_rotary_positional_embeddings(
                height=height * vae_scale_factor_spatial,
                width=width * vae_scale_factor_spatial,
                num_frames=num_frames,
                transformer_config=transformer_config,
                vae_scale_factor_spatial=vae_scale_factor_spatial,
                device=self.accelerator.device,
            )
            if transformer_config.use_rotary_positional_embeddings
            else None
        )
        
        
        print(f"debug: line 267, latent_img_noisy: {latent_img_noisy.dtype}, prompot_embedding: {prompt_embedding.dtype}, timesteps: {timesteps.dtype}")
        

        # Predict noise, For CogVideoX1.5 Only.
        ofs_emb = (
            None
            if self.state.transformer_config.ofs_embed_dim is None
            else latent.new_full((1,), fill_value=2.0)
        )
        predicted_noise = self.components.transformer(
            hidden_states=latent_img_noisy,
            encoder_hidden_states=prompt_embedding,
            timestep=timesteps,
            ofs=ofs_emb,
            image_rotary_emb=rotary_emb,
            return_dict=False,
        )[0]

        # Denoise
        latent_pred = self.components.scheduler.get_velocity(
            predicted_noise, latent_noisy, timesteps
        )

        alphas_cumprod = self.components.scheduler.alphas_cumprod[timesteps]
        weights = 1 / (1 - alphas_cumprod)
        while len(weights.shape) < len(latent_pred.shape):
            weights = weights.unsqueeze(-1)

        loss = torch.mean(
            (weights * (latent_pred - latent) ** 2).reshape(batch_size, -1),
            dim=1,
        )
        loss = loss.mean()

        return loss

    @override
    def validation_step(
        self, eval_data: dict[str, Any], pipe: CogVideoXImageToVideoPipeline
    ) -> list[tuple[str, Image.Image | list[Image.Image]]]:
        """
        Return the data that needs to be saved. For videos, the data format is List[PIL],
        and for images, the data format is PIL
        """
        prompt, prompt_embedding, image, trajectory, _ = (
            eval_data["prompt"],
            eval_data["prompt_embedding"],
            eval_data["image"],
            eval_data["trajectory"],
            eval_data["video"],
        )

        # fuse trajectory with text prompt embedding
        traj_embedding = self.components.transformer.encode_trajectory(trajectory)
        prompt_embedding = self.components.transformer.fuse_trajectory(prompt_embedding, traj_embedding)

        video_generate = pipe(
            num_frames=self.state.train_resolution[0],
            height=self.state.train_resolution[1],
            width=self.state.train_resolution[2],
            prompt_embeds=prompt_embedding,
            image=image,
            generator=self.state.generator,
        ).frames[0]
        return [("prompt", prompt), ("image", image), ("video", video_generate)]

    def prepare_rotary_positional_embeddings(
        self,
        height: int,
        width: int,
        num_frames: int,
        transformer_config: dict,
        vae_scale_factor_spatial: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        grid_height = height // (vae_scale_factor_spatial * transformer_config.patch_size)
        grid_width = width // (vae_scale_factor_spatial * transformer_config.patch_size)

        if transformer_config.patch_size_t is None:
            base_num_frames = num_frames
        else:
            base_num_frames = (
                num_frames + transformer_config.patch_size_t - 1
            ) // transformer_config.patch_size_t

        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=transformer_config.attention_head_dim,
            crops_coords=None,
            grid_size=(grid_height, grid_width),
            temporal_size=base_num_frames,
            grid_type="slice",
            max_size=(grid_height, grid_width),
            device=device,
        )

        return freqs_cos, freqs_sin


# register("cogvideox-i2v", "lora", CogVideoXI2VLoraTrainer)
