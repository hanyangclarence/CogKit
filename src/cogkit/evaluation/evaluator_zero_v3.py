import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
import typing as tp

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

from cogkit.finetune.diffusion.schemas import TrainableModel
from cogkit.utils.utils import instantiate_from_config, get_obj_from_str
from cogkit.datasets.utils import preprocess_image_with_resize
from cogkit.finetune.diffusion.schemas.components import TrainableModel
from cogkit.evaluation.utils import call_with_depth
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint



class Evaluator:
    def __init__(
            self, model_name: str, ckpt_dir: str, additional_cfg: str,
            resolution: tp.Tuple[int, int, int] = (17, 480, 720),
            device: str = "cuda", dtype: torch.dtype = torch.float16
    ):
        self.device = device
        self.dtype = dtype
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
        
        modify_model_config = additional_cfg["modify_model"]
        modify_fn = get_obj_from_str(modify_model_config["target"])
        self = modify_fn(self)

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

        self.pipe.to(self.dtype)
        self.pipe.to(self.device)
        self.transformer.trajectory_encoder.to(self.device)
        self.transformer.trajectory_fuser.to(self.device)
        self.transformer.trajectory_encoder.to(self.dtype)
        self.transformer.trajectory_fuser.to(self.dtype)

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
    
    def preprocess_depth(self, depth: Image.Image) -> torch.Tensor:
        depth = depth.resize((self.resolution[2], self.resolution[1]), Image.Resampling.BILINEAR)
        depth = torch.from_numpy(np.array(depth)).permute(2, 0, 1).float().contiguous()
        
        assert depth.shape == (3, self.resolution[1], self.resolution[2])
        depth = transforms.Compose(
            [transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)]
        )(depth)
        return depth
        

    @torch.no_grad()
    def generate(self, prompt: str, image: Image.Image, depth: Image.Image, trajectory: np.ndarray, save_dir: str):
        prompt_embedding = self.encode_text(prompt)  # (seq_len, hidden_size)

        image = preprocess_image_with_resize(image, self.resolution[1], self.resolution[2], torch.device(self.device))
        image = transforms.Compose(
            [transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)]
        )(image)  # (C, H, W)
        
        depth = self.preprocess_depth(depth)  # (C, H, W)

        trajectory = torch.tensor(trajectory).float()

        prompt_embedding = prompt_embedding[None, ...].to(self.device)  # (1, seq_len, hidden_size)
        trajectory = trajectory[None, ...].to(self.device)  # (1, T, D)

        traj_embedding = self.transformer.encode_trajectory(trajectory)
        prompt_embedding = self.transformer.fuse_trajectory(prompt_embedding, traj_embedding)

        generator = torch.Generator(device=self.device)
        video_generate = call_with_depth(
            self.pipe,
            num_frames=self.resolution[0],
            height=self.resolution[1],
            width=self.resolution[2],
            prompt_embeds=prompt_embedding,
            image=image,
            depth=depth,
            generator=generator,
            dtype=self.dtype,
        ).frames

        for idx, videos in enumerate(video_generate):
            filename = f"{save_dir}/video_{idx}.mp4"
            print(f"Saving video to {filename}")

            # resize the video to 480 x 480
            videos = [img.resize((480, 480)) for img in videos]

            export_to_video(videos, filename, fps=5)
