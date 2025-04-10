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
from transformers import AutoTokenizer, T5EncoderModel
from peft import (
    set_peft_model_state_dict,
)
from torchvision import transforms
from decord import VideoReader
from decord import cpu

from cogkit.finetune.diffusion.schemas import TrainableModel
from cogkit.utils.utils import instantiate_from_config
from cogkit.datasets.utils import preprocess_image_with_resize
from cogkit.finetune.utils.rlbench_utils import interpolate_joint_trajectory




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

        # load pretrained lora
        lora_state_dict = self.pipeline_cls.lora_state_dict(ckpt_dir)
        transformer_state_dict = {
            f"{k.replace('transformer.', '')}": v
            for k, v in lora_state_dict.items()
            if k.startswith("transformer.")
        }
        incompatible_keys = set_peft_model_state_dict(
            self.transformer.transformer, transformer_state_dict, adapter_name="default"
        )
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                print(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

        # load the trajectory encoder/fusor
        traj_module_path = Path(ckpt_dir, "trajectory_modules.pt")
        assert traj_module_path.exists(), f"Cannot find {traj_module_path}"
        traj_module_dict = torch.load(traj_module_path, map_location="cpu")
        self.transformer.trajectory_encoder.load_state_dict(
            traj_module_dict["trajectory_encoder"]
        )
        self.transformer.trajectory_fuser.load_state_dict(
            traj_module_dict["trajectory_fuser"]
        )
        print(f"Loaded trajectory encoder/fusor from {traj_module_path}")

        self.pipe = self.pipeline_cls(
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            vae=self.vae,
            transformer=self.transformer.transformer,
            scheduler=self.scheduler,
        )

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
        video_generate = self.pipe(
            num_frames=self.resolution[0],
            height=self.resolution[1],
            width=self.resolution[2],
            prompt_embeds=prompt_embedding,
            image=image,
            generator=generator,
        ).frames[0]

        for idx, (type, value) in enumerate(video_generate):
            if type == "video":
                filename = f"{save_dir}/video_{idx}.mp4"
                print(f"Saving video to {filename}")
                export_to_video(value, filename, fps=5)


if __name__ == "__main__":
    ckpt = ""
    config = ""
    model_name = "THUDM/CogVideoX-5B-I2V"
    additional_cfg = ""
    data_dir = "/gpfs/u/scratch/LMCG/LMCGhazh/enroot/rlbench_data/root/RACER-DataGen/racer_datagen/rlbench_videos/test"
    save_dir = "generated_videos"
    
    os.makedirs(save_dir, exist_ok=True)

    config = OmegaConf.load(config)
    evaluator = LoraEvaluator(
        model_name=model_name,
        ckpt_dir=ckpt,
        additional_cfg=additional_cfg,
        resolution=(17, 480, 720),
        device="cuda",
    )
    
    test_data = json.load(open(f"{data_dir}/metadata.jsonl", "r"))
    video_path_data = json.load(open(f"{data_dir}/videos/metadata.jsonl", "r"))
    video_path_data = {data["id"]: data["file_name"] for data in video_path_data}
    target_traj_length = config["trajectory_encoder"]["params"]["target_traj_len"]
    for data in test_data:
        idx = data["id"]
        prompt = data["prompt"]
        
        traj_path = data["trajectory"]
        demo = pickle.load(open(f"{data_dir}/trajectories/{traj_path}", "rb"))
        joint_traj = demo[-1].gt_path  # (T, 7)
        joint_traj = interpolate_joint_trajectory(joint_traj, target_traj_length)

        gripper_start = demo[0].gripper_open
        gripper_end = demo[-1].gripper_open
        gripper_status_traj = np.zeros((target_traj_length, 2))
        gripper_status_traj[:, 0] = gripper_start
        gripper_status_traj[:, 1] = gripper_end
        trajectory = np.concatenate([joint_traj, gripper_status_traj], axis=1)
        
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

