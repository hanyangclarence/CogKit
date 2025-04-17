from PIL import Image
from omegaconf import OmegaConf
import json
import pickle
import os
import argparse
from decord import VideoReader
from decord import cpu
from torchvision import transforms

from cogkit.utils.utils import get_obj_from_str

# This version evaluates the model that conditioned on trajectory, first frame and first frame depth


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--config", type=str, default="")
    args = parser.parse_args()
    
    ckpt = args.ckpt
    config_path = args.config
    model_name = "THUDM/CogVideoX-5B-I2V"
    data_dir = "/gpfs/u/scratch/LMCG/LMCGhazh/enroot/rlbench_data/root/RACER-DataGen/racer_datagen/rlbench_videos/test"
    save_dir = "generated_videos"
    use_lora = False
    
    # Create the save directory
    os.makedirs(save_dir, exist_ok=True)
    config_name = os.path.basename(config_path).split(".")[0]
    config_name += "_lora" if use_lora else "_zero"
    save_dir = os.path.join(save_dir, config_name)
    os.makedirs(save_dir, exist_ok=True)

    config = OmegaConf.load(config_path)
    
    evaluator_class = get_obj_from_str(config["evaluation"]["lora_evaluator"]) if use_lora else get_obj_from_str(config["evaluation"]["zero_evaluator"])
    
    evaluator = evaluator_class(
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
    with open(f"{data_dir}/depth/metadata.jsonl", "r") as f:
        depth_path_data = [json.loads(line) for line in f]
    video_path_data = {data["id"]: data["file_name"] for data in video_path_data}
    depth_path_data = {data["id"]: data["file_name"] for data in depth_path_data}
    
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
        
        depth_path = f"{data_dir}/depth/{depth_path_data[idx]}"
        depth_vr = VideoReader(depth_path, ctx=cpu(0))
        first_frame_depth = Image.fromarray(depth_vr[0].asnumpy())
        
        os.makedirs(f"{save_dir}/{idx}", exist_ok=True)
        evaluator.generate(
            prompt=prompt,
            image=first_frame,
            trajectory=trajectory,
            save_dir=f"{save_dir}/{idx}",
        )
        print(f"Generated video for {idx}: {prompt}")