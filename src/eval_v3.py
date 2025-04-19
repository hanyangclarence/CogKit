from PIL import Image
from omegaconf import OmegaConf
import json
import pickle
import os
import argparse
from decord import VideoReader
from decord import cpu
import random

from cogkit.utils.utils import get_obj_from_str

task_list = [
    "put_item_in_drawer",             # 0                 
    "reach_and_drag",                 # 1                 
    "turn_tap",                       # 2  --> [0:3]      
    "slide_block_to_color_target",    # 3                 
    "open_drawer",                    # 4                 
    "put_groceries_in_cupboard",      # 5  --> [3:6]      
    "place_shape_in_shape_sorter",    # 6                 
    "put_money_in_safe",              # 7                 
    "push_buttons",                   # 8  --> [6:9]      
    "close_jar",                      # 9                 
    "stack_blocks",                   # 10                
    "place_cups",                     # 11 --> [9:12]     
    "place_wine_at_rack_location",    # 12                
    "light_bulb_in",                  # 13                
    "sweep_to_dustpan_of_size",       # 14 --> [12:15]    
    "insert_onto_square_peg",         # 15                
    "meat_off_grill",                 # 16                
    "stack_cups",                     # 17 --> [15:18]
]

# This version evaluates the model that conditioned on trajectory, first frame and first frame depth


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--max_num_per_task", type=int, default=5)
    args = parser.parse_args()
    
    ckpt = args.ckpt
    config_path = args.config
    model_name = "THUDM/CogVideoX-5B-I2V"
    data_dir = "/gpfs/u/scratch/LMCG/LMCGhazh/enroot/rlbench_data/root/RACER-DataGen/racer_datagen/rlbench_videos/test"
    save_dir = "generated_videos"
    use_lora = False
    max_num_per_task = args.max_num_per_task
    
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
    get_traj_fn = get_obj_from_str(config["dataset"]["params"]["get_traj_fn_name"])
    
    random.seed(0)
    random.shuffle(test_data)
    task_num_count = {task: 0 for task in task_list}
    
    for data in test_data:
        idx = data["id"]
        prompt = data["prompt"]
        
        # parse filename to get task name
        filename = video_path_data[idx]
        filename = filename.split("perturb")[0] if "perturb" in filename else filename.split("expert")[0]
        filename = "_".join(filename.split("_")[:-3])
        if filename not in task_list:
            print(f"Task {filename} not in task list, skipping...")
            continue
        if task_num_count[filename] >= max_num_per_task:
            print(f"Task {filename} already has {max_num_per_task} samples, skipping...")
            continue
        task_num_count[filename] += 1
        
        traj_path = data["trajectory"]
        demo = pickle.load(open(f"{data_dir}/trajectories/{traj_path}", "rb"))
        trajectory = get_traj_fn(demo, target_traj_length)
        
        video_path = f"{data_dir}/videos/{video_path_data[idx]}"
        vr = VideoReader(video_path, ctx=cpu(0))
        first_frame = Image.fromarray(vr[0].asnumpy())
        
        depth_path = f"{data_dir}/depth/{depth_path_data[idx]}"
        depth_vr = VideoReader(depth_path, ctx=cpu(0))
        first_frame_depth = Image.fromarray(depth_vr[0].asnumpy())
        
        os.makedirs(f"{save_dir}/{idx}_{video_path_data[idx]}", exist_ok=True)
        evaluator.generate(
            prompt=prompt,
            image=first_frame,
            depth=first_frame_depth,
            trajectory=trajectory,
            save_dir=f"{save_dir}/{idx}_{video_path_data[idx]}",
        )
        print(f"Generated video for {idx}: {prompt}")