import pickle
import torch
import numpy as np

from rlbench.demo import Demo
from cogkit.finetune.utils.rlbench_utils import interpolate_joint_trajectory, interpolate_joint_gripper_trajectory

def get_joints(demo: Demo, target_traj_length: int) -> np.ndarray:
    
    joint_traj = demo[-1].gt_path  # (T, 7)
    joint_traj = interpolate_joint_trajectory(joint_traj, target_traj_length)
    
    gripper_start = demo[0].gripper_open
    gripper_end = demo[-1].gripper_open
    gripper_status_traj = np.zeros((target_traj_length, 2))
    gripper_status_traj[:, 0] = gripper_start
    gripper_status_traj[:, 1] = gripper_end
    trajectory = np.concatenate([joint_traj, gripper_status_traj], axis=1)  # (T', 9)
    return trajectory

def get_traj_from_obs(demo: Demo, target_traj_length: int) -> np.ndarray:
    traj = np.concatenate(
        [np.concatenate([obs.gripper_pose, [obs.gripper_open]])[None, ...] for obs in demo], 
        axis=0
    )  # (T, 8)
    traj = interpolate_joint_gripper_trajectory(traj, target_traj_length)  # (T', 8)
    return traj
