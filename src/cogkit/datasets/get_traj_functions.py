import pickle
import torch
import numpy as np
import quaternion

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

def get_traj_with_cam_coordinates(demo: Demo, target_traj_length: int) -> np.ndarray:
    traj = np.concatenate(
        [np.concatenate([obs.gripper_pose, [obs.gripper_open]])[None, ...] for obs in demo], 
        axis=0
    )  # (T, 8)
    traj = interpolate_joint_gripper_trajectory(traj, target_traj_length)  # (T', 8)
    
    traj_cam_frame = []
    traj_pixel_frame = []
    for obs in demo:
        T_w2c = np.linalg.inv(obs.misc["front_camera_extrinsics"])
        K = obs.misc["front_camera_intrinsics"]
        T_world = obs.gripper_matrix
        
        T_cam = T_w2c @ T_world
        
        R_cam = T_cam[:3, :3]
        t_cam = T_cam[:3, 3]
        R_cam = quaternion.from_rotation_matrix(R_cam)
        R_cam = quaternion.as_float_array(R_cam)[[1, 2, 3, 0]]
        gripper_pose_cam = np.concatenate([t_cam, R_cam, [obs.gripper_open]])  # (8,)
        traj_cam_frame.append(gripper_pose_cam[None, ...])
        
        u, v, w = K @ t_cam
        t_pix = np.array([u/w, v/w])
        # scale to -1, 1
        t_pix = (t_pix - 256) / 256
        traj_pixel_frame.append(t_pix[None, ...])
        
    traj_cam_frame = np.concatenate(traj_cam_frame, axis=0)  # (T, 8)
    traj_cam_frame = interpolate_joint_gripper_trajectory(traj_cam_frame, target_traj_length)  # (T', 8)
    traj_pixel_frame = np.concatenate(traj_pixel_frame, axis=0)  # (T, 2)
    traj_pixel_frame = interpolate_joint_trajectory(traj_pixel_frame, target_traj_length)  # (T', 2)
    
    traj = np.concatenate([traj, traj_cam_frame, traj_pixel_frame], axis=1)  # (T', 18)
    return traj
