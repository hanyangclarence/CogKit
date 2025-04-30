import torch
import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


def normalise_quat(x: torch.Tensor):
    return x / torch.clamp(x.square().sum(dim=-1).sqrt().unsqueeze(-1), min=1e-10)


def interpolate_joint_gripper_trajectory(traj: np.ndarray, target_traj_length: int) -> np.ndarray:
    """
    Interpolates a trajectory containing 3D position, xyzw quaternion rotation, and gripper state.
    Uses linear interpolation for position and gripper state, and SLERP for rotation.

    Args:
        traj: Input trajectory, shape (T, 8) where columns are [x, y, z, qx, qy, qz, qw, gripper_open].
        target_traj_length: The desired length of the interpolated trajectory.

    Returns:
        Interpolated trajectory, shape (target_traj_length, 8).
    """
    assert traj.shape[1] == 8, f"Input trajectory must have shape (T, 8), but got {traj.shape}."
    original_length = traj.shape[0]
    if original_length == target_traj_length:
        return traj
    if original_length < 2:
        # Cannot interpolate with less than 2 points, repeat the single point
        return np.tile(traj, (target_traj_length, 1))

    # Time points for original and target trajectories
    original_time = np.linspace(0, 1, original_length)
    target_time = np.linspace(0, 1, target_traj_length)

    # Separate components
    position = traj[:, 0:3]  # Indices 0, 1, 2
    # IMPORTANT: scipy Rotation expects quaternions as [x, y, z, w]
    # Ensure your input `traj` follows this convention for indices 3:7
    quat_xyzw = traj[:, 3:7] # Indices 3, 4, 5, 6
    gripper_state = traj[:, 7:8] # Index 7 (keep as 2D array)

    # --- Linear Interpolation for Position and Gripper State ---
    interp_position = np.zeros((target_traj_length, 3))
    interp_gripper_state = np.zeros((target_traj_length, 1))

    for i in range(3): # x, y, z
        interp_position[:, i] = np.interp(target_time, original_time, position[:, i])
    interp_gripper_state[:, 0] = np.interp(target_time, original_time, gripper_state[:, 0])
    interp_gripper_state = (interp_gripper_state > 0.5).astype(traj.dtype)


    # --- SLERP Interpolation for Rotation ---
    try:
        # Create Rotation objects from the quaternions
        rotations = R.from_quat(quat_xyzw)

        # Create the SLERP interpolator object
        slerp_interpolator = Slerp(original_time, rotations)

        # Interpolate rotations at the target time points
        interp_rotations = slerp_interpolator(target_time)

        # Convert interpolated rotations back to xyzw quaternions
        interp_quat_xyzw = interp_rotations.as_quat()

    except ValueError as e:
        print(f"Warning: SLERP failed ({e}). Falling back to linear interpolation for quaternions.")
        # Fallback: Linear interpolation for quaternions (less accurate, but avoids crashing)
        # Normalize after linear interpolation
        interp_quat_xyzw = np.zeros((target_traj_length, 4))
        for i in range(4):
             interp_quat_xyzw[:, i] = np.interp(target_time, original_time, quat_xyzw[:, i])
        # Normalize the linearly interpolated quaternions
        norms = np.linalg.norm(interp_quat_xyzw, axis=1, keepdims=True)
        # Avoid division by zero for zero quaternions (though unlikely)
        norms[norms == 0] = 1.0
        interp_quat_xyzw /= norms


    # --- Combine Interpolated Components ---
    interpolated_traj = np.concatenate(
        [interp_position, interp_quat_xyzw, interp_gripper_state],
        axis=1
    )

    return interpolated_traj

def interpolate_joint_trajectory(trajectory, interpolation_length: int):
    if type(trajectory) == torch.Tensor:
        trajectory = trajectory.numpy()
    # Calculate the current number of steps
    old_num_steps = len(trajectory)

    # Create a 1D array for the old and new steps
    old_steps = np.linspace(0, 1, old_num_steps)
    new_steps = np.linspace(0, 1, interpolation_length)

    # Interpolate each dimension separately
    resampled = np.empty((interpolation_length, trajectory.shape[1]))
    for i in range(trajectory.shape[1]):
        interpolator = CubicSpline(old_steps, trajectory[:, i])
        resampled[:, i] = interpolator(new_steps)
    resampled = torch.tensor(resampled)
    return resampled