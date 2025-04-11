import torch
import numpy as np
from scipy.interpolate import CubicSpline, interp1d


def normalise_quat(x: torch.Tensor):
    return x / torch.clamp(x.square().sum(dim=-1).sqrt().unsqueeze(-1), min=1e-10)


def interpolate_joint_gripper_trajectory(trajectory, interpolation_length: int):
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
        if i == (trajectory.shape[1] - 1):  # gripper opening
            interpolator = interp1d(old_steps, trajectory[:, i])
        else:
            interpolator = CubicSpline(old_steps, trajectory[:, i])

        resampled[:, i] = interpolator(new_steps)

    resampled = torch.tensor(resampled)
    if trajectory.shape[1] == 8:
        resampled[:, 3:7] = normalise_quat(resampled[:, 3:7])
    return resampled

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