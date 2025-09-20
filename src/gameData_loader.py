from .EXR_loader import loadEXR
import torch
import numpy as np

def load_backward_velocity(exr_path):
    exr_data = loadEXR(exr_path)  # HWC, float32
    height, width, _ = exr_data.shape

    # Extract the backward velocity channels (assuming they are in the first two channels)
    motion_1_to_0 = np.stack([exr_data[..., 2], exr_data[..., 1]], axis=-1)  # HWC, float32
    motion_1_to_0[..., 0] = -1 * width * motion_1_to_0[..., 0]   # x 軸
    motion_1_to_0[..., 1] = height * motion_1_to_0[..., 1]  # y 軸反向
    backward_velocity = torch.from_numpy(motion_1_to_0).permute(2, 0, 1).unsqueeze(0).float().cuda()  # NCHW, float32

    # Extract the depth channel (assuming it's in the third channel)
    depth_0 = exr_data[..., 0]  # HW, float32
    depth_0 = torch.from_numpy(depth_0).unsqueeze(0).unsqueeze(0).float().cuda()  # NCHW, float32
    return backward_velocity, depth_0