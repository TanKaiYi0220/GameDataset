from .EXR_loader import loadEXR
from .utils import flow_to_image
import torch
import numpy as np
import cv2

def nan_checker(np_array):
    return np.isnan(np_array).any()

def load_backward_velocity(exr_path):
    exr_data = loadEXR(exr_path)  # HWC, float32
    height, width, _ = exr_data.shape


    # Extract the backward velocity channels (assuming they are in the first two channels)
    motion_1_to_0 = np.stack([exr_data[..., 2], exr_data[..., 1]], axis=-1)  # HWC, float32
    motion_1_to_0[..., 0] = -1 * width * motion_1_to_0[..., 0]   # x 軸
    motion_1_to_0[..., 1] = height * motion_1_to_0[..., 1]  # y 軸反向

    failed_counter = 0
    while nan_checker(motion_1_to_0):
        print("NaN detected in backward velocity, reloading EXR...")
        exr_data = loadEXR(exr_path)  # HWC, float32
        motion_1_to_0 = np.stack([exr_data[..., 2], exr_data[..., 1]], axis=-1)  # HWC, float32
        motion_1_to_0[..., 0] = -1 * width * motion_1_to_0[..., 0]   # x 軸
        motion_1_to_0[..., 1] = height * motion_1_to_0[..., 1]  # y 軸反向
        debug = flow_to_image(motion_1_to_0)

        failed_counter += 1
        if failed_counter > 5:
            cv2.imshow("debug_flow", debug)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            raise ValueError(f"Failed to load valid EXR data from {exr_path} due to NaN values. {nan_checker(motion_1_to_0)}")
    
    backward_velocity = torch.from_numpy(motion_1_to_0).permute(2, 0, 1).unsqueeze(0).float().cuda()  # NCHW, float32

    # Extract the depth channel (assuming it's in the third channel)
    depth_0 = exr_data[..., 0]  # HW, float32
    depth_0 = torch.from_numpy(depth_0).unsqueeze(0).unsqueeze(0).float().cuda()  # NCHW, float32
    return backward_velocity, depth_0

def load_forward_velocity(exr_path):
    exr_data = loadEXR(exr_path)  # HWC, float32
    height, width, _ = exr_data.shape

    # Extract the backward velocity channels (assuming they are in the first two channels)
    motion_1_to_0 = np.stack([exr_data[..., 2], exr_data[..., 1]], axis=-1)  # HWC, float32
    motion_1_to_0[..., 0] = -1 * width * motion_1_to_0[..., 0]   # x 軸
    motion_1_to_0[..., 1] = height * motion_1_to_0[..., 1]  # y 軸反向

    failed_counter = 0
    while nan_checker(motion_1_to_0):
        exr_data = loadEXR(exr_path)  # HWC, float32
        motion_1_to_0 = np.stack([exr_data[..., 2], exr_data[..., 1]], axis=-1)  # HWC, float32
        motion_1_to_0[..., 0] = -1 * width * motion_1_to_0[..., 0]   # x 軸
        motion_1_to_0[..., 1] = height * motion_1_to_0[..., 1]  # y 軸反向
        debug = flow_to_image(motion_1_to_0)

        failed_counter += 1
        if failed_counter > 5:
            cv2.imshow("debug_flow", debug)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            raise ValueError(f"Failed to load valid EXR data from {exr_path} due to NaN values. {nan_checker(motion_1_to_0)}")

    forward_velocity = torch.from_numpy(motion_1_to_0).permute(2, 0, 1).unsqueeze(0).float().cuda()  # NCHW, float32

    # Extract the depth channel (assuming it's in the third channel)
    depth_0 = exr_data[..., 0]  # HW, float32
    depth_0 = torch.from_numpy(depth_0).unsqueeze(0).unsqueeze(0).float().cuda()  # NCHW, float32
    return forward_velocity, depth_0