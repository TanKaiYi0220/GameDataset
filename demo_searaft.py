import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
from src.EXR_loader import loadEXR
from src.warp_module import Warping

import sys
sys.path.append('models/SEARAFT/core')

import argparse
import cv2
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from models.SEARAFT.config.parser import parse_args

from models.SEARAFT.core.raft import RAFT
from models.SEARAFT.core.utils.flow_viz import flow_to_image
from models.SEARAFT.core.utils.utils import load_ckpt


def create_color_bar(height, width, color_map):
    """
    Create a color bar image using a specified color map.

    :param height: The height of the color bar.
    :param width: The width of the color bar.
    :param color_map: The OpenCV colormap to use.
    :return: A color bar image.
    """
    # Generate a linear gradient
    gradient = np.linspace(0, 255, width, dtype=np.uint8)
    gradient = np.repeat(gradient[np.newaxis, :], height, axis=0)

    # Apply the colormap
    color_bar = cv2.applyColorMap(gradient, color_map)

    return color_bar

def add_color_bar_to_image(image, color_bar, orientation='vertical'):
    """
    Add a color bar to an image.

    :param image: The original image.
    :param color_bar: The color bar to add.
    :param orientation: 'vertical' or 'horizontal'.
    :return: Combined image with the color bar.
    """
    if orientation == 'vertical':
        return cv2.vconcat([image, color_bar])
    else:
        return cv2.hconcat([image, color_bar])

def vis_heatmap(name, image, heatmap):
    # theta = 0.01
    # print(heatmap.max(), heatmap.min(), heatmap.mean())
    heatmap = heatmap[:, :, 0]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    # heatmap = heatmap > 0.01
    heatmap = (heatmap * 255).astype(np.uint8)
    colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = image * 0.3 + colored_heatmap * 0.7
    # Create a color bar
    height, width = image.shape[:2]
    color_bar = create_color_bar(50, width, cv2.COLORMAP_JET)  # Adjust the height and colormap as needed
    # Add the color bar to the image
    overlay = overlay.astype(np.uint8)
    combined_image = add_color_bar_to_image(overlay, color_bar, 'vertical')
    cv2.imwrite(name, cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))

def get_heatmap(info, args):
    raw_b = info[:, 2:]
    log_b = torch.zeros_like(raw_b)
    weight = info[:, :2].softmax(dim=1)              
    log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=args.var_max)
    log_b[:, 1] = torch.clamp(raw_b[:, 1], min=args.var_min, max=0)
    heatmap = (log_b * weight).sum(dim=1, keepdim=True)
    return heatmap

def forward_flow(args, model, image1, image2):
    output = model(image1, image2, iters=args.iters, test_mode=True)
    flow_final = output['flow'][-1]
    info_final = output['info'][-1]
    return flow_final, info_final

def calc_flow(args, model, image1, image2):
    img1 = F.interpolate(image1, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
    img2 = F.interpolate(image2, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
    H, W = img1.shape[2:]
    flow, info = forward_flow(args, model, img1, img2)
    flow_down = F.interpolate(flow, scale_factor=0.5 ** args.scale, mode='bilinear', align_corners=False) * (0.5 ** args.scale)
    info_down = F.interpolate(info, scale_factor=0.5 ** args.scale, mode='area')
    return flow_down, info_down

def demo_warping(warping_module, src_image, target_image, flow, path, name):
    # demo warping function
    warping_module.warp(src_image, flow)
    warped_img, _ = warping_module.get_warping_result(mode="average")
    hit_vis = warping_module.visualize_hit()
    diff_img = np.abs(warped_img.cpu().numpy() - target_image.cpu().numpy())

    cv2.imwrite(f"{path}warped_img_{name}.jpg", warped_img.cpu().numpy()[0].transpose(1, 2, 0).astype(np.uint8))
    cv2.imwrite(f"{path}hit_img_{name}.jpg", hit_vis)
    cv2.imwrite(f"{path}diff_img_{name}.jpg", diff_img[0].transpose(1, 2, 0))
    print(f"Diff mean: {diff_img.mean()}, max: {diff_img.max()}")

def visualize_hit(hit):
    """
    hit: [N,1,H,W] torch.Tensor, int (0,1,2,...)
    return: [H,W,3] uint8 (BGR)
    """
    hmap = hit[0,0].cpu().numpy().astype(np.int32)
    H, W = hmap.shape

    vis = np.zeros((H,W,3), dtype=np.uint8)

    vis[hmap == 0] = (255, 0, 0)   # none-to-one → 藍 (BGR)
    vis[hmap == 1] = (0, 255, 0)   # one-to-one  → 綠
    vis[hmap >= 2] = (0, 0, 255)   # many-to-one → 紅

    return vis

@torch.no_grad()
def demo_data(name, args, model, warping_module, image1, image2, flow_gt, bmv_0_path=None):
    path = f"demo/{name}/"
    os.system(f"mkdir -p {path}")
    H, W = image1.shape[2:]
    cv2.imwrite(f"{path}image1.jpg", image1[0].permute(1, 2, 0).cpu().numpy())
    cv2.imwrite(f"{path}image2.jpg", image2[0].permute(1, 2, 0).cpu().numpy())
    flow_gt_vis = flow_to_image(flow_gt[0].permute(1, 2, 0).cpu().numpy(), convert_to_bgr=False)
    cv2.imwrite(f"{path}gt.jpg", flow_gt_vis)
    flow, info = calc_flow(args, model, image1, image2)
    flow_vis = flow_to_image(flow[0].permute(1, 2, 0).cpu().numpy(), convert_to_bgr=False)
    cv2.imwrite(f"{path}flow_final.jpg", flow_vis)
    diff = flow_gt - flow
    diff_vis = flow_to_image(diff[0].permute(1, 2, 0).cpu().numpy(), convert_to_bgr=False)
    cv2.imwrite(f"{path}error_final.jpg", diff_vis)
    heatmap = get_heatmap(info, args)
    vis_heatmap(f"{path}heatmap_final.jpg", image1[0].permute(1, 2, 0).cpu().numpy(), heatmap[0].permute(1, 2, 0).cpu().numpy())
    epe = torch.sum((flow - flow_gt)**2, dim=1).sqrt()
    print(f"EPE: {epe.mean().cpu().item()}")

    # forward warp for flow direction check
    demo_warping(warping_module, image1, image2, flow, path, name="forward")

    # backward warping for flow direction check
    bmv, depth = load_backward_velocity(bmv_0_path)
    demo_warping(warping_module, image1, image2, bmv, path, name="backward")

def FRPG_loader(name, model, warping_module, img_0_path, img_1_path, img_gt_path, args, bmv_0_path=None):
    image1 = cv2.imread(img_0_path)
    image2 = cv2.imread(img_1_path)

    image1 = torch.from_numpy(image1).permute(2, 0, 1).unsqueeze(0).float().cuda()
    image2 = torch.from_numpy(image2).permute(2, 0, 1).unsqueeze(0).float().cuda()

    # initialize a empty flow_gt as we do not have ground truth
    if img_gt_path is None:
        flow_gt = torch.zeros(1, 2, image1.shape[2], image1.shape[3]).float().cuda()
    else:
        flow_gt = cv2.imread(img_gt_path)
        flow_gt = torch.from_numpy(flow_gt).permute(2, 0, 1).unsqueeze(0).float().cuda()

    demo_data(name, args, model, warping_module, image1, image2, flow_gt, bmv_0_path)

def load_backward_velocity(exr_path):
    exr_data = loadEXR(exr_path)  # HWC, float32
    height, width, _ = exr_data.shape

    # Extract the backward velocity channels (assuming they are in the first two channels)
    motion_1_to_0 = np.stack([exr_data[..., 2], exr_data[..., 1]], axis=-1)  # HWC, float32
    motion_1_to_0[..., 0] = width * motion_1_to_0[..., 0]   # x 軸
    motion_1_to_0[..., 1] = -1 * height * motion_1_to_0[..., 1]  # y 軸反向
    backward_velocity = torch.from_numpy(motion_1_to_0).permute(2, 0, 1).unsqueeze(0).float().cuda()  # NCHW, float32

    # Extract the depth channel (assuming it's in the third channel)
    depth_0 = exr_data[..., 0]  # HW, float32
    depth_0 = torch.from_numpy(depth_0).unsqueeze(0).unsqueeze(0).float().cuda()  # NCHW, float32
    return backward_velocity, depth_0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--model', help='checkpoint path', required=True, type=str)
    # parser.add_argument('')
    args = parse_args(parser)
    model = RAFT(args)
    load_ckpt(model, args.model)
    model = model.cuda()
    model.eval()

    warping_module = Warping(warp_mode="nearest")

    # demo on custom images
    # img_0_path = "/home/kevin/Desktop/VFI/offline_dataset/models/SEARAFT/custom/image1.jpg"
    # img_1_path = "/home/kevin/Desktop/VFI/offline_dataset/models/SEARAFT/custom/image2.jpg"
    # img_gt_path = None

    # template for loading FRPG images
    img_0_path = "/home/kevin/Desktop/VFI/offline_dataset/datasets/Fantasy_RPG/FRPG_1_0_0/fps_30_png/colorNoScreenUI_10.png"
    img_1_path = "/home/kevin/Desktop/VFI/offline_dataset/datasets/Fantasy_RPG/FRPG_1_0_0/fps_30_png/colorNoScreenUI_11.png"
    bmv_0_path = "/home/kevin/Desktop/VFI/offline_dataset/datasets/Fantasy_RPG/FRPG_1_0_0/fps_30/backwardVel_Depth_11.exr"
    img_gt_path = None
    FRPG_loader("SEARAFT/FRPG/", model, warping_module, img_0_path, img_1_path, img_gt_path, args, bmv_0_path)


if __name__ == '__main__':
    main()