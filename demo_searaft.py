import sys
sys.path.append('models/SEARAFT/core')

import argparse
import os
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

def forward_warp_nearest(src, flow):
    """
    src:  [N,C,H,W]
    flow: [N,2,H,W]  (dx, dy), 定義為從 source 座標 (x,y) -> target (x+dx, y+dy)
    回傳:
      out:  [N,C,H,W]
      hit:  [N,1,H,W]  累積權重(命中次數)，可用來看洞的位置
    """
    N, C, H, W = src.shape
    device = src.device

    # 建 (y,x) 網格
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device), torch.arange(W, device=device),
        indexing='ij'
    )  # [H,W]
    xx = xx.unsqueeze(0).expand(N, -1, -1).float()
    yy = yy.unsqueeze(0).expand(N, -1, -1).float()

    tx = xx + flow[:,0]  # [N,H,W]
    ty = yy + flow[:,1]  # [N,H,W]

    # 最近鄰
    txn = tx.round().long()
    tyn = ty.round().long()

    # 有效範圍
    mask = (txn >= 0) & (txn < W) & (tyn >= 0) & (tyn < H)

    out = torch.zeros_like(src)
    hit = torch.zeros((N,1,H,W), device=device, dtype=src.dtype)

    # 展平 index，做 scatter_add
    linear_idx = tyn.clamp(0, H-1) * W + txn.clamp(0, W-1)  # [N,H,W]
    base = (torch.arange(N, device=device) * (H*W)).view(N,1,1)
    flat_idx = (linear_idx + base).view(-1)  # [N*H*W]

    src_flat = src.permute(0,2,3,1).reshape(-1, C)  # [N*H*W, C]
    mask_flat = mask.view(-1)

    out_flat = torch.zeros((N*H*W, C), device=device, dtype=src.dtype)
    out_flat.index_add_(0, flat_idx[mask_flat], src_flat[mask_flat])
    out = out_flat.view(N, H, W, C).permute(0,3,1,2)

    hit_flat = torch.zeros((N*H*W, 1), device=device, dtype=src.dtype)
    one = torch.ones((mask_flat.sum(),1), device=device, dtype=src.dtype)
    hit_flat.index_add_(0, flat_idx[mask_flat], one)
    hit = hit_flat.view(N, H, W, 1).permute(0,3,1,2)
    return out, hit

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
def demo_data(name, args, model, image1, image2, flow_gt):
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

    # forward warp 檢查 flow direction 是否正確
    warped_img, hit = forward_warp_nearest(image1, flow)
    warped_avg = warped_img / (hit + 1e-6)
    warped_avg = torch.where(hit > 0, warped_avg, torch.zeros_like(warped_avg))  # 沒命中就置零/保持無值
    cv2.imwrite(f"{path}warped_img.jpg", warped_avg.cpu().numpy()[0].transpose(1, 2, 0).astype(np.uint8))
    hit_vis = visualize_hit(hit)
    cv2.imwrite(f"{path}hit_img.jpg", hit_vis)
    diff_img = np.abs(warped_avg.cpu().numpy() - image2.cpu().numpy())
    cv2.imwrite(f"{path}diff_img.jpg", diff_img[0].transpose(1, 2, 0))
    print(f"Diff mean: {diff_img.mean()}, max: {diff_img.max()}")

def FRPG_loader(name, model, img_0_path, img_1_path, img_gt_path, args):
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

    demo_data(name, args, model, image1, image2, flow_gt)

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

    # demo on custom images
    # img_0_path = "/home/kevin/Desktop/VFI/offline_dataset/models/SEARAFT/custom/image1.jpg"
    # img_1_path = "/home/kevin/Desktop/VFI/offline_dataset/models/SEARAFT/custom/image2.jpg"
    # img_gt_path = None

    # template for loading FRPG images
    img_0_path = "/home/kevin/Desktop/VFI/offline_dataset/datasets/Fantasy_RPG/FRPG_0_0_0/fps_60_png/colorNoScreenUI_0.png"
    img_1_path = "/home/kevin/Desktop/VFI/offline_dataset/datasets/Fantasy_RPG/FRPG_0_0_0/fps_60_png/colorNoScreenUI_3.png"
    img_gt_path = None
    FRPG_loader("SEARAFT/FRPG/", model, img_0_path, img_1_path, img_gt_path, args)


if __name__ == '__main__':
    main()