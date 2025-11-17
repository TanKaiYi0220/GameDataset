import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
from src.gameData_loader import load_backward_velocity
from src.warp_module import ForwardWarpingNearest, BackwardWarpingNearest, ForwardWarpingNearestWithDepth
from src.utils import find_max_index_in_dir, save_img, EXR_to_PNG, get_all_modes, parse_mode_name, save_np_array

import sys
sys.path.append('models/SEARAFT/core')

import argparse
import cv2
import numpy as np
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from models.SEARAFT.config.parser import parse_args

from models.SEARAFT.core.raft import RAFT
from models.SEARAFT.core.raftGM import RAFTGM
from models.SEARAFT.core.utils.flow_viz import flow_to_image
from models.SEARAFT.core.utils.utils import load_ckpt

from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt

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
    # print(heatmap.max(), heatmap.min(), heatmap.meang error: Write Errorn())
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
    save_img(name, cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))

def get_heatmap(info, args):
    raw_b = info[:, 2:]
    log_b = torch.zeros_like(raw_b)
    weight = info[:, :2].softmax(dim=1)              
    log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=args.var_max)
    log_b[:, 1] = torch.clamp(raw_b[:, 1], min=args.var_min, max=0)
    heatmap = (log_b * weight).sum(dim=1, keepdim=True)
    return heatmap

def create_color_bar_with_labels(
    colormap=cv2.COLORMAP_JET, vmin=0.0, vmax=1.0,
    width=40, height=400, n_ticks=6, font_scale=0.5, thickness=1
):
    # 1) 建立垂直色條 (top=vmax, bottom=vmin)
    grad = np.linspace(1, 0, height, dtype=np.float32).reshape(height, 1)
    grad = np.repeat(grad, width, axis=1)
    # 用 matplotlib 產生 RGB，再轉 BGR
    cmap = plt.get_cmap('jet')
    bar_rgb = (cmap(grad)[:, :, :3] * 255).astype(np.uint8)
    bar = cv2.cvtColor(bar_rgb, cv2.COLOR_RGB2BGR)

    # 2) 先算所有 label 的最大寬度 → 動態留白
    tick_vals = np.linspace(vmax, vmin, n_ticks)
    labels = [f"{v:.2f}" for v in tick_vals]
    max_w = 0; max_h = 0; max_base = 0
    for s in labels:
        (tw, th), base = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        max_w = max(max_w, tw)
        max_h = max(max_h, th)
        max_base = max(max_base, base)

    pad_left = 6                  # 色條右側到刻度線間距
    tick_len = 6                  # 刻度線長度
    pad_right = max_w + 8         # 文字到右邊界留白
    pad = pad_left + tick_len + pad_right

    # 3) 擴寬畫布，並畫上刻度＆文字（白字黑邊，避免淹沒）
    canvas = np.zeros((height, width + pad, 3), dtype=np.uint8)
    canvas[:, :width] = bar

    for i, (val, lab) in enumerate(zip(tick_vals, labels)):
        y = int(round(i * (height - 1) / (n_ticks - 1)))

        # 小刻度線
        x0 = width + pad_left
        cv2.line(canvas, (x0, y), (x0 + tick_len, y), (255, 255, 255), 1, cv2.LINE_AA)

        # 文字位置（夾在安全範圍內，避免出上下邊）
        (tw, th), base = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        y_text = np.clip(y + th // 2, th + 2, height - 2)  # baseline 安全
        x_text = x0 + tick_len + 4

        # 先黑邊再白字
        cv2.putText(canvas, lab, (x_text, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness+2, cv2.LINE_AA)
        cv2.putText(canvas, lab, (x_text, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return canvas

def vis_flow_diff(name, src_image, flow1, flow2):
    # --- Compute flow difference ---
    flow_diff = torch.norm(flow1 - flow2, dim=1)
    flow_diff_np = flow_diff[0].detach().cpu().numpy()

    # --- Compute actual EPE range ---
    epe_min = float(flow_diff_np.min())
    epe_max = float(flow_diff_np.max())

    # --- Normalize for visualization ---
    norm_flow_diff = (flow_diff_np - epe_min) / (epe_max - epe_min + 1e-8)
    norm_flow_diff = (norm_flow_diff * 255).astype(np.uint8)

    # --- Apply colormap ---
    colored_flow_diff = cv2.applyColorMap(norm_flow_diff, cv2.COLORMAP_JET)

    # --- Overlay with source image ---
    overlay = src_image * 0.3 + colored_flow_diff * 0.7
    overlay = overlay.astype(np.uint8)

    # --- Draw colorbar with actual EPE range ---
    color_bar = create_color_bar_with_labels(cv2.COLORMAP_JET, epe_min, epe_max, width=30, height=overlay.shape[0])

    # --- Combine ---
    combined_image = np.hstack((overlay, color_bar))
    cv2.imwrite(name, cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))

def forward_flow(args, model, image1, image2, game_motion_flow=None):
    output = model(image1, image2, iters=args.iters, test_mode=True)
    flow_net_init = output['flow'][0]
    flow_gm_init = output["flow"][1]
    flow_final = output['flow'][-1]
    info_final = output['info'][-1]
    return flow_net_init, flow_gm_init, flow_final, info_final

def calc_flow(args, model, image1, image2, game_motion_flow=None):
    img1 = F.interpolate(image1, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
    img2 = F.interpolate(image2, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
    H, W = img1.shape[2:]
    # downscale to be multiples of 8
    game_motion_flow_8x = F.interpolate(game_motion_flow, size=(H // 8, W // 8), mode='bilinear', align_corners=False)
    game_motion_flow_8x = game_motion_flow_8x / 8.0
    flow_net_init, flow_gm_init, flow, info = forward_flow(args, model, img1, img2)
    flow_net_init = F.interpolate(flow_net_init, scale_factor=0.5 ** args.scale, mode='bilinear', align_corners=False) * (0.5 ** args.scale)
    flow_gm_init = F.interpolate(flow_gm_init, scale_factor=0.5 ** args.scale, mode='bilinear', align_corners=False) * (0.5 ** args.scale)
    flow_down = F.interpolate(flow, scale_factor=0.5 ** args.scale, mode='bilinear', align_corners=False) * (0.5 ** args.scale)
    info_down = F.interpolate(info, scale_factor=0.5 ** args.scale, mode='area')
    return flow_net_init, flow_gm_init, flow_down, info_down

def warping(warping_module, src_image, target_image, flow, path, name):
    # demo warping function
    warping_module.warp(src_image, flow)
    warped_img, _ = warping_module.get_warping_result(mode="average")
    hit_vis = warping_module.visualize_hit()
    diff_img = np.abs(warped_img.cpu().numpy() - target_image.cpu().numpy())

    save_img(f"{path}warped_img_{name}.png", warped_img.cpu().numpy()[0].transpose(1, 2, 0).astype(np.uint8))
    save_img(f"{path}hit_img_{name}.png", hit_vis)
    save_img(f"{path}diff_img_{name}.png", diff_img[0].transpose(1, 2, 0))

    exit()

    return warped_img

@torch.no_grad()
def inference(name, args, model, image1, image2, bmv):
    path = name
    os.system(f"mkdir -p {path}")
    H, W = image1.shape[2:]
    save_img(f"{path}image1.png", image1[0].permute(1, 2, 0).cpu().numpy())
    save_img(f"{path}image2.png", image2[0].permute(1, 2, 0).cpu().numpy())

    flow_net_init, flow_gm_init, flow, info = calc_flow(args, model, image2, image1, game_motion_flow=bmv)
    flow_vis = flow_to_image(flow[0].permute(1, 2, 0).cpu().numpy(), convert_to_bgr=False)
    save_img(f"{path}flow_final.png", flow_vis)
    save_np_array(f"{path}flow_final.npy", flow[0].permute(1, 2, 0).cpu().numpy())

    flow_vis = flow_to_image(flow_net_init[0].permute(1, 2, 0).cpu().numpy(), convert_to_bgr=False)
    save_img(f"{path}flow_net_init.png", flow_vis)

    flow_vis = flow_to_image(flow_gm_init[0].permute(1, 2, 0).cpu().numpy(), convert_to_bgr=False)
    save_img(f"{path}flow_gm_init.png", flow_vis)

    vis_flow_diff(f"{path}gameMotion_vs_opticalFlow.png", image2[0].permute(1, 2, 0).cpu().numpy(), bmv, flow)

    heatmap = get_heatmap(info, args)
    vis_heatmap(f"{path}heatmap_final.png", image1[0].permute(1, 2, 0).cpu().numpy(), heatmap[0].permute(1, 2, 0).cpu().numpy())
    
    bmv_vis = flow_to_image(bmv[0].permute(1, 2, 0).cpu().numpy(), convert_to_bgr=False)
    save_img(f"{path}flow_gameData.png", bmv_vis)

    return flow, info

def warping(warping_module, src_image, target_image, flow, path, name):
    # demo warping function
    warping_module.warp(src_image, flow)
    warped_img, _ = warping_module.get_warping_result(mode="average")
    hit_vis = warping_module.visualize_hit()
    diff_img = np.abs(warped_img.cpu().numpy() - target_image.cpu().numpy())

    save_img(f"{path}warped_img_{name}.png", warped_img.cpu().numpy()[0].transpose(1, 2, 0).astype(np.uint8))
    save_img(f"{path}hit_img_{name}.png", hit_vis)
    save_img(f"{path}diff_img_{name}.png", diff_img[0].transpose(1, 2, 0))

    return warped_img

def warping_with_depth(warping_module, src_image, target_image, flow, depth, path, name):
    # demo warping function
    warping_module.warp(src_image, flow, depth)
    warped_img, _ = warping_module.get_warping_result(mode="average")
    hit_vis = warping_module.visualize_hit()
    diff_img = np.abs(warped_img.cpu().numpy() - target_image.cpu().numpy())

    save_img(f"{path}warped_img_{name}.png", warped_img.cpu().numpy()[0].transpose(1, 2, 0).astype(np.uint8))
    save_img(f"{path}hit_img_{name}.png", hit_vis)
    save_img(f"{path}diff_img_{name}.png", diff_img[0].transpose(1, 2, 0))

    return warped_img

def evaluation(optical_flow, game_motion, warped_img_flow, warped_img_motion, target_image):
    # epe score
    epe = torch.norm(optical_flow - game_motion, dim=1).mean().cpu().item()

    # psnr score (skimage.metrics.peak_signal_noise_ratio)
    warped_img_flow_np = warped_img_flow[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    warped_img_motion_np = warped_img_motion[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    target_image_np = target_image[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    psnr_flow = psnr(target_image_np, warped_img_flow_np, data_range=255)
    psnr_motion = psnr(target_image_np, warped_img_motion_np, data_range=255)
    if np.isfinite(psnr_flow) == False:
        print("Identical images (PSNR=inf). Setting PSNR to 0.")
        psnr_flow = 0
    if np.isfinite(psnr_motion) == False:
        print("Identical images (PSNR=inf). Setting PSNR to 0.")
        psnr_motion = 0

    return epe, psnr_flow, psnr_motion

def stats_summary(scores, clip_list, clip_len):
    max_score = max(scores)
    min_score = min(scores)
    max_idx = scores.index(max_score)
    min_idx = scores.index(min_score)
    max_clip_name = clip_list[max_idx // clip_len]
    min_clip_name = clip_list[min_idx // clip_len]
    max_clip_idx = max_idx % clip_len
    min_clip_idx = min_idx % clip_len

    return {
        "mean": sum(scores) / len(scores),
        "max": {
            "score": max_score,
            "clip_idx": max_clip_idx,
            "clip": max_clip_name
        },
        "min": {
            "score": min_score,
            "clip_idx": min_clip_idx,
            "clip": min_clip_name
        },
        "all": [
            {
                "score": score,
                "clip": clip_list[idx // clip_len],
                "clip_idx": idx % clip_len
            } for idx, score in enumerate(scores)
        ]
    }

def save_results_json(name, clip_json_path, args, epe_scores, psnr_flow_scores, psnr_motion_scores, clip_list, clip_len):
    # Prepare JSON result
    # epe stats
    epe_stats = stats_summary(epe_scores, clip_list, clip_len)
    
    # psnr flow stats
    psnr_flow_stats = stats_summary(psnr_flow_scores, clip_list, clip_len)

    # psnr motion stats
    psnr_motion_stats = stats_summary(psnr_motion_scores, clip_list, clip_len)

    # results json for epe, psnr_flow, psnr_motion
    results = {
        "exp_config": {
            "clip_json": clip_json_path,
            "model_args": {
                "cfg": args.cfg,
                "model": args.model,
            },
        },
        "epe_results": epe_stats,
        "psnr_flow_results": psnr_flow_stats,
        "psnr_motion_results": psnr_motion_stats
    }

    # Save results to JSON file
    with open(f"{name}/results.json", "w") as f:
        json.dump(results, f, indent=4)

def FRPG_loader(name, model, forward_warping_module, forward_warping_with_depth_module, backward_warping_module, clip_color_seq, clip_velD_seq, args):
    max_index = len(clip_color_seq)

    epe_scores = []
    psnr_flow_scores = []
    psnr_motion_scores = []

    with tqdm(range(max_index - 1)) as pbar:
        for i in pbar:
            image_1 = cv2.imread(clip_color_seq[i])
            image_2 = cv2.imread(clip_color_seq[i + 1])

            image_1 = torch.from_numpy(image_1).permute(2, 0, 1).unsqueeze(0).float().cuda()
            image_2 = torch.from_numpy(image_2).permute(2, 0, 1).unsqueeze(0).float().cuda()

            bmv, depth = load_backward_velocity(clip_velD_seq[i + 1])

            folder_name = f"{name}/results/frame_{i:04d}_to_{i+1:04d}/"

            flow, info = inference(folder_name, args, model, image_1, image_2, bmv)

            warped_img_fw_flow = warping(forward_warping_module, image_2, image_1, flow, folder_name, "opticalFlow_forward")
            warped_img_fw_motion = warping(forward_warping_module, image_2, image_1, bmv, folder_name, "gameMotion_forward")
            
            warped_img_fw_flow_with_depth = warping_with_depth(forward_warping_with_depth_module, image_2, image_1, flow, depth, folder_name, "opticalFlow_depth_forward")
            warped_img_fw_motion_with_depth = warping_with_depth(forward_warping_with_depth_module, image_2, image_1, bmv, depth, folder_name, "gameMotion_depth_forward")
            
            warped_img_bw_flow = warping(backward_warping_module, image_1, image_2, flow, folder_name, "opticalFlow_backward")
            warped_img_bw_motion = warping(backward_warping_module, image_1, image_2, bmv, folder_name, "gameMotion_backward")

            epe, psnr_flow, psnr_motion = evaluation(flow, bmv, warped_img_bw_flow, warped_img_bw_motion, image_1)

            pbar.set_postfix({"EPE": f"{epe:.4f}", "PSNR Flow": f"{psnr_flow:.4f}", "PSNR Motion": f"{psnr_motion:.4f}"})

            epe_scores.append(epe)
            psnr_flow_scores.append(psnr_flow)
            psnr_motion_scores.append(psnr_motion)

    return epe_scores, psnr_flow_scores, psnr_motion_scores

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

    forward_warping_module = ForwardWarpingNearest()
    forward_warping_with_depth_module = ForwardWarpingNearestWithDepth()
    backward_warping_module = BackwardWarpingNearest()

    ROOT_PATH = "/datasets/VFI/datasets/AnimeFantasyRPG/"
    RECORD_NAME = ["AnimeFantasyRPG_3_60"]
    FPS = "fps_60"
    MAIN_INDEX = ["0", "1", "2", "3"]
    DIFFICULTY = ["Easy", "Medium"]
    SUB_INDEX = "0"
    MODES = get_all_modes(MAIN_INDEX, DIFFICULTY, SUB_INDEX, FPS)
    MAX_INDEX = 800
        
    CLEAN_ROOT_PATH = f"/datasets/VFI/GFI_datasets/"

    for record_name in RECORD_NAME:
        dataset_root_path = f"{CLEAN_ROOT_PATH}/{record_name}/"
        print("Dataset Root Path:", dataset_root_path)

        clip_json_path = f"{dataset_root_path}/overall_{FPS}_clip_info.json"
        with open(clip_json_path, "r", encoding="utf-8") as f:
            clip_json = json.load(f)
        print(clip_json.keys())

        for mode in MODES:
            mode_name = parse_mode_name(mode)

            for difficult in mode:
                epe_scores_list = []
                psnr_flow_scores_list = []
                psnr_motion_scores_list = []
                clip_list = list(clip_json[mode_name].keys())
                clip_len = 0

                for clip in clip_list:
                    clip_info = clip_json[mode_name][clip]
                    print("Processing Clip:", clip, "Difficult:", difficult)

                    clip_color_seq = clip_info[difficult]["colorNoScreenUI"]
                    clip_velD_seq = clip_info[difficult]["backwardVel_Depth"]

                    output_path = f"output/SEARAFT/AnimeFantasyRPG/{record_name}_clip/{record_name}/{difficult}/{clip}"

                    epe_scores, psnr_flow_scores, psnr_motion_scores = FRPG_loader(
                        output_path, 
                        model, 
                        forward_warping_module, forward_warping_with_depth_module, backward_warping_module, 
                        clip_color_seq, clip_velD_seq,
                        args
                    )

                    epe_scores_list.extend(epe_scores)
                    psnr_flow_scores_list.extend(psnr_flow_scores)
                    psnr_motion_scores_list.extend(psnr_motion_scores)

                    clip_len = len(epe_scores)

                    print(clip_len)

                save_results_json(
                    f"output/SEARAFT/AnimeFantasyRPG/{record_name}_clip/{record_name}/{difficult}/", 
                    clip_json_path, args, 
                    epe_scores_list, psnr_flow_scores_list, psnr_motion_scores_list, clip_list, clip_len
                )
                
            

    # for mode in dataset_mode_path:
    #     dataset_dir_path = os.path.join(dataset_root_path, mode)
    #     FRPG_loader(f"SEARAFT_GM/AnimeFantasyRPG/{record_name}/{mode}", model, forward_warping_module, forward_warping_with_depth_module, backward_warping_module, dataset_dir_path, args)


if __name__ == '__main__':
    main()