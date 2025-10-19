import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
from src.gameData_loader import load_backward_velocity
from src.warp_module import ForwardWarpingNearest, BackwardWarpingNearest, ForwardWarpingNearestWithDepth
from src.utils import find_max_index_in_dir, save_img, EXR_to_PNG

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

def vis_flow_diff(name, src_image, flow1, flow2):
    flow_diff = torch.norm(flow1 - flow2, dim=1)
    flow_diff = flow_diff[0].cpu().numpy()
    flow_diff = (flow_diff - flow_diff.min()) / (flow_diff.max() - flow_diff.min())
    flow_diff = (flow_diff * 255).astype(np.uint8)
    colored_flow_diff = cv2.applyColorMap(flow_diff, cv2.COLORMAP_JET)
    save_img(name, colored_flow_diff)

    overlay = src_image * 0.3 + colored_flow_diff * 0.7
    # Create a color bar
    height, width = src_image.shape[:2]
    color_bar = create_color_bar(50, width, cv2.COLORMAP_JET)  # Adjust the height and colormap as needed
    # Add the color bar to the image
    overlay = overlay.astype(np.uint8)
    combined_image = add_color_bar_to_image(overlay, color_bar, 'vertical')
    save_img(name, cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))

def forward_flow(args, model, image1, image2, game_motion_flow=None):
    output = model(image1, image2, iters=args.iters, test_mode=True, game_motion_flow=game_motion_flow)
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
    flow_net_init, flow_gm_init, flow, info = forward_flow(args, model, img1, img2, game_motion_flow=game_motion_flow_8x)
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

def save_results_json(name, dataset_dir_path, args, epe_scores, psnr_flow_scores, psnr_motion_scores):
    # Prepare JSON result
    # epe stats
    max_epe = max(epe_scores)
    min_epe = min(epe_scores)
    max_idx = epe_scores.index(max_epe)
    min_idx = epe_scores.index(min_epe)

    # psnr flow stats
    max_psnr_flow = max(psnr_flow_scores)
    min_psnr_flow = min(psnr_flow_scores)
    max_idx_flow = psnr_flow_scores.index(max_psnr_flow)
    min_idx_flow = psnr_flow_scores.index(min_psnr_flow)

    # psnr motion stats
    max_psnr_motion = max(psnr_motion_scores)
    min_psnr_motion = min(psnr_motion_scores)
    max_idx_motion = psnr_motion_scores.index(max_psnr_motion)
    min_idx_motion = psnr_motion_scores.index(min_psnr_motion)

    # results json for epe, psnr_flow, psnr_motion
    results = {
        "exp_config": {
            "dataset_dir": dataset_dir_path,
            "model_args": {
                "cfg": args.cfg,
                "model": args.model,
            },
        },
        "epe_results": {
            "mean": sum(epe_scores) / len(epe_scores),
            "max": {
                "score": max_epe,
                "index": max_idx
            },
            "min": {
                "score": min_epe,
                "index": min_idx
            },
            "all": [
                {
                    "score": score,
                    "index": idx,
                } for idx, score in enumerate(epe_scores)
            ]
        },
        "psnr_flow_results": {
            "mean": sum(psnr_flow_scores) / len(psnr_flow_scores),
            "max": {
                "score": max_psnr_flow,
                "index": max_idx_flow
            },
            "min": {
                "score": min_psnr_flow,
                "index": min_idx_flow
            },
            "all": [
                {
                    "score": score,
                    "index": idx,
                } for idx, score in enumerate(psnr_flow_scores)
            ]
        },
        "psnr_motion_results": {
            "mean": sum(psnr_motion_scores) / len(psnr_motion_scores),
            "max": {
                "score": max_psnr_motion,
                "index": max_idx_motion
            },
            "min": {
                "score": min_psnr_motion,
                "index": min_idx_motion
            },
            "all": [
                {
                    "score": score,
                    "index": idx,
                } for idx, score in enumerate(psnr_motion_scores)
            ]
        }
    }

    # Save results to JSON file
    with open(f"output/{name}/results.json", "w") as f:
        json.dump(results, f, indent=4)

def FRPG_loader(name, model, forward_warping_module, forward_warping_with_depth_module, backward_warping_module, dataset_dir_path, args):
    max_index = find_max_index_in_dir(dataset_dir_path)
    print(max_index)

    epe_scores = []
    psnr_flow_scores = []
    psnr_motion_scores = []

    print(name)

    with tqdm(range(max_index - 1)) as pbar:
        for i in pbar:
            write_flag = False

            image_1_flag = os.path.exists(f"{dataset_dir_path}colorNoScreenUI_{i}.png")
            image_2_flag = os.path.exists(f"{dataset_dir_path}colorNoScreenUI_{i + 1}.png")

            if not image_1_flag or not image_2_flag:
                image_1_path = f"{dataset_dir_path}colorNoScreenUI_{i}.exr"
                image_2_path = f"{dataset_dir_path}colorNoScreenUI_{i + 1}.exr"
                image_1 = EXR_to_PNG(image_1_path)
                image_2 = EXR_to_PNG(image_2_path)

                save_img(f"{dataset_dir_path}colorNoScreenUI_{i}.png", image_1)
                save_img(f"{dataset_dir_path}colorNoScreenUI_{i + 1}.png", image_2)

                write_flag = True

            image_1 = cv2.imread(f"{dataset_dir_path}colorNoScreenUI_{i}.png")
            image_2 = cv2.imread(f"{dataset_dir_path}colorNoScreenUI_{i + 1}.png")

            image_1 = torch.from_numpy(image_1).permute(2, 0, 1).unsqueeze(0).float().cuda()
            image_2 = torch.from_numpy(image_2).permute(2, 0, 1).unsqueeze(0).float().cuda()

            bmv_0_path = f"{dataset_dir_path}backwardVel_Depth_{i + 1}.exr"
            bmv, depth = load_backward_velocity(bmv_0_path)

            output_path = f"output/{name}"

            folder_name = f"{output_path}/results/frame_{i:04d}_to_{i+1:04d}/"

            flow, info = inference(folder_name, args, model, image_1, image_2, bmv)

            warped_img_fw_flow = warping(forward_warping_module, image_2, image_1, flow, folder_name, "opticalFlow_forward")
            warped_img_fw_motion = warping(forward_warping_module, image_2, image_1, bmv, folder_name, "gameMotion_forward")
            
            warped_img_fw_flow_with_depth = warping_with_depth(forward_warping_with_depth_module, image_2, image_1, flow, depth, folder_name, "opticalFlow_depth_forward")
            warped_img_fw_motion_with_depth = warping_with_depth(forward_warping_with_depth_module, image_2, image_1, bmv, depth, folder_name, "gameMotion_depth_forward")
            
            warped_img_bw_flow = warping(backward_warping_module, image_1, image_2, flow, folder_name, "opticalFlow_backward")
            warped_img_bw_motion = warping(backward_warping_module, image_1, image_2, bmv, folder_name, "gameMotion_backward")

            epe, psnr_flow, psnr_motion = evaluation(flow, bmv, warped_img_bw_flow, warped_img_bw_motion, image_1)

            pbar.set_postfix({"EPE": f"{epe:.4f}", "PSNR Flow": f"{psnr_flow:.4f}", "PSNR Motion": f"{psnr_motion:.4f}", "NEW": write_flag})

            epe_scores.append(epe)
            psnr_flow_scores.append(psnr_flow)
            psnr_motion_scores.append(psnr_motion)

    save_results_json(name, dataset_dir_path, args, epe_scores, psnr_flow_scores, psnr_motion_scores)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--model', help='checkpoint path', required=True, type=str)
    # parser.add_argument('')
    args = parse_args(parser)
    model = RAFTGM(args)
    load_ckpt(model, args.model)
    model = model.cuda()
    model.eval()

    forward_warping_module = ForwardWarpingNearest()
    forward_warping_with_depth_module = ForwardWarpingNearestWithDepth()
    backward_warping_module = BackwardWarpingNearest()

    # template for loading FRPG images
    record_name = "AnimeFantasyRPG_3_60"
    dataset_root_path = f"/datasets/VFI/datasets/AnimeFantasyRPG/{record_name}/"
    dataset_mode_path = [
        "0_Easy/0_Easy_0/fps_30/", 
        "0_Medium/0_Medium_0/fps_30/", 
        # "0_Difficult/0_Difficult_0/fps_30/",

        # "0_Easy/0_Easy_0/fps_60/", 
        # "0_Medium/0_Medium_0/fps_60/", 
        # "0_Difficult/0_Difficult_0/fps_60/",

        # "4_Easy/4_Easy_0/fps_30/", 
        # "4_Medium/4_Medium_0/fps_30/",
        # "4_Difficult/4_Difficult_0/fps_30/", 

        # "4_Easy/4_Easy_0/fps_60/", 
        # "4_Medium/4_Medium_0/fps_60/", 
        # "4_Difficult/4_Difficult_0/fps_60/", 

        # "0_Easy/0_Easy_1/fps_30/", 
        # "0_Medium/0_Medium_1/fps_30/", 
        # "0_Difficult/0_Difficult_0/fps_30/",

        # "0_Easy/0_Easy_1/fps_60/", 
        # "0_Medium/0_Medium_1/fps_60/", 
        # "0_Difficult/0_Difficult_0/fps_60/",

        # "4_Easy/4_Easy_1/fps_30/", 
        # "4_Medium/4_Medium_1/fps_30/",
        # "4_Difficult/4_Difficult_0/fps_30/", 

        # "4_Easy/4_Easy_1/fps_60/", 
        # "4_Medium/4_Medium_1/fps_60/", 
        # "4_Difficult/4_Difficult_0/fps_60/", 
    ]

    for mode in dataset_mode_path:
        dataset_dir_path = os.path.join(dataset_root_path, mode)
        FRPG_loader(f"SEARAFT_GM/AnimeFantasyRPG/{record_name}/{mode}", model, forward_warping_module, forward_warping_with_depth_module, backward_warping_module, dataset_dir_path, args)


if __name__ == '__main__':
    main()