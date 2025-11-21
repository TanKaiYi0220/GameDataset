import os

import cv2
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
from src.gameData_loader import load_backward_velocity
from src.warp_module import ForwardWarpingNearest, BackwardWarpingNearest, ForwardWarpingNearestWithDepth
from src.utils import flow_to_image, get_all_modes, parse_mode_name, show_images_switchable, save_img

import sys
sys.path.append('models/IFRNet')

from models.IFRNet import Model
from utils import warp

import numpy as np
import torch
from utils import read
from imageio import mimsave
from tqdm import tqdm
import json

def inference(folder_name, model, image_1, image_2):
    embt = torch.tensor(1/2).view(1, 1, 1, 1).float().cuda()

    imgt_pred, up_flow0_1, up_flow1_1, up_mask_1 = model.inference(image_1, image_2, embt)

    imgt_pred_np = (imgt_pred[0].data.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    up_flow0_1_np = flow_to_image(up_flow0_1[0].data.permute(1, 2, 0).cpu().numpy())
    up_flow1_1_np = flow_to_image(up_flow1_1[0].data.permute(1, 2, 0).cpu().numpy())
    up_mask_1_np = (up_mask_1[0, 0].data.cpu().numpy() * 255.0).astype(np.uint8)

    # Save results
    # BGR to RGB
    save_img(f'{folder_name}/out_2x_test.png', cv2.cvtColor(imgt_pred_np, cv2.COLOR_RGB2BGR))

    # Flow and mask
    save_img(f'{folder_name}/out_2x_flow0.png', up_flow0_1_np)
    save_img(f'{folder_name}/out_2x_flow1.png', up_flow1_1_np)
    save_img(f'{folder_name}/out_2x_mask.png', up_mask_1_np)

    return imgt_pred_np, up_flow0_1_np, up_flow1_1_np, up_mask_1_np

def FRPG_loader(name, model, forward_warping_module, forward_warping_with_depth_module, backward_warping_module, clip_color_seq, clip_velD_seq):
    max_index = len(clip_color_seq)

    epe_scores = []
    psnr_flow_scores = []
    psnr_motion_scores = []

    with tqdm(range(max_index - 1)) as pbar:
        for i in pbar:
            image_1 = cv2.imread(clip_color_seq[i])
            image_2 = cv2.imread(clip_color_seq[i + 1])

            # image_1 = torch.from_numpy(image_1).permute(2, 0, 1).unsqueeze(0).float().cuda()
            # image_2 = torch.from_numpy(image_2).permute(2, 0, 1).unsqueeze(0).float().cuda()

            image_1 = (torch.tensor(image_1.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).cuda()
            image_2 = (torch.tensor(image_2.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).cuda()

            bmv, depth = load_backward_velocity(clip_velD_seq[i + 1])

            folder_name = f"{name}/results/frame_{i:04d}_to_{i+1:04d}/"

            imgt_pred, up_flow0_1, up_flow1_1, up_mask_1 = inference(folder_name, model, image_1, image_2)

            # warped_img_fw_flow = warping(forward_warping_module, image_2, image_1, flow, folder_name, "opticalFlow_forward")
            # warped_img_fw_motion = warping(forward_warping_module, image_2, image_1, bmv, folder_name, "gameMotion_forward")
            
            # warped_img_fw_flow_with_depth = warping_with_depth(forward_warping_with_depth_module, image_2, image_1, flow, depth, folder_name, "opticalFlow_depth_forward")
            # warped_img_fw_motion_with_depth = warping_with_depth(forward_warping_with_depth_module, image_2, image_1, bmv, depth, folder_name, "gameMotion_depth_forward")
            
            # warped_img_bw_flow = warping(backward_warping_module, image_1, image_2, flow, folder_name, "opticalFlow_backward")
            # warped_img_bw_motion = warping(backward_warping_module, image_1, image_2, bmv, folder_name, "gameMotion_backward")

            # epe, psnr_flow, psnr_motion = evaluation(flow, bmv, warped_img_bw_flow, warped_img_bw_motion, image_1)

            # pbar.set_postfix({"EPE": f"{epe:.4f}", "PSNR Flow": f"{psnr_flow:.4f}", "PSNR Motion": f"{psnr_motion:.4f}"})

            # epe_scores.append(epe)
            # psnr_flow_scores.append(psnr_flow)
            # psnr_motion_scores.append(psnr_motion)

            image_1_np = (image_1[0].data.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
            image_2_np = (image_2[0].data.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)

            show_images_switchable(
                [image_1_np,
                 imgt_pred,
                 image_2_np],
                ["Image 1", "Interpolated Image", "Image 2"]
            )

    return epe_scores, psnr_flow_scores, psnr_motion_scores


def main():
    model = Model().cuda().eval()
    model.load_state_dict(torch.load('./models/IFRNet/checkpoints/IFRNet/IFRNet_Vimeo90K.pth'))

    forward_warping_module = ForwardWarpingNearest()
    forward_warping_with_depth_module = ForwardWarpingNearestWithDepth()
    backward_warping_module = BackwardWarpingNearest()

    ROOT_PATH = "/datasets/VFI/datasets/AnimeFantasyRPG/"
    RECORD_NAME = ["AnimeFantasyRPG_3_60"]
    FPS = "fps_30"
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

                    output_path = f"output/IFRNet/AnimeFantasyRPG/{record_name}_clip/{record_name}/{difficult}/{clip}"

                    epe_scores, psnr_flow_scores, psnr_motion_scores = FRPG_loader(
                        output_path, 
                        model, 
                        forward_warping_module, forward_warping_with_depth_module, backward_warping_module, 
                        clip_color_seq, clip_velD_seq,
                    )

                    epe_scores_list.extend(epe_scores)
                    psnr_flow_scores_list.extend(psnr_flow_scores)
                    psnr_motion_scores_list.extend(psnr_motion_scores)

                    clip_len = len(epe_scores)

                    print(clip_len)


if __name__ == "__main__":
    main()