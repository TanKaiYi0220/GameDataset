
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import re
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.metrics import cal_psnr
from src.utils import read_npy, flow_to_image, save_img
from src.gameData_loader import load_backward_velocity

def get_all_modes(main_index_list, difficulty_list, sub_index, fps):
    modes = []
    for main_index in main_index_list:
        mode = []
        for difficulty in difficulty_list:
            mode.append(f"{main_index}_{difficulty}/{main_index}_{difficulty}_{sub_index}/{fps}")
        modes.append(mode)
    return modes

def parse_mode_name(modes):
    """
    從 mode list 取出簡化名稱。
    e.g.
    ["0_Easy/0_Easy_0/fps_60", "0_Medium/0_Medium_0/fps_60"] → "0_Easy_Medium_0"
    """
    # 拿第一個 mode 拆開
    first = modes[0].split('/')
    main_index = first[0].split('_')[0]         # 例如 '0'
    sub_index = first[1].split('_')[-1]         # 例如 '0'
    fps = first[2]

    # 把所有 mode 的第二個部分的難度名稱抓出來
    difficulties = [m.split('/')[0].split('_')[1] for m in modes]

    # 拼接結果
    combined_name = f"{main_index}_{'_'.join(difficulties)}_{sub_index}_{fps}"
    return combined_name

def show_images_switchable(images, titles):
    """
    images: list[np.ndarray]   要顯示的圖片
    titles: list[str]          每張圖的標題
    """
    assert len(images) == len(titles)
    idx = 0
    n = len(images)

    while True:
        img = images[idx].copy()

        # 顯示標題 (目前第幾張)
        text = f"[{idx+1}/{n}] {titles[idx]}"
        cv2.putText(img, text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Overlay", img)
        key = cv2.waitKey(0) & 0xFF

        # ← 或 ↑：上一張
        if key in [ord('a'), 81, 82]:  # 'a' 或 左/上箭頭
            idx = (idx - 1) % n
        # → 或 ↓：下一張
        elif key in [ord('d'), 83, 84]:  # 'd' 或 右/下箭頭
            idx = (idx + 1) % n
        # q 或 ESC 離開
        elif key in [ord('q'), 27]:
            break

    cv2.destroyAllWindows()

def convert_hit_img_to_binary(hit_img: np.ndarray):
    """
    將 hit image (BGR) 轉換為 binary mask。
    - BLUE (255, 0, 0): NOT VALID (0)
    - GREEN (0, 255, 0): VALID (1)
    """
    green = np.array([0, 255, 0], dtype=np.uint8)
    blue  = np.array([255, 0, 0], dtype=np.uint8)

    valid_mask   = np.all(hit_img == green, axis=-1)
    invalid_mask = np.all(hit_img == blue,  axis=-1)

    binary_mask = np.zeros(hit_img.shape[:2], dtype=np.uint8)
    binary_mask[valid_mask] = 1
    return binary_mask

def load_warped_and_hit(root_path: str, suffix: str):
    warped_path = os.path.join(root_path, f"warped_img_{suffix}.png")
    warped = cv2.imread(warped_path) # H, W, 3

    hit_warped_path = os.path.join(root_path, f"hit_img_{suffix}.png")
    hit_warped = cv2.imread(hit_warped_path)

    hit_warped = convert_hit_img_to_binary(hit_warped)

    return warped, hit_warped

if __name__ == "__main__":
    ROOT_PATH = "/datasets/VFI/datasets/AnimeFantasyRPG/"
    RECORD_NAME = "AnimeFantasyRPG_3_60"
    ORIGINAL_FPS = 30
    FPS = f"fps_{ORIGINAL_FPS}"
    MAIN_INDEX = ["0", "1", "2", "3"]
    DIFFICULTY = ["Easy", "Medium"]
    SUB_INDEX = "0"
    MODES = get_all_modes(MAIN_INDEX, DIFFICULTY, SUB_INDEX, FPS)
    MAX_INDEX = 800
        
    IMG_FOLDER = f"{ROOT_PATH}/{RECORD_NAME}"

    RESULT_ROOT_PATH = f"output/SEARAFT/AnimeFantasyRPG/{RECORD_NAME}_clip/{RECORD_NAME}/"
    RESULT_GM_ROOT_PATH = f"output/SEARAFT_GM/AnimeFantasyRPG/{RECORD_NAME}_clip/{RECORD_NAME}/"
    
    CLEAN_ROOT_PATH = f"/datasets/VFI/GFI_datasets/"
    OUTPUT_ROOT_PATH = f"analysis_results/1103_PSNR_Mask/{RECORD_NAME}_clip/"
    if not os.path.exists(OUTPUT_ROOT_PATH):
        os.makedirs(OUTPUT_ROOT_PATH)

    dataset_root_path = f"{CLEAN_ROOT_PATH}/{RECORD_NAME}/"
    print("Dataset Root Path:", dataset_root_path)

    clip_json_path = f"{dataset_root_path}/overall_{FPS}_clip_info.json"
    with open(clip_json_path, "r", encoding="utf-8") as f:
        clip_json = json.load(f)
    print(clip_json.keys())

    overall_df = pd.DataFrame()

    for mode in MODES:
        mode_name = parse_mode_name(mode)
        print(mode_name)

        for difficult in mode:
            clip_list = list(clip_json[mode_name].keys())
            clip_len = 0

            record = {
                "difficult": [],
                "clip": [],
                "frame": [],
                "psnr bmv": [],
                "psnr flow": [],
                "psnr flow gm": []
            }

            print(difficult)
            for clip in clip_list:
                result_path = f"{RESULT_ROOT_PATH}/{difficult}/{clip}/results"
                result_gm_path = f"{RESULT_GM_ROOT_PATH}/{difficult}/{clip}/results"
                flow_dir_path = os.listdir(result_path)
                flow_gm_dir_path = os.listdir(result_gm_path)
                flow_dir_path = sorted(flow_dir_path, key=lambda x:int(x.split("_")[-1]))
                flow_gm_dir_path = sorted(flow_gm_dir_path, key=lambda x:int(x.split("_")[-1]))

                clip_info = clip_json[mode_name][clip]
                clip_color_seq = clip_info[difficult]["colorNoScreenUI"]
                clip_velD_seq = clip_info[difficult]["backwardVel_Depth"]


                print(clip)
                # with tqdm(range(5)) as pbar:
                with tqdm(range(len(flow_dir_path))) as pbar:
                    for i in pbar:
                        clip_root_path = os.path.join(result_path, flow_dir_path[i])
                        clip_gm_root_path = os.path.join(result_gm_path, flow_gm_dir_path[i])

                        src_path = os.path.join(clip_root_path, clip_color_seq[i])
                        src_img = cv2.imread(src_path)

                        target_path = os.path.join(clip_root_path, clip_color_seq[i + 1])
                        target_img = cv2.imread(target_path)

                        bmv_path = os.path.join(clip_root_path, clip_velD_seq[i + 1])
                        bmv, depth = load_backward_velocity(bmv_path)
                        bmv = bmv[0].permute(1, 2, 0).cpu().numpy()

                        warped_bmv, hit_warped_bmv = load_warped_and_hit(clip_root_path, "gameMotion_backward")
                        warped_flow, hit_warped_flow = load_warped_and_hit(clip_root_path, "opticalFlow_backward")
                        warped_flow_gm, hit_warped_flow_gm = load_warped_and_hit(clip_gm_root_path, "opticalFlow_backward")

                        psnr_bmv = cal_psnr(target_img, warped_bmv, hit_warped_bmv)
                        psnr_flow = cal_psnr(target_img, warped_flow, hit_warped_flow)
                        psnr_flow_gm = cal_psnr(target_img, warped_flow_gm, hit_warped_flow_gm)


                        record["difficult"].append(difficult)
                        record["clip"].append(difficult)
                        record["frame"].append(i)
                        record["psnr bmv"].append(psnr_bmv)
                        record["psnr flow"].append(psnr_flow)
                        record["psnr flow gm"].append(psnr_flow_gm)

                        pbar.set_postfix({"PSNR BMV": psnr_bmv, "PSNR Flow": psnr_flow, "PSNR Flow GM": psnr_flow_gm})


                    # Debug
                    # show_images_switchable(
                    #     [src_img, target_img, warped_bmv, hit_warped_bmv * 255.0, warped_flow, hit_warped_flow * 255.0, warped_flow_gm, hit_warped_flow_gm * 255.0], 
                    #     ["Source Image", "Target Image", "Warped BMV", "Hit map for Warped BMV", "Warped Flow", "Hit map for Warped Flow", "Warped Flow GM", "Hit map for Warped Flow GM"]
                    # )

            temp_df = pd.DataFrame(record)
            overall_df = pd.concat([overall_df, temp_df], ignore_index=True)

    overall_df.to_csv(f"{OUTPUT_ROOT_PATH}/overall_psnr.csv")
