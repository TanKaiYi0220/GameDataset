
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

    df_path = f"{OUTPUT_ROOT_PATH}/overall_psnr.csv"
    overall_df = pd.read_csv(df_path)

    print(overall_df.head())

    for mode in MODES:
        mode_name = parse_mode_name(mode)
        print(mode_name)

        for difficult in mode:
            clip_list = list(clip_json[mode_name].keys())
            clip_len = 0

            print(difficult)

            difficult_df = overall_df[overall_df["difficult"] == difficult]
            psnr_mean_bmv = difficult_df["psnr bmv"].mean()
            psnr_mean_flow = difficult_df["psnr flow"].mean()
            psnr_mean_flow_gm = difficult_df["psnr flow gm"].mean()

            data = [
                difficult_df['psnr bmv'],
                difficult_df['psnr flow'],
                difficult_df['psnr flow gm']
            ]

            labels = [
                f'Mean PSNR BMV = {psnr_mean_bmv:.2f}', 
                f'Mean PSNR Flow = {psnr_mean_flow:.2f}', 
                f'Mean PSNR Flow GM = {psnr_mean_flow_gm:.2f}'
            ]

            print(difficult_df.head())

            index = difficult_df.index - difficult_df.index[0]
            print(index)
            print(len(data[0]))

            plt.figure(figsize=(8, 6))
            for d, l in zip(data, labels):
                plt.plot(index, d, label=l)

            plt.xlabel('Index')
            plt.ylabel('PSNR')
            plt.title(f'PSNR - {difficult}')
            plt.legend()
            plt.grid(True)
            os.makedirs(f"{OUTPUT_ROOT_PATH}/{difficult}", exist_ok=True)
            plt.savefig(f"{OUTPUT_ROOT_PATH}/{difficult}/psnr.png")

            plt.figure(figsize=(8, 6))

            # 建立 violin plot
            parts = plt.violinplot(
                dataset=data,
                showmeans=True,       # 顯示平均值
                showmedians=True,     # 顯示中位數
                showextrema=True      # 顯示 min/max
            )

            # 設定外觀
            for pc in parts['bodies']:
                pc.set_facecolor('#87CEFA')   # 改成你喜歡的顏色（天藍）
                pc.set_edgecolor('black')
                pc.set_alpha(0.7)

            # 軸與標籤
            plt.xticks(range(1, len(labels) + 1), labels)
            plt.ylabel('PSNR')
            plt.title(f'PSNR Distribution (Violin Plot) - {difficult}')
            plt.grid(True, axis='y')
            # plt.show()
            plt.savefig(f"{OUTPUT_ROOT_PATH}/{difficult}/psnr_violin.png")