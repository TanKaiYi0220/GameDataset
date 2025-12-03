import cv2
import os
import pandas as pd
import numpy as np

from dataset_config import DATASET_CONFIGS
from utils import load_backward_velocity, flow_to_image

ROOT_DIR = DATASET_CONFIGS["root_dir"]

def show_images(img: np.ndarray, img_path: str, i: int, max_i: int, status: str, win: str):
    if status:
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)

    cv2.rectangle(img, (5, 5), (img.shape[1] - 5, img.shape[0] - 5), color, 3)
    text = f"[{i+1}/{max_i}] {os.path.basename(img_path)} | Status: {status}"
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow(win, img)

def review_images(easy_df, medium_df):
    """
    一個簡單的 OpenCV 審圖工具：
    ←/→ or ↑/↓ 翻頁
    Y: 採用
    N: 不採用
    S: 存檔
    Q / ESC: 退出
    """
    i = 1
    difficult_idx = 1  # 0: easy, 1: medium
    flow_flag = 0      # 0: RGB, 1: flow
    win = "Reviewer"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        current_easy_row = easy_df.iloc[i]
        current_medium_row = medium_df.iloc[i]

        easy_dir_path = os.path.join(ROOT_DIR, current_easy_row["record"], current_easy_row["mode"])
        medium_dir_path = os.path.join(ROOT_DIR, current_medium_row["record"], current_medium_row["mode"])

        easy_img_path = os.path.join(easy_dir_path, f"colorNoScreenUI_{i}.png")
        medium_img_path = os.path.join(medium_dir_path, f"colorNoScreenUI_{i}.png")
        backward_vel_easy_path = os.path.join(easy_dir_path, f"backwardVel_Depth_{i}.exr")
        backward_vel_medium_path = os.path.join(medium_dir_path, f"backwardVel_Depth_{i}.exr")

        try:
            # 根據目前的 difficult_idx / flow_flag 決定要讀哪種圖
            if difficult_idx == 0:  # easy
                display_path = easy_img_path   # show_images 裡面顯示的路徑還是用 RGB 那個
                if flow_flag == 0:
                    img = cv2.imread(easy_img_path)
                else:
                    mv_easy, _ = load_backward_velocity(backward_vel_easy_path)
                    img = flow_to_image(mv_easy)
            else:  # medium
                display_path = medium_img_path
                if flow_flag == 0:
                    img = cv2.imread(medium_img_path)
                else:
                    mv_medium, _ = load_backward_velocity(backward_vel_medium_path)
                    img = flow_to_image(mv_medium)

            # 根據標記顯示不同顏色邊框
            status = current_easy_row["is_valid"] and current_medium_row["is_valid"]

            show_images(img, display_path, i, len(easy_df), status, win)

        except Exception as e:
            # 若讀取錯誤就跳到下一張
            i = min(i + 1, len(easy_df) - 1)

        key = cv2.waitKey(0) & 0xFF

        # ↑: 82, ↓: 84, →: 83, ←: 81，另外加上 WASD 方便
        if key in (82, ord('w')):  # Up: diff++
            difficult_idx = (difficult_idx + 1) % 2  # 0 <-> 1
        elif key in (84, ord('s')):  # Down: diff--
            difficult_idx = (difficult_idx - 1) % 2
        elif key in (83, ord('d')):  # Right: frameIndex++
            i = min(i + 1, len(easy_df) - 1)
        elif key in (81, ord('a')):  # Left: frameIndex--
            i = max(i - 1, 0)
        elif key in (ord('f'), ord('F')):  # Flow toggle
            flow_flag = 1 - flow_flag      # 0 <-> 1
        elif key in (ord('y'), ord('Y')):
            easy_df.at[i, "is_valid"] = True
            medium_df.at[i, "is_valid"] = True
            i = min(i + 1, len(easy_df) - 1)
        elif key in (ord('n'), ord('N')):
            easy_df.at[i, "is_valid"] = False
            medium_df.at[i, "is_valid"] = False
            easy_df.at[i, "reason"] = "Rendering BUG (Player/Camera)"
            medium_df.at[i, "reason"] = "Rendering BUG (Player/Camera)"
            i = min(i + 1, len(easy_df) - 1)
        elif key in (ord('q'), ord('Q'), 27):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    easy_df = pd.read_csv("./data/AnimeFantasyRPG_3_60/0_Easy_0_fps_60_frame_index.csv", dtype={"reason": "string"})
    medium_df = pd.read_csv("./data/AnimeFantasyRPG_3_60/0_Medium_0_fps_60_frame_index.csv", dtype={"reason": "string"})

    review_images(easy_df, medium_df)

    easy_df.to_csv("./data/AnimeFantasyRPG_3_60/0_Easy_0_fps_60_frame_index.csv", index=False)
    medium_df.to_csv("./data/AnimeFantasyRPG_3_60/0_Medium_0_fps_60_frame_index.csv", index=False)