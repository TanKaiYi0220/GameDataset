
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import re
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

def calc_epe(flow: np.ndarray, flow_gt: np.ndarray):
    diff = flow - flow_gt
    return np.sqrt(diff[..., 0] ** 2 + diff[..., 1] ** 2)

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

def vis_epe_heatmap(epe: np.ndarray, img: np.ndarray):
    epe_min = float(epe.min())
    epe_max = float(epe.max())

    norm_epe = (epe - epe_min) / (epe_max - epe_min + 1e-8)
    norm_epe = (norm_epe * 255).astype(np.uint8)

    colored_map = cv2.applyColorMap(norm_epe, cv2.COLORMAP_JET)

    # --- Overlay with source image ---
    overlay = img * 0.3 + colored_map * 0.7
    overlay = overlay.astype(np.uint8)

    # --- Draw colorbar with actual EPE range ---
    color_bar = create_color_bar_with_labels(cv2.COLORMAP_JET, epe_min, epe_max, width=30, height=overlay.shape[0])

    # --- Combine ---
    combined_image = np.hstack((overlay, color_bar))

    return combined_image

import numpy as np
import cv2

def vis_epe_discrete(epe_delta: np.ndarray, img: np.ndarray):
    """
    ΔEPE (B-A) 離散視覺化 + 圖例 + 百分比
    """
    H, W = epe_delta.shape
    vis = np.zeros((H, W, 3), np.uint8)

    bins   = [-np.inf, -10, -5, -2, -0.5, 0.5, 2, 5, 10, np.inf]
    labels = ["(-inf,-10]", "(-10,-5]", "(-5,-2]", "(-2,-0.5]",
              "[-0.5,0.5]", "(0.5,2]", "(2,5]", "(5,10]", "(10,inf)"]
    colors = [  # BGR
        (128, 64,  0), (144, 96, 16), (176,144, 64), (208,192,128),
        (170,170,170),
        (32,176,255), (16,128,255), (0,64,255), (0,0,255)
    ]

    finite = np.isfinite(epe_delta)
    total_valid = int(finite.sum())
    counts = []

    for i in range(len(bins)-1):
        lo, hi = bins[i], bins[i+1]
        if i == 4:  # 中間段包含邊界
            mask = finite & (epe_delta >= lo) & (epe_delta <= hi)
        else:
            mask = finite & (epe_delta > lo) & (epe_delta <= hi)
        vis[mask] = colors[i]
        counts.append(mask.sum())

    # 將百分比加入標籤文字
    if total_valid > 0:
        labels = [f"{lab}: {cnt / total_valid * 100:.1f}%" for lab, cnt in zip(labels, counts)]

    # 疊圖
    overlay = cv2.addWeighted(img, 0.35, vis, 0.65, 0)

    # 畫 legend
    legend_w = 260
    legend = np.full((H, legend_w, 3), 250, np.uint8)
    cv2.putText(legend, "ΔEPE bins (B-A)", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2, cv2.LINE_AA)

    top = 54
    row_h = 30
    for i, (lab, col) in enumerate(zip(labels, colors)):
        y = top + i * row_h
        cv2.rectangle(legend, (12, y-18), (42, y+6), col, -1)
        cv2.rectangle(legend, (12, y-18), (42, y+6), (40,40,40), 1)
        cv2.putText(legend, lab, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30,30,30), 1, cv2.LINE_AA)

    return np.hstack([overlay, legend])


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
    OUTPUT_ROOT_PATH = f"analysis_results/1027/{RECORD_NAME}_clip/"
    if not os.path.exists(OUTPUT_ROOT_PATH):
        os.makedirs(OUTPUT_ROOT_PATH)

    dataset_root_path = f"{CLEAN_ROOT_PATH}/{RECORD_NAME}/"
    print("Dataset Root Path:", dataset_root_path)

    clip_json_path = f"{dataset_root_path}/overall_{FPS}_clip_info.json"
    with open(clip_json_path, "r", encoding="utf-8") as f:
        clip_json = json.load(f)
    print(clip_json.keys())

    for mode in MODES:
        mode_name = parse_mode_name(mode)
        print(mode_name)

        for difficult in mode:
            clip_list = list(clip_json[mode_name].keys())
            clip_len = 0

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
                with tqdm(range(len(flow_dir_path))) as pbar:
                    for i in pbar:
                        pbar.set_postfix({"Frame": flow_dir_path[i]})
                        color_path = os.path.join(result_path, flow_dir_path[i], clip_color_seq[i + 1])
                        img = cv2.imread(color_path)

                        bmv_path = os.path.join(result_path, flow_dir_path[i], clip_velD_seq[i + 1])
                        bmv, depth = load_backward_velocity(bmv_path)
                        bmv = bmv[0].permute(1, 2, 0).cpu().numpy()

                        flow_path = os.path.join(result_path, flow_dir_path[i], "flow_final.npy")
                        flow = read_npy(flow_path) # H, W, 2

                        flow_gm_path = os.path.join(result_gm_path, flow_gm_dir_path[i], "flow_gm_final.npy")
                        flow_gm = read_npy(flow_gm_path) # H, W, 2

                        epe_flow = calc_epe(flow, bmv)
                        epe_flow_gm = calc_epe(flow_gm, bmv)

                        epe_diff = epe_flow - epe_flow_gm
                        
                        overlay = vis_epe_heatmap(epe_diff, img)
                        overlay_discrete = vis_epe_discrete(epe_diff, img)

                        output_path = f"{OUTPUT_ROOT_PATH}/{difficult}/{clip}/{flow_dir_path[i]}/"

                        save_img(f"{output_path}/epe_improve_heatmap.png", overlay)
                        save_img(f"{output_path}/epe_improve_discrete_heatmap.png", overlay_discrete)

                        # Debug
                        # show_images_switchable(
                        #     [overlay, overlay_discrete, flow_to_image(flow), flow_to_image(flow_gm), flow_to_image(bmv)], 
                        #     ["Overlay", "Overlay Discrete", "Flow", "Flow GM", "GM"]
                        # )

