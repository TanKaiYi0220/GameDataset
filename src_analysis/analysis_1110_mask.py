
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import re
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import read_npy, flow_to_image, save_img
from src.gameData_loader import load_backward_velocity
from src.warp_module import BackwardWarpingNearest

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

def extract_previous_index(root_dir_path: str, path: str) -> str:
    """
    將完整路徑轉成簡短檔名，如：
    /path/.../colorNoScreenUI_246_241.png → colorNoScreenUI_246.png
    """
    basename = os.path.basename(path)  # colorNoScreenUI_246_241.png
    name, ext = os.path.splitext(basename)

    # 以 '_' 拆分，保留前兩段 (文字 + 第一個數字)
    parts = name.split('_')
    previous_path = f"{'_'.join(parts[:2])}{ext}"

    return os.path.join(root_dir_path, previous_path)

def vis_color_diff(img1, img2):
    diff = img1 - img2

    cv2.imshow("Diff", diff)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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

def smape_np(a, b, eps=1e-6):
    """Symmetric Mean Absolute Percentage Error (element-wise)."""
    return np.abs(a - b) / (np.abs(a) + np.abs(b) + eps)

def mape_np(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Element-wise MAPE = |a-b| / (|b| + eps)."""
    return np.abs(a - b) / (np.abs(b) + eps)

def mae_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise MAE = |a-b|."""
    return np.abs(a - b)

def gray_to_rgb(img):
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    return np.repeat(img[..., None], 3, axis=2)  # R=G=B

def min_neighborhood_smape_np(img1, img2, kernel_size=3):
    """
    Compute min_{x' in N(x)} SMAPE(img1[x], img2[x']) for each pixel.
    Args:
        img1, img2: np.ndarray, shape (H, W, C)
    Returns:
        min_smape: np.ndarray, shape (H, W)
    """
    H, W, C = img1.shape
    pad = kernel_size // 2

    # padding 以避免邊界問題
    img2_padded = np.pad(img2, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')

    # 初始化結果圖
    min_smape = np.full((H, W), np.inf, dtype=np.float32)

    # 遍歷 3x3 鄰域（這樣寫比較簡單直觀）
    for dy in range(-pad, pad + 1):
        for dx in range(-pad, pad + 1):
            shifted = img2_padded[pad + dy:pad + dy + H, pad + dx:pad + dx + W, :]
            diff = smape_np(img1, shifted).mean(axis=2)  # 對 channel 求平均
            min_smape = np.minimum(min_smape, diff)      # 取最小值

    return gray_to_rgb(min_smape)

def min_neighborhood_error_np_threshold(
    img1: np.ndarray,
    img2: np.ndarray,
    error_fn,
    kernel_size: int = 3,
    threshold: float = 0.1
) -> np.ndarray:
    """
    Compute min_{x' in N(x)} SMAPE(img1[x], img2[x']) for each pixel.
    Args:
        img1, img2: np.ndarray, shape (H, W, C)
    Returns:
        min_smape: np.ndarray, shape (H, W)
    """
    H, W, C = img1.shape
    pad = kernel_size // 2

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    if img1.max() > 2.0:
        img1 = img1 / 255.0
        img2 = img2 / 255.0

    # padding 以避免邊界問題
    img2_padded = np.pad(img2, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')

    min_error = np.full((H, W), np.inf, dtype=np.float32)
    # 初始化結果圖

    # 遍歷 3x3 鄰域（這樣寫比較簡單直觀）
    for dy in range(-pad, pad + 1):
        for dx in range(-pad, pad + 1):
            shifted = img2_padded[pad + dy:pad + dy + H, pad + dx:pad + dx + W, :]
            diff = error_fn(img1, shifted).mean(axis=2)  # 對 channel 求平均
            min_error = np.minimum(min_error, diff)      # 取最小值

    if threshold != 0.0:
        mask = (min_error > threshold)
        return gray_to_rgb(mask)
    else:
        return gray_to_rgb(min_error)

def diff_mask(img1, img2):
    """
    計算 img1 - img2 並 normalize 到 [0,1]
    Args:
        img1, img2: np.ndarray, shape (H, W, 3)
    Returns:
        diff_norm: np.ndarray, shape (H, W, 3), float32 in [0,1]
    """
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    # 若輸入為 0~255，先轉成 0~1
    if img1.max() > 2.0:
        img1 /= 255.0
        img2 /= 255.0

    # 計算差異並取絕對值（避免正負混亂）
    diff = np.abs(img1 - img2)
    diff_gray = diff.mean(axis=2)

    return gray_to_rgb(diff_gray)

def diff_threshold_mask(img1, img2, threshold=0.1):
    """
    Compute a binary mask where pixel-wise difference > threshold.
    Args:
        img1, img2: np.ndarray of shape (H, W, C)
        threshold: float, difference threshold (e.g., 0.1 for 10%)
    Returns:
        mask: np.ndarray of shape (H, W), dtype=uint8 (0 or 1)
    """
    # 確保是 float 計算
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    # 如果圖像是 0~255，先轉 0~1
    if img1.max() > 1.5:
        img1 /= 255.0
        img2 /= 255.0

    # 計算 per-pixel 差異（取 channel 平均）
    diff = np.abs(img1 - img2).mean(axis=2)

    # 超過 threshold 標記為 1
    mask = (diff > threshold)

    return gray_to_rgb(mask)

def warping(warping_module, src_image, flow):
    # demo warping function
    src_image_torch = torch.from_numpy(src_image).permute(2, 0, 1).unsqueeze(0).float().cuda()
    warping_module.warp(src_image_torch, flow)
    warped_img, _ = warping_module.get_warping_result(mode="average")

    warped_img = warped_img.cpu().numpy()[0].transpose(1, 2, 0).astype(np.uint8)

    return warped_img

if __name__ == "__main__":
    ROOT_PATH = "/datasets/VFI/datasets/AnimeFantasyRPG/"
    RECORD_NAME = "AnimeFantasyRPG_3_60"
    ORIGINAL_FPS = 30
    FPS = f"fps_{ORIGINAL_FPS}"
    MAIN_INDEX = "0"
    DIFFICULTY = ["Easy", "Medium"]
    SUB_INDEX = "0"
    MODES = get_all_modes(MAIN_INDEX, DIFFICULTY, SUB_INDEX, FPS)
    MAX_INDEX = 800
        
    DIFFICULT_DIR_PATH = f"{ROOT_PATH}/{RECORD_NAME}/0_Difficult/0_Difficult_0/{FPS}/"
    CLEAN_ROOT_PATH = f"/datasets/VFI/GFI_datasets/"
    OUTPUT_ROOT_PATH = f"analysis_results/1027_mask/{RECORD_NAME}_clip/"
    if not os.path.exists(OUTPUT_ROOT_PATH):
        os.makedirs(OUTPUT_ROOT_PATH)

    
    backward_warping_module = BackwardWarpingNearest()

    dataset_root_path = f"{CLEAN_ROOT_PATH}/{RECORD_NAME}/"
    print("Dataset Root Path:", dataset_root_path)

    clip_json_path = f"{dataset_root_path}/overall_{FPS}_clip_info.json"
    with open(clip_json_path, "r", encoding="utf-8") as f:
        clip_json = json.load(f)
    print(clip_json.keys())

    mode_name = parse_mode_name(MODES[0])

    clip_list = list(clip_json[mode_name].keys())
    clip_len = 0

    medium_difficulty = MODES[0][1]
    threshold_list = [0.05, 0.10, 0.2, 0.3]

    for clip in clip_list:
        clip_info = clip_json[mode_name][clip]
        clip_color_seq = clip_info[medium_difficulty]["colorNoScreenUI"]
        clip_velD_seq = clip_info[medium_difficulty]["backwardVel_Depth"]

        with tqdm(range(len(clip_color_seq))) as pbar:
            for i in pbar:
                difficult_gt_frame_path = extract_previous_index(DIFFICULT_DIR_PATH, clip_color_seq[i + 1])
                difficult_src_frame_path = extract_previous_index(DIFFICULT_DIR_PATH, clip_color_seq[i])

                bmv, depth = load_backward_velocity(clip_velD_seq[i + 1])

                img_target = cv2.imread(difficult_gt_frame_path)
                img_src = cv2.imread(difficult_src_frame_path)
                
                warped_img_bw_motion = warping(backward_warping_module, img_src, bmv)

                show_images_switchable(
                    [
                        img_target, warped_img_bw_motion, 
                        diff_mask(img_target, warped_img_bw_motion), 
                        min_neighborhood_error_np_threshold(img_target, warped_img_bw_motion, error_fn=smape_np, threshold=0.0, kernel_size=1),
                        min_neighborhood_error_np_threshold(img_target, warped_img_bw_motion, error_fn=smape_np, threshold=0.0),
                        min_neighborhood_error_np_threshold(img_target, warped_img_bw_motion, error_fn=smape_np, threshold=0.0, kernel_size=5),
                        min_neighborhood_error_np_threshold(img_target, warped_img_bw_motion, error_fn=mae_np, threshold=0.0),
                        min_neighborhood_error_np_threshold(img_target, warped_img_bw_motion, error_fn=mae_np, threshold=0.0, kernel_size=5),
                    ],
                    [
                        "Target", "Warped Image", 
                        "MAE Mask", 
                        "SMAPE Mask",
                        "SMAPE Mask with 3x3 kernel",
                        "SMAPE Mask with 5X5 kernel", 
                        "MAE Mask with 3x3 kernel", 
                        "MAE Mask with 5x5 kernel", 
                    ]
                )

                break

        