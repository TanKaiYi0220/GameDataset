import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, random_split

from gamedataset.data.loader import FlowEstimationTrainDataset
from gamedataset.data.config import (
    iter_dataset_configs,
    TRAIN_DATASET_CONFIGS,
    TEST_DATASET_CONFIGS,
    STAIR_DATASET_CONFIG,
    VFX_DATASET_CONFIGS,
    TRAIN_VFX_0326_DATASET_CONFIGS, 
    TEST_VFX_0326_DATASET_CONFIGS,
    TRAIN_VFX_0416_DATASET_CONFIGS,
    TEST_VFX_0416_DATASET_CONFIGS
)

from gamedataset.utils.image import flow_to_image

DATASET = TEST_VFX_0416_DATASET_CONFIGS
ROOT_DIR = TEST_VFX_0416_DATASET_CONFIGS["root_dir"]
DF_ROOT = "./datasets/data"

def calc_epe(flow: np.ndarray, flow_gt: np.ndarray):
    diff = flow - flow_gt
    return np.sqrt(diff[..., 0] ** 2 + diff[..., 1] ** 2)

def flow_approx(flow, time, forward=True):
    # 1/t
    return time * flow if forward else (1 - time) * flow

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

def vis_epe_heatmap(epe: np.ndarray, img: np.ndarray, vmin=None, vmax=None):
    """
    epe: (H, W)
    img: (H, W, 3) uint8
    """

    # ---- 1. set range ----
    if vmin is None:
        vmin = 0.0

    if vmax is None:
        # avoid outlier → use percentile
        vmax = np.percentile(epe, 99)

    epe_clipped = np.clip(epe, vmin, vmax)

    # ---- 2. normalize → uint8 ----
    norm = (epe_clipped - vmin) / (vmax - vmin + 1e-8)
    norm = (norm * 255).astype(np.uint8)   # (H,W)

    # ---- 3. colormap ----
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)

    # ---- 4. overlay ----
    overlay = (
        img.astype(np.float32) * 0.4 +
        colored.astype(np.float32) * 0.6
    )
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # ---- 5. colorbar ----
    color_bar = create_color_bar_with_labels(
        vmin=vmin,
        vmax=vmax,
        width=30,
        height=overlay.shape[0]
    )

    return np.hstack((overlay, color_bar))

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

def valid_distance_indexing(distance_indexing, threshold=0.1):
    if distance_indexing > 0.5 + threshold or distance_indexing < 0.5 - threshold:
        return False
    return True

def main():
    # Load Dataset
    for cfg in iter_dataset_configs(DATASET):
        if cfg.fps != 60:
            continue

        if cfg.difficulty != "Difficult":
            continue

        df = pd.read_csv(f"{DF_ROOT}/{cfg.record_name}_preprocessed/{cfg.mode_index}_raw_sequence_frame_index.csv")
        df["record"] = cfg.record
        df["mode"] = cfg.mode_path

        dataset = FlowEstimationTrainDataset(df, ROOT_DIR, input_fps=30, transform=False)

        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False
        )

        pbar = tqdm(loader)

        fmv_epe = []
        bmv_epe = []
        invalid_count = 0

        for batch in pbar:
            img0, imgt, img1, bmv_60, fmv_60, bmv_30, fmv_30, img0_30, img1_30, embt, info = batch

            if not valid_distance_indexing(info["distance_indexing"][0], 0.1) or not valid_distance_indexing(info["distance_indexing"][1], 0.1):
                # print("Not Valid")
                invalid_count += 1
                continue

            B = img0.shape[0]

            for b in range(B):
                img0_np = (img0[b].permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8)
                imgt_np = (imgt[b].permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8)
                img1_np = (img1[b].permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8)
                bmv_30_np = bmv_30[b].permute(1, 2, 0).cpu().numpy()
                fmv_30_np = fmv_30[b].permute(1, 2, 0).cpu().numpy()
                bmv_60_np = bmv_60[b].permute(1, 2, 0).cpu().numpy()
                fmv_60_np = fmv_60[b].permute(1, 2, 0).cpu().numpy()
                img0_30_np = (img0_30[b].permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8)
                img1_30_np = (img1_30[b].permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8)

                bmv_60_approx = flow_approx(bmv_30_np, 0.5, forward=False)
                fmv_60_approx = flow_approx(fmv_30_np, 0.5, forward=True)

                epe_diff = calc_epe(bmv_30_np, bmv_60_np)
                epe_approx_diff = calc_epe(bmv_60_approx, bmv_60_np)

                # 用同一個 scale（關鍵）
                global_max = max(
                    np.percentile(epe_diff, 99),
                    np.percentile(epe_approx_diff, 99)
                )

                overlay = vis_epe_heatmap(epe_diff, imgt_np, vmin=0.0, vmax=global_max)
                overlay_approx = vis_epe_heatmap(epe_approx_diff, imgt_np, vmin=0.0, vmax=global_max)
                # overlay_discrete = vis_epe_discrete(epe_diff, img)

                fmv_epe.append(calc_epe(bmv_60_approx, bmv_60_np))
                bmv_epe.append(calc_epe(fmv_60_approx, fmv_60_np))

                # show_images_switchable(
                #     [img0_np, img1_np, flow_to_image(bmv_60_np), flow_to_image(bmv_60_approx), flow_to_image(bmv_30_np), overlay, overlay_approx],
                #     ["img0", "img1", "fmv 60", "fmv 60 (approx)", "fmv 30", "epe diff", "epe approx diff"]
                # )

                # show_images_switchable(
                #     [img0_np, img0_30_np, img1_np, img1_30_np],
                #     ["img0", "img0 30", "img1", "img1 30"]
                # )

        print("EPE =", sum(fmv_epe) / len(fmv_epe))
        print("Invalid Count =", invalid_count)
        exit()



if __name__ == "__main__":
    main()