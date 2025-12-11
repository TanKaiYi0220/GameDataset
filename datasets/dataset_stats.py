import pandas as pd
from tqdm import tqdm
import os
import numpy as np
from typing import Optional, Sequence, Tuple, Dict, List, Any
import matplotlib.pyplot as plt

import sys
sys.path.append('../')
from datasets.utils import load_backward_velocity
from datasets.dataset_loader import VFIDataset
from datasets.dataset_config import DATASET_CONFIGS, MINOR_DATASET_CONFIGS, VFX_DATASET_CONFIGS, STAIR_DATASET_CONFIG, iter_dataset_configs


DATASET = STAIR_DATASET_CONFIG
ROOT_DIR = DATASET["root_dir"]

def calculate_magnitude(motion):
    magnitude = motion[:, :, 0] ** 2 + motion[:, :, 0] ** 2
    magnitude = np.sqrt(magnitude)

    return magnitude

def bin_stats_from_raw(
    magnitudes: np.ndarray,
    bins: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    bins = np.asarray(bins, dtype=float)
    edges = np.concatenate([[0.0], bins])  # (0,b1], (b1,b2], ...
    upper_edges = edges[1:]               # 每個 bin 的右邊界

    hist, _ = np.histogram(magnitudes.astype(float), bins=edges)
    freq = hist / hist.sum()            # 橫軸每個 bin 的頻率（和=1）
    return upper_edges, freq


# plot frequency for motion magnitude distribution
def plot_frequency(
    bins: Sequence[float],
    motion_list: Sequence[float],
    seq_valid: Sequence[Any],
    bias_value: float,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    motion = np.asarray(motion_list, dtype=float)
    valid = np.asarray(seq_valid)

    # 避免 NaN 影響（因為你只在偶數 frame_idx 寫入 mean）
    mask = ~np.isnan(motion)
    motion = motion[mask]
    valid = valid[mask]

    # 假設 valid 欄位是 0/1 或 True/False
    valid_mask = (valid.astype(int) == 1)

    # 三種集合
    motion_all = motion
    motion_valid = motion[valid_mask]
    motion_invalid = motion[~valid_mask]

    # 各自算 histogram
    upper_edges, freq_all = bin_stats_from_raw(motion_all, bins=bins + bias_value)
    _, freq_valid = bin_stats_from_raw(motion_valid, bins=bins + bias_value)
    _, freq_invalid = bin_stats_from_raw(motion_invalid, bins=bins + bias_value)

    plt.figure(figsize=(6, 4))

    # 假設 bins 間距固定，用 offset 做並排柱狀圖
    bins = np.asarray(bins, dtype=float)
    if len(bins) > 1:
        bin_width = bins[1] - bins[0]
    else:
        bin_width = 1.0

    bar_width = bin_width * 0.25
    offset = bar_width

    # x 軸位置使用 bin 的右邊界（upper_edges）
    x = upper_edges - bias_value

    plt.bar(x - offset, freq_all,   width=bar_width, label="All",     linewidth=0.3)
    plt.bar(x,          freq_valid, width=bar_width, label="Valid",   linewidth=0.3)
    plt.bar(x + offset, freq_invalid, width=bar_width, label="Invalid", linewidth=0.3)

    plt.xlabel("Motion Magnitude")
    plt.ylabel("Frequency")
    plt.xticks(bins)
    plt.xlim(0, max(bins) * 1.05)

    ymax = max(freq_all.max(), freq_valid.max(), freq_invalid.max(), 0.0)
    if ymax > 1.0:
        plt.yticks(np.arange(0, ymax + 1e-6, step=5))
    else:
        plt.yticks(np.arange(0, ymax + 0.1, step=0.1))

    plt.grid(True, axis="y", linestyle="--", alpha=0.35)
    if title:
        plt.title(title)

    plt.legend()

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    # Load Dataset
    for cfg in iter_dataset_configs(DATASET):
        if cfg.difficulty != "Medium":
            continue

        if cfg.fps != 60:
            continue

        raw_seq_df = pd.read_csv(f"./data/{cfg.record_name}_preprocessed/{cfg.mode_index}_raw_sequence_frame_index.csv")

        bmv_2_0_magnitude_list = []
        bmv_1_0_magnitude_list = []
        distance_mean_list = []

        with tqdm(range(0, len(raw_seq_df), 2)) as pbar:
            for frame_idx in pbar:
                distance_index_mean = raw_seq_df.at[frame_idx, "D_index Mean"]
                seq_valid = raw_seq_df.at[frame_idx, "valid"]

                img_2_idx = frame_idx + 2

                # fps 30
                fps_30_mode_path = cfg.mode_path.replace("fps_60", "fps_30")
                fps_30_dir = os.path.join(ROOT_DIR, cfg.record_name, fps_30_mode_path)
                backwardVel_fps30_2_0_path = f"{fps_30_dir}/backwardVel_Depth_{img_2_idx // 2}.exr"

                # fps 60
                fps_60_dir = os.path.join(ROOT_DIR, cfg.record_name, cfg.mode_path)
                backwardVel_fps60_1_0_path = f"{fps_60_dir}/backwardVel_Depth_{frame_idx + 1}.exr"

                # Load backward
                backwardVel_2_0, _ = load_backward_velocity(backwardVel_fps30_2_0_path)
                backwardVel_1_0, _ = load_backward_velocity(backwardVel_fps60_1_0_path)

                bmv_2_0_magnitude = calculate_magnitude(backwardVel_2_0)
                bmv_1_0_magnitude = calculate_magnitude(backwardVel_1_0)

                bmv_2_0_magnitude_mean = np.mean(bmv_2_0_magnitude)
                bmv_1_0_magnitude_mean = np.mean(bmv_1_0_magnitude)

                raw_seq_df.at[frame_idx, "bmv_2_0_magnitude_mean"] = bmv_2_0_magnitude_mean
                raw_seq_df.at[frame_idx, "bmv_1_0_magnitude_mean"] = bmv_1_0_magnitude_mean

                pbar.set_postfix(
                    {
                        "Motion Magnitude Mean 2-0": f"{bmv_2_0_magnitude_mean:.4f}",
                        "Motion Magnitude Mean 1-0": f"{bmv_1_0_magnitude_mean:.4f}",
                        "D_index Mean": f"{distance_index_mean:.4f}"
                    }
                )

        
        bmv_2_0_magnitude_list = raw_seq_df["bmv_2_0_magnitude_mean"]
        bmv_1_0_magnitude_list = raw_seq_df["bmv_1_0_magnitude_mean"]
        seq_valid = raw_seq_df["valid"]
        bins = np.arange(0, 60 + 1, step=15)

        plot_frequency(
            bins, bmv_2_0_magnitude_list, seq_valid, 0.0,
            title="Motion Magnitude Mean 2-0", 
            save_path=f"./data/{cfg.record_name}_preprocessed/{cfg.mode_index}_mean_motion_magnitude_2_to_0.png"
        )

        plot_frequency(
            bins, bmv_1_0_magnitude_list, seq_valid, 0.0,
            title="Motion Magnitude Mean 1-0", 
            save_path=f"./data/{cfg.record_name}_preprocessed/{cfg.mode_index}_mean_motion_magnitude_1_to_0.png"
        )

        float_bins = np.linspace(0.1, 1.0, num=10)

        plot_frequency(
            float_bins, raw_seq_df["D_index Mean"], seq_valid, 0.05,
            title="Distance Indexing Mean", 
            save_path=f"./data/{cfg.record_name}_preprocessed/{cfg.mode_index}_mean_distance_indexing.png"
        )