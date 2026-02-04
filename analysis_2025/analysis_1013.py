import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Sequence, Tuple, Dict

import torch
from tqdm import tqdm

import os
import sys
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import find_max_index_in_dir
from src.gameData_loader import load_backward_velocity


def mode_rename(mode):
    difficult, _, fps, _ = mode.split("/")
    return f"{difficult}_{fps}"

# 1) 把原始 magnitudes 分箱 -> 頻率與累積
#    bins 為每個區間的右邊界，例如 [30, 60, 90, 120, 150]
#    分箱範圍: (0,30], (30,60], ...
# ------------------------------------------------------------
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
    bin_centers: np.ndarray,
    freq: np.ndarray,
    bins: Sequence[float],
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    plt.figure(figsize=(6,4))
    width = 1.0
    plt.bar(bin_centers, freq, width=width, linewidth=0.3)
    plt.xlabel("Motion Magnitude")
    plt.ylabel("Frequency")
    plt.xticks(bins)
    plt.xlim(0, max(bins) * 1.05)
    if max(freq) > 1.0:
        plt.yticks(np.arange(0, max(freq), step=5))
    else:
        plt.yticks(np.arange(0, max(freq)+0.1, step=0.1))
    plt.grid(True, axis="y", linestyle="--", alpha=0.35)
    if title: plt.title(title)
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
    plt.close()

if __name__ == "__main__":
    record_name = "AnimeFantasyRPG_2_60"
    fps = "fps_60"
    dataset_root_path = f"/datasets/VFI/datasets/AnimeFantasyRPG/{record_name}/"
    dataset_mode_path = [
        ## AnimeFantasyRPG_3_60
        f"0_Easy/0_Easy_0/{fps}/", 
        f"0_Medium/0_Medium_0/{fps}/", 

        f"4_Easy/4_Easy_0/{fps}/", 
        f"4_Medium/4_Medium_0/{fps}/",

        ## AnimeFantasyRPG_2_60
        # f"0_Easy/0_Easy_1/{fps}/", 
        # f"0_Medium/0_Medium_1/{fps}/",

        # f"4_Easy/4_Easy_1/{fps}/", 
        # f"4_Medium/4_Medium_1/{fps}/",
    ]

    analysis_path = f"./analysis_results/1013/{record_name}/"
    os.makedirs(analysis_path, exist_ok=True)

    for mode in dataset_mode_path:
        dataset_dir_path = f"{dataset_root_path}/{mode}/"
        max_index = find_max_index_in_dir(dataset_dir_path)

        motion_magnitude_list = []


        with tqdm(range(20, max_index)) as pbar:
            for i in pbar:
                pbar.set_description(f"Processing {mode} - Frame {i}")

                bmv_path = f"{dataset_root_path}/{mode}/backwardVel_Depth_{i + 1}.exr"

                bmv, depth = load_backward_velocity(bmv_path)

                _, C, H, W = bmv.shape
                magnitude = bmv[:, 0, :, :] ** 2 + bmv[:, 1, :, :] ** 2
                magnitude = torch.sqrt(magnitude)

                motion_magnitude_mean = magnitude.mean().cpu().numpy().item()

                pbar.set_postfix({"Motion Magnitude Mean": f"{motion_magnitude_mean:.4f}"})

                motion_magnitude_list.append(motion_magnitude_mean)

        # visualize motion magnitude distribution
        motion_magnitude_list = np.array(motion_magnitude_list, dtype=float)
        bins = np.arange(0, 60 + 1, step=15)
        upper_edges, freq = bin_stats_from_raw(motion_magnitude_list, bins=bins)

        save_name = mode_rename(mode)
        print(W, H, freq)
        plot_frequency(upper_edges, freq, bins, title="Mean Motion Magnitude", save_path=f"{analysis_path}/{save_name}_mean_motion_magnitude.png")
        
        plot_frequency(
            [idx + 20 for idx in range(len(motion_magnitude_list))], 
            motion_magnitude_list, 
            bins=[idx for idx in range(20, len(motion_magnitude_list) + 20, 20)], 
            title="Mean Motion Magnitude per Frame", 
            save_path=f"{analysis_path}/{save_name}_mean_motion_magnitude_per_frame.png"
        )
