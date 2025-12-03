import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import os
import sys
sys.path.append('/datasets/VFI/offline_dataset/')
from datasets.dataset_config import DATASET_CONFIGS, STAIR_DATASET_CONFIG, iter_dataset_configs

OUTPUT_DIR = "./output/IFRNet/"
ANALYSIS_DIR = f"./analysis_results/1117_IFRNet/"
LABEL_LOC = {
    "psnr": "lower right",
    "epe_1_to_0": "upper right",
    "epe_1_to_2": "upper right"
}

# 存每個 exp_name 底下的各個 mode_name 的 dataframe
# 結構會是：{ exp_name: { mode_name: df, ... }, ... }
mode_to_df = {}

priority = {"Easy": 0, "Medium": 1, "Difficult": 2}

def get_mode_priority(name):
    # 根據你的 mode_name 格式提取難度字串
    # e.g. "0_Easy_1_fps_60" → Easy
    for key in priority.keys():
        if key in name:
            return priority[key]
    return 999  # 萬一沒找到就排最後


def plot_metric(exp_name, mode_dict, metric_name, title_prefix=None):
    if title_prefix is None:
        title_prefix = metric_name

    mode_names = sorted(mode_dict.keys(), key=get_mode_priority)
    cmap = plt.get_cmap("tab10")
    colors = {m: cmap(i % cmap.N) for i, m in enumerate(mode_names)}

    fig, ax = plt.subplots(figsize=(16, 6))

    for mode_name in mode_names:
        df = mode_dict[mode_name]
        x = df["frame_index"].values
        y = df[metric_name].values
        color = colors[mode_name]

        # ---- 新增：依 valid 分組 ----
        mask_valid = df["valid"].astype(bool).values
        x_valid, y_valid = x[mask_valid], y[mask_valid]
        x_invalid, y_invalid = x[~mask_valid], y[~mask_valid]

        # 統計線用全部資料
        mean_val = y.mean()
        median_val = np.median(y)
        std_val = y.std()

        # invalid 點（小、淡）
        ax.scatter(
            x_invalid, y_invalid,
            s=24,               # 比較小
            color='red',
            alpha=0.2,
            label=None         # 不進 legend
        )

        # valid 點（大、顯眼、有邊框）
        ax.scatter(
            x_valid, y_valid,
            s=12,              # 比較大
            color=color,
            edgecolors='black',
            linewidths=0.6,
            alpha=0.9,
            label=f"{mode_name}={mean_val:.2f}/{median_val:.2f}"
        )

        # mean 線 (solid)
        ax.axhline(mean_val, color=color, linestyle='-', linewidth=2.2, alpha=0.9)

        # median 線 (dashed)
        ax.axhline(median_val, color=color, linestyle='--', linewidth=2.2, alpha=0.9)

        # mean ± std 陰影
        ax.fill_between(
            x,
            mean_val - std_val,
            mean_val + std_val,
            color=color,
            alpha=0.10,
            linewidth=0
        )

    ax.set_title(f"{title_prefix} over Frames (exp_name={exp_name}) (total valid frames={len(x_valid)})")
    ax.set_xlabel("frame_index")
    ax.set_ylabel(metric_name)
    ax.grid(True)

    # -------- Legend 區分模式與線型 --------

    # A. 線型 / 點型 legend（左上）
    style_handles = [
        Line2D([0], [0], marker='o', linestyle='None', color='black', alpha=0.2, label='per-frame (invalid)'),
        Line2D([0], [0], marker='o', linestyle='None', color='black', markersize=8, label='per-frame (valid)'),
        Line2D([0], [0], color='black', linestyle='-', label='mean'),
        Line2D([0], [0], color='black', linestyle='--', label='median'),
        Patch(facecolor='gray', alpha=0.2, label='mean ± std'),
    ]
    style_legend = ax.legend(handles=style_handles, loc="upper left", title="Style")
    ax.add_artist(style_legend)

    # B. mode name legend（右下）
    ax.legend(title="mode_name", loc=LABEL_LOC[metric_name])

    os.makedirs(f"{ANALYSIS_DIR}/{exp_name}/", exist_ok=True)
    plt.savefig(f"{ANALYSIS_DIR}/{exp_name}/{metric_name}.png")
    # plt.show()




if __name__ == "__main__":
    for cfg in iter_dataset_configs(DATASET_CONFIGS):
        if cfg.fps != 60:
            continue
        
        result_df_path = f"{OUTPUT_DIR}/{cfg.record}/{cfg.mode_name}_evaluation_results.csv"
        print(result_df_path)
        df = pd.read_csv(result_df_path)

        print()
        print(result_df_path)
        print(df.describe())

        valid_df = df[df["valid"]==True]
        print()
        print("Only Valid")
        print(valid_df.describe())

        # 如果沒有 frame_index，用 index 代替
        if "frame_index" not in df.columns:
            df["frame_index"] = df.index

        # 初始化該 mode_index 的 dict
        exp_name = f"{cfg.record_name}/{cfg.mode_index}"
        if exp_name not in mode_to_df:
            mode_to_df[exp_name] = {}

        # 以 mode_name 存進該 mode_index 群組
        mode_to_df[exp_name][cfg.mode_name] = df

    # -------------------------------
    # 依照 mode_index 分組畫圖
    # 每個 mode_index 會有三張圖（psnr, epe_1_to_0, epe_1_to_2）
    # 圖中多條線：同一個 mode_index 下的不同 mode_name
    # -------------------------------
    for exp_name, mode_dict in mode_to_df.items():
        plot_metric(exp_name, mode_dict, "psnr",       title_prefix="PSNR")
        plot_metric(exp_name, mode_dict, "epe_1_to_0", title_prefix="EPE 1→0")
        plot_metric(exp_name, mode_dict, "epe_1_to_2", title_prefix="EPE 1→2")
