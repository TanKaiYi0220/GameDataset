import math
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
CHECKPOINTS_DIR = "./output/IFRNet/checkpoints/IFRNet/"
ANALYSIS_DIR = f"./analysis_results/1222_IFRNet/"

EXP_NAME = "IFRNet_Residual_WarpBiDir_60"
OUTPUT_DIR = f"./output/{EXP_NAME}/"
CHECKPOINTS_DIR = f"./output/{EXP_NAME}/checkpoints/IFRNet/"
ANALYSIS_DIR = f"./analysis_results/1222_{EXP_NAME}/"

def sorted_int(epoch_name):
    epoch_name = epoch_name.replace(".csv", "")
    epoch_name = epoch_name.split("_")
    return int(epoch_name[-1])

def plot_psnr_curve(epochs, psnr_means, title, y_label, save_path=None):
    """Plot PSNR curve. x=epoch, y=mean psnr."""
    if len(epochs) == 0:
        print("[WARN] Empty curve, skip plotting.")
        return

    plt.figure(figsize=(10, 4))
    plt.plot(epochs, psnr_means, marker="o", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"[OK] saved: {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    for cfg in iter_dataset_configs(STAIR_DATASET_CONFIG):
        if cfg.fps != 60:
            continue

        if cfg.difficulty != "Difficult":
            continue
        
        print(cfg.record, cfg.mode_name)
        epochs_dir = f"{CHECKPOINTS_DIR}/{cfg.record}/{cfg.mode_path}/"
        epochs_path_list = os.listdir(epochs_dir)
        epochs_path_list = [p for p in epochs_path_list if p.endswith(".csv")]
        epochs_path_list = sorted(epochs_path_list, key=lambda x: sorted_int(x))

        epochs = []
        psnr_means = []
        loss_rec_means = []
        loss_geo_means = []
        loss_dis_means = []

        for idx, epoch_name in enumerate(epochs_path_list):
            path = f"{CHECKPOINTS_DIR}/{cfg.record}/{cfg.mode_path}/{epoch_name}"
            print(path)

            df = pd.read_csv(path)

            epochs.append(idx)
            psnr_means.append(df["psnr"].mean())
            loss_rec_means.append(df["loss_rec"].mean())
            loss_geo_means.append(df["loss_geo"].mean())
            loss_dis_means.append(df["loss_dis"].mean())
            # loss_rec_means.append(math.log(df["loss_rec"].mean()))
            # loss_geo_means.append(math.log(df["loss_geo"].mean()))
            # loss_dis_means.append(math.log(df["loss_dis"].mean()))

            if idx >= 30:
                break


        plot_title = f"PSNR Curve for {cfg.record} - {cfg.mode_name} (fps={cfg.fps})"
        save_path = f"{ANALYSIS_DIR}/psnr_curve_{cfg.record}_{cfg.mode_name}_fps{cfg.fps}.png"
        plot_psnr_curve(epochs, psnr_means, plot_title, "PSNR", save_path=save_path)

        plot_title = f"Loss Rec Curve for {cfg.record} - {cfg.mode_name} (fps={cfg.fps})"
        save_path = f"{ANALYSIS_DIR}/loss_rec_curve_{cfg.record}_{cfg.mode_name}_fps{cfg.fps}.png"
        plot_psnr_curve(epochs, loss_rec_means, plot_title, "Loss Rec", save_path=save_path)
        
        plot_title = f"Loss Geo Curve for {cfg.record} - {cfg.mode_name} (fps={cfg.fps})"
        save_path = f"{ANALYSIS_DIR}/loss_geo_curve_{cfg.record}_{cfg.mode_name}_fps{cfg.fps}.png"
        plot_psnr_curve(epochs, loss_geo_means, plot_title, "Loss Geo", save_path=save_path)

        plot_title = f"Loss Dis Curve for {cfg.record} - {cfg.mode_name} (fps={cfg.fps})"
        save_path = f"{ANALYSIS_DIR}/loss_dis_curve_{cfg.record}_{cfg.mode_name}_fps{cfg.fps}.png"
        plot_psnr_curve(epochs, loss_dis_means, plot_title, "Loss Dis", save_path=save_path)