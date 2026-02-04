import os
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append('/datasets/VFI/offline_dataset/')
from datasets.dataset_config import DATASET_CONFIGS, STAIR_DATASET_CONFIG, TEST_DATASET_CONFIGS, iter_dataset_configs

# ---- your existing utils ----
def sorted_int(epoch_name):
    epoch_name = epoch_name.replace(".csv", "")
    epoch_name = epoch_name.split("_")
    return int(epoch_name[-1])

def plot_psnr_curve(epochs, values, title, y_label, save_path=None, vlines=None, vline_labels=None):
    if len(epochs) == 0:
        print("[WARN] Empty curve, skip plotting.")
        return

    plt.figure(figsize=(10, 4))
    plt.plot(epochs, values, marker="o", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)

    # draw boundaries between parts
    if vlines is not None:
        for i, x in enumerate(vlines):
            plt.axvline(x=x, linestyle="--", alpha=0.6)
            if vline_labels is not None and i < len(vline_labels):
                plt.text(x + 0.3, plt.ylim()[0], vline_labels[i], rotation=90, va="bottom", alpha=0.8)

    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"[OK] saved: {save_path}")
    else:
        plt.show()

# ---- new: load and stitch multiple exp parts ----
def load_curve_for_one_cfg(exp_parts, cfg, metric_keys, max_local_epochs=30):
    """
    exp_parts: list of (exp_name, offset)
    returns:
        global_epochs, metrics_dict, boundary_xs, boundary_labels
    """
    global_epochs = []
    metrics = {k: [] for k in metric_keys}

    boundary_xs = []       # where a new part begins (global epoch)
    boundary_labels = []   # label for that boundary

    for part_i, (exp_name, offset) in enumerate(exp_parts):
        checkpoints_dir = f"./output/{exp_name}/checkpoints/IFRNet/"
        # epochs_dir = f"{checkpoints_dir}/{cfg.record}/{cfg.mode_path}/"
        epochs_dir = f"{checkpoints_dir}/merged_fps60_Difficult"
        if not os.path.isdir(epochs_dir):
            print(f"[WARN] missing dir: {epochs_dir} (skip {exp_name})")
            continue

        epochs_path_list = [p for p in os.listdir(epochs_dir) if p.endswith(".csv")]
        epochs_path_list = sorted(epochs_path_list, key=lambda x: sorted_int(x))

        # boundary mark: start of this part (except the very first actual loaded point)
        if len(global_epochs) > 0:
            boundary_xs.append(offset)
            boundary_labels.append(exp_name)

        for local_idx, epoch_name in enumerate(epochs_path_list):
            if local_idx >= max_local_epochs:
                break
            path = f"{epochs_dir}/{epoch_name}"
            df = pd.read_csv(path)

            ge = offset + local_idx
            global_epochs.append(ge)
            for k in metric_keys:
                metrics[k].append(df[k].mean())

    return global_epochs, metrics, boundary_xs, boundary_labels


if __name__ == "__main__":
    # 你要接起來的三段
    EXP_NAME = "IFRNet_FineTuning_Val"
    EXP_PARTS = [
        (f"{EXP_NAME}", 0),
        (f"{EXP_NAME}_30", 30),
        (f"{EXP_NAME}_60", 60),
        # (f"{EXP_NAME}_90", 90),
    ]

    ANALYSIS_DIR = f"./analysis_results/0124/{EXP_NAME}/"
    metric_keys = ["psnr", "loss_rec", "loss_geo", "loss_dis"]

    for cfg in iter_dataset_configs(TEST_DATASET_CONFIGS):
        if cfg.fps != 60:
            continue
        # if cfg.difficulty != "Difficult":
        #     continue

        print("[CFG]", cfg.record, cfg.mode_name)

        epochs, metrics, boundary_xs, boundary_labels = load_curve_for_one_cfg(
            EXP_PARTS, cfg, metric_keys, max_local_epochs=30
        )

        # PSNR
        plot_title = f"PSNR (stitched) for {cfg.record} - {cfg.mode_name} (fps={cfg.fps})"
        save_path = f"{ANALYSIS_DIR}/psnr_stitched_{cfg.record}_{cfg.mode_name}_fps{cfg.fps}.png"
        plot_psnr_curve(epochs, metrics["psnr"], plot_title, "PSNR",
                        save_path=save_path, vlines=boundary_xs, vline_labels=boundary_labels)

        # loss_rec
        plot_title = f"Loss Rec (stitched) for {cfg.record} - {cfg.mode_name} (fps={cfg.fps})"
        save_path = f"{ANALYSIS_DIR}/loss_rec_stitched_{cfg.record}_{cfg.mode_name}_fps{cfg.fps}.png"
        plot_psnr_curve(epochs, metrics["loss_rec"], plot_title, "Loss Rec",
                        save_path=save_path, vlines=boundary_xs, vline_labels=boundary_labels)

        # loss_geo
        plot_title = f"Loss Geo (stitched) for {cfg.record} - {cfg.mode_name} (fps={cfg.fps})"
        save_path = f"{ANALYSIS_DIR}/loss_geo_stitched_{cfg.record}_{cfg.mode_name}_fps{cfg.fps}.png"
        plot_psnr_curve(epochs, metrics["loss_geo"], plot_title, "Loss Geo",
                        save_path=save_path, vlines=boundary_xs, vline_labels=boundary_labels)

        # loss_dis
        plot_title = f"Loss Dis (stitched) for {cfg.record} - {cfg.mode_name} (fps={cfg.fps})"
        save_path = f"{ANALYSIS_DIR}/loss_dis_stitched_{cfg.record}_{cfg.mode_name}_fps{cfg.fps}.png"
        plot_psnr_curve(epochs, metrics["loss_dis"], plot_title, "Loss Dis",
                        save_path=save_path, vlines=boundary_xs, vline_labels=boundary_labels)
