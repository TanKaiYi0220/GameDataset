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

def plot_metric_curve_3splits(
    epochs_by_split,
    values_by_split,
    title,
    y_label,
    save_path=None,
    vlines=None,
    vline_labels=None,
):
    """
    epochs_by_split: dict {split: [global_epoch,...]}
    values_by_split: dict {split: [value,...]}
    """
    plt.figure(figsize=(10, 4))

    # 固定順序（想換順序自己改）
    # split_order = ["train", "val", "test"]
    split_order = ["train", "test"]
    for sp in split_order:
        if sp not in epochs_by_split or len(epochs_by_split[sp]) == 0:
            continue
        plt.plot(
            epochs_by_split[sp],
            values_by_split[sp],
            marker="o",
            linewidth=2,
            label=sp,
        )

    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()

    if vlines is not None:
        for i, x in enumerate(vlines):
            plt.axvline(x=x, linestyle="--", alpha=0.6)
            if vline_labels is not None and i < len(vline_labels):
                plt.text(x + 0.3, plt.ylim()[0], vline_labels[i],
                         rotation=90, va="bottom", alpha=0.8)

    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"[OK] saved: {save_path}")
    else:
        plt.show()


def parse_split_and_local_epoch(filename: str):
    # e.g. train_epoch_12.csv
    name = filename.replace(".csv", "")
    parts = name.split("_")
    split = parts[0]                   # train / val / test
    local_epoch = int(parts[-1])       # 12
    return split, local_epoch


def load_curves_for_one_cfg_all_splits(exp_parts, metric_keys, max_local_epochs=30):
    """
    returns:
        epochs_by_split: dict {split: [global_epoch,...]}
        metrics_by_split: dict {split: {metric: [values...]}}
        boundary_xs, boundary_labels
    """
    splits = ["train", "val", "test"]
    epochs_by_split = {sp: [] for sp in splits}
    metrics_by_split = {sp: {k: [] for k in metric_keys} for sp in splits}

    boundary_xs = []
    boundary_labels = []

    loaded_any_part = False

    for part_i, (exp_name, offset) in enumerate(exp_parts):
        checkpoints_dir = f"./output/{exp_name}/checkpoints/"
        epochs_dir = f"{checkpoints_dir}/"

        if not os.path.isdir(epochs_dir):
            print(f"[WARN] missing dir: {epochs_dir} (skip {exp_name})")
            continue

        # boundary mark: start of this part (only after we have loaded something before)
        if loaded_any_part:
            boundary_xs.append(offset)
            boundary_labels.append(exp_name)

        files = [p for p in os.listdir(epochs_dir) if p.endswith(".csv")]

        # 分 split 收集
        files_by_split = {sp: [] for sp in splits}
        for fn in files:
            try:
                sp, le = parse_split_and_local_epoch(fn)
            except Exception:
                continue
            if sp in files_by_split:
                files_by_split[sp].append((le, fn))

        # 每個 split 都按照 local epoch 排序，最多取 max_local_epochs
        for sp in splits:
            files_by_split[sp].sort(key=lambda x: x[0])
            for local_epoch, fn in files_by_split[sp][:max_local_epochs]:
                path = f"{epochs_dir}/{fn}"
                df = pd.read_csv(path)

                ge = offset + local_epoch
                epochs_by_split[sp].append(ge)
                for k in metric_keys:
                    metrics_by_split[sp][k].append(df[k].mean())

                loaded_any_part = True

    return epochs_by_split, metrics_by_split, boundary_xs, boundary_labels



if __name__ == "__main__":
    EXP_NAME = "IFRNet_FineTuning_Resume_0416"
    EXP_PARTS = [
        (f"{EXP_NAME}", 0),
        (f"{EXP_NAME}_30", 0),
        # (f"{EXP_NAME}_60", 0),
    ]

    ANALYSIS_DIR = f"./analysis_results/0423_FineTuning/{EXP_NAME}/"
    metric_keys = ["psnr", "loss_total"]
    # metric_keys = ["psnr"]

    for cfg in iter_dataset_configs(TEST_DATASET_CONFIGS):
        if cfg.fps != 60:
            continue

        if cfg.difficulty != "Difficult":
            continue

        print("[CFG]", cfg.record, cfg.mode_name)

        epochs_by_split, metrics_by_split, boundary_xs, boundary_labels = \
            load_curves_for_one_cfg_all_splits(EXP_PARTS, metric_keys, max_local_epochs=30)

        # psnr
        plot_title = f"PSNR (fps={cfg.fps})"
        save_path = f"{ANALYSIS_DIR}/psnr_{EXP_NAME}_{cfg.fps}.png"
        plot_metric_curve_3splits(
            epochs_by_split,
            # {sp: metrics_by_split[sp]["psnr"] for sp in ["train","val","test"]},
            {sp: metrics_by_split[sp]["psnr"] for sp in ["train","test"]},
            plot_title,
            "PSNR",
            save_path=save_path,
            vlines=boundary_xs,
            vline_labels=boundary_labels,
        )

        # loss_total
        plot_title = f"Loss Total (fps={cfg.fps})"
        save_path = f"{ANALYSIS_DIR}/loss_total_{EXP_NAME}_{cfg.fps}.png"
        plot_metric_curve_3splits(
            epochs_by_split,
            # {sp: metrics_by_split[sp]["loss_total"] for sp in ["train","val","test"]},
            {sp: metrics_by_split[sp]["loss_total"] for sp in ["train","test"]},
            plot_title,
            "Loss Rec",
            save_path=save_path,
            vlines=boundary_xs,
            vline_labels=boundary_labels,
        )

        # # loss_geo
        # plot_title = f"Loss Geo (stitched) (fps={cfg.fps})"
        # save_path = f"{ANALYSIS_DIR}/loss_geo_stitched_{cfg.record}_{cfg.mode_name}_fps{cfg.fps}.png"
        # plot_metric_curve_3splits(
        #     epochs_by_split,
        #     {sp: metrics_by_split[sp]["loss_geo"] for sp in ["train","val","test"]},
        #     plot_title,
        #     "Loss Geo",
        #     save_path=save_path,
        #     vlines=boundary_xs,
        #     vline_labels=boundary_labels,
        # )

        # # loss_dis
        # plot_title = f"Loss Dis (stitched) (fps={cfg.fps})"
        # save_path = f"{ANALYSIS_DIR}/loss_dis_stitched_{cfg.record}_{cfg.mode_name}_fps{cfg.fps}.png"
        # plot_metric_curve_3splits(
        #     epochs_by_split,
        #     {sp: metrics_by_split[sp]["loss_dis"] for sp in ["train","val","test"]},
        #     plot_title,
        #     "Loss Dis",
        #     save_path=save_path,
        #     vlines=boundary_xs,
        #     vline_labels=boundary_labels,
        # )

