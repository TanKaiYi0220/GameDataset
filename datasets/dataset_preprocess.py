import pandas as pd
from glob import glob
import os

from dataset_config import DATASET_CONFIGS, iter_dataset_configs

def build_frame_index_for_mode(root_dir, record, mode):
    rows = []
    mode_root = os.path.join(root_dir, record, mode)

    # 假設你以 colorScreenNoUI 為基準判斷幀數
    frame_files = sorted(glob(os.path.join(mode_root, "colorNoScreenUI_*.exr")))
    for f in frame_files:
        # 從檔名 parse 出 frame_idx
        # e.g. colorScreenNoUI_000123.png

        name = os.path.basename(f)
        idx_str = name.split("_")[-1].split(".")[0]
        frame_idx = int(idx_str)

        rows.append({
            "record": record,
            "mode": mode,
            "frame_idx": frame_idx,
            "is_valid": True,
            "reason": "",
        })

    return pd.DataFrame(rows)

def remove_identical_images():
    # TODO: implement this function to remove identical images in the dataset
    pass

def labeling_invalid_frames():
    # TODO: label invalid frames by manual
    pass


if __name__ == "__main__":
    ROOT_DIR = DATASET_CONFIGS["root_dir"]

    for cfg in iter_dataset_configs(DATASET_CONFIGS):
        raw_df = build_frame_index_for_mode(ROOT_DIR, cfg.record, cfg.mode_path)
        raw_df = raw_df.sort_values(by="frame_idx").reset_index(drop=True)
        print(raw_df.head())

        os.makedirs(f"./data/{cfg.record_name}/", exist_ok=True)

        raw_df.to_csv(f"./data/{cfg.record_name}/{cfg.mode_name}_frame_index.csv", index=False)
