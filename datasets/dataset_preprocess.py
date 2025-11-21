import pandas as pd
from glob import glob
import os

from dataset_config import DATASET_CONFIGS, MINOR_DATASET_CONFIGS, iter_dataset_configs
from remove_identical import identical_images
from manual_labeling import review_images
from utils import loadPNG
from tqdm import tqdm
import numpy as np

ROOT_DIR = DATASET_CONFIGS["root_dir"]

def build_frame_index_for_mode(record, mode):
    rows = []
    mode_root = os.path.join(ROOT_DIR, record, mode)

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

    df = pd.DataFrame(rows)
    df["reason"] = df["reason"].astype(str)

    return df

def remove_identical_images(raw_df: pd.DataFrame) -> pd.DataFrame:
    with tqdm(range(1, len(raw_df))) as pbar:
        for i in pbar:
            current_row = raw_df.iloc[i]
            previous_row = raw_df.iloc[i - 1]

            # Construct image paths
            current_img_name = f"colorNoScreenUI_{current_row['frame_idx']}.png"
            previous_img_name = f"colorNoScreenUI_{previous_row['frame_idx']}.png"

            dir_path = os.path.join(ROOT_DIR, current_row["record"], current_row["mode"])

            img1_path = os.path.join(dir_path, previous_img_name)
            img2_path = os.path.join(dir_path, current_img_name)

            img1 = loadPNG(img1_path).astype(np.uint8)
            img2 = loadPNG(img2_path).astype(np.uint8)

            # Check if images are identical
            if identical_images(img1, img2):
                raw_df.at[i, "is_valid"] = False
                raw_df.at[i, "reason"] = "Identical to previous frame"

            pbar.set_postfix({"Current Frame": current_row["frame_idx"], "Valid": raw_df.at[i, "is_valid"]})

    return raw_df

if __name__ == "__main__":
    REMOVE_IDENTICAL = False
    MANUAL_LABELING = True

    if REMOVE_IDENTICAL:
        for cfg in iter_dataset_configs(MINOR_DATASET_CONFIGS):
            # initialize dataframe to store frame indices for each mode
            print(f"Processing record: {cfg.record_name}, mode: {cfg.mode_name}")
            raw_df = build_frame_index_for_mode(cfg.record, cfg.mode_path)
            raw_df = raw_df.sort_values(by="frame_idx").reset_index(drop=True)
            print(raw_df.head())

            # check identical images and mark them as invalid
            raw_df = remove_identical_images(raw_df)

            # save to csv
            os.makedirs(f"./data/{cfg.record_name}/", exist_ok=True)
            raw_df.to_csv(f"./data/{cfg.record_name}/{cfg.mode_name}_frame_index.csv", index=False)

    if MANUAL_LABELING:
        for cfg in iter_dataset_configs(MINOR_DATASET_CONFIGS):
            # only label in Medium difficulty
            if cfg.difficulty != "Medium":
                continue

            easy_mode_name = cfg.mode_name.replace("Medium", "Easy")
            easy_df = pd.read_csv(f"./data/{cfg.record_name}/{easy_mode_name}_frame_index.csv", dtype={"reason": "string"})
            medium_df = pd.read_csv(f"./data/{cfg.record_name}/{cfg.mode_name}_frame_index.csv", dtype={"reason": "string"})

            print(f"Manual labeling for record: {cfg.record_name}, mode: {cfg.mode_name}")
            review_images(easy_df, medium_df)

            # save to csv
            easy_df.to_csv(f"./data/{cfg.record_name}/{easy_mode_name}_frame_index.csv", index=False)
            medium_df.to_csv(f"./data/{cfg.record_name}/{cfg.mode_name}_frame_index.csv", index=False)