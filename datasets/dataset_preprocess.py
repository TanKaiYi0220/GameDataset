import pandas as pd
from glob import glob
import os

from dataset_config import DATASET_CONFIGS, MINOR_DATASET_CONFIGS, STAIR_DATASET_CONFIG, iter_dataset_configs
from remove_identical import identical_images, visualize_color_difference
from manual_labeling import review_images
from clipping import get_valid_continuous_segments, check_valid_in_high_fps
from utils import loadPNG
from tqdm import tqdm
import numpy as np

DATA_CONFIG = STAIR_DATASET_CONFIG
ROOT_DIR = DATA_CONFIG["root_dir"]

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
    invalid_count = 0
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
                invalid_count += 1

            pbar.set_postfix({"Current Frame": current_row["frame_idx"], "Invalid Count": invalid_count})

    return raw_df

def check_identical_images_cross_fps(fps_30_df: pd.DataFrame, fps_60_df: pd.DataFrame):
    with tqdm(range(1, len(fps_30_df))) as pbar:
        for i in pbar:
            current_row_30 = fps_30_df.iloc[i]
            current_row_60 = fps_60_df.iloc[2 * i]

            # Construct image paths
            img_name_30 = f"colorNoScreenUI_{current_row_30['frame_idx']}.png"
            img_name_60 = f"colorNoScreenUI_{current_row_60['frame_idx']}.png"

            dir_path_30 = os.path.join(ROOT_DIR, current_row_30["record"], current_row_30["mode"])
            dir_path_60 = os.path.join(ROOT_DIR, current_row_60["record"], current_row_60["mode"])

            img_30_path = os.path.join(dir_path_30, img_name_30)
            img_60_path = os.path.join(dir_path_60, img_name_60)

            img_30 = loadPNG(img_30_path).astype(np.uint8)
            img_60 = loadPNG(img_60_path).astype(np.uint8)

            # Check if images are identical
            if not identical_images(img_30, img_60):
                print(f"Non-identical frames found at fps 30 frame {current_row_30['frame_idx']} and fps 60 frame {current_row_60['frame_idx']}")
                visualize_color_difference(img_30, img_60)
            pbar.set_postfix({"Frame 30": current_row_30["frame_idx"], "Frame 60": current_row_60["frame_idx"]})

def get_valid_clip(df: pd.DataFrame, target_frames_count: int) -> pd.DataFrame:
    # get valid contiguous clip with target number of frames
    df = df.sort_values(by="frame_idx").reset_index(drop=True)
    idx_array = df["frame_idx"].to_numpy()

    segments = []

    start = 0
    for i in range(1, len(idx_array) + 1):
        if i == len(idx_array) or idx_array[i] != idx_array[i-1] + 1:
            length = i - start
            if length >= target_frames_count:
                # sliding window（stride=1，可依需求改）
                for j in range(start, i - target_frames_count + 1):
                    s30 = int(idx_array[j])
                    e30 = s30 + target_frames_count - 1
                    segments.append({
                        "start": s30,
                        "end": e30,
                    })
            start = i

    return segments

if __name__ == "__main__":
    REMOVE_IDENTICAL = False                # initial raw frame index generation with identical images removed
    CHECK_IDENTICAL_CROSS_FPS = False       # check identical images between fps 30 and fps 60
    MANUAL_LABELING = False                 # manual labeling based on Medium difficulty and Easy difficulty
    MERGE_DATASETS = False                  # merge Easy and Medium difficulties into one dataframe with global validity
    DATA_CLIPPING = False                   # clip data to only valid continuous segments across fps 30 and fps 60
    RAW_SEQUENCE = True                    # generate sequence from 0 to MAX INDEX with valid flag


    if REMOVE_IDENTICAL:
        for cfg in iter_dataset_configs(DATA_CONFIG):
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

    if CHECK_IDENTICAL_CROSS_FPS:
        for ctg in iter_dataset_configs(DATA_CONFIG):
            # only check in fps 30
            if ctg.fps != 30:
                continue

            fps_60_name = ctg.mode_name.replace("fps_30", "fps_60")
            fps_30_df = pd.read_csv(f"./data/{ctg.record_name}/{ctg.mode_name}_frame_index.csv", dtype={"reason": "string"})
            fps_60_df = pd.read_csv(f"./data/{ctg.record_name}/{fps_60_name}_frame_index.csv", dtype={"reason": "string"})

            print(f"Checking identical frames between fps 30 and fps 60 for record: {ctg.record_name}, mode: {ctg.mode_name}")
            check_identical_images_cross_fps(fps_30_df, fps_60_df)
            

    if MANUAL_LABELING:
        for cfg in iter_dataset_configs(DATA_CONFIG):
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

    if MERGE_DATASETS:
        for cfg in iter_dataset_configs(DATA_CONFIG):
            # merge Easy and Medium difficulties
            if cfg.difficulty != "Medium":
                continue

            easy_mode_name = cfg.mode_name.replace("Medium", "Easy")
            easy_df = pd.read_csv(f"./data/{cfg.record_name}/{easy_mode_name}_frame_index.csv", dtype={"reason": "string"})
            medium_df = pd.read_csv(f"./data/{cfg.record_name}/{cfg.mode_name}_frame_index.csv", dtype={"reason": "string"})

            merged_df = easy_df.merge(
                medium_df,
                on=["record", "frame_idx"],
                how="inner",
                suffixes=("_easy", "_medium")
            )

            merged_df["global_is_valid"] = merged_df["is_valid_easy"] & merged_df["is_valid_medium"]

            os.makedirs(f"./data/{cfg.record_name}_preprocessed/", exist_ok=True)
            merged_df.to_csv(f"./data/{cfg.record_name}_preprocessed/{cfg.mode_index}_merged_frame_index.csv", index=False)

    if DATA_CLIPPING:
        for cfg in iter_dataset_configs(DATA_CONFIG):
            # only clip Medium difficulty since Easy & Medium have been merged
            if cfg.difficulty != "Medium":
                continue

            if cfg.fps != 60:
                continue

            print(cfg.mode_index)

            fps_30_name = cfg.mode_name.replace("fps_60", "fps_30")
            df_30 = pd.read_csv(f"./data/{cfg.record_name}_preprocessed/{cfg.mode_index.replace('fps_60', 'fps_30')}_merged_frame_index.csv", dtype={"reason_easy": "string", "reason_medium": "string"})
            df_60 = pd.read_csv(f"./data/{cfg.record_name}_preprocessed/{cfg.mode_index}_merged_frame_index.csv", dtype={"reason_easy": "string", "reason_medium": "string"})

            # Clip data to only valid frames
            segments_low = get_valid_continuous_segments(df_30, target_frames_count=60)

            clipped_df = pd.DataFrame()

            for start_low, end_low in segments_low:
                # check if all frames in this segment are valid in fps 60
                is_valid_in_60 = check_valid_in_high_fps(df_60, (start_low, end_low))

                if is_valid_in_60:
                    print(f"Valid clip from frame {start_low} to {end_low} in both fps 30 and fps 60")
                    for frame_idx in range(start_low * 2, end_low * 2 + 2, 1):
                        # each row: [frame_idx, frame_idx + 1, frame_idx + 2]
                        row = pd.DataFrame({
                            "record": [cfg.record],
                            "fps": [cfg.fps],
                            "img0": [frame_idx],
                            "img1": [frame_idx + 1],
                            "img2": [frame_idx + 2],
                            "valid": True
                        })
                        clipped_df = pd.concat([clipped_df, row], ignore_index=True)
                else:
                    print(f"Clip from frame {start_low} to {end_low} is NOT valid in fps 60")

            if clipped_df.empty:
                print(f"No valid clips found for record: {cfg.record_name}, mode: {cfg.mode_name}")
                continue

            print(f"Total valid clips found: {len(clipped_df)}")

            clipped_df.to_csv(f"./data/{cfg.record_name}_preprocessed/{cfg.mode_index}_clipped_frame_index.csv", index=False)

    if RAW_SEQUENCE:
        for cfg in iter_dataset_configs(DATA_CONFIG):
            # only clip Medium difficulty since Easy & Medium have been merged
            if cfg.difficulty != "Medium":
                continue

            if cfg.fps != 60:
                continue

            print(cfg.mode_index)

            fps_30_name = cfg.mode_name.replace("fps_60", "fps_30")
            df_30 = pd.read_csv(f"./data/{cfg.record_name}_preprocessed/{cfg.mode_index.replace('fps_60', 'fps_30')}_merged_frame_index.csv", dtype={"reason_easy": "string", "reason_medium": "string"})
            df_60 = pd.read_csv(f"./data/{cfg.record_name}_preprocessed/{cfg.mode_index}_merged_frame_index.csv", dtype={"reason_easy": "string", "reason_medium": "string"})

            # Generate full sequence with valid flag
            raw_seq_df = pd.DataFrame()

            for frame_idx in range(0, cfg.max_index - 1, 1):
                valid_flag = True
                if frame_idx % 2 == 0: # even
                    #       HERE
                    # 1 ---- 2 ---- 3 ---- 4
                    # True   False  True   True
                    # 2 ---- 2 ---- 3 ---- 4 ==> Still Valid
                    # True   True   False  True
                    # 1 ---- 2 ---- 2 ---- 4 ==> Invalid
                    # True   True   True   False
                    # 1 ---- 2 ---- 4 ---- 4 ==> Invalid
                    # Thus, only need to check about [frame_idx + 1] and [frame_idx + 2] 
                    fps_30_img_2_flag = df_30.at[frame_idx // 2 + 1, "global_is_valid"]

                    fps_60_img_1_flag = df_60.at[frame_idx + 1, "global_is_valid"]
                    fps_60_img_2_flag = df_60.at[frame_idx + 2, "global_is_valid"]

                    valid_flag = fps_30_img_2_flag and fps_60_img_1_flag and fps_60_img_2_flag
                else: # odds
                    #       HERE
                    # 0 ---- 1 ---- 2 ---- 3
                    # True   False  True   True
                    # 0 ---- 0 ---- 2 ---- 3 ==> Invalid
                    # True   True   False  True
                    # 0 ---- 2 ---- 2 ---- 3 ==> Invalid
                    # True   True   True   False
                    # 0 ---- 1 ---- 2 ---- 2 ==> Invalid
                    # Thus, only need to check about [frame_idx + 1] and [frame_idx + 2] 
                    fps_30_img_2_flag = df_30.at[frame_idx // 2 + 1, "global_is_valid"]

                    fps_60_img_0_flag = df_60.at[frame_idx, "global_is_valid"]
                    fps_60_img_1_flag = df_60.at[frame_idx + 1, "global_is_valid"]
                    fps_60_img_2_flag = df_60.at[frame_idx + 2, "global_is_valid"]

                    valid_flag = fps_30_img_2_flag and fps_60_img_0_flag and fps_60_img_1_flag and fps_60_img_2_flag

                row = pd.DataFrame({
                    "record": [cfg.record],
                    "fps": [cfg.fps],
                    "img0": [frame_idx],
                    "img1": [frame_idx + 1],
                    "img2": [frame_idx + 2],
                    "valid": [valid_flag]
                })

                raw_seq_df = pd.concat([raw_seq_df, row], ignore_index=True)

            raw_seq_df.to_csv(f"./data/{cfg.record_name}_preprocessed/{cfg.mode_index}_raw_sequence_frame_index.csv", index=False)
            print(f"./data/{cfg.record_name}_preprocessed/{cfg.mode_index}_raw_sequence_frame_index.csv")