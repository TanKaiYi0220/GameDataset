import os
import shutil
import cv2
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
import json

def get_all_modes(main_index_list, difficulty_list, sub_index, fps):
    modes = []
    for main_index in main_index_list:
        for difficulty in difficulty_list:
            mode = f"{main_index}_{difficulty}/{main_index}_{difficulty}_{sub_index}/{fps}"
            modes.append(mode)
    return modes

def identical_images(img1_path: str, img2_path: str) -> bool:
    img1 = cv2.imread(img1_path).astype(np.uint8)
    img2 = cv2.imread(img2_path).astype(np.uint8)

    # visualize different between two images
    # color_diff = cv2.absdiff(img1, img2)
    # cv2.imshow("Color Difference", color_diff)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    difference = psnr(img1, img2, data_range=255)

    if np.isfinite(difference) == False:
        print("Identical images (PSNR=inf). Setting PSNR to 0.")
        return True

    if difference > 48: # maximum PSNR for 8-bit images is around 48 dB
        return True
    
    return False

if __name__ == "__main__":
    ROOT_PATH = "/datasets/VFI/datasets/AnimeFantasyRPG/"
    RECORD_NAME = "AnimeFantasyRPG_3_60"
    FPS = "fps_60"
    MAIN_INDEX = ["0", "1", "2", "3", "4"]
    MAIN_INDEX = ["0", "4"]
    DIFFICULTY = ["Easy", "Medium"]
    SUB_INDEX = "0"
    MODES = get_all_modes(MAIN_INDEX, DIFFICULTY, SUB_INDEX, FPS)
    MAX_INDEX = 800
    print(MODES)
        
    IMG_FOLDER = f"{ROOT_PATH}/{RECORD_NAME}"

    OUTPUT_PATH = f"/datasets/VFI/GFI_datasets/{RECORD_NAME}"

    TARGET_FILES = [
        ("colorNoScreenUI_", ".png"),
        ("backwardVel_Depth_", ".exr")
    ]

    skipped_records = {}
    for mode in MODES:
        print(f"Processing mode: {mode}")
        mode_output_path = f"{OUTPUT_PATH}/{mode}/overall/"
        if not os.path.exists(mode_output_path):
            os.makedirs(mode_output_path)

        prev_img_path = None
        skipped_indices = []  # 儲存這個 mode 被跳過的 frame index

        with tqdm(range(0, MAX_INDEX)) as pbar:
            pbar.set_description(f"Processing mode: {mode}")
            dst_frame_count = 0

            for frame_index in pbar:
                for prefix, ext in TARGET_FILES:
                    src = f"{IMG_FOLDER}/{mode}/{prefix}{frame_index}{ext}"
                    dst = f"{mode_output_path}/{prefix}{frame_index}{ext}"

                    # check if identical
                    if prev_img_path is not None and src.endswith(".png"):
                        if identical_images(src, prev_img_path):
                            print("[Skipped] Identical image detected, skipping copy:", src)
                            skipped_indices.append(src)
                            continue
                    shutil.copyfile(src, dst)

                    if src.endswith(".png"):
                        prev_img_path = src
                        dst_frame_count += 1

            print(f"Total copied frames for mode {mode}: {dst_frame_count}")
            skipped_records[mode] = {
                "Total Available": dst_frame_count,  # 記錄當前 mode 的結果
                "Skip Indices": skipped_indices  # 記錄當前 mode 的結果
            }
        
    json_output_path = os.path.join(OUTPUT_PATH, "skipped_indices.json")
    with open(json_output_path, "w") as f:
        json.dump(skipped_records, f, indent=4)