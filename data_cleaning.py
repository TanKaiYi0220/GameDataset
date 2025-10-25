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
        mode = []
        for difficulty in difficulty_list:
            mode.append(f"{main_index}_{difficulty}/{main_index}_{difficulty}_{sub_index}/{fps}")
        modes.append(mode)
    return modes

def parse_mode_name(modes):
    """
    從 mode list 取出簡化名稱。
    e.g.
    ["0_Easy/0_Easy_0/fps_60", "0_Medium/0_Medium_0/fps_60"] → "0_Easy_Medium_0"
    """
    # 拿第一個 mode 拆開
    first = modes[0].split('/')
    main_index = first[0].split('_')[0]         # 例如 '0'
    sub_index = first[1].split('_')[-1]         # 例如 '0'

    # 把所有 mode 的第二個部分的難度名稱抓出來
    difficulties = [m.split('/')[0].split('_')[1] for m in modes]

    # 拼接結果
    combined_name = f"{main_index}_{'_'.join(difficulties)}_{sub_index}"
    return combined_name

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
    print(MODES)
    for mode in MODES:
        print("Processing Mode :", mode)

        skipped_indices = []
        status = {i: "keep" for i in range(MAX_INDEX)}

        # 1: Check identical images and mark them as "drop"
        for difficult in mode:
            prev_img_path = None
            with tqdm(range(MAX_INDEX)) as pbar:
                pbar.set_description(f"Check Identical: {difficult}")
                for frame_index in pbar:
                    src = f"{IMG_FOLDER}/{difficult}/colorNoScreenUI_{frame_index}.png"

                    if prev_img_path is not None:
                        if identical_images(src, prev_img_path):
                            print("[Skipped] Identical image detected, skipping copy:", src)
                            status[frame_index] = "drop"

                    prev_img_path = src

        # 2: Copy files according to the status
        for difficult in mode:
            dst_frame_count = 0
            with tqdm(range(MAX_INDEX)) as pbar:
                pbar.set_description(f"Copying Files: {difficult}")
                for frame_index in pbar:
                    if status[frame_index] == "drop":
                        skipped_indices.append(frame_index)
                        continue
                    else:
                        for prefix, ext in TARGET_FILES:
                            mode_output_path = f"{OUTPUT_PATH}/{difficult}/overall/"
                            if not os.path.exists(mode_output_path):
                                os.makedirs(mode_output_path)

                            src = f"{IMG_FOLDER}/{difficult}/{prefix}{frame_index}{ext}"
                            dst = f"{mode_output_path}/{prefix}{frame_index}_{dst_frame_count}{ext}"
                            shutil.copyfile(src, dst)
                        dst_frame_count += 1

        mode_name = parse_mode_name(mode)
        print(f"Total copied frames for mode {mode_name}: {dst_frame_count}")
        skipped_records[mode_name] = {
            "Total Available": dst_frame_count,  # 記錄當前 mode 的結果
            "Skip Indices": skipped_indices  # 記錄當前 mode 的結果
        }
        
    json_output_path = os.path.join(OUTPUT_PATH, "skipped_indices.json")
    with open(json_output_path, "w") as f:
        json.dump(skipped_records, f, indent=4)