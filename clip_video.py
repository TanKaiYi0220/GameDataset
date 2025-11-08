import os
import shutil
import cv2
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np

def mode_rename(mode):
    difficult, _, fps, _ = mode.split("/")
    return f"{difficult}_{fps}"

def identical_images(img1_path: str, img2_path: str) -> bool:
    img1 = cv2.imread(img1_path).astype(np.uint8)
    img2 = cv2.imread(img2_path).astype(np.uint8)

    # visualize different between two images
    color_diff = cv2.absdiff(img1, img2)
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

def PNG_to_video(img_folder: str, output_path: str, fps: int, start_index: int, end_index: int):
    if os.path.exists(output_path):
        print(f"[Skipped] Video already exists: {output_path}")
        return
    
    images = [f"{img_folder}/colorNoScreenUI_{idx}.png" for idx in range(start_index, end_index)]


    height, width, channel = cv2.imread(images[0]).shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 建議用 mp4v 保證跨平台
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(output_path)

    # 寫入每一張圖片
    for img_path in images:
        img = cv2.imread(img_path)
        video.write(img)

    video.release()

def copy_files(img_folder: str, output_folder: str, start_index: int, end_index: int):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    target_files = [
        ("colorNoScreenUI_", ".png"),
        ("backwardVel_Depth_", ".exr")
    ]


    with tqdm(range(start_index, end_index)) as pbar:
        prev_img_path = None
        for idx in pbar:
            for prefix, ext in target_files:
                src = f"{img_folder}/{prefix}{idx}{ext}"
                dst = f"{output_folder}/{prefix}{idx}{ext}"

                if prev_img_path is not None and ext == ".png":
                    if identical_images(src, prev_img_path):
                        print("[Skipped] Identical image detected, skipping copy:", src)
                        continue
                if ext == ".png":
                    prev_img_path = src

                if not os.path.exists(src):
                    raise FileNotFoundError(f"Source file not found: {src}")
                else:
                    shutil.copy2(src, dst)

                pbar.set_postfix({"Index": f"{idx:.4f}", "Prefix": prefix})


if __name__ == "__main__":
    ROOT_PATH = "/datasets/VFI/datasets/AnimeFantasyRPG/"
    RECORD_NAME = "AnimeFantasyRPG_3_60"
    FPS = "fps_60"
    MODE = f"0_Easy/0_Easy_0/{FPS}"
    IMG_FOLDER = f"{ROOT_PATH}/{RECORD_NAME}"

    OUTPUT_PATH = f"/datasets/VFI/GFI_datasets/{RECORD_NAME}"

    INDEX_RANGE = [
        (140, 260),
        (400, 520),
    ]

    INDEX_RANGE = [(idx, idx + 120) for idx in range(0, 800, 120)]

    ORIGINAL_FPS = 60
    FPS = 15

    clip_idx = 0
    for start_index, end_index in INDEX_RANGE:
        copy_files(
            f"{IMG_FOLDER}/{MODE}",
            f"{OUTPUT_PATH}/{MODE}/{start_index}_{end_index}_frames",
            start_index,
            end_index
        )

        PNG_to_video(
            f"{IMG_FOLDER}/{MODE}",
            f"{OUTPUT_PATH}/{MODE}/{start_index}_{end_index}_{1}.mp4",
            1,
            start_index,
            end_index
        )

        PNG_to_video(
            f"{IMG_FOLDER}/{MODE}",
            f"{OUTPUT_PATH}/{MODE}/{start_index}_{end_index}_{FPS}.mp4",
            FPS,
            start_index,
            end_index
        )

        PNG_to_video(
            f"{IMG_FOLDER}/{MODE}",
            f"{OUTPUT_PATH}/{MODE}/{start_index}_{end_index}_{ORIGINAL_FPS}.mp4",
            ORIGINAL_FPS,
            start_index,
            end_index
        )
