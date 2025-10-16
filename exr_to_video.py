import os
import cv2

def mode_rename(mode):
    difficult, _, fps, _ = mode.split("/")
    return f"{difficult}_{fps}"

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

if __name__ == "__main__":
    ROOT_PATH = "/datasets/VFI/datasets/AnimeFantasyRPG/"
    RECORD_NAME = "AnimeFantasyRPG_3_60"
    MODES = [
        "0_Easy/0_Easy_0/fps_30/", 
        "0_Medium/0_Medium_0/fps_30/", 
        # "0_Difficult/0_Difficult_0/fps_30/"
    ]
    IMG_FOLDER = f"{ROOT_PATH}/{RECORD_NAME}"

    OUTPUT_PATH = f"./analysis_results/1006/{RECORD_NAME}"

    START_INDEX = 360
    END_INDEX = 380
    ORIGINAL_FPS = 30
    FPS = 1

    for mode in MODES:
        PNG_to_video(
            f"{IMG_FOLDER}/{mode}",
            f"{OUTPUT_PATH}/{mode_rename(mode)}_{START_INDEX}_{END_INDEX}_{FPS}.mp4",
            FPS,
            START_INDEX,
            END_INDEX
        )

        PNG_to_video(
            f"{IMG_FOLDER}/{mode}",
            f"{OUTPUT_PATH}/{mode_rename(mode)}_{START_INDEX}_{END_INDEX}_{ORIGINAL_FPS}.mp4",
            ORIGINAL_FPS,
            START_INDEX,
            END_INDEX
        )