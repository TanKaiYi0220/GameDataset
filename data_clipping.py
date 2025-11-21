import os
import cv2
import re
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
    fps = first[2]

    # 把所有 mode 的第二個部分的難度名稱抓出來
    difficulties = [m.split('/')[0].split('_')[1] for m in modes]

    # 拼接結果
    combined_name = f"{main_index}_{'_'.join(difficulties)}_{sub_index}_{fps}"
    return combined_name

def get_prefix_key(path):
    folder = os.path.dirname(path)
    name = os.path.splitext(os.path.basename(path))[0]
    prefix = re.sub(r'\d+$', '', name)
    return (folder, prefix)

def get_idx(path):
    return int(os.path.basename(path).split("_")[-1].split(".")[0])

def convert_color_to_velDepth(path: str) -> str:
    """
    將 'colorNoScreenUI_{}_{}.png' 換成 'backwardVel_Depth_{}_{}.exr'
    例如:
      colorNoScreenUI_12_45.png → backwardVel_Depth_12_45.exr
    """
    return path.replace("colorNoScreenUI_", "backwardVel_Depth_").replace(".png", ".exr")

def write_clip_json(root_path, mode, clip_seq):
    clip_json = {}
    for difficult in mode:
        colorNoScreenUI_path = []
        gameMotion_path = []
        for color_file in clip_seq:
            img_path = os.path.join(root_path, difficult, "overall", color_file)

            velD_file = convert_color_to_velDepth(color_file)
            velD_path = os.path.join(root_path, difficult, "overall", velD_file)
            
            colorNoScreenUI_path.append(img_path)
            gameMotion_path.append(velD_path)
        clip_json[difficult] = {
            "colorNoScreenUI": colorNoScreenUI_path,
            "backwardVel_Depth": gameMotion_path
        }
    return clip_json

def write_clip(paths, out_path, fps=1):
    """將圖片列表輸出為影片"""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    first = cv2.imread(paths[0])
    if first is None:
        return
    h, w = first.shape[:2]
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for p in paths:
        img = cv2.imread(p)
        if img is not None and img.shape[:2] == (h, w):
            vw.write(img)
    vw.release()

if __name__ == "__main__":
    ROOT_PATH = "/datasets/VFI/datasets/AnimeFantasyRPG/"
    RECORD_NAME = "AnimeFantasyRPG_3_60"
    ORIGINAL_FPS = 30
    FPS = f"fps_{ORIGINAL_FPS}"
    MAIN_INDEX = ["0", "1", "2", "3"]
    DIFFICULTY = ["Easy", "Medium"]
    SUB_INDEX = "0"
    MODES = get_all_modes(MAIN_INDEX, DIFFICULTY, SUB_INDEX, FPS)
    MAX_INDEX = 800
        
    IMG_FOLDER = f"{ROOT_PATH}/{RECORD_NAME}"

    CLEAN_PATH = f"/datasets/VFI/GFI_datasets/{RECORD_NAME}"

    CLIP_LEN = ORIGINAL_FPS * 2

    TARGET_FPS = [1, 15, ORIGINAL_FPS]

    overall_clip_json = {}

    for mode in MODES:
        mode_name = parse_mode_name(mode)
        print("Mode Name:", mode_name)

        overall_clip_json[mode_name] = {}

        json_path = f"{CLEAN_PATH}/{mode_name}_review_result.json"
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)


        avail_seq = [os.path.basename(x) for x in data["available_list"]]
        avail_indices = [get_idx(x) for x in avail_seq]
        avail_indices = sorted(avail_indices)


        runs = []
        start = 0
        for i in range(1, len(avail_indices) + 1):
            if i == len(avail_indices) or avail_indices[i] != avail_indices[i - 1] + 1:
                runs.append((start, i - 1))
                start = i

        total = 0
        for s, e in runs:
            if e - s + 1 < CLIP_LEN:
                continue
            pos = s
            while pos + CLIP_LEN - 1 <= e:
                clip_seq = avail_seq[pos:pos + CLIP_LEN]
                start_idx = avail_indices[pos]
                end_idx   = avail_indices[pos + CLIP_LEN - 1]

                clip_json = write_clip_json(CLEAN_PATH, mode, clip_seq)
                for difficult in mode:
                    color_paths = clip_json[difficult]["colorNoScreenUI"]

                    for fps in TARGET_FPS:
                        out_name = f"{CLEAN_PATH}/{difficult}/clip_{start_idx:06d}_{end_idx:06d}_fps60_to_fps{fps}.mp4"
                        write_clip(color_paths, out_name, fps=fps)
                        print("Clip written:", out_name)

                overall_clip_json[mode_name][f"clip_{start_idx:06d}_{end_idx:06d}"] = clip_json

                total += 1
                pos += CLIP_LEN

    with open(f"{CLEAN_PATH}/overall_{FPS}_clip_info.json", "w", encoding="utf-8") as f:
        json.dump(overall_clip_json, f, indent=4)


