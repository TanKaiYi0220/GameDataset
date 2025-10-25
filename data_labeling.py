import cv2
import json
import os

def get_all_modes(main_index_list, difficulty_list, sub_index, fps):
    modes = []
    for main_index in main_index_list:
        for difficulty in difficulty_list:
            mode = f"{main_index}_{difficulty}/{main_index}_{difficulty}_{sub_index}/{fps}"
            modes.append(mode)
    return modes

def save_json(image_paths, status, i, json_path="review_result.json"):
    available = [image_paths[k] for k, v in status.items() if v == "keep"]
    rejected  = [image_paths[k] for k, v in status.items() if v == "drop"]
    out = {
        "available_list": available,
        "rejected_list": rejected,
        "unlabeled": [image_paths[k] for k, v in status.items() if v is None],
        "progress": {
            "total": len(image_paths),
            "labeled": len(available) + len(rejected),
            "current_index": i,
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[Saved] {json_path}")

def review_images(image_paths, json_path="review_result.json"):
    """
    一個簡單的 OpenCV 審圖工具：
    ←/→ or ↑/↓ 翻頁
    Y: 採用
    N: 不採用
    S: 存檔
    Q / ESC: 退出
    """
    assert len(image_paths) > 0, "image_paths 不可為空"

    status = {i: "keep" for i in range(len(image_paths))}
    i = 0
    win = "Reviewer"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        img = cv2.imread(image_paths[i])

        if img is None:
            print(f"[Warning] 無法讀取 {image_paths[i]}")
            i = min(i + 1, len(image_paths) - 1)
            continue

        # 根據標記顯示不同顏色邊框
        if status[i] == "keep":
            color = (0, 255, 0)
        elif status[i] == "drop":
            color = (0, 0, 255)
        else:
            color = (128, 128, 128)

        cv2.rectangle(img, (5, 5), (img.shape[1] - 5, img.shape[0] - 5), color, 3)
        text = f"[{i+1}/{len(image_paths)}] {os.path.basename(image_paths[i])} | 狀態: {status[i]}"
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow(win, img)

        key = cv2.waitKey(0) & 0xFF
        if key in (ord('d'), 83, 81, 82):  # → ↓
            i = min(i + 1, len(image_paths) - 1)
        elif key in (ord('a'), 80, 84):  # ← ↑
            i = max(i - 1, 0)
        elif key in (ord('y'), ord('Y')):
            status[i] = "keep"
            i = min(i + 1, len(image_paths) - 1)
        elif key in (ord('n'), ord('N')):
            status[i] = "drop"
            i = min(i + 1, len(image_paths) - 1)
        elif key in (ord('s'), ord('S')):
            save_json(image_paths, status, i, json_path)
        elif key in (ord('q'), ord('Q'), 27):
            break

    cv2.destroyAllWindows()
    save_json(image_paths, status, i, json_path)
    available = [p for idx, p in enumerate(image_paths) if status[idx] == "keep"]
    rejected  = [p for idx, p in enumerate(image_paths) if status[idx] == "drop"]
    return available, rejected

if __name__ == "__main__":
    ROOT_PATH = "/datasets/VFI/datasets/AnimeFantasyRPG/"
    RECORD_NAME = "AnimeFantasyRPG_3_60"
    FPS = "fps_60"
    MAIN_INDEX = "4"
    DIFFICULTY = ["Easy", "Medium"]
    SUB_INDEX = "0"
    MODES = get_all_modes([MAIN_INDEX], DIFFICULTY, SUB_INDEX, FPS)
    MAX_INDEX = 800
        
    IMG_FOLDER = f"{ROOT_PATH}/{RECORD_NAME}"

    CLEAN_PATH = f"/datasets/VFI/GFI_datasets/{RECORD_NAME}"

    skip_indices = json.load(open(f"{CLEAN_PATH}/skipped_indices.json", "r"))

    image_paths = []
    for i in range(MAX_INDEX):
        available = True
        for mode in MODES:
            img_path = f"{CLEAN_PATH}/{mode}/overall/colorNoScreenUI_{i}.png"

            if not os.path.exists(img_path) or img_path in skip_indices[mode]["Skip Indices"]:
                available = False
            else:
                print("[Cleanned] ", img_path)
        
        if available:
            image_paths.append(img_path)

    if not image_paths:
        print("請提供 image_paths 再執行。")
    else:
        available, rejected = review_images(image_paths, json_path=f"{CLEAN_PATH}/{MAIN_INDEX}_{DIFFICULTY[-1]}_{SUB_INDEX}_{FPS}_review_result.json")
        print("Available:", len(available))
        print("Rejected:", len(rejected))