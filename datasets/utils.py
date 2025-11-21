import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np

def show_images_switchable(images, titles):
    """
    images: list[np.ndarray]   要顯示的圖片
    titles: list[str]          每張圖的標題
    """
    assert len(images) == len(titles)
    idx = 0
    n = len(images)

    while True:
        img = images[idx].copy()

        # 顯示標題 (目前第幾張)
        text = f"[{idx+1}/{n}] {titles[idx]}"
        cv2.putText(img, text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Overlay", img)
        key = cv2.waitKey(0) & 0xFF

        # ← 或 ↑：上一張
        if key in [ord('a'), 81, 82]:  # 'a' 或 左/上箭頭
            idx = (idx - 1) % n
        # → 或 ↓：下一張
        elif key in [ord('d'), 83, 84]:  # 'd' 或 右/下箭頭
            idx = (idx + 1) % n
        # q 或 ESC 離開
        elif key in [ord('q'), 27]:
            break

    cv2.destroyAllWindows()

def retry_save_img(path: str, img: np.ndarray, retry_times: int):
    if retry_times == 5:
        raise Exception("Over Retries Limit")
    
    try:
        cv2.imwrite(path, img)
    except:
        retry_save_img(path, img, retry_times + 1)

def save_img(path: str, img: np.ndarray):
    dir_name = os.path.dirname(path)
    if not os.path.exists(dir_name):
        # print(f"Created {dir_name}")
        os.makedirs(dir_name, exist_ok=True)
    try:
        cv2.imwrite(path, img)
    except Exception as e:
        print(f"{path}: {e}")
        retry_save_img(path, img, 0)

def loadEXR(filename: str) -> np.ndarray:
    exrImg = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    return exrImg

def loadPNG(filepath: str) -> np.ndarray:
    if os.path.exists(filepath) == False:
        # replaced path extension from .png to .exr
        name, _ = os.path.splitext(filepath)
        exr_path = f"{name}.exr"
        EXRToPNG(exr_path, filepath)
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    return img

def EXRToPNG(filepath: str, imgPath: str):
    exrImg = loadEXR(filepath)
    exrImg = exrImg * 255.0
    exrImg = exrImg.astype(np.uint8)  # Fixed: assign the conversion

    save_img(imgPath, exrImg)