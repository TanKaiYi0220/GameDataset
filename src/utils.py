import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import re
from .EXR_loader import loadEXR


def mode_rename(mode):
    difficult, _, fps, _ = mode.split("/")
    return f"{difficult}_{fps}"

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
        print(f"Created {dir_name}")
        os.makedirs(dir_name, exist_ok=True)
    try:
        cv2.imwrite(path, img)
    except Exception as e:
        print(f"{path}: {e}")
        retry_save_img(path, img, 0)


def find_max_index_in_dir(path: str) -> int:
    """
    掃描指定資料夾，回傳檔名中最大的 index 整數。
    """
    max_index = -1
    pattern = re.compile(r'_(\d+)\.')  # 抓取最後一個 '_' 與 '.' 之間的數字

    for filename in os.listdir(path):
        match = pattern.search(filename)
        if match:
            index = int(match.group(1))
            if index > max_index:
                max_index = index

    return max_index

def EXR_to_PNG(filepath: str):
    exrImg = loadEXR(filepath)
    exrImg = exrImg * 255.0
    exrImg = exrImg.astype(np.uint8)  # Fixed: assign the conversion

    return exrImg