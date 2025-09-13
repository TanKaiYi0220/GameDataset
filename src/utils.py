import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def save_img(path: str, img: np.ndarray):
    dir_name = os.path.dirname(path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
    cv2.imwrite(path, img)
