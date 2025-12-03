import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from utils import show_images_switchable

def visualize_color_difference(img1: np.ndarray, img2: np.ndarray):
    difference = psnr(img1, img2, data_range=255)

    # visualize different between two images
    color_diff = cv2.absdiff(img1, img2)
    show_images_switchable(
        [img1, img2, color_diff],
        ["Image 1", "Image 2", f"Color Difference (PSNR: {difference:.2f} dB)"]
    )

def identical_images(img1: np.ndarray, img2: np.ndarray) -> bool:
    difference = psnr(img1, img2, data_range=255)

    if np.isfinite(difference) == False:
        # print("Identical images (PSNR=inf). Setting PSNR to 0.")
        return True

    if difference > 48: # maximum PSNR for 8-bit images is around 48 dB
        return True
    
    return False

if __name__ == "__main__":
    # Show identical images examples
    img1_path = "/datasets/VFI/datasets/AnimeFantasyRPG/AnimeFantasyRPG_3_60/0_Easy/0_Easy_0/fps_60/colorNoScreenUI_1.png"
    img2_path = "/datasets/VFI/datasets/AnimeFantasyRPG/AnimeFantasyRPG_3_60/0_Easy/0_Easy_0/fps_60/colorNoScreenUI_2.png"

    img1 = cv2.imread(img1_path).astype(np.uint8)
    img2 = cv2.imread(img2_path).astype(np.uint8)

    visualize_color_difference(img1, img2)
    if identical_images(img1, img2):
        print("Images are identical.")
    else:
        print("Images are different.")

    # Show different images examples
    img1_path = "/datasets/VFI/datasets/AnimeFantasyRPG/AnimeFantasyRPG_3_60/0_Easy/0_Easy_0/fps_60/colorNoScreenUI_11.png"
    img2_path = "/datasets/VFI/datasets/AnimeFantasyRPG/AnimeFantasyRPG_3_60/0_Easy/0_Easy_0/fps_60/colorNoScreenUI_12.png"

    img1 = cv2.imread(img1_path).astype(np.uint8)
    img2 = cv2.imread(img2_path).astype(np.uint8)

    visualize_color_difference(img1, img2)
    if identical_images(img1, img2):
        print("Images are identical.")
    else:
        print("Images are different.")