
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np

def cal_psnr(img_true: np.ndarray, img_pred: np.ndarray, mask: np.ndarray):
    """
    計算 mask 範圍內的 PSNR。
    img1, img2: np.ndarray, shape = (H, W) 或 (H, W, C)
    mask: np.ndarray, shape = (H, W)，值為 0 或 1
    data_range = 255.0
    """
    valid_idx = mask > 0
    img_true_valid = img_true[valid_idx]
    img_pred_valid = img_pred[valid_idx]

    psnr_score = psnr(img_true_valid, img_pred_valid, data_range=255)

    return psnr_score