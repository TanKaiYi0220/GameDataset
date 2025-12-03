from datasets.dataset_loader import VFIDataset
from datasets.dataset_config import DATASET_CONFIGS, MINOR_DATASET_CONFIGS, VFX_DATASET_CONFIGS, STAIR_DATASET_CONFIG, iter_dataset_configs
import pandas as pd
from src.gameData_loader import load_backward_velocity, load_forward_velocity
from src.utils import show_images_switchable, flow_to_image, save_img, save_np_array
from evaluation import TaskEvaluator, VFI_METRICS

import cv2
import torch
import numpy as np
import os
import time

import sys
sys.path.append('models/IFRNet')
from tqdm import tqdm

from models.IFRNet import Model
from skimage.metrics import peak_signal_noise_ratio as psnr
from utils import warp


ROOT_DIR = "./datasets/data/"
MODEL_PATH = "./models/IFRNet/checkpoints/IFRNet/IFRNet_Vimeo90K.pth"
OUTPUT_DIR = "./output/IFRNet/"
DATASET = DATASET_CONFIGS

def save_video_from_images(images, video_path, fps=60):
    """
    將一串 BGR image (np.ndarray) 存成 mp4 影片。

    images: List[np.ndarray]，每張 shape = (H, W, 3)
    video_path: 輸出影片路徑
    fps: 影格率
    """
    if len(images) == 0:
        print(f"[WARN] No images for video: {video_path}")
        return

    # 影像尺寸 (OpenCV 要 (width, height))
    h, w, _ = images[0].shape

    os.makedirs(os.path.dirname(video_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

    for img in images:
        # 可選：檢查 None / 尺寸不一致
        if img is None:
            continue
        if img.shape[0] != h or img.shape[1] != w:
            img = cv2.resize(img, (w, h))
        writer.write(img)

    writer.release()
    print(f"Video saved to: {video_path}")


def main():
    # Load Dataset
    for cfg in iter_dataset_configs(DATASET):
        if cfg.fps != 60:
            continue

        df = pd.read_csv(f"{ROOT_DIR}/{cfg.record_name}_preprocessed/{cfg.mode_index}_raw_sequence_frame_index.csv")
        
        dataset = VFIDataset(
            df=df,
            root_dir=DATASET["root_dir"],
            record=cfg.record,
            mode=cfg.mode_path,
            input_fps=30,
        )

        print(cfg.mode_name, len(dataset))

        images_GT_list = []
        images_pred_list = []

        with tqdm(range(len(dataset))) as pbar:
            for i in pbar:
                sample = dataset[i]
                input = sample["input"]
                gt = sample["ground_truth"]

                img0_path = input["colorNoScreenUI"][0]
                img1_path = input["colorNoScreenUI"][1]
                imgGT_path = gt["colorNoScreenUI"]

                img0_np = cv2.imread(img0_path)
                img1_np = cv2.imread(img1_path)
                imgGT_np = cv2.imread(imgGT_path)

                # stores results
                save_dir = f"{OUTPUT_DIR}/{cfg.record}/{cfg.mode_path}/{sample['frame_range']}/"
                imgPred_path = f"{save_dir}/image_pred.png" 
                imgPred_np = cv2.imread(imgPred_path)

                if i == 0:
                    images_GT_list.extend([img0_np, imgGT_np, img1_np])
                    images_pred_list.extend([img0_np, imgPred_np, img1_np])
                else:
                    images_GT_list.extend([imgGT_np, img1_np])
                    images_pred_list.extend([imgPred_np, img1_np])

        # save to video mp4 with fps 60
        if len(images_GT_list) > 0 and len(images_pred_list) > 0:
            video_dir = os.path.join(OUTPUT_DIR, cfg.record, "mp4")
            os.makedirs(video_dir, exist_ok=True)

            target_fps = 60
            gt_video_path = os.path.join(video_dir, f"{cfg.mode_name}_GT_fps_{target_fps}.mp4")
            pred_video_path = os.path.join(video_dir, f"{cfg.mode_name}_Pred_fps_{target_fps}.mp4")

            save_video_from_images(images_GT_list, gt_video_path, fps=target_fps)
            save_video_from_images(images_pred_list, pred_video_path, fps=target_fps)

            target_fps = 5
            gt_video_path = os.path.join(video_dir, f"{cfg.mode_name}_GT_fps_{target_fps}.mp4")
            pred_video_path = os.path.join(video_dir, f"{cfg.mode_name}_Pred_fps_{target_fps}.mp4")

            save_video_from_images(images_GT_list, gt_video_path, fps=target_fps)
            save_video_from_images(images_pred_list, pred_video_path, fps=target_fps)
        else:
            print("No images collected for this config; skip video export.")


            
if __name__ == "__main__":
    main()