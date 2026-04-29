from datasets.dataset_loader import VFIDataset
from datasets.dataset_config import DATASET_CONFIGS, MINOR_DATASET_CONFIGS, VFX_DATASET_CONFIGS, STAIR_DATASET_CONFIG, TEST_DATASET_CONFIGS, iter_dataset_configs
import pandas as pd
from src.gameData_loader import load_backward_velocity, load_forward_velocity
from src.utils import show_images_switchable, flow_to_image, save_img, save_np_array
from evaluation import TaskEvaluator, VFI_METRICS

import cv2
import torch
import numpy as np
import os
import time

from tqdm import tqdm

# from models.IFRNet import Model
from models.IFRNet_Residual import Model
from skimage.metrics import peak_signal_noise_ratio as psnr
from utils import warp


ROOT_DIR = "./datasets/data/"
# MODEL_PATH = "./models/IFRNet/checkpoints/IFRNet/IFRNet_Vimeo90K.pth"
MODEL_PATH = "./output/IFRNet_R_0124_60/checkpoints/IFRNet/merged_fps60_Difficult/"
OUTPUT_DIR = "./output/IFRNet_R_0124_60/checkpoints/IFRNet/merged_fps60_Difficult/inference/"
DATASET = TEST_DATASET_CONFIGS

def main():

    # Load Dataset
    for cfg in iter_dataset_configs(DATASET):
        if cfg.fps != 60:
            continue

        # if cfg.difficulty == "Difficult":
        #     continue

        # Load Eval CSV
        eval_path = os.path.join(OUTPUT_DIR, f"{cfg.record}/{cfg.mode_name}_evaluation_results.csv")
        eval_df = pd.read_csv(eval_path)
        print(eval_df.describe())
        print(f"Saving Evaluation Result into {eval_path}")

            
if __name__ == "__main__":
    main()