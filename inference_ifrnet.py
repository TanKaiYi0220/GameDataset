from datasets.dataset_loader import VFIDataset
from datasets.dataset_config import MINOR_DATASET_CONFIGS, iter_dataset_configs
import pandas as pd
from src.utils import show_images_switchable, flow_to_image, save_img
import cv2
import torch
import numpy as np
import os
import time

import sys
sys.path.append('models/IFRNet')
from tqdm import tqdm

from models.IFRNet import Model
from utils import warp

ROOT_DIR = "./datasets/data/AnimeFantasyRPG_3_60_preprocessed"
MODEL_PATH = "./models/IFRNet/checkpoints/IFRNet/IFRNet_Vimeo90K.pth"
OUTPUT_DIR = "./output/IFRNet/"


def main():
    # Load Model
    model = Model().cuda().eval()
    model.load_state_dict(torch.load(MODEL_PATH))

    # Load Dataset
    for cfg in iter_dataset_configs(MINOR_DATASET_CONFIGS):
        if cfg.fps != 60:
            continue

        df = pd.read_csv(f"{ROOT_DIR}/{cfg.mode_index}_clipped_frame_index.csv")
        
        dataset = VFIDataset(
            df=df,
            root_dir=MINOR_DATASET_CONFIGS["root_dir"],
            record=cfg.record,
            mode=cfg.mode_path,
            input_fps=30,
        )

        print(cfg.mode_name, len(dataset))

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

                # Inference
                img0 = (torch.tensor(img0_np.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).cuda()
                img1 = (torch.tensor(img1_np.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).cuda()
                embt = torch.tensor(1/2).view(1, 1, 1, 1).float().cuda()

                # ------------------------ insert timing ------------------------
                torch.cuda.synchronize()
                start = time.time()

                imgPred, up_flow0_1, up_flow1_1, up_mask_1 = model.inference(img0, img1, embt)

                torch.cuda.synchronize()
                end = time.time()
                infer_time = end - start
                # ---------------------------------------------------------------

                imgPred_np = (imgPred[0].data.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
                up_flow0_1_np = flow_to_image(up_flow0_1[0].data.permute(1, 2, 0).cpu().numpy())
                up_flow1_1_np = flow_to_image(up_flow1_1[0].data.permute(1, 2, 0).cpu().numpy())
                up_mask_1_np = (up_mask_1[0, 0].data.cpu().numpy() * 255.0).astype(np.uint8)

                # Warped images
                img0_warped = warp(img0, up_flow0_1)
                img1_warped = warp(img1, up_flow1_1)

                img0_warped_np = img0_warped[0].data.permute(1, 2, 0).cpu().numpy() * 255.0
                img1_warped_np = img1_warped[0].data.permute(1, 2, 0).cpu().numpy() * 255.0

                # stores results
                save_dir = f"{OUTPUT_DIR}/{cfg.record}/{cfg.mode_path}/{sample['frame_range']}/"
                os.makedirs(f"{save_dir}", exist_ok=True)
                save_img(f"{save_dir}/image_0.png", img0_np)
                save_img(f"{save_dir}/image_1.png", img1_np)
                save_img(f"{save_dir}/image_gt.png", imgGT_np)
                save_img(f"{save_dir}/image_pred.png", imgPred_np)
                save_img(f"{save_dir}/flow_1_to_0.png", up_flow0_1_np)
                save_img(f"{save_dir}/flow_1_to_2.png", up_flow1_1_np)
                save_img(f"{save_dir}/flow_mask.png", up_mask_1_np)
                save_img(f"{save_dir}/image_0_warped.png", img0_warped_np)
                save_img(f"{save_dir}/image_1_warped.png", img1_warped_np)

                pbar.set_postfix({
                    "FrameRange": sample["frame_range"],
                    "InferenceTime": infer_time
                })

            





            


if __name__ == "__main__":
    main()