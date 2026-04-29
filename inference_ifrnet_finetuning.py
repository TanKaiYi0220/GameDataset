import os

import cv2
import numpy as np
import torch

from config.config import get_config
from datasets.dataset_config import TEST_3D_VFX_DATASET_CONFIGS
from evaluation import TaskEvaluator, VFI_METRICS
from inference_ifrnet_common import load_model, run_inference_loop
from src.gameData_loader import load_backward_velocity, load_forward_velocity
from src.utils import flow_to_image
from utils import warp, save_img

ROOT_DIR = get_config("data_root", "./datasets/data")
MODEL_PATH = os.path.join(get_config("output_root", "./output"), "IFRNet_FineTuning_0326", "checkpoints")
OUTPUT_DIR = os.path.join(get_config("output_root", "./output"), "IFRNet_FineTuning_0326", "checkpoints", "inference")
DATASET = TEST_3D_VFX_DATASET_CONFIGS


def save_sample_fn(*, outputs, sample, cfg, output_dir, img0_np, img1_np, imgGT_np, bmv, fmv, img0, img1, embt, device, vfi_evaluator, **kwargs):
    imgPred, up_flow0_1, up_flow1_1, up_mask_1 = outputs

    imgPred_np = (imgPred[0].detach().permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    up_flow0_1_np = flow_to_image(up_flow0_1[0].detach().permute(1, 2, 0).cpu().numpy())
    up_flow1_1_np = flow_to_image(up_flow1_1[0].detach().permute(1, 2, 0).cpu().numpy())
    up_mask_1_np = (up_mask_1[0, 0].detach().cpu().numpy() * 255.0).astype(np.uint8)

    img0_warped = warp(img0, up_flow0_1)
    img1_warped = warp(img1, up_flow1_1)
    img0_warped_np = img0_warped[0].detach().permute(1, 2, 0).cpu().numpy() * 255.0
    img1_warped_np = img1_warped[0].detach().permute(1, 2, 0).cpu().numpy() * 255.0

    save_dir = f"{output_dir}/{cfg.record}/{cfg.mode_path}/{sample['frame_range']}/"
    os.makedirs(save_dir, exist_ok=True)

    save_img(f"{save_dir}/image_0.png", img0_np)
    save_img(f"{save_dir}/image_1.png", img1_np)
    save_img(f"{save_dir}/image_gt.png", imgGT_np)
    save_img(f"{save_dir}/image_pred.png", imgPred_np)
    save_img(f"{save_dir}/flow_1_to_0.png", up_flow0_1_np)
    save_img(f"{save_dir}/flow_1_to_2.png", up_flow1_1_np)
    save_img(f"{save_dir}/flow_mask.png", up_mask_1_np)
    save_img(f"{save_dir}/image_0_warped.png", img0_warped_np)
    save_img(f"{save_dir}/image_1_warped.png", img1_warped_np)

    result = vfi_evaluator.evaluate(
        meta={
            "record": cfg.record,
            "mode": cfg.mode_path,
            "frame_range": sample["frame_range"],
            "valid": sample["valid"],
            "distance_indexing": sample["distance_indexing"],
        },
        img_gt=imgGT_np,
        img_pred=imgPred_np,
        flow_1_to_0=up_flow0_1,
        flow_1_to_2=up_flow1_1,
        bmv=bmv,
        fmv=fmv,
    )
    return {
        "PSNR": float(result["psnr"]),
        "EPE_1_to_0": float(result["epe_1_to_0"]),
        "EPE_1_to_2": float(result["epe_1_to_2"]),
    }


def filter_cfg_fn(cfg):
    return cfg.difficulty != "Difficult"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("IFRNet", os.path.join(MODEL_PATH, "best.pth"), device).eval()
    run_inference_loop(
        model_name="IFRNet",
        model=model,
        dataset_configs=DATASET,
        root_dir=ROOT_DIR,
        output_dir=OUTPUT_DIR,
        device=device,
        save_sample_fn=save_sample_fn,
        filter_cfg_fn=filter_cfg_fn,
    )


if __name__ == "__main__":
    main()
