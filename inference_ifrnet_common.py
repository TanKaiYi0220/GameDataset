import os
import time
import pandas as pd
from tqdm import tqdm

import cv2
import numpy as np
import torch
from torch import Tensor

from datasets.dataset_loader import VFIDataset
from src.gameData_loader import load_backward_velocity, load_forward_velocity
from evaluation import TaskEvaluator, VFI_METRICS
from model_registry import get_model_class, get_model_config
from utils import warp, save_img, save_np_array
from src.utils import flow_to_image


def load_model(model_name: str, checkpoint_path: str, device: torch.device):
    model_class = get_model_class(model_name)
    model = model_class().to(device)
    if checkpoint_path and os.path.isfile(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
    return model


def tensorize_image(image_np: np.ndarray, device: torch.device):
    tensor = torch.tensor(image_np.transpose(2, 0, 1)).float() / 255.0
    return tensor.unsqueeze(0).to(device)


def inference_step(model_name: str, model, img0: Tensor, img1: Tensor, embt: Tensor, bmv: Tensor, fmv: Tensor):
    if model_name == "IFRNet":
        return model.inference(img0, img1, embt)

    return model.inference(img0, img1, embt, init_flow0=bmv, init_flow1=fmv)


def ensure_images_readable(paths, max_retries=5, delay_s=1.0):
    imgs = [cv2.imread(path) for path in paths]
    retries = 0
    while any(img is None for img in imgs) and retries < max_retries:
        time.sleep(delay_s)
        imgs = [cv2.imread(path) for path in paths]
        retries += 1
    if any(img is None for img in imgs):
        missing = [path for img, path in zip(imgs, paths) if img is None]
        raise RuntimeError(f"Unable to read image files: {missing}")
    return imgs


def run_inference_loop(
    model_name: str,
    model,
    dataset_configs,
    root_dir: str,
    output_dir: str,
    device: torch.device,
    save_sample_fn,
    filter_cfg_fn=None,
):
    for cfg in dataset_configs:
        if cfg.fps != 60:
            continue
        if filter_cfg_fn is not None and filter_cfg_fn(cfg):
            continue

        csv_path = f"{root_dir}/{cfg.record_name}_preprocessed/{cfg.mode_index}_raw_sequence_frame_index.csv"
        df = pd.read_csv(csv_path)
        df["record"] = cfg.record
        df["mode"] = cfg.mode_path

        vfi_evaluator = TaskEvaluator(task_name="VFI", metric_fns=VFI_METRICS)
        dataset = VFIDataset(df=df, root_dir=cfg.root_dir, input_fps=30)

        print(cfg.record, cfg.mode_name, len(dataset))
        with tqdm(range(len(dataset))) as pbar:
            for i in pbar:
                sample = dataset[i]
                input_data = sample["input"]
                gt = sample["ground_truth"]

                img0_path = input_data["colorNoScreenUI"][0]
                img1_path = input_data["colorNoScreenUI"][1]
                imgGT_path = gt["colorNoScreenUI"]
                bmv_path = gt["backwardVel_Depth"]
                fmv_path = gt["forwardVel_Depth"]

                img0_np, img1_np, imgGT_np = ensure_images_readable([img0_path, img1_path, imgGT_path])
                bmv, _ = load_backward_velocity(bmv_path)
                fmv, _ = load_forward_velocity(fmv_path)

                img0 = tensorize_image(img0_np, device)
                img1 = tensorize_image(img1_np, device)
                embt = torch.tensor(1 / 2).view(1, 1, 1, 1).float().to(device)

                outputs = inference_step(model_name, model, img0, img1, embt, bmv, fmv)

                result = save_sample_fn(
                    outputs=outputs,
                    sample=sample,
                    cfg=cfg,
                    output_dir=output_dir,
                    img0_np=img0_np,
                    img1_np=img1_np,
                    imgGT_np=imgGT_np,
                    bmv=bmv,
                    fmv=fmv,
                    img0=img0,
                    img1=img1,
                    embt=embt,
                    device=device,
                    vfi_evaluator=vfi_evaluator,
                )

                if isinstance(result, dict):
                    pbar.set_postfix(result)

        eval_df = vfi_evaluator.to_dataframe()
        eval_path = os.path.join(output_dir, f"{cfg.record}/{cfg.mode_name}_evaluation_results.csv")
        eval_df.to_csv(eval_path, index=False)
        print(eval_df.describe())
        print(f"Saving Evaluation Result into {eval_path}")
