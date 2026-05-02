import os
import time
from collections.abc import Iterable

import cv2
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from tqdm import tqdm

from gamedataset.data.config import iter_dataset_configs
from gamedataset.data.loader import VFIDataset
from gamedataset.data.io import load_backward_velocity
from gamedataset.evaluation import TaskEvaluator, VFI_METRICS
from gamedataset.models.registry import get_model_class


def load_model(model_name: str, checkpoint_path: str, device: torch.device):
    if checkpoint_path == "":
        raise ValueError("checkpoint_path must not be empty")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model_class = get_model_class(model_name)
    model = model_class().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint format in {checkpoint_path}")

    model.load_state_dict(state_dict)
    return model


def tensorize_image(image_np: np.ndarray, device: torch.device) -> Tensor:
    tensor = torch.tensor(image_np.transpose(2, 0, 1)).float() / 255.0
    return tensor.unsqueeze(0).to(device)


def tensorize_flow(flow_np: np.ndarray, device: torch.device) -> Tensor:
    tensor = torch.from_numpy(flow_np.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor.to(device)


def inference_step(
    model_name: str,
    model,
    img0: Tensor,
    img1: Tensor,
    embt: Tensor,
    bmv: Tensor,
    fmv: Tensor,
):
    if model_name == "IFRNet":
        return model.inference(img0, img1, embt)

    return model.inference(img0, img1, embt, init_flow0=bmv, init_flow1=fmv)


def ensure_images_readable(paths: list[str], max_retries: int, delay_s: float) -> list[np.ndarray]:
    images = [cv2.imread(path) for path in paths]
    retries = 0

    while any(image is None for image in images) and retries < max_retries:
        time.sleep(delay_s)
        images = [cv2.imread(path) for path in paths]
        retries += 1

    if any(image is None for image in images):
        missing_paths = [path for image, path in zip(images, paths) if image is None]
        raise RuntimeError(f"Unable to read image files after retries: {missing_paths}")

    return images


def iter_inference_configs(dataset_configs):
    if isinstance(dataset_configs, dict):
        return iter_dataset_configs(dataset_configs)
    if isinstance(dataset_configs, Iterable):
        return dataset_configs
    raise TypeError(f"Unsupported dataset_configs type: {type(dataset_configs)!r}")


def run_inference_loop(
    model_name: str,
    model,
    dataset_configs,
    root_dir: str,
    output_dir: str,
    device: torch.device,
    save_sample_fn,
    filter_cfg_fn=None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    for cfg in iter_inference_configs(dataset_configs):
        if cfg.fps != 60:
            continue
        if filter_cfg_fn is not None and filter_cfg_fn(cfg):
            continue

        csv_path = os.path.join(
            root_dir,
            f"{cfg.record_name}_preprocessed",
            f"{cfg.mode_index}_raw_sequence_frame_index.csv",
        )
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")

        dataframe = pd.read_csv(csv_path)
        dataframe["record"] = cfg.record
        dataframe["mode"] = cfg.mode_path

        vfi_evaluator = TaskEvaluator(task_name="VFI", metric_fns=VFI_METRICS)
        dataset = VFIDataset(df=dataframe, root_dir=cfg.root_dir, input_fps=30)

        print(cfg.record, cfg.mode_name, len(dataset))
        with tqdm(range(len(dataset))) as progress:
            for index in progress:
                sample = dataset[index]
                input_data = sample["input"]
                ground_truth = sample["ground_truth"]

                img0_path = input_data["colorNoScreenUI"][0]
                img1_path = input_data["colorNoScreenUI"][1]
                img_gt_path = ground_truth["colorNoScreenUI"]
                bmv_path = ground_truth["backwardVel_Depth"]
                fmv_path = ground_truth["forwardVel_Depth"]

                img0_np, img1_np, img_gt_np = ensure_images_readable(
                    [img0_path, img1_path, img_gt_path],
                    max_retries=5,
                    delay_s=1.0,
                )
                bmv_np, _ = load_backward_velocity(bmv_path)
                fmv_np, _ = load_backward_velocity(fmv_path)

                img0 = tensorize_image(img0_np, device)
                img1 = tensorize_image(img1_np, device)
                bmv = tensorize_flow(bmv_np, device)
                fmv = tensorize_flow(fmv_np, device)
                embt = torch.tensor(1 / 2).view(1, 1, 1, 1).float().to(device)

                with torch.no_grad():
                    outputs = inference_step(model_name, model, img0, img1, embt, bmv, fmv)

                result = save_sample_fn(
                    outputs=outputs,
                    sample=sample,
                    cfg=cfg,
                    output_dir=output_dir,
                    img0_np=img0_np,
                    img1_np=img1_np,
                    imgGT_np=img_gt_np,
                    bmv=bmv,
                    fmv=fmv,
                    img0=img0,
                    img1=img1,
                    embt=embt,
                    device=device,
                    vfi_evaluator=vfi_evaluator,
                )

                if isinstance(result, dict):
                    progress.set_postfix(result)

        eval_df = vfi_evaluator.to_dataframe()
        eval_path = os.path.join(output_dir, cfg.record, f"{cfg.mode_name}_evaluation_results.csv")
        os.makedirs(os.path.dirname(eval_path), exist_ok=True)
        eval_df.to_csv(eval_path, index=False)
        print(eval_df.describe())
        print(f"Saving Evaluation Result into {eval_path}")
