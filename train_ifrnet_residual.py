import os
import time
import math
import argparse
import random
import logging

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from datasets.dataset_loader import VFIDataset
from datasets.dataset_config import (
    STAIR_DATASET_CONFIG,
    iter_dataset_configs,
)

from src.gameData_loader import load_backward_velocity, load_forward_velocity
from evaluation import TaskEvaluator, VFI_METRICS

import sys
sys.path.append("models/IFRNet")

from models.IFRNet_Residual import Model


# -----------------------------
# Utils
# -----------------------------
def get_lr(lr_start, lr_end, cur_iter, total_iter):
    # cosine decay
    ratio = 0.5 * (1.0 + math.cos(cur_iter / total_iter * math.pi))
    return (lr_start - lr_end) * ratio + lr_end


def set_lr(optimizer, lr):
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def to_tensor_bchw(img_np_uint8):
    # cv2.imread -> HWC BGR uint8
    x = torch.from_numpy(img_np_uint8.transpose(2, 0, 1)).float() / 255.0
    return x.unsqueeze(0)  # 1,C,H,W


def make_flow_tensor(bmv, fmv):
    """
    你最需要依照實際資料改的地方。

    假設：
    - bmv / fmv : (H, W, >=2) 或 (H, W, 2)
    - 前兩個 channel 是 (dx, dy)
    - IFRNet training 需要 flow = concat(flow_t->0, flow_t->1) => (H, W, 4)
    回傳: torch float tensor shape [1, 4, H, W]
    """
    if bmv.ndim == 2:
        raise ValueError("bmv looks like 2D array; expected HxWxC.")

    if fmv.ndim == 2:
        raise ValueError("fmv looks like 2D array; expected HxWxC.")

    bmv_xy = bmv[..., :2]
    fmv_xy = fmv[..., :2]
    flow = np.concatenate([bmv_xy, fmv_xy], axis=-1).astype(np.float32)  # H,W,4
    flow = torch.from_numpy(flow).permute(2, 0, 1).unsqueeze(0)  # 1,4,H,W
    return flow


# -----------------------------
# Dataset Wrapper (VFIDataset -> IFRNet train tuple)
# -----------------------------
class VFITrainWrapper(Dataset):
    def __init__(self, vfi_dataset, use_flow=True, only_valid=True):
        self.vfi_dataset = vfi_dataset
        self.use_flow = use_flow

        # --- method 1: filter valid=True at init ---
        if only_valid:
            self.indices = [i for i in range(len(vfi_dataset)) if vfi_dataset[i]["valid"]]
        else:
            self.indices = list(range(len(vfi_dataset)))

        print(f"[VFITrainWrapper] only_valid={only_valid}, kept {len(self.indices)}/{len(vfi_dataset)} samples")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        sample = self.vfi_dataset[real_idx]
        input_ = sample["input"]
        gt = sample["ground_truth"]

        img0_path = input_["colorNoScreenUI"][0]
        img1_path = input_["colorNoScreenUI"][1]
        imgt_path = gt["colorNoScreenUI"]

        bmv_path = gt.get("backwardVel_Depth", None)
        fmv_path = gt.get("forwardVel_Depth", None)

        img0_np = cv2.imread(img0_path)
        img1_np = cv2.imread(img1_path)
        imgt_np = cv2.imread(imgt_path)

        failed_counter = 0
        while img0_np is None or img1_np is None or imgt_np is None:
            img0_np = cv2.imread(img0_path)
            img1_np = cv2.imread(img1_path)
            imgt_np = cv2.imread(imgt_path)

            failed_counter += 1
            if failed_counter > 5:
                raise FileNotFoundError(f"Failed to read images at idx={real_idx}")

        img0 = to_tensor_bchw(img0_np)[0]  # [3,H,W]
        img1 = to_tensor_bchw(img1_np)[0]
        imgt = to_tensor_bchw(imgt_np)[0]

        # embt per-sample must be [1,1,1] so batch -> [B,1,1,1]
        embt = torch.tensor(0.5, dtype=torch.float32).view(1, 1, 1)

        # flow per-sample must be [4,H,W] so batch -> [B,4,H,W]
        if self.use_flow:
            if bmv_path is None or fmv_path is None:
                raise KeyError("use_flow=True but backwardVel_Depth/forwardVel_Depth not found in gt dict.")

            bmv, _ = load_backward_velocity(bmv_path)
            fmv, _ = load_forward_velocity(fmv_path)

            flow = make_flow_tensor(bmv, fmv)  # [4,H,W] (no [0]!)
        else:
            _, H, W = img0.shape
            flow = torch.zeros((4, H, W), dtype=torch.float32)

        return img0, imgt, img1, flow, embt, sample


# -----------------------------
# Train / Eval
# -----------------------------
@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    evaluator = TaskEvaluator(task_name="VFI", metric_fns=VFI_METRICS)

    for batch in tqdm(dataloader, desc="Eval", leave=False):
        img0, imgt, img1, flow, embt, samples = batch
        img0 = img0.to(device)
        img1 = img1.to(device)
        imgt = imgt.to(device)
        flow = flow.to(device)
        embt = embt.to(device)

        # IFRNet training forward (same signature as your DDP training code)
        imgt_pred, loss_rec, loss_geo, loss_dis, up_flow0_1, up_flow1_1, up_mask_1 = model(img0, img1, embt, imgt, flow)

        # per-sample metric
        for b in range(img0.shape[0]):
            # convert to uint8 for evaluator (keep consistent with your original pipeline)
            pred_np = (imgt_pred[b].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
            gt_np = (imgt[b].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)

            meta = {
                "frame_range": samples["frame_range"][b] if isinstance(samples, dict) and "frame_range" in samples else None,
                "valid": samples["valid"][b] if isinstance(samples, dict) and "valid" in samples else None,
            }

            evaluator.evaluate(
                meta=meta,
                img_gt=gt_np,
                img_pred=pred_np,
                # evaluator 也需要 flow_* 的話，你可改成用 model.inference() 取出 up_flow/up_mask
                flow_1_to_0=up_flow0_1,
                flow_1_to_2=up_flow1_1,
                bmv=flow[:, 0:2],
                fmv=flow[:, 2:4],
            )

    df = evaluator.to_dataframe()
    # 以 psnr mean 當主要指標（你 evaluator 裡通常會有 psnr 欄位）
    if "psnr" in df.columns and len(df) > 0:
        return float(df["psnr"].mean()), df
    return float("nan"), df


def train_one_cfg(args, cfg, device, logger):
    df = pd.read_csv(f"{args.root_dir}/{cfg.record_name}_preprocessed/{cfg.mode_index}_raw_sequence_frame_index.csv")

    base_dataset = VFIDataset(
        df=df,
        root_dir=args.dataset_root_dir,
        record=cfg.record,
        mode=cfg.mode_path,
        input_fps=args.input_fps,
    )

    train_dataset = VFITrainWrapper(base_dataset, use_flow=args.use_flow)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # (可選) eval loader：這裡先用同一份資料做 quick sanity check
    eval_loader = DataLoader(
        train_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = Model().to(device)
    if args.resume_path is not None and os.path.isfile(args.resume_path):
        model.load_state_dict(torch.load(args.resume_path, map_location="cpu"))
        logger.info(f"Resumed from {args.resume_path}")

    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr_start, weight_decay=0.0)

    save_dir = os.path.join(args.output_dir, "checkpoints", args.model_name, cfg.record, cfg.mode_path)
    os.makedirs(save_dir, exist_ok=True)

    total_iters = args.epochs * len(train_loader)
    iters = 0
    best_psnr = -1e9

    logger.info(f"Start training: {cfg.record}/{cfg.mode_path}, samples={len(train_dataset)}, it/epoch={len(train_loader)}")

    for epoch in range(args.epochs):
        model.train()
        loss_rec_avg = 0.0
        loss_geo_avg = 0.0
        loss_dis_avg = 0.0

        pbar = tqdm(train_loader, desc=f"Train E{epoch+1}/{args.epochs}", leave=False)
        for batch in pbar:
            img0, imgt, img1, flow, embt, _samples = batch
            img0 = img0.to(device)
            img1 = img1.to(device)
            imgt = imgt.to(device)
            flow = flow.to(device)
            embt = embt.to(device)

            lr = get_lr(args.lr_start, args.lr_end, iters, total_iters)
            set_lr(optimizer, lr)

            optimizer.zero_grad(set_to_none=True)
            
            init_flow0_full = flow[:, 0:2]  # [B,2,H,W]
            init_flow1_full = flow[:, 2:4]  # [B,2,H,W]

            imgt_pred, loss_rec, loss_geo, loss_dis, up_flow0_1, up_flow1_1, up_mask_1 = model(
                img0, img1, embt, imgt, flow,
                init_flow0_full=init_flow0_full, init_flow1_full=init_flow1_full
            )
            loss = loss_rec + loss_geo + loss_dis
            loss.backward()
            optimizer.step()

            loss_rec_avg += float(loss_rec.detach().cpu())
            loss_geo_avg += float(loss_geo.detach().cpu())
            loss_dis_avg += float(loss_dis.detach().cpu())

            iters += 1
            pbar.set_postfix({
                "lr": f"{lr:.2e}",
                "rec": f"{loss_rec_avg/(pbar.n+1):.3e}",
                "geo": f"{loss_geo_avg/(pbar.n+1):.3e}",
                "dis": f"{loss_dis_avg/(pbar.n+1):.3e}",
            })

        logger.info(
            f"[{cfg.record}/{cfg.mode_path}] "
            f"epoch {epoch+1}/{args.epochs} "
            f"loss_rec={loss_rec_avg/len(train_loader):.4e} "
            f"loss_geo={loss_geo_avg/len(train_loader):.4e} "
            f"loss_dis={loss_dis_avg/len(train_loader):.4e}"
        )

        # save latest
        torch.save(model.state_dict(), os.path.join(save_dir, "latest.pth"))

        # eval
        if (epoch + 1) % args.eval_interval == 0:
            psnr_mean, eval_df = evaluate(model, eval_loader, device)
            logger.info(f"[{cfg.record}/{cfg.mode_path}] eval epoch {epoch+1}: psnr_mean={psnr_mean:.3f}")
            eval_df.to_csv(os.path.join(save_dir, f"eval_epoch_{epoch+1}.csv"), index=False)

            if not math.isnan(psnr_mean) and psnr_mean > best_psnr:
                best_psnr = psnr_mean
                torch.save(model.state_dict(), os.path.join(save_dir, "best.pth"))
                logger.info(f"New best PSNR={best_psnr:.3f} -> saved best.pth")


def build_logger(log_root):
    os.makedirs(log_root, exist_ok=True)
    run_dir = os.path.join(log_root, time.strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(run_dir, exist_ok=True)

    logger = logging.getLogger("IFRNetSingleTrain")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")

    # console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # file
    fh = logging.FileHandler(os.path.join(run_dir, "train.log"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info(f"Log dir: {run_dir}")
    return logger, run_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="IFRNet", type=str)
    parser.add_argument("--root_dir", default="./datasets/data/", type=str)  # your ROOT_DIR
    parser.add_argument("--dataset_root_dir", default=STAIR_DATASET_CONFIG["root_dir"], type=str)
    parser.add_argument("--output_dir", default="./output/IFRNet_Residual/", type=str)
    parser.add_argument("--resume_path", default=None, type=str)

    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--eval_batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=2, type=int)

    parser.add_argument("--lr_start", default=1e-4, type=float)
    parser.add_argument("--lr_end", default=1e-5, type=float)
    parser.add_argument("--eval_interval", default=1, type=int)

    parser.add_argument("--input_fps", default=30, type=int)
    parser.add_argument("--only_fps", default=60, type=int)

    parser.add_argument("--use_flow", action="store_true", help="use bmv/fmv to build flow supervision")

    args = parser.parse_args()
    logger, log_dir = build_logger(os.path.join(args.output_dir, "logs"))

    # Reproducibility
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Args: {args}")

    # Train over configs
    dataset_cfg = STAIR_DATASET_CONFIG
    for cfg in iter_dataset_configs(dataset_cfg):
        if cfg.fps != args.only_fps:
            continue
        if cfg.difficulty != "Difficult":
            continue
        train_one_cfg(args, cfg, device, logger)


if __name__ == "__main__":
    main()
