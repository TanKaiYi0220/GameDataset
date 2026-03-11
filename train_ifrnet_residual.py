import os
import argparse
import logging
import random
import time
import math

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from datasets.dataset_loader import VFITrainDataset
from datasets.dataset_config import (
    iter_dataset_configs,
    TRAIN_DATASET_CONFIGS,
    TEST_DATASET_CONFIGS,
    STAIR_DATASET_CONFIG,
    VFX_DATASET_CONFIGS,
)

from src.gameData_loader import load_backward_velocity, load_forward_velocity
from evaluation import TaskEvaluator, VFI_METRICS

import sys
sys.path.append("models/IFRNet")

from models.IFRNet_Residual import Model


# -------------------------------------------------
# Utils
# -------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_tensor(img):
    return torch.from_numpy(img.transpose(2,0,1)).float() / 255


def get_lr(args):
    ratio = 0.5 * (1.0 + np.cos(args.iters / (args.epochs * args.iters_per_epoch) * math.pi))
    lr = (args.lr_start - args.lr_end) * ratio + args.lr_end
    return lr


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# -------------------------------------------------
# Dataset builder
# -------------------------------------------------
def build_merged_dataframe(root_dir, dataset_cfgs, only_fps, logger):
    dfs = []
    for cfg in iter_dataset_configs(dataset_cfgs):
        if cfg.fps != only_fps:
            continue

        csv_path = f"{root_dir}/{cfg.record_name}_preprocessed/{cfg.mode_index}_raw_sequence_frame_index.csv"

        if not os.path.isfile(csv_path):
            logger.warning(f"CSV missing: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        df["record"] = cfg.record
        df["mode"] = cfg.mode_path
        dfs.append(df)

        logger.info(f"Loaded {cfg.record}/{cfg.mode_path} rows={len(df)}")

    if len(dfs) == 0:
        raise RuntimeError("No dataset found")

    merged = pd.concat(dfs, ignore_index=True)
    logger.info(f"Merged dataset size = {len(merged)}")
    return merged


# -------------------------------------------------
# Evaluation
# -------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    evaluator = TaskEvaluator("VFI", VFI_METRICS)

    for batch in tqdm(loader, leave=False):
        img0, imgt, img1, bmv, fmv, embt, info = batch
        img0 = img0.to(device)
        img1 = img1.to(device)
        imgt = imgt.to(device)
        bmv = bmv.to(device)
        fmv = fmv.to(device)
        embt = embt.to(device)

        imgt_pred, loss_rec, loss_geo, loss_dis, up_flow0_1, up_flow1_1, up_mask_1 = model(
            img0, img1, embt, imgt,
            init_flow0=fmv, init_flow1=bmv
        )

        B = imgt_pred.shape[0]

        for b in range(B):
            pred_np = (imgt_pred[b].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
            gt_np = (imgt[b].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)

            evaluator.evaluate(
                meta={},
                img_gt=gt_np,
                img_pred=pred_np,
                flow_1_to_0=up_flow0_1[b],
                flow_1_to_2=up_flow1_1[b],
                bmv=bmv[b],
                fmv=fmv[b]
            )

    df = evaluator.to_dataframe()
    return df["psnr"].mean(), df

# -------------------------------------------------
# Train
# -------------------------------------------------

def train(args, model, train_loader, val_loader, test_loader, device, logger):
    optimizer = optim.AdamW(model.parameters(), lr=args.lr_start, weight_decay=0)

    iters = 0
    best_psnr = 0.0

    for epoch in range(args.resume_epoch, args.epochs):
        model.train()

        pbar = tqdm(train_loader)

        for batch in pbar:
            img0, imgt, img1, bmv, fmv, embt, info = batch
            img0 = img0.to(device)
            img1 = img1.to(device)
            imgt = imgt.to(device)
            bmv = bmv.to(device)
            fmv = fmv.to(device)
            embt = embt.to(device)

            lr = get_lr(args)
            set_lr(optimizer, lr)
            optimizer.zero_grad()

            imgt_pred, loss_rec, loss_geo, loss_dis, up_flow0_1, up_flow1_1, up_mask_1 = model(
                img0, img1, embt, imgt,
                init_flow0=fmv, init_flow1=bmv
            )

            loss = loss_rec + loss_geo + loss_dis
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=float(loss))

            iters += 1

        if (epoch + 1) % args.eval_interval == 0:
            psnr, val_df = evaluate(model, val_loader, device)
            val_df.to_csv(os.path.join(f"{args.output_dir}/checkpoints", f"val_epoch_{epoch+1}.csv"), index=False)
            logger.info(f"Epoch {epoch+1} Validation PSNR {psnr}")

            test_psnr, test_df = evaluate(model, test_loader, device)
            test_df.to_csv(os.path.join(f"{args.output_dir}/checkpoints", f"test_epoch_{epoch+1}.csv"), index=False)
            logger.info(f"Epoch {epoch+1} Test PSNR {test_psnr}")

            if psnr > best_psnr:
                best_psnr = psnr
                torch.save(model.state_dict(), os.path.join(f"{args.output_dir}/checkpoints", "best.pth"))

# -----------------------------
# Logger
# -----------------------------
def build_logger(log_root):
    os.makedirs(log_root, exist_ok=True)
    run_dir = os.path.join(log_root, time.strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(run_dir, exist_ok=True)

    logger = logging.getLogger("IFRNetTrainValTest")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(os.path.join(run_dir, "train.log"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info(f"Log dir: {run_dir}")
    return logger, run_dir

# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_dir", default="./datasets/data")
    parser.add_argument("--dataset_root_dir", default=STAIR_DATASET_CONFIG["root_dir"], type=str)

    parser.add_argument("--resume_epoch", default=0, type=int)
    parser.add_argument("--epochs", default=90, type=int)
    parser.add_argument("--resume_path", default=None, type=str)
    # parser.add_argument("--resume_path", default="./output/IFRNet_Residual_Small_Cropping_30/checkpoints/best.pth", type=str)
    parser.add_argument("--eval_interval", default=1, type=int)

    parser.add_argument("--lr_start", default=1e-4, type=float)
    parser.add_argument("--lr_end", default=1e-5, type=float)

    parser.add_argument("--val_ratio", default=0.1, type=float)

    parser.add_argument("--seed", default=1234, type=int)

    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--output_dir", default="./output/IFRNet_Residual_Small_Cropping_Full", type=str)



    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    logger, log_dir = build_logger(os.path.join(args.output_dir, "logs"))

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Device {device}")

    merged_df = build_merged_dataframe(
        args.root_dir,
        TRAIN_DATASET_CONFIGS,
        only_fps=60,
        logger=logger
    )

    test_df = build_merged_dataframe(
        args.root_dir,
        TEST_DATASET_CONFIGS,
        only_fps=60,
        logger=logger
    )

    merged_df = merged_df[merged_df["valid"] == True]

    dataset = VFITrainDataset(
        merged_df,
        args.dataset_root_dir,
        augment=True,
        input_fps=30,
    )

    test_dataset = VFITrainDataset(
        test_df,
        args.dataset_root_dir,
        augment=False,
        input_fps=30,
    )

    train_len = int(len(dataset)*(1-args.val_ratio))
    val_len = len(dataset)-train_len

    train_set, val_set = random_split(dataset,[train_len,val_len])

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False
    )

    args.iters_per_epoch = train_loader.__len__()
    args.iters = args.resume_epoch * args.iters_per_epoch

    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False
    )

    model = Model().to(device)

    if args.resume_path is not None and os.path.isfile(args.resume_path):
        model.load_state_dict(torch.load(args.resume_path))
        logger.info(f"Resumed from {args.resume_path}")

    train(
        args,
        model,
        train_loader,
        val_loader,
        test_loader,
        device,
        logger
    )


if __name__ == "__main__":
    main()