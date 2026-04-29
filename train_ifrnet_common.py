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
from torch.utils.data import DataLoader

from datasets.dataset_loader import VFITrainDataset
from datasets.dataset_config import (
    TRAIN_VFX_0416_DATASET_CONFIGS,
    TEST_VFX_0416_DATASET_CONFIGS,
    iter_dataset_configs,
)
from src.gameData_loader import load_backward_velocity, load_forward_velocity
from evaluation import TaskEvaluator, VFI_METRICS
from model_registry import get_model_class, get_model_config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_lr(args):
    ratio = 0.5 * (1.0 + np.cos(args.iters / (args.epochs * args.iters_per_epoch) * math.pi))
    lr = (args.lr_start - args.lr_end) * ratio + args.lr_end
    return lr


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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


def forward_model(model_name, model, img0, img1, embt, imgt, bmv, fmv):
    if model_name == "IFRNet":
        flow = torch.cat([bmv, fmv], dim=1).float()
        return model(img0, img1, embt, imgt, flow)

    return model(img0, img1, embt, imgt, init_flow0=bmv, init_flow1=fmv)


@torch.no_grad()
def evaluate(model_name, model, loader, device):
    model.eval()
    evaluator = TaskEvaluator("VFI", VFI_METRICS)
    records = []
    pbar = tqdm(loader)

    for batch in pbar:
        img0, imgt, img1, bmv, fmv, embt, info = batch
        img0 = img0.to(device)
        img1 = img1.to(device)
        imgt = imgt.to(device)
        bmv = bmv.to(device)
        fmv = fmv.to(device)
        embt = embt.to(device)

        imgt_pred, loss_rec, loss_geo, loss_dis, up_flow0_1, up_flow1_1, up_mask_1 = forward_model(
            model_name, model, img0, img1, embt, imgt, bmv, fmv
        )

        total_loss = loss_rec + loss_geo + loss_dis
        B = imgt_pred.shape[0]

        for b in range(B):
            pred_np = (imgt_pred[b].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            gt_np = (imgt[b].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

            evaluator.evaluate(
                meta={},
                img_gt=gt_np,
                img_pred=pred_np,
                flow_1_to_0=up_flow0_1[b],
                flow_1_to_2=up_flow1_1[b],
                bmv=bmv[b],
                fmv=fmv[b],
            )

            records.append({
                "loss_rec": float(loss_rec.detach().cpu()),
                "loss_geo": float(loss_geo.detach().cpu()),
                "loss_dis": float(loss_dis.detach().cpu()),
                "loss_total": float(total_loss.detach().cpu()),
            })

            pbar.set_postfix(loss=f"Evaluate Loss {float(total_loss.detach().cpu()):.6f}")

    df = evaluator.to_dataframe()
    loss_df = pd.DataFrame(records)
    df = pd.concat([df.reset_index(drop=True), loss_df.reset_index(drop=True)], axis=1)
    return df["psnr"].mean(), df


def train(args, model_name, model, optimizer, train_loader, test_loader, device, logger):
    iters = 0
    best_psnr = args.best_psnr

    for epoch in range(args.resume_epoch, args.epochs):
        model.train()
        pbar = tqdm(train_loader)
        train_evaluator = TaskEvaluator("VFI", VFI_METRICS)
        train_loss = []

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

            imgt_pred, loss_rec, loss_geo, loss_dis, up_flow0_1, up_flow1_1, up_mask_1 = forward_model(
                model_name, model, img0, img1, embt, imgt, bmv, fmv
            )

            total_loss = loss_rec + loss_geo + loss_dis
            total_loss.backward()
            optimizer.step()

            B = imgt_pred.shape[0]
            for b in range(B):
                pred_np = (imgt_pred[b].detach().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                gt_np = (imgt[b].detach().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

                train_evaluator.evaluate(
                    meta={},
                    img_gt=gt_np,
                    img_pred=pred_np,
                    flow_1_to_0=up_flow0_1[b],
                    flow_1_to_2=up_flow1_1[b],
                    bmv=bmv[b],
                    fmv=fmv[b],
                )

                train_loss.append({
                    "loss_rec": float(loss_rec.detach().cpu()),
                    "loss_geo": float(loss_geo.detach().cpu()),
                    "loss_dis": float(loss_dis.detach().cpu()),
                    "loss_total": float(total_loss.detach().cpu()),
                })

            pbar.set_postfix(loss=f"Training Loss {float(total_loss.detach().cpu()):.6f}")
            iters += 1

        if (epoch + 1) % args.eval_interval == 0:
            val_df = train_evaluator.to_dataframe()
            val_loss_df = pd.DataFrame(train_loss)
            val_df = pd.concat([val_df.reset_index(drop=True), val_loss_df.reset_index(drop=True)], axis=1)
            val_psnr = val_df["psnr"].mean()
            val_df.to_csv(os.path.join(f"{args.output_dir}/checkpoints", f"train_epoch_{epoch+1}.csv"), index=False)
            logger.info(f"Epoch {epoch+1} Train PSNR {val_psnr}")

            test_psnr, test_df = evaluate(model_name, model, test_loader, device)
            test_df.to_csv(os.path.join(f"{args.output_dir}/checkpoints", f"test_epoch_{epoch+1}.csv"), index=False)
            logger.info(f"Epoch {epoch+1} Test PSNR {test_psnr}")

            if test_psnr > best_psnr:
                best_psnr = test_psnr
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_psnr": best_psnr,
                }, os.path.join(f"{args.output_dir}/checkpoints", "best.pth"))

        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_psnr": best_psnr,
        }, os.path.join(f"{args.output_dir}/checkpoints", "latest.pth"))


def parse_train_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="IFRNet", choices=["IFRNet", "IFRNet_Residual"])
    parser.add_argument("--root_dir", default="./datasets/data")
    parser.add_argument("--dataset_root_dir", default=TRAIN_VFX_0416_DATASET_CONFIGS["root_dir"], type=str)
    parser.add_argument("--resume_epoch", default=None, type=int)
    parser.add_argument("--epochs", default=60, type=int)
    parser.add_argument("--resume_path", default=None, type=str)
    parser.add_argument("--eval_interval", default=1, type=int)
    parser.add_argument("--lr_start", default=1e-4, type=float)
    parser.add_argument("--lr_end", default=1e-5, type=float)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--output_dir", default=None, type=str)
    return parser.parse_args(argv)


def prepare_args(args):
    config = get_model_config(args.model_name)
    args.output_dir = args.output_dir or config["default_output_dir"]
    args.resume_path = args.resume_path or config["default_resume_path"]
    args.resume_epoch = args.resume_epoch if args.resume_epoch is not None else config["default_resume_epoch"]
    return args


def main(argv=None):
    args = parse_train_args(argv)
    args = prepare_args(args)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    logger, _ = build_logger(os.path.join(args.output_dir, "logs"))
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device {device}")

    merged_df = build_merged_dataframe(args.root_dir, TRAIN_VFX_0416_DATASET_CONFIGS, only_fps=60, logger=logger)
    test_df = build_merged_dataframe(args.root_dir, TEST_VFX_0416_DATASET_CONFIGS, only_fps=60, logger=logger)
    merged_df = merged_df[merged_df["valid"] == True]

    dataset = VFITrainDataset(merged_df, args.dataset_root_dir, augment=True, input_fps=30)
    test_dataset = VFITrainDataset(test_df, args.dataset_root_dir, augment=False, input_fps=30)

    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    args.iters_per_epoch = len(train_loader)
    args.iters = args.resume_epoch * args.iters_per_epoch

    model_class = get_model_class(args.model_name)
    model = model_class().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr_start, weight_decay=0)
    args.best_psnr = 0.0

    if args.resume_path is not None and os.path.isfile(args.resume_path):
        ckpt = torch.load(args.resume_path)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        args.resume_epoch = ckpt["epoch"] + 1
        args.best_psnr = ckpt.get("best_psnr", 0.0)
        logger.info(f"Resumed from {args.resume_path}")
    elif args.model_name == "IFRNet":
        logger.info(f"Loading pretrained checkpoint from {get_model_config(args.model_name)['pretrained_checkpoint']}")
        model.load_state_dict(torch.load(get_model_config(args.model_name)["pretrained_checkpoint"]))
    else:
        logger.info("Training IFRNet_Residual from scratch")

    train(args, args.model_name, model, optimizer, train_loader, test_loader, device, logger)


if __name__ == "__main__":
    main()
