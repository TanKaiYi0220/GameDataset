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
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split  # <- add random_split

from datasets.dataset_loader import VFIDataset
from datasets.dataset_config import (
    STAIR_DATASET_CONFIG,
    TRAIN_DATASET_CONFIGS,
    iter_dataset_configs,
    TEST_DATASET_CONFIGS
)

from src.gameData_loader import load_backward_velocity, load_forward_velocity
from evaluation import TaskEvaluator, VFI_METRICS

import sys
sys.path.append("models/IFRNet")

from models.IFRNet_Residual import Model

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


# -----------------------------
# Utils
# -----------------------------
def get_lr(lr_start, lr_end, cur_iter, total_iter):
    ratio = 0.5 * (1.0 + math.cos(cur_iter / total_iter * math.pi))
    return (lr_start - lr_end) * ratio + lr_end


def set_lr(optimizer, lr):
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def to_tensor_bchw(img_np_uint8):
    x = torch.from_numpy(img_np_uint8.transpose(2, 0, 1)).float() / 255.0
    return x.unsqueeze(0)  # 1,C,H,W


def split_train_val(dataset, val_ratio: float, seed: int):
    assert 0.0 < val_ratio < 1.0
    n = len(dataset)
    val_len = max(1, int(round(n * val_ratio)))
    train_len = n - val_len
    g = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(dataset, [train_len, val_len], generator=g)
    return train_set, val_set


# -----------------------------
# Dataset Wrapper (VFIDataset -> IFRNet train tuple)
# -----------------------------
class VFITrainWrapper(Dataset):
    def __init__(self, vfi_dataset, use_flow=True, only_valid=True):
        self.vfi_dataset = vfi_dataset
        self.use_flow = use_flow

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

        embt = torch.tensor(0.5, dtype=torch.float32).view(1, 1, 1)

        if self.use_flow:
            if bmv_path is None or fmv_path is None:
                raise KeyError("use_flow=True but backwardVel_Depth/forwardVel_Depth not found in gt dict.")

            bmv, _ = load_backward_velocity(bmv_path)

            fmv, _ = load_forward_velocity(fmv_path)

            flow = torch.cat([bmv, fmv], dim=1)[0].float()  # [4,H,W]
        else:
            _, H, W = img0.shape
            flow = torch.zeros((4, H, W), dtype=torch.float32)

        # ---- FLOW SANITY CHECK (FAIL FAST) ----
        if not torch.isfinite(flow).all():
            raise RuntimeError(
                f"[FLOW INVALID] idx={real_idx}, "
                f"has NaN/Inf in backward/forward velocity"
            )

        # optional: clamp extreme flow to avoid grid_sample OOB
        flow = torch.clamp(flow, min=-500.0, max=500.0)

        return img0, imgt, img1, flow, embt, sample


# -----------------------------
# Merge multiple configs -> one dataset
# -----------------------------
def build_merged_dataset(args, dataset_cfgs, logger, only_valid=True):
    wrapped_list = []
    kept_cfgs = []

    for cfg in iter_dataset_configs(dataset_cfgs):
        if cfg.fps != args.only_fps:
            continue
        if cfg.difficulty != "Difficult":
            continue

        csv_path = f"{args.root_dir}/{cfg.record_name}_preprocessed/{cfg.mode_index}_raw_sequence_frame_index.csv"
        if not os.path.isfile(csv_path):
            logger.warning(f"CSV not found, skip: {csv_path}")
            continue

        df = pd.read_csv(csv_path)

        base_dataset = VFIDataset(
            df=df,
            root_dir=args.dataset_root_dir,
            record=cfg.record,
            mode=cfg.mode_path,
            input_fps=args.input_fps,
        )

        wrapped = VFITrainWrapper(base_dataset, use_flow=args.use_flow, only_valid=only_valid)
        if len(wrapped) == 0:
            logger.warning(f"0 valid samples, skip cfg: {cfg.record}/{cfg.mode_path}")
            continue

        wrapped_list.append(wrapped)
        kept_cfgs.append(cfg)
        logger.info(f"Added cfg: {cfg.record}/{cfg.mode_path} | samples={len(wrapped)}")

    if len(wrapped_list) == 0:
        raise RuntimeError("No configs produced a non-empty dataset. Check filters / paths.")

    merged = ConcatDataset(wrapped_list)
    logger.info(f"Merged dataset size = {len(merged)} (configs={len(wrapped_list)})")
    return merged, kept_cfgs


# -----------------------------
# Eval
# -----------------------------
@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    evaluator = TaskEvaluator(task_name="VFI", metric_fns=VFI_METRICS)
    loss_rows = []

    def _pick_loss(loss_tensor, b):
        if loss_tensor is None:
            return float("nan")
        if not torch.is_tensor(loss_tensor):
            return float(loss_tensor)
        if loss_tensor.ndim == 0:
            return float(loss_tensor.detach().cpu().item())
        if loss_tensor.ndim == 1:
            return float(loss_tensor.detach().cpu()[b].item())
        return float(loss_tensor.detach().cpu().mean().item())

    for batch in tqdm(dataloader, desc="Eval", leave=False):
        img0, imgt, img1, flow, embt, samples = batch
        img0 = img0.to(device)
        img1 = img1.to(device)
        imgt = imgt.to(device)
        flow = flow.to(device)
        embt = embt.to(device)

        init_flow0_full = flow[:, 0:2]
        init_flow1_full = flow[:, 2:4]

        imgt_pred, loss_rec, loss_geo, loss_dis, up_flow0_1, up_flow1_1, up_mask_1 = model(
            img0, img1, embt, imgt, flow,
            init_flow0_full=init_flow0_full, init_flow1_full=init_flow1_full
        )

        B = img0.shape[0]
        for b in range(B):
            pred_np = (imgt_pred[b].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
            gt_np   = (imgt[b].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)

            rec = _pick_loss(loss_rec, b)
            geo = _pick_loss(loss_geo, b)
            dis = _pick_loss(loss_dis, b)
            tot = rec + geo + dis

            meta = {
                "frame_range": samples["frame_range"][b] if isinstance(samples, dict) and "frame_range" in samples else None,
                "valid": samples["valid"][b] if isinstance(samples, dict) and "valid" in samples else None,
                "loss_rec": rec,
                "loss_geo": geo,
                "loss_dis": dis,
                "loss_total": tot,
            }

            evaluator.evaluate(
                meta=meta,
                img_gt=gt_np,
                img_pred=pred_np,
                flow_1_to_0=up_flow0_1,
                flow_1_to_2=up_flow1_1,
                bmv=flow[:, 0:2],
                fmv=flow[:, 2:4],
            )

            loss_rows.append({
                "loss_rec": rec,
                "loss_geo": geo,
                "loss_dis": dis,
                "loss_total": tot,
            })

    df = evaluator.to_dataframe()
    if df is not None and len(df) == len(loss_rows):
        for k in ["loss_rec", "loss_geo", "loss_dis", "loss_total"]:
            if k not in df.columns:
                df[k] = [r[k] for r in loss_rows]

    if df is not None and "psnr" in df.columns and len(df) > 0:
        return float(df["psnr"].mean()), df
    return float("nan"), df


# -----------------------------
# Train (train/val) + optional test
# -----------------------------
def train_with_val_and_test(args, model, optimizer, train_loader, val_loader, test_loader, device, logger, save_dir):
    total_iters = args.epochs * len(train_loader)
    iters = 0
    best_val_psnr = -1e9

    logger.info(f"Start training: train_samples={len(train_loader.dataset)}, it/epoch={len(train_loader)}")
    val_psnr0, val_df0 = evaluate(model, val_loader, device)
    logger.info(f"[VAL] epoch 0 (before FT): psnr_mean={val_psnr0:.3f}")
    if val_df0 is not None:
        val_df0.to_csv(os.path.join(save_dir, "val_epoch_0.csv"), index=False)

    if test_loader is not None:
        test_psnr0, test_df0 = evaluate(model, test_loader, device)
        logger.info(f"[TEST] epoch 0 (before FT): psnr_mean={test_psnr0:.3f}")
        if test_df0 is not None:
            test_df0.to_csv(os.path.join(save_dir, "test_epoch_0.csv"), index=False)

    if train_loader is not None:
        train_psnr0, train_df0 = evaluate(model, train_loader, device)
        logger.info(f"[TRAIN-SET] epoch 0 (before FT): psnr_mean={train_psnr0:.3f}")
        if train_df0 is not None:
            train_df0.to_csv(os.path.join(save_dir, "train_epoch_0.csv"), index=False)

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

            init_flow0_full = flow[:, 0:2]
            init_flow1_full = flow[:, 2:4]

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
            f"[TRAIN] epoch {epoch+1}/{args.epochs} "
            f"loss_rec={loss_rec_avg/len(train_loader):.4e} "
            f"loss_geo={loss_geo_avg/len(train_loader):.4e} "
            f"loss_dis={loss_dis_avg/len(train_loader):.4e}"
        )

        torch.save(model.state_dict(), os.path.join(save_dir, "latest.pth"))

        if (epoch + 1) % args.eval_interval == 0:
            # ---- Validation ----
            val_psnr, val_df = evaluate(model, val_loader, device)
            logger.info(f"[VAL] epoch {epoch+1}: psnr_mean={val_psnr:.3f}")
            if val_df is not None:
                val_df.to_csv(os.path.join(save_dir, f"val_epoch_{epoch+1}.csv"), index=False)

            # ---- Save best by VAL ----
            if not math.isnan(val_psnr) and val_psnr > best_val_psnr:
                best_val_psnr = val_psnr
                torch.save(model.state_dict(), os.path.join(save_dir, "best.pth"))
                logger.info(f"New best VAL PSNR={best_val_psnr:.3f} -> saved best.pth")

            if test_loader is not None:
                test_psnr, test_df = evaluate(model, test_loader, device)
                logger.info(f"[TEST] epoch {epoch+1} (best-val checkpoint): psnr_mean={test_psnr:.3f}")
                if test_df is not None:
                    test_df.to_csv(os.path.join(save_dir, f"test_epoch_{epoch+1}.csv"), index=False)
            
            if train_loader is not None:
                train_psnr, train_df = evaluate(model, train_loader, device)
                logger.info(f"[TRAIN-SET] epoch {epoch+1} (best-val checkpoint): psnr_mean={train_psnr:.3f}")
                if train_df is not None:
                    train_df.to_csv(os.path.join(save_dir, f"train_epoch_{epoch+1}.csv"), index=False)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="IFRNet", type=str)
    parser.add_argument("--root_dir", default="./datasets/data/", type=str)
    parser.add_argument("--dataset_root_dir", default=STAIR_DATASET_CONFIG["root_dir"], type=str)

    parser.add_argument("--output_dir", default="./output/IFRNet_R_0228_60/", type=str)
    # parser.add_argument("--resume_path", default=None, type=str)
    # parser.add_argument("--resume_path", default="./models/IFRNet/checkpoints/IFRNet/IFRNet_Vimeo90K.pth", type=str)
    parser.add_argument("--resume_path", default="./output/IFRNet_R_0228_30/checkpoints/IFRNet/merged_fps60_Difficult/best.pth", type=str)


    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--eval_batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=0, type=int)

    parser.add_argument("--lr_start", default=1e-4, type=float)
    parser.add_argument("--lr_end", default=1e-5, type=float)
    parser.add_argument("--eval_interval", default=1, type=int)

    parser.add_argument("--input_fps", default=30, type=int)
    parser.add_argument("--only_fps", default=60, type=int)

    parser.add_argument("--use_flow", default=True, type=bool)

    # ✅ new: split + test behavior
    parser.add_argument("--val_ratio", default=0.1, type=float)
    parser.add_argument("--split_seed", default=1234, type=int)
    parser.add_argument("--eval_test_on_best", action="store_true", help="run test only when val best improves")
    parser.add_argument("--eval_test_at_start", action="store_true", help="also eval test at epoch 0")

    args = parser.parse_args()
    logger, log_dir = build_logger(os.path.join(args.output_dir, "logs"))

    # ---- Deterministic / Debug mode ----
    random.seed(args.split_seed)
    np.random.seed(args.split_seed)
    torch.manual_seed(args.split_seed)
    torch.cuda.manual_seed_all(args.split_seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Args: {args}")

    # -----------------------------
    # TRAIN/VAL: build merged dataset then split 90/10
    # -----------------------------
    merged_dataset, kept_cfgs = build_merged_dataset(args, TRAIN_DATASET_CONFIGS, logger, only_valid=True)
    train_set, val_set = split_train_val(merged_dataset, args.val_ratio, args.split_seed)
    logger.info(f"Split: train={len(train_set)} val={len(val_set)} (val_ratio={args.val_ratio})")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
    )

    # -----------------------------
    # TEST: build from TEST_DATASET_CONFIG(S)
    # -----------------------------
    test_dataset, _ = build_merged_dataset(args, TEST_DATASET_CONFIGS, logger, only_valid=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
    )
    logger.info(f"Test dataset size = {len(test_dataset)}")

    # -----------------------------
    # Train
    # -----------------------------
    model = Model().to(device)
    if args.resume_path is not None and os.path.isfile(args.resume_path):
        model.load_state_dict(torch.load(args.resume_path))
        logger.info(f"Resumed from {args.resume_path}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr_start, weight_decay=0.0)

    save_dir = os.path.join(
        args.output_dir,
        "checkpoints",
        args.model_name,
        f"merged_fps{args.only_fps}_Difficult"
    )
    os.makedirs(save_dir, exist_ok=True)

    train_with_val_and_test(args, model, optimizer, train_loader, val_loader, test_loader, device, logger, save_dir)


if __name__ == "__main__":
    main()