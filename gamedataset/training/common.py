import argparse
import logging
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from gamedataset.data.config import (
    TEST_VFX_0416_DATASET_CONFIGS,
    TRAIN_VFX_0416_DATASET_CONFIGS,
    iter_dataset_configs,
)
from gamedataset.data.loader import VFITrainDataset
from gamedataset.evaluation import TaskEvaluator, VFI_METRICS
from gamedataset.models.registry import get_model_class, get_model_config


@dataclass(frozen=True)
class TrainingState:
    start_epoch: int
    global_step: int
    best_psnr: float
    resume_path: str | None
    mode: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_lr(args: argparse.Namespace, step: int) -> float:
    total_steps = max(args.epochs * args.iters_per_epoch, 1)
    ratio = 0.5 * (1.0 + np.cos(step / total_steps * math.pi))
    return (args.lr_start - args.lr_end) * ratio + args.lr_end


def set_lr(optimizer: optim.Optimizer, lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def build_merged_dataframe(
    root_dir: str,
    dataset_cfgs: dict,
    only_fps: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    dataframe_list: list[pd.DataFrame] = []

    for cfg in iter_dataset_configs(dataset_cfgs):
        if cfg.fps != only_fps:
            continue

        csv_path = os.path.join(
            root_dir,
            f"{cfg.record_name}_preprocessed",
            f"{cfg.mode_index}_raw_sequence_frame_index.csv",
        )
        if not os.path.isfile(csv_path):
            logger.warning("Dataset CSV missing", extra={"csv_path": csv_path})
            continue

        dataframe = pd.read_csv(csv_path)
        dataframe["record"] = cfg.record
        dataframe["mode"] = cfg.mode_path
        dataframe_list.append(dataframe)
        logger.info("Loaded dataset CSV %s rows=%s", csv_path, len(dataframe))

    if len(dataframe_list) == 0:
        raise RuntimeError(f"No dataset CSV found under root_dir={root_dir}")

    merged = pd.concat(dataframe_list, ignore_index=True)
    logger.info("Merged dataset size=%s", len(merged))
    return merged


def build_logger(log_root: str) -> tuple[logging.Logger, str]:
    os.makedirs(log_root, exist_ok=True)
    run_dir = os.path.join(log_root, time.strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(run_dir, exist_ok=True)

    logger = logging.getLogger("IFRNetTrainValTest")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(os.path.join(run_dir, "train.log"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("Log dir: %s", run_dir)
    return logger, run_dir


def forward_model(
    model_name: str,
    model: torch.nn.Module,
    img0: torch.Tensor,
    img1: torch.Tensor,
    embt: torch.Tensor,
    imgt: torch.Tensor,
    bmv: torch.Tensor,
    fmv: torch.Tensor,
):
    if model_name == "IFRNet":
        flow = torch.cat([bmv, fmv], dim=1).float()
        return model(img0, img1, embt, imgt, flow)

    return model(img0, img1, embt, imgt, init_flow0=bmv, init_flow1=fmv)


def build_loss_record(
    loss_rec: torch.Tensor,
    loss_geo: torch.Tensor,
    loss_dis: torch.Tensor,
    total_loss: torch.Tensor,
) -> dict[str, float]:
    return {
        "loss_rec": float(loss_rec.detach().cpu()),
        "loss_geo": float(loss_geo.detach().cpu()),
        "loss_dis": float(loss_dis.detach().cpu()),
        "loss_total": float(total_loss.detach().cpu()),
    }


def resolve_path(path_value: str) -> Path:
    return Path(path_value).expanduser().resolve()


def read_resume_start_epoch(checkpoint_path: Path) -> int | None:
    if not checkpoint_path.is_file():
        return None

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    epoch = checkpoint.get("epoch")
    if epoch is None:
        return None
    return int(epoch) + 1


def build_unique_output_dir(base_dir: Path) -> str:
    if not base_dir.exists():
        return str(base_dir)

    suffix_index = 1
    while True:
        candidate = base_dir.parent / f"{base_dir.name}_{suffix_index:02d}"
        if not candidate.exists():
            return str(candidate)
        suffix_index += 1


def build_resume_output_dir(resume_path: str, start_epoch: int | None) -> str:
    checkpoint_path = resolve_path(resume_path)
    if checkpoint_path.parent.name != "checkpoints":
        raise ValueError(
            f"resume_path must point inside a checkpoints directory, got {checkpoint_path}"
        )

    source_output_dir = checkpoint_path.parent.parent
    checkpoint_name = checkpoint_path.stem
    resume_epoch_label = f"e{start_epoch}" if start_epoch is not None else "resume"
    base_dir = source_output_dir.parent / f"{source_output_dir.name}_{checkpoint_name}_{resume_epoch_label}"
    return build_unique_output_dir(base_dir)


def resolve_resume_path(user_resume_path: str | None, default_resume_path: str | None) -> str | None:
    if user_resume_path is not None:
        resume_path = resolve_path(user_resume_path)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        return str(resume_path)

    if default_resume_path is None:
        return None

    resume_path = resolve_path(default_resume_path)
    if resume_path.is_file():
        return str(resume_path)
    return None


def resolve_output_dir(
    output_dir: str | None,
    resume_path: str | None,
    default_output_dir: str,
) -> tuple[str, str]:
    if output_dir is not None:
        return str(resolve_path(output_dir)), "user"

    if resume_path is not None:
        start_epoch = read_resume_start_epoch(resolve_path(resume_path))
        return build_resume_output_dir(resume_path, start_epoch), "auto_resume"

    return build_unique_output_dir(resolve_path(default_output_dir)), "auto_fresh"


def require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def save_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    best_psnr: float,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_psnr": best_psnr,
        },
        checkpoint_path,
    )


@torch.no_grad()
def evaluate(
    model_name: str,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, pd.DataFrame]:
    model.eval()
    evaluator = TaskEvaluator("VFI", VFI_METRICS)
    records: list[dict[str, float]] = []
    progress = tqdm(loader)

    for batch in progress:
        img0, imgt, img1, bmv, fmv, embt, info = batch
        img0 = img0.to(device)
        img1 = img1.to(device)
        imgt = imgt.to(device)
        bmv = bmv.to(device)
        fmv = fmv.to(device)
        embt = embt.to(device)

        imgt_pred, loss_rec, loss_geo, loss_dis, up_flow0_1, up_flow1_1, up_mask_1 = forward_model(
            model_name,
            model,
            img0,
            img1,
            embt,
            imgt,
            bmv,
            fmv,
        )

        total_loss = loss_rec + loss_geo + loss_dis
        loss_record = build_loss_record(loss_rec, loss_geo, loss_dis, total_loss)
        batch_size = imgt_pred.shape[0]

        for batch_index in range(batch_size):
            pred_np = (imgt_pred[batch_index].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            gt_np = (imgt[batch_index].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

            evaluator.evaluate(
                meta={},
                img_gt=gt_np,
                img_pred=pred_np,
                flow_1_to_0=up_flow0_1[batch_index],
                flow_1_to_2=up_flow1_1[batch_index],
                bmv=bmv[batch_index],
                fmv=fmv[batch_index],
            )
            records.append(loss_record)

        progress.set_postfix(loss=f"Evaluate Loss {loss_record['loss_total']:.6f}")

    dataframe = evaluator.to_dataframe()
    loss_df = pd.DataFrame(records)
    dataframe = pd.concat([dataframe.reset_index(drop=True), loss_df.reset_index(drop=True)], axis=1)
    return float(dataframe["psnr"].mean()), dataframe


def train(
    args: argparse.Namespace,
    model_name: str,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    logger: logging.Logger,
    training_state: TrainingState,
) -> None:
    best_psnr = training_state.best_psnr
    global_step = training_state.global_step

    for epoch in range(training_state.start_epoch, args.epochs):
        model.train()
        progress = tqdm(train_loader)
        train_evaluator = TaskEvaluator("VFI", VFI_METRICS)
        train_loss_records: list[dict[str, float]] = []

        for batch in progress:
            img0, imgt, img1, bmv, fmv, embt, info = batch
            img0 = img0.to(device)
            img1 = img1.to(device)
            imgt = imgt.to(device)
            bmv = bmv.to(device)
            fmv = fmv.to(device)
            embt = embt.to(device)

            lr = get_lr(args, global_step)
            set_lr(optimizer, lr)
            optimizer.zero_grad()

            imgt_pred, loss_rec, loss_geo, loss_dis, up_flow0_1, up_flow1_1, up_mask_1 = forward_model(
                model_name,
                model,
                img0,
                img1,
                embt,
                imgt,
                bmv,
                fmv,
            )

            total_loss = loss_rec + loss_geo + loss_dis
            total_loss.backward()
            optimizer.step()

            loss_record = build_loss_record(loss_rec, loss_geo, loss_dis, total_loss)
            batch_size = imgt_pred.shape[0]

            for batch_index in range(batch_size):
                pred_np = (imgt_pred[batch_index].detach().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                gt_np = (imgt[batch_index].detach().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

                train_evaluator.evaluate(
                    meta={},
                    img_gt=gt_np,
                    img_pred=pred_np,
                    flow_1_to_0=up_flow0_1[batch_index],
                    flow_1_to_2=up_flow1_1[batch_index],
                    bmv=bmv[batch_index],
                    fmv=fmv[batch_index],
                )
                train_loss_records.append(loss_record)

            progress.set_postfix(
                loss=f"Training Loss {loss_record['loss_total']:.6f}",
                lr=f"{lr:.2e}",
            )
            global_step += 1

        if (epoch + 1) % args.eval_interval == 0:
            train_df = train_evaluator.to_dataframe()
            train_loss_df = pd.DataFrame(train_loss_records)
            train_df = pd.concat([train_df.reset_index(drop=True), train_loss_df.reset_index(drop=True)], axis=1)
            train_psnr = float(train_df["psnr"].mean())
            train_csv_path = os.path.join(args.output_dir, "checkpoints", f"train_epoch_{epoch + 1}.csv")
            train_df.to_csv(train_csv_path, index=False)
            logger.info("Epoch %s Train PSNR %.6f", epoch + 1, train_psnr)

            test_psnr, test_df = evaluate(model_name, model, test_loader, device)
            test_csv_path = os.path.join(args.output_dir, "checkpoints", f"test_epoch_{epoch + 1}.csv")
            test_df.to_csv(test_csv_path, index=False)
            logger.info("Epoch %s Test PSNR %.6f", epoch + 1, test_psnr)

            if test_psnr > best_psnr:
                best_psnr = test_psnr
                save_checkpoint(
                    os.path.join(args.output_dir, "checkpoints", "best.pth"),
                    model,
                    optimizer,
                    epoch,
                    best_psnr,
                )

        save_checkpoint(
            os.path.join(args.output_dir, "checkpoints", "latest.pth"),
            model,
            optimizer,
            epoch,
            best_psnr,
        )


def parse_train_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train IFRNet variants on the VFI dataset.")
    parser.add_argument("--model-name", default="IFRNet", choices=["IFRNet", "IFRNet_Residual"])
    parser.add_argument("--root_dir", default="./datasets/data", help="Directory containing preprocessed CSV indexes.")
    parser.add_argument(
        "--dataset_root_dir",
        default=TRAIN_VFX_0416_DATASET_CONFIGS["root_dir"],
        type=str,
        help="Root directory containing frame and velocity assets.",
    )
    parser.add_argument("--resume_epoch", default=None, type=int, help="Optional sanity check for the next epoch when resuming.")
    parser.add_argument("--epochs", default=60, type=int, help="Total number of epochs to run.")
    parser.add_argument("--resume_path", default=None, type=str, help="Checkpoint to resume from. Leave empty to use the model default if it exists.")
    parser.add_argument("--eval_interval", default=1, type=int, help="Run validation every N epochs.")
    parser.add_argument("--lr_start", default=1e-4, type=float, help="Initial learning rate.")
    parser.add_argument("--lr_end", default=1e-5, type=float, help="Final learning rate after cosine decay.")
    parser.add_argument("--seed", default=1234, type=int, help="Random seed.")
    parser.add_argument("--batch_size", default=8, type=int, help="Training batch size.")
    parser.add_argument("--output_dir", default=None, type=str, help="Output directory for checkpoints and logs.")
    return parser.parse_args(argv)


def prepare_args(args: argparse.Namespace) -> argparse.Namespace:
    model_config = get_model_config(args.model_name)
    args.resume_path = resolve_resume_path(args.resume_path, model_config["default_resume_path"])
    args.output_dir, args.output_dir_reason = resolve_output_dir(
        args.output_dir,
        args.resume_path,
        model_config["default_output_dir"],
    )

    return args


def validate_args(args: argparse.Namespace) -> None:
    require_positive("epochs", args.epochs)
    require_positive("eval_interval", args.eval_interval)
    require_positive("batch_size", args.batch_size)
    if args.lr_start <= 0 or args.lr_end < 0:
        raise ValueError(f"Learning rates must satisfy lr_start > 0 and lr_end >= 0, got {args.lr_start}, {args.lr_end}")
    if args.lr_end > args.lr_start:
        raise ValueError(f"lr_end must be <= lr_start, got lr_start={args.lr_start}, lr_end={args.lr_end}")


def load_training_state(
    args: argparse.Namespace,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    logger: logging.Logger,
) -> TrainingState:
    if args.resume_path is not None:
        checkpoint = torch.load(args.resume_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        start_epoch = int(checkpoint["epoch"]) + 1
        if args.resume_epoch is not None and args.resume_epoch != start_epoch:
            raise ValueError(f"resume_epoch={args.resume_epoch} does not match checkpoint next epoch={start_epoch}")

        logger.info("Resumed from %s at epoch %s", args.resume_path, start_epoch)
        return TrainingState(
            start_epoch=start_epoch,
            global_step=start_epoch * args.iters_per_epoch,
            best_psnr=float(checkpoint.get("best_psnr", 0.0)),
            resume_path=args.resume_path,
            mode="resume",
        )

    pretrained_path = get_model_config(args.model_name)["pretrained_checkpoint"]
    if pretrained_path is not None:
        pretrained_path = str(resolve_path(pretrained_path))
        if not os.path.isfile(pretrained_path):
            raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_path}")

        logger.info("Loading pretrained checkpoint from %s", pretrained_path)
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
        return TrainingState(
            start_epoch=0,
            global_step=0,
            best_psnr=0.0,
            resume_path=None,
            mode="pretrained",
        )

    logger.info("Training %s from scratch", args.model_name)
    return TrainingState(
        start_epoch=0,
        global_step=0,
        best_psnr=0.0,
        resume_path=None,
        mode="scratch",
    )


def log_run_summary(
    args: argparse.Namespace,
    training_state: TrainingState,
    train_dataset: VFITrainDataset,
    test_dataset: VFITrainDataset,
    logger: logging.Logger,
    device: torch.device,
) -> None:
    logger.info(
        "Starting run with model=%s mode=%s device=%s output_dir=%s output_dir_reason=%s train_samples=%s test_samples=%s start_epoch=%s epochs=%s batch_size=%s",
        args.model_name,
        training_state.mode,
        device,
        args.output_dir,
        args.output_dir_reason,
        len(train_dataset),
        len(test_dataset),
        training_state.start_epoch,
        args.epochs,
        args.batch_size,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_train_args(argv)
    args = prepare_args(args)
    validate_args(args)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    logger, run_dir = build_logger(os.path.join(args.output_dir, "logs"))
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device %s", device)

    merged_df = build_merged_dataframe(args.root_dir, TRAIN_VFX_0416_DATASET_CONFIGS, only_fps=60, logger=logger)
    test_df = build_merged_dataframe(args.root_dir, TEST_VFX_0416_DATASET_CONFIGS, only_fps=60, logger=logger)
    merged_df = merged_df[merged_df["valid"] == True]

    train_dataset = VFITrainDataset(merged_df, args.dataset_root_dir, augment=True, input_fps=30)
    test_dataset = VFITrainDataset(test_df, args.dataset_root_dir, augment=False, input_fps=30)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    args.iters_per_epoch = len(train_loader)
    if args.iters_per_epoch == 0:
        raise RuntimeError("Training dataset is empty after filtering")

    model_class = get_model_class(args.model_name)
    model = model_class().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr_start, weight_decay=0)
    training_state = load_training_state(args, model, optimizer, device, logger)

    if training_state.start_epoch >= args.epochs:
        raise ValueError(
            f"start_epoch={training_state.start_epoch} must be smaller than epochs={args.epochs}"
        )

    log_run_summary(args, training_state, train_dataset, test_dataset, logger, device)
    logger.info("Run log directory %s", run_dir)

    train(
        args,
        args.model_name,
        model,
        optimizer,
        train_loader,
        test_loader,
        device,
        logger,
        training_state,
    )


if __name__ == "__main__":
    main()
