from datasets.dataset_loader import VFIDataset
from datasets.dataset_config import (
    DATASET_CONFIGS, 
    MINOR_DATASET_CONFIGS, 
    VFX_DATASET_CONFIGS, 
    STAIR_DATASET_CONFIG, 
    TEST_DATASET_CONFIGS, 
    TEST_VFX_DATASET_CONFIGS,
    TEST_VFX_0326_DATASET_CONFIGS, 
    TEST_UNSEEN_VFX_0326_DATASET_CONFIGS,
    TRAIN_VFX_0416_DATASET_CONFIGS,
    TEST_VFX_0416_DATASET_CONFIGS,
    iter_dataset_configs
)
import pandas as pd
from src.gameData_loader import load_backward_velocity, load_forward_velocity
from src.utils import show_images_switchable, flow_to_image, save_img, save_np_array
from evaluation import TaskEvaluator, VFI_METRICS
from config.config import get_config

import cv2
import torch
import numpy as np
import os
import time

import sys
sys.path.append('models/IFRNet')
from tqdm import tqdm

# from models.IFRNet import Model
from models.IFRNet_Residual import Model
from skimage.metrics import peak_signal_noise_ratio as psnr
from utils import warp


ROOT_DIR = get_config("data_root", "./datasets/data")
# MODEL_PATH = "./models/IFRNet/checkpoints/IFRNet/IFRNet_Vimeo90K.pth"
MODEL_PATH = os.path.join(get_config("output_root", "./output"), "IFRNet_VFX_0326", "checkpoints")
OUTPUT_DIR = os.path.join(get_config("output_root", "./output"), "IFRNet_VFX_0326", "checkpoints", "inference")
DATASET = TRAIN_VFX_0416_DATASET_CONFIGS

def add_colorbar_cv2(
    heatmap_bgr: np.ndarray,
    vmin: float,
    vmax: float,
    *,
    colormap: int = cv2.COLORMAP_TURBO,
    bar_width: int = 28,
    label_width: int = 90,
    pad: int = 8,
    ticks: int = 5,
    font_scale: float = 0.45,
    thickness: int = 1,
):
    """
    Append a vertical colorbar with numeric tick labels to the right of a BGR heatmap image.
    heatmap_bgr: [H,W,3] uint8, already color-mapped.
    vmin/vmax: numeric range you want to show on the bar.
    """
    H, W = heatmap_bgr.shape[:2]

    # gradient: top=high, bottom=low
    grad = np.linspace(1.0, 0.0, H, dtype=np.float32)[:, None]
    bar_u8 = (grad * 255).astype(np.uint8)                      # [H,1]
    bar_bgr = cv2.applyColorMap(bar_u8, colormap)               # [H,1,3]
    bar_bgr = cv2.resize(bar_bgr, (bar_width, H), interpolation=cv2.INTER_NEAREST)

    out_w = W + pad + bar_width + label_width
    out = np.full((H, out_w, 3), 255, dtype=np.uint8)           # white background
    out[:, :W] = heatmap_bgr
    out[:, W:W+pad] = 255
    out[:, W+pad:W+pad+bar_width] = bar_bgr

    x_bar0 = W + pad
    x_bar1 = x_bar0 + bar_width - 1
    x_text = x_bar1 + 6

    # ticks & labels
    if ticks < 2:
        ticks = 2
    for i in range(ticks):
        t = i / (ticks - 1)                                     # 0..1
        y = int((1.0 - t) * (H - 1))                             # top high
        val = vmin + t * (vmax - vmin)

        # tick line
        cv2.line(out, (x_bar0, y), (x_bar1, y), (0, 0, 0), 1)

        # label
        y_text = min(max(y + 4, 12), H - 6)
        cv2.putText(
            out,
            f"{val:.2f}",
            (x_text, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

    # title
    cv2.putText(
        out,
        "|Δflow|",
        (x_bar0, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )
    return out


def visualize_and_save_flow_diff(
    *,
    up_flow: torch.Tensor,          # [B,2,H,W]
    init_flow: torch.Tensor,        # [B,2,H,W]
    bg_img_np: np.ndarray,          # [H,W,3] BGR (cv2.imread)
    save_dir: str,
    name: str = "1_to_0",
    thr: float = 1.0,               # flow difference threshold (in pixels)
    percentile: float = 99.0,       # robust scaling for heatmap
    overlay_alpha_bg: float = 0.2,
    overlay_alpha_heat: float = 0.8,
    colormap: int = cv2.COLORMAP_TURBO,
):
    """
    Save visualizations to save_dir:

      - init_flow_{name}.png              (init flow visualization)
      - diff_flow_{name}.png              (vector diff visualization)
      - diff_mag_{name}.png               (L2 magnitude heatmap)
      - diff_mag_overlay_{name}.png       (heatmap overlay on bg_img_np)
      - diff_changed_thr_{thr}_{name}.png (binary mask: mag > thr)

    Returns a dict of stats for logging.
    """

    os.makedirs(save_dir, exist_ok=True)

    # ensure same device
    init_flow = init_flow.to(up_flow.device)

    # diff flow (tensor)
    diff = (up_flow - init_flow).detach()  # [B,2,H,W]

    # to numpy [H,W,2]
    diff_np = diff[0].permute(1, 2, 0).cpu().numpy()
    init_np = init_flow[0].detach().permute(1, 2, 0).cpu().numpy()

    # (A) vector visualization
    init_vis = flow_to_image(init_np)
    diff_vis = flow_to_image(diff_np)

    # (B) magnitude heatmap
    diff_mag = np.linalg.norm(diff_np, axis=2)  # [H,W]
    p = float(np.percentile(diff_mag, percentile))
    scale = max(p, 1e-6)

    diff_mag_u8 = (np.clip(diff_mag / scale, 0, 1) * 255).astype(np.uint8)
    diff_mag_color = cv2.applyColorMap(diff_mag_u8, colormap)
    diff_mag_color_cb = add_colorbar_cv2(
        diff_mag_color,
        vmin=0.0,
        vmax=scale,                 # 這裡 scale 就是你用 percentile 算的上限（p99）
        colormap=colormap,
        ticks=5
    )

    # overlay (make sure both are uint8 BGR)
    if bg_img_np.dtype != np.uint8:
        bg_vis = np.clip(bg_img_np, 0, 255).astype(np.uint8)
    else:
        bg_vis = bg_img_np

    overlay = cv2.addWeighted(bg_vis, overlay_alpha_bg, diff_mag_color, overlay_alpha_heat, 0)
    overlay_cb = add_colorbar_cv2(
        overlay,
        vmin=0.0,
        vmax=scale,
        colormap=colormap,
        ticks=5
    )

    # (C) changed mask
    changed = (diff_mag > thr).astype(np.uint8) * 255

    # save
    # print(os.path.join(save_dir, f"init_flow_{name}.png"))
    save_img(os.path.join(save_dir, f"init_flow_{name}.png"), init_vis)
    save_img(os.path.join(save_dir, f"diff_flow_{name}.png"), diff_vis)
    save_img(os.path.join(save_dir, f"diff_mag_{name}.png"), diff_mag_color)
    save_img(os.path.join(save_dir, f"diff_mag_cb_{name}.png"), diff_mag_color_cb)

    save_img(os.path.join(save_dir, f"diff_mag_overlay_{name}.png"), overlay)
    save_img(os.path.join(save_dir, f"diff_mag_overlay_cb_{name}.png"), overlay_cb)
    save_img(os.path.join(save_dir, f"diff_changed_thr_{thr:.2f}_{name}.png"), changed)

    stats = {
        "pctl_value": p,
        "mag_mean": float(diff_mag.mean()),
        "mag_max": float(diff_mag.max()),
        "changed_ratio": float((changed > 0).mean()),
    }
    return stats

def main():
    # Load Model
    model = Model().cuda().eval()
    print(f"{MODEL_PATH}/best.pth")
    model.load_state_dict(torch.load(f"{MODEL_PATH}/best.pth"))

    # Load Dataset
    for cfg in iter_dataset_configs(DATASET):
        if cfg.fps != 60:
            continue

        # if cfg.difficulty != "Difficult":
        #     continue


        df = pd.read_csv(f"{ROOT_DIR}/{cfg.record_name}_preprocessed/{cfg.mode_index}_raw_sequence_frame_index.csv")
        df["record"] = cfg.record
        df["mode"] = cfg.mode_path
        
        vfi_evaluator = TaskEvaluator(task_name="VFI", metric_fns=VFI_METRICS)

        dataset = VFIDataset(
            df=df,
            root_dir=DATASET["root_dir"],
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
                bmv_path = gt["backwardVel_Depth"]
                fmv_path = gt["forwardVel_Depth"]


                bmv, _ = load_backward_velocity(bmv_path)
                fmv, _ = load_forward_velocity(fmv_path)

                # concat along channel: [1,4,H,W] -> take [0] => [4,H,W]
                flow = torch.cat([bmv, fmv], dim=1).float()

                img0_np = cv2.imread(img0_path)
                img1_np = cv2.imread(img1_path)
                imgGT_np = cv2.imread(imgGT_path)

                retries = 0
                while img0_np is None or img1_np is None or imgGT_np is None:
                    print(f"Warning: Failed to read images for sample {i} in {cfg.mode_name}. Retrying...")
                    time.sleep(1)  # wait a bit before retrying
                    img0_np = cv2.imread(img0_path)
                    img1_np = cv2.imread(img1_path)
                    imgGT_np = cv2.imread(imgGT_path)
                    retries += 1
                    if retries > 5:
                        raise RuntimeError(f"Failed to read images after 5 retries for sample {i} in {cfg.mode_name}. Check file paths and integrity.")

                # Inference
                img0 = (torch.tensor(img0_np.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).cuda()
                img1 = (torch.tensor(img1_np.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).cuda()
                embt = torch.tensor(1/2).view(1, 1, 1, 1).float().cuda()

                # # ------------------------ insert timing ------------------------
                # torch.cuda.synchronize()
                # start = time.time()

                # imgPred, up_flow0_1, up_flow1_1, up_mask_1 = model.inference(img0, img1, embt)
                init_flow0_full = flow[:, 0:2]  # [B,2,H,W]
                init_flow1_full = flow[:, 2:4]  # [B,2,H,W]
                imgPred, up_flow0_1, up_flow1_1, up_mask_1, up_res_1, imgt_merge = model.inference(
                    img0, img1, embt,
                    init_flow0=init_flow0_full, init_flow1=init_flow1_full
                )
                # print("flow0_1 mean", up_flow0_1.abs().mean().item(), "max", up_flow0_1.abs().max().item())
                # print("bias flow mean abs", (up_flow0_1 - init_flow0_full).abs().mean().item(), "max", (up_flow0_1 - init_flow0_full).abs().max().item())
                # print("mask saturation", ((up_mask_1 < 0.05) | (up_mask_1 > 0.95)).float().mean().item())
                # print("residual mean abs", up_res_1.abs().mean().item(), "max", up_res_1.abs().max().item())
                # torch.cuda.synchronize()
                # end = time.time()
                # infer_time = end - start
                # # ---------------------------------------------------------------

                imgPred_np = (imgPred[0].data.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
                imgt_merge_np = (imgt_merge[0].data.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
                up_flow0_1_np = flow_to_image(up_flow0_1[0].data.permute(1, 2, 0).cpu().numpy())
                up_flow1_1_np = flow_to_image(up_flow1_1[0].data.permute(1, 2, 0).cpu().numpy())
                bmv_np = flow_to_image(bmv[0].permute(1, 2, 0).cpu().numpy())
                fmv_np = flow_to_image(fmv[0].permute(1, 2, 0).cpu().numpy())
                up_mask_1_np = (up_mask_1[0, 0].data.cpu().numpy() * 255.0).astype(np.uint8)

                # Warped images
                img0_warped = warp(img0, up_flow0_1)
                img1_warped = warp(img1, up_flow1_1)
                img0_bmv_warped = warp(img0, bmv)
                img1_fmv_warped = warp(img1, fmv)

                img0_warped_np = img0_warped[0].data.permute(1, 2, 0).cpu().numpy() * 255.0
                img1_warped_np = img1_warped[0].data.permute(1, 2, 0).cpu().numpy() * 255.0
                img0_bmv_warped_np = img0_bmv_warped[0].data.permute(1, 2, 0).cpu().numpy() * 255.0
                img1_fmv_warped_np = img1_fmv_warped[0].data.permute(1, 2, 0).cpu().numpy() * 255.0

                # stores results
                save_dir = f"{OUTPUT_DIR}/{cfg.record}/{cfg.mode_path}/{sample['frame_range']}/"

                # ======= NEW: visualize flow update (up_flow0_1 vs init_flow0_full) =======
                diff_stats = visualize_and_save_flow_diff(
                    up_flow=up_flow0_1,
                    init_flow=init_flow0_full,
                    bg_img_np=img0_np,
                    save_dir=save_dir,
                    name="1_to_0",
                    thr=1.0,          # 你可以改 0.5 / 1.0 / 2.0 看敏感度
                    percentile=99.0,  # 你也可以改 95 讓差異更顯眼
                )
                # print("[FlowDiffStats]", diff_stats)

                os.makedirs(f"{save_dir}", exist_ok=True)
                save_img(f"{save_dir}/image_0.png", img0_np)
                save_img(f"{save_dir}/image_1.png", img1_np)
                save_img(f"{save_dir}/image_gt.png", imgGT_np)
                save_img(f"{save_dir}/image_pred.png", imgPred_np)
                save_img(f"{save_dir}/image_merge.png", imgt_merge_np)
                # save_np_array(f"{save_dir}/flow_1_to_0.npy", up_flow0_1_np)
                # save_np_array(f"{save_dir}/flow_1_to_2.npy", up_flow1_1_np)
                save_img(f"{save_dir}/bmv.png", bmv_np)
                save_img(f"{save_dir}/fmv.png", fmv_np)
                save_img(f"{save_dir}/flow_1_to_0.png", up_flow0_1_np)
                save_img(f"{save_dir}/flow_1_to_2.png", up_flow1_1_np)
                save_img(f"{save_dir}/flow_mask.png", up_mask_1_np)
                save_img(f"{save_dir}/image_0_warped.png", img0_warped_np)
                save_img(f"{save_dir}/image_1_warped.png", img1_warped_np)
                save_img(f"{save_dir}/image_0_bmv_warped.png", img0_bmv_warped_np)
                save_img(f"{save_dir}/image_1_fmv_warped.png", img1_fmv_warped_np)

                # evaluation
                bmv, _ = load_backward_velocity(bmv_path)
                fmv, _ = load_forward_velocity(fmv_path)
                
                meta = {
                    "record": cfg.record,
                    "mode": cfg.mode_path,
                    "frame_range": sample["frame_range"],
                    # "inference_time": infer_time,
                    "valid": sample["valid"],
                    "distance_indexing": sample["distance_indexing"]
                }

                result = vfi_evaluator.evaluate(
                    meta=meta,
                    img_gt=imgGT_np,
                    img_pred=imgPred_np,
                    flow_1_to_0=up_flow0_1,
                    flow_1_to_2=up_flow1_1,
                    bmv=bmv,
                    fmv=fmv
                )
                

                pbar.set_postfix({
                    "FrameRange": sample["frame_range"],
                    "PSNR": f"{result['psnr']:.2f}",
                    "EPE_1_to_0": f"{result['epe_1_to_0']:.3f}",
                    "EPE_1_to_2": f"{result['epe_1_to_2']:.3f}",
                    # "InferenceTime": f"{result['inference_time']:.4f}",
                    "Valid": sample["valid"],
                    "D(t)": f"{sample['distance_indexing'][0]:.3f}"
                })

        eval_df = vfi_evaluator.to_dataframe()
        eval_path = os.path.join(OUTPUT_DIR, f"{cfg.record}/{cfg.mode_name}_evaluation_results.csv")
        eval_df.to_csv(eval_path, index=False)
        print(eval_df.describe())
        print(f"Saving Evaluation Result into {eval_path}")

            
if __name__ == "__main__":
    main()