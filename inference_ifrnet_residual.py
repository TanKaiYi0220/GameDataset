import os

import cv2
import numpy as np
import torch

from config.config import get_config
from datasets.dataset_config import TRAIN_VFX_0416_DATASET_CONFIGS
from inference_ifrnet_common import load_model, run_inference_loop
from models.IFRNet.utils import warp
from src.utils import flow_to_image
from src.utils import save_img

ROOT_DIR = get_config("data_root", "./datasets/data")
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
    H, W = heatmap_bgr.shape[:2]
    grad = np.linspace(1.0, 0.0, H, dtype=np.float32)[:, None]
    bar_u8 = (grad * 255).astype(np.uint8)
    bar_bgr = cv2.applyColorMap(bar_u8, colormap)
    bar_bgr = cv2.resize(bar_bgr, (bar_width, H), interpolation=cv2.INTER_NEAREST)
    out_w = W + pad + bar_width + label_width
    out = np.full((H, out_w, 3), 255, dtype=np.uint8)
    out[:, :W] = heatmap_bgr
    out[:, W:W+pad] = 255
    out[:, W+pad:W+pad+bar_width] = bar_bgr
    x_bar0 = W + pad
    x_bar1 = x_bar0 + bar_width - 1
    x_text = x_bar1 + 6
    if ticks < 2:
        ticks = 2
    for i in range(ticks):
        t = i / (ticks - 1)
        y = int((1.0 - t) * (H - 1))
        val = vmin + t * (vmax - vmin)
        cv2.line(out, (x_bar0, y), (x_bar1, y), (0, 0, 0), 1)
        y_text = min(max(y + 4, 12), H - 6)
        cv2.putText(out, f"{val:.2f}", (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    cv2.putText(out, "|Δflow|", (x_bar0, 18), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return out


def visualize_and_save_flow_diff(*, up_flow: torch.Tensor, init_flow: torch.Tensor, bg_img_np: np.ndarray, save_dir: str, name: str = "1_to_0", thr: float = 1.0, percentile: float = 99.0, overlay_alpha_bg: float = 0.2, overlay_alpha_heat: float = 0.8, colormap: int = cv2.COLORMAP_TURBO):
    os.makedirs(save_dir, exist_ok=True)
    init_flow = init_flow.to(up_flow.device)
    diff = (up_flow - init_flow).detach()
    diff_np = diff[0].permute(1, 2, 0).cpu().numpy()
    init_np = init_flow[0].detach().permute(1, 2, 0).cpu().numpy()
    init_vis = flow_to_image(init_np)
    diff_vis = flow_to_image(diff_np)
    diff_mag = np.linalg.norm(diff_np, axis=2)
    p = float(np.percentile(diff_mag, percentile))
    scale = max(p, 1e-6)
    diff_mag_u8 = (np.clip(diff_mag / scale, 0, 1) * 255).astype(np.uint8)
    diff_mag_color = cv2.applyColorMap(diff_mag_u8, colormap)
    diff_mag_color_cb = add_colorbar_cv2(diff_mag_color, vmin=0.0, vmax=scale, colormap=colormap, ticks=5)
    if bg_img_np.dtype != np.uint8:
        bg_vis = np.clip(bg_img_np, 0, 255).astype(np.uint8)
    else:
        bg_vis = bg_img_np
    overlay = cv2.addWeighted(bg_vis, overlay_alpha_bg, diff_mag_color, overlay_alpha_heat, 0)
    overlay_cb = add_colorbar_cv2(overlay, vmin=0.0, vmax=scale, colormap=colormap, ticks=5)
    changed = (diff_mag > thr).astype(np.uint8) * 255
    save_img(os.path.join(save_dir, f"init_flow_{name}.png"), init_vis)
    save_img(os.path.join(save_dir, f"diff_flow_{name}.png"), diff_vis)
    save_img(os.path.join(save_dir, f"diff_mag_{name}.png"), diff_mag_color)
    save_img(os.path.join(save_dir, f"diff_mag_cb_{name}.png"), diff_mag_color_cb)
    save_img(os.path.join(save_dir, f"diff_mag_overlay_{name}.png"), overlay)
    save_img(os.path.join(save_dir, f"diff_mag_overlay_cb_{name}.png"), overlay_cb)
    save_img(os.path.join(save_dir, f"diff_changed_thr_{thr:.2f}_{name}.png"), changed)
    return {
        "pctl_value": p,
        "mag_mean": float(diff_mag.mean()),
        "mag_max": float(diff_mag.max()),
        "changed_ratio": float((changed > 0).mean()),
    }


def save_sample_fn(*, outputs, sample, cfg, output_dir, img0_np, img1_np, imgGT_np, bmv, fmv, img0, img1, embt, device, vfi_evaluator, **kwargs):
    imgPred, up_flow0_1, up_flow1_1, up_mask_1, up_res_1, imgt_merge = outputs
    imgPred_np = (imgPred[0].detach().permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    imgt_merge_np = (imgt_merge[0].detach().permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    up_flow0_1_np = flow_to_image(up_flow0_1[0].detach().permute(1, 2, 0).cpu().numpy())
    up_flow1_1_np = flow_to_image(up_flow1_1[0].detach().permute(1, 2, 0).cpu().numpy())
    bmv_np = flow_to_image(bmv[0].permute(1, 2, 0).cpu().numpy())
    fmv_np = flow_to_image(fmv[0].permute(1, 2, 0).cpu().numpy())
    up_mask_1_np = (up_mask_1[0, 0].detach().cpu().numpy() * 255.0).astype(np.uint8)
    img0_warped = warp(img0, up_flow0_1)
    img1_warped = warp(img1, up_flow1_1)
    img0_bmv_warped = warp(img0, bmv)
    img1_fmv_warped = warp(img1, fmv)
    img0_warped_np = img0_warped[0].detach().permute(1, 2, 0).cpu().numpy() * 255.0
    img1_warped_np = img1_warped[0].detach().permute(1, 2, 0).cpu().numpy() * 255.0
    img0_bmv_warped_np = img0_bmv_warped[0].detach().permute(1, 2, 0).cpu().numpy() * 255.0
    img1_fmv_warped_np = img1_fmv_warped[0].detach().permute(1, 2, 0).cpu().numpy() * 255.0
    save_dir = f"{output_dir}/{cfg.record}/{cfg.mode_path}/{sample['frame_range']}/"
    os.makedirs(save_dir, exist_ok=True)
    visualize_and_save_flow_diff(
        up_flow=up_flow0_1,
        init_flow=bmv,
        bg_img_np=img0_np,
        save_dir=save_dir,
        name="1_to_0",
        thr=1.0,
        percentile=99.0,
    )
    save_img(f"{save_dir}/image_0.png", img0_np)
    save_img(f"{save_dir}/image_1.png", img1_np)
    save_img(f"{save_dir}/image_gt.png", imgGT_np)
    save_img(f"{save_dir}/image_pred.png", imgPred_np)
    save_img(f"{save_dir}/image_merge.png", imgt_merge_np)
    save_img(f"{save_dir}/bmv.png", bmv_np)
    save_img(f"{save_dir}/fmv.png", fmv_np)
    save_img(f"{save_dir}/flow_1_to_0.png", up_flow0_1_np)
    save_img(f"{save_dir}/flow_1_to_2.png", up_flow1_1_np)
    save_img(f"{save_dir}/flow_mask.png", up_mask_1_np)
    save_img(f"{save_dir}/image_0_warped.png", img0_warped_np)
    save_img(f"{save_dir}/image_1_warped.png", img1_warped_np)
    save_img(f"{save_dir}/image_0_bmv_warped.png", img0_bmv_warped_np)
    save_img(f"{save_dir}/image_1_fmv_warped.png", img1_fmv_warped_np)
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("IFRNet_Residual", os.path.join(MODEL_PATH, "best.pth"), device).eval()
    run_inference_loop(
        model_name="IFRNet_Residual",
        model=model,
        dataset_configs=DATASET,
        root_dir=ROOT_DIR,
        output_dir=OUTPUT_DIR,
        device=device,
        save_sample_fn=save_sample_fn,
    )


if __name__ == "__main__":
    main()
