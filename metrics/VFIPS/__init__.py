import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from . import networks
from .utils import *


def calc_vfips(dis_dir, ref_dir):
    moduleNetwork = networks.get_model("multiscale_v33")
    moduleNetwork.load_state_dict(torch.load("metrics/VFIPS/checkpoints/VFIPS.pytorch"))

    moduleNetwork.cuda().eval()

    @torch.no_grad()
    def estimate(tenRef, tenVideo):
        tenDis = moduleNetwork(tenRef, tenVideo)
        return tenDis

    transform_list = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    transform = transforms.Compose(transform_list)

    disLst = sorted(os.listdir(dis_dir))
    refLst = sorted(os.listdir(ref_dir))

    frame_count = len(disLst)

    # stride 12
    # for start_id in range(0, frame_count - 12):
    scores = []
    for start_id in range(0, frame_count - 12, 12):
        video = []
        gt = []
        for i in range(12):
            videoimg = cv2.imread(os.path.join(dis_dir, disLst.pop(0)))
            videoimg = cv2.cvtColor(videoimg, cv2.COLOR_BGR2RGB)
            videoimg = Image.fromarray(videoimg)
            videoimg = transform(videoimg).unsqueeze(0)

            gtimg = cv2.imread(os.path.join(ref_dir, refLst.pop(0)))
            gtimg = cv2.cvtColor(gtimg, cv2.COLOR_BGR2RGB)
            gtimg = Image.fromarray(gtimg)
            gtimg = transform(gtimg).unsqueeze(0)

            video.append(videoimg)
            gt.append(gtimg)

        video = torch.cat(video, dim=0)
        gt = torch.cat(gt, dim=0)

        video = video.unsqueeze(0).cuda()
        gt = gt.unsqueeze(0).cuda()

        dis = estimate(gt, video)
        dis = dis.data.cpu().numpy().flatten()

        scores.append(dis)
    return np.mean(scores)

def calc_vfips_mp4(
    pred_mp4: str,
    gt_mp4: str,
    *,
    ckpt_path: str = "metrics/VFIPS/checkpoints/VFIPS.pytorch",
    model_name: str = "multiscale_v33",
    clip_len: int = 12,
    stride: int = 12,                 # 原本你的版本是每 12 幀取一段（非 sliding）
    device: str = "cuda",
) -> float:
    """
    Compute VFIPS between two mp4 videos by chunking into clips.

    pred_mp4: video_pred.mp4
    gt_mp4:   video_gt.mp4

    Returns: mean VFIPS score across clips.
    """
    # -----------------------------
    # Load VFIPS model
    # -----------------------------
    moduleNetwork = networks.get_model(model_name)
    moduleNetwork.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    dev = torch.device(device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    moduleNetwork = moduleNetwork.to(dev).eval()

    @torch.no_grad()
    def estimate(tenRef, tenVideo):
        # keep your original call convention
        return moduleNetwork(tenRef, tenVideo)

    # -----------------------------
    # Preprocess
    # -----------------------------
    transform = transforms.Compose([
        transforms.ToTensor(),  # -> [3,H,W] in 0~1
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # -> [-1,1]
    ])

    pred_frames = read_all_frames(pred_mp4)
    gt_frames = read_all_frames(gt_mp4)

    # align length by min T
    T = min(len(pred_frames), len(gt_frames))
    pred_frames = pred_frames[:T]
    gt_frames = gt_frames[:T]

    if T < clip_len:
        raise RuntimeError(f"Video too short: T={T} < clip_len={clip_len}")

    # -----------------------------
    # Compute VFIPS per clip
    # -----------------------------
    scores = []
    for start in range(0, T - clip_len + 1, stride):
        pred_clip = []
        gt_clip = []
        for i in range(clip_len):
            pred_img = Image.fromarray(pred_frames[start + i])
            gt_img = Image.fromarray(gt_frames[start + i])

            pred_t = transform(pred_img).unsqueeze(0)  # [1,3,H,W]
            gt_t = transform(gt_img).unsqueeze(0)      # [1,3,H,W]

            pred_clip.append(pred_t)
            gt_clip.append(gt_t)

        # [T,1,3,H,W] -> cat -> [T,3,H,W] -> unsqueeze -> [1,T,3,H,W]
        pred_clip = torch.cat(pred_clip, dim=0).unsqueeze(0).to(dev)
        gt_clip = torch.cat(gt_clip, dim=0).unsqueeze(0).to(dev)

        dis = estimate(gt_clip, pred_clip)  # keep same order as your original
        if isinstance(dis, (tuple, list)):
            dis = dis[0]

        # 有些实现会回 [B] / [B,1] / scalar，这里统一取 mean
        dis = dis.detach().float().mean().cpu().item()
        scores.append(dis)

    return float(np.mean(scores))