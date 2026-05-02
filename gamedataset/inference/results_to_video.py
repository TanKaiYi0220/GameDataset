# png_to_video.py
# Export VFI inference PNG results into MP4 videos.
# Behavior:
#   - Grid video: one frame_dir -> one composed frame (basic/debug)  [optional]
#   - Single videos: for each PNG stream, export a video using VFI60 timeline:
#         dir0: img0, pred, img1
#         dir1..: pred, img1
#     (no resize; assumes all PNG sizes are identical)

import re
import cv2
import argparse
from pathlib import Path
from typing import List, Optional, Tuple


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_frame_dirs(seq_dir: Path) -> List[Path]:
    dirs = [p for p in seq_dir.iterdir() if p.is_dir()]
    dirs.sort(key=lambda p: natural_key(p.name))
    return dirs


def read_bgr(path: Path) -> Optional[cv2.Mat]:
    if not path.exists():
        return None
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def find_first_existing(frame_dir: Path, candidates: List[str]) -> Optional[Path]:
    for name in candidates:
        p = frame_dir / name
        if p.exists():
            return p
    return None


# -----------------------
# Grid (optional)
# -----------------------
def collect_tiles(frame_dir: Path, layout: str):
    """
    layout presets:
      - basic: 2x3 => img0 img1 gt / pred merge diff_overlay
      - debug: 3x3 => img0 img1 gt / pred merge mask / flow0 flow1 diff_overlay
    """
    if layout == "basic":
        diff = find_first_existing(frame_dir, [
            "diff_mag_overlay_cb_1_to_0.png",
            "diff_mag_overlay_1_to_0.png",
            "diff_mag_cb_1_to_0.png",
            "diff_mag_1_to_0.png",
        ])
        tiles = [
            read_bgr(frame_dir / "image_0.png"),
            read_bgr(frame_dir / "image_1.png"),
            read_bgr(frame_dir / "image_gt.png"),
            read_bgr(frame_dir / "image_pred.png"),
            read_bgr(frame_dir / "image_merge.png"),
            read_bgr(diff) if diff else None,
        ]
        labels = ["img0", "img1", "gt", "pred", "merge", "flow (1 to 0) overlay"]
        grid = (2, 3)
        return tiles, labels, grid

    if layout == "debug":
        diff = find_first_existing(frame_dir, [
            "diff_mag_overlay_cb_1_to_0.png",
            "diff_mag_overlay_1_to_0.png",
            "diff_mag_cb_1_to_0.png",
            "diff_mag_1_to_0.png",
        ])
        tiles = [
            read_bgr(frame_dir / "image_0.png"),
            read_bgr(frame_dir / "image_1.png"),
            read_bgr(frame_dir / "image_gt.png"),
            read_bgr(frame_dir / "image_pred.png"),
            read_bgr(frame_dir / "image_merge.png"),
            read_bgr(frame_dir / "flow_mask.png"),
            read_bgr(frame_dir / "flow_1_to_0.png"),
            read_bgr(frame_dir / "flow_1_to_2.png"),
            read_bgr(diff) if diff else None,
        ]
        labels = ["img0", "img1", "gt", "pred", "merge", "mask", "flow 1->0", "flow 1->2", "Δflow overlay"]
        grid = (3, 3)
        return tiles, labels, grid

    raise ValueError(f"Unknown layout: {layout}")


def get_first_image_size(frame_dirs: List[Path], prefer_names: List[str]) -> Tuple[int, int]:
    for d in frame_dirs:
        for nm in prefer_names:
            img = read_bgr(d / nm)
            if img is not None:
                return (img.shape[1], img.shape[0])  # (W,H)
    return (960, 540)


def safe_resize(img, size: Tuple[int, int]):
    if img is None:
        return None
    W, H = size
    if img.shape[1] == W and img.shape[0] == H:
        return img
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)


def make_grid_frame(
    frame_dir: Path,
    layout: str,
    tile_size: Tuple[int, int],
    pad: int,
    show_frame_name: bool = True,
):
    import numpy as np

    tiles, labels, grid = collect_tiles(frame_dir, layout)
    rows, cols = grid
    tw, th = tile_size

    out_w = cols * tw + (cols + 1) * pad
    out_h = rows * th + (rows + 1) * pad
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    canvas[:] = (0, 0, 0)

    for i in range(rows * cols):
        r = i // cols
        c = i % cols
        x0 = pad + c * (tw + pad)
        y0 = pad + r * (th + pad)

        tile = tiles[i] if i < len(tiles) else None
        if tile is None:
            continue

        tile = safe_resize(tile, (tw, th))
        canvas[y0:y0 + th, x0:x0 + tw] = tile

        if i < len(labels) and labels[i]:
            cv2.rectangle(canvas, (x0, y0), (x0 + tw, y0 + 26), (0, 0, 0), -1)
            cv2.putText(
                canvas, labels[i], (x0 + 8, y0 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),
                1, cv2.LINE_AA
            )

    if show_frame_name:
        cv2.putText(
            canvas, frame_dir.name, (12, out_h - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),
            2, cv2.LINE_AA
        )

    return canvas


def write_video(frames: List[cv2.Mat], out_path: Path, fps: int):
    if len(frames) == 0:
        print(f"[WARN] empty frames, skip: {out_path}")
        return

    H, W = frames[0].shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))
    if not vw.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter: {out_path}")

    for f in frames:
        vw.write(f)

    vw.release()
    print(f"[OK] saved: {out_path} ({len(frames)} frames @ {fps}fps)")


def export_grid_video(
    frame_dirs: List[Path],
    out_path: Path,
    fps: int,
    layout: str,
    tile_scale: float,
    pad: int,
):
    base_tw, base_th = get_first_image_size(
        frame_dirs,
        prefer_names=["image_pred.png", "image_gt.png", "image_0.png"]
    )
    tw = max(16, int(base_tw * tile_scale))
    th = max(16, int(base_th * tile_scale))

    frames = []
    for d in frame_dirs:
        frames.append(make_grid_frame(d, layout, (tw, th), pad, show_frame_name=True))

    write_video(frames, out_path, fps)


# -----------------------
# VFI60 timeline export (single videos)
# -----------------------
def build_vfi60_timeline_frames(
    frame_dirs: List[Path],
    filename: str,
) -> List[cv2.Mat]:
    """
    Export frames in VFI60 timeline style:
      - For image_0.png: take only from the first dir (t=0)
      - For image_pred.png and image_1.png: take from every dir
      - For any other filename: take from every dir (dir-wise stream)
        (still no resize)
    """
    frames: List[cv2.Mat] = []

    if filename == "image_0.png":
        img = read_bgr(frame_dirs[0] / filename)
        if img is not None:
            frames.append(img)
        return frames

    # pred & img1 are the VFI60 "new" frames each dir
    if filename in ("image_pred.png", "image_1.png"):
        for d in frame_dirs:
            img = read_bgr(d / filename)
            if img is not None:
                frames.append(img)
        return frames

    # default: just a per-dir stream
    for d in frame_dirs:
        img = read_bgr(d / filename)
        if img is not None:
            frames.append(img)
    return frames


def export_vfi60_rgb_video(
    frame_dirs: List[Path],
    out_path: Path,
    fps: int = 60,
    *,
    img0_name: str = "image_0.png",
    pred_name: str = "image_pred.png",
    img1_name: str = "image_1.png",
    put_text: bool = False,
):
    """
    One RGB video in desired timeline:
      dir0: img0, pred, img1
      dir1..: pred, img1
    No resize; assumes identical resolution.
    """
    frames: List[cv2.Mat] = []

    for i, d in enumerate(frame_dirs):
        img0 = read_bgr(d / img0_name)
        pred = read_bgr(d / pred_name)
        img1 = read_bgr(d / img1_name)

        if i == 0 and img0 is not None:
            frames.append(img0)

        if pred is not None:
            frames.append(pred)
        if img1 is not None:
            frames.append(img1)

    if put_text:
        for idx, f in enumerate(frames):
            cv2.putText(
                f, f"t={idx}", (12, f.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),
                2, cv2.LINE_AA
            )

    write_video(frames, out_path, fps)


def export_all_single_videos_vfi60_timeline(
    frame_dirs: List[Path],
    out_dir: Path,
    record: str,
    mode: str,
    fps: int,
    files: List[str],
):
    """
    Export each filename into its own mp4.
    For image_0/pred/image_1 it follows VFI60 timeline semantics:
      - image_0: only first dir (t=0)
      - image_pred: every dir (t=1,3,5,...)
      - image_1: every dir (t=2,4,6,...)
    For others: per-dir stream (one frame per dir).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    for fn in files:
        frames = build_vfi60_timeline_frames(frame_dirs, fn)

        tag = fn.replace(".png", "").replace("/", "_")
        out_path = out_dir / f"{record}_{mode}_{tag}.mp4"

        write_video(frames, out_path, fps)


def main():
    model_name = "IFRNet_FineTuning_0326"  # adjust as needed

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=str,
        default=f"./output/{model_name}/checkpoints/inference/",
        help="Inference output root (contains record/mode/frame_range folders)"
    )
    parser.add_argument(
        "--record", type=str,
        default="ARPG_2",
    )
    parser.add_argument(
        "--mode", type=str, 
        default="4_Difficult/4_Difficult_2/fps_60"
    )

    parser.add_argument("--out-dir", type=str, default="./videos")
    parser.add_argument("--fps", type=int, default=60)

    # Grid (optional)
    parser.add_argument("--no-grid", action="store_true")
    parser.add_argument("--layout", type=str, default="basic", choices=["basic", "debug"])
    parser.add_argument("--tile-scale", type=float, default=0.5)
    parser.add_argument("--pad", type=int, default=8)

    # Single exports
    parser.add_argument("--export-all", action="store_true",
                        help="Export each png stream as its own mp4 using VFI60 timeline semantics")
    parser.add_argument("--single-files", type=str, default="",
                        help="Comma-separated filenames to export. If empty, uses a default list.")

    # One main VFI60 RGB video (img0,pred,img1 | pred,img1 ...)
    parser.add_argument("--vfi60", action="store_true",
                        help="Export one RGB video with timeline: dir0(img0,pred,img1), next dirs(pred,img1).")

    args = parser.parse_args()

    root = Path(args.root)
    seq_dir = root / args.record / args.mode
    if not seq_dir.exists():
        raise FileNotFoundError(f"Missing: {seq_dir}")

    frame_dirs = list_frame_dirs(seq_dir)
    if len(frame_dirs) == 0:
        raise RuntimeError(f"No frame dirs under: {seq_dir}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # (A) Grid video (optional)
    if not args.no_grid:
        out_grid = out_dir / f"{model_name}_{args.record}_{args.mode}_{args.layout}_grid.mp4"
        export_grid_video(
            frame_dirs=frame_dirs,
            out_path=out_grid,
            fps=args.fps,
            layout=args.layout,
            tile_scale=args.tile_scale,
            pad=args.pad,
        )

    # (B) One RGB timeline video
    if args.vfi60:
        pred_vfi = out_dir / f"{model_name}_{args.record}_{args.mode}_vfi60_pred.mp4"
        gt_vfi = out_dir / f"{model_name}_{args.record}_{args.mode}_vfi60_gt.mp4"
        export_vfi60_rgb_video(
            frame_dirs=frame_dirs,
            out_path=pred_vfi,
            fps=args.fps,   # typically 60
            img0_name="image_0.png",
            pred_name="image_pred.png",
            img1_name="image_1.png",
            put_text=False,
        )

        export_vfi60_rgb_video(
            frame_dirs=frame_dirs,
            out_path=gt_vfi,
            fps=args.fps,   # typically 60
            img0_name="image_0.png",
            pred_name="image_gt.png",
            img1_name="image_1.png",
            put_text=False,
        )

    # (C) Export each PNG stream to its own mp4 (VFI60 semantics where relevant)
    if args.export_all:
        if args.single_files.strip():
            files = [x.strip() for x in args.single_files.split(",") if x.strip()]
        else:
            files = [
                "image_0.png",
                "image_pred.png",
                "image_1.png",
                "image_gt.png",
                "image_merge.png",
                "image_0_warped.png",
                "image_1_warped.png",
                "flow_1_to_0.png",
                "flow_1_to_2.png",
                "flow_mask.png",
                "diff_mag_overlay_cb_1_to_0.png",
                "diff_mag_overlay_1_to_0.png",
                "diff_mag_cb_1_to_0.png",
                "diff_mag_1_to_0.png",
                "diff_flow_1_to_0.png",
                "init_flow_1_to_0.png",
                "diff_changed_thr_1.00_1_to_0.png",
            ]

        single_out_dir = out_dir / "single"
        export_all_single_videos_vfi60_timeline(
            frame_dirs=frame_dirs,
            out_dir=single_out_dir,
            record=args.record,
            mode=args.mode,
            fps=args.fps,
            files=files,
        )


if __name__ == "__main__":
    main()
