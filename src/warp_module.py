import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


# =========================
# Base class (common utils)
# =========================
class WarpingBase:
    """
    共用:結果緩存 (self.out, self.hit)、平均輸出、命中可視化、建網格/索引等工具。
    子類別只需實作 self.warp(image, flow) 內部把 self.out / self.hit 設好。
    """
    def __init__(self):
        self.out: Optional[torch.Tensor] = None   # [N,C,H,W] 或同型
        self.hit: Optional[torch.Tensor] = None   # [N,1,H,W]:forward=累積次數；backward=0/1 有效遮罩

    # ---- helpers (共用) ----
    @staticmethod
    def _make_uv_grid(N:int, H:int, W:int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        v, u = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij"
        )
        u = u[None].float().expand(N, -1, -1)  # [N,H,W]
        v = v[None].float().expand(N, -1, -1)  # [N,H,W]
        return u, v

    @staticmethod
    def _flat_index(N:int, H:int, W:int, x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
        """將 (x,y) -> 線性索引 x,y 已經 clamp 到合法範圍。"""
        base = (torch.arange(N, device=x.device) * (H * W)).view(N, 1, 1)
        return (y * W + x + base).view(-1)  # [N*H*W]

    # ---- public APIs (共用) ----
    def get_warping_result(self, mode="average"):
        valid_modes = ["average", "raw"]
        if self.out is None or self.hit is None:
            raise ValueError("No warping has been performed yet.")

        if mode == "raw":
            return self.out, self.hit

        if mode == "average":
            # forward: out=累加值 / hit=次數；backward: out=已填值 / hit=0/1
            denom = torch.where(self.hit > 0, self.hit, torch.ones_like(self.hit))
            avg = self.out / denom
            avg = torch.where(self.hit > 0, avg, torch.zeros_like(avg))
            return avg, self.hit

        raise ValueError(f"Invalid mode: {mode}")

    def visualize_hit(self) -> np.ndarray:
        """
        forward: 0 (none)、1 (one-to-one)、>=2 (many-to-one)
        backward: 0 (越界/無效)、1 (有效)
        """
        if self.hit is None:
            raise ValueError("No warping has been performed yet.")
        hmap = self.hit[0, 0].detach().cpu().numpy()
        hmap_i = hmap.astype(np.int32)
        H, W = hmap_i.shape
        vis = np.zeros((H, W, 3), dtype=np.uint8)
        vis[hmap_i == 0] = (255, 0, 0)   # none   → 藍 (B,G,R)
        vis[hmap_i == 1] = (0, 255, 0)   # one    → 綠
        vis[hmap_i >= 2] = (0, 0, 255)   # many   → 紅
        return vis

    # 子類別需實作
    def warp(self, image: torch.Tensor, flow: torch.Tensor):
        raise NotImplementedError


# =========================
# Forward (nearest / splat)
# =========================
class ForwardWarpingNearest(WarpingBase):
    """
    flow 定義:source→target (對來源像素 (x,y) 給出位移 dx,dy 到目標）
    行為:把來源像素 splat 到目標 (最近鄰）,累加到 self.out,累計命中次數 self.hit。
    """
    def warp(self, src_image: torch.Tensor, flow_s2t: torch.Tensor):
        N, C, H, W = src_image.shape
        device = src_image.device

        # 來源網格 (x,y)
        u, v = self._make_uv_grid(N, H, W, device)  # 這裡 u,v 正好也是 source 的 (x,y)

        tx = u + flow_s2t[:, 0]  # [N,H,W]
        ty = v + flow_s2t[:, 1]  # [N,H,W]

        # 最近鄰
        txn = tx.round().long()
        tyn = ty.round().long()

        # 有效範圍
        mask = (txn >= 0) & (txn < W) & (tyn >= 0) & (tyn < H)

        # scatter 累加
        linear_idx = tyn.clamp(0, H - 1) * W + txn.clamp(0, W - 1)
        base = (torch.arange(N, device=device) * (H * W)).view(N, 1, 1)
        flat_idx = (linear_idx + base).view(-1)

        src_flat = src_image.permute(0, 2, 3, 1).reshape(-1, C)  # [N*H*W, C]
        mask_flat = mask.view(-1)

        out_flat = torch.zeros((N * H * W, C), device=device, dtype=src_image.dtype)
        out_flat.index_add_(0, flat_idx[mask_flat], src_flat[mask_flat])
        out = out_flat.view(N, H, W, C).permute(0, 3, 1, 2)

        hit_flat = torch.zeros((N * H * W, 1), device=device, dtype=src_image.dtype)
        one = torch.ones((mask_flat.sum(), 1), device=device, dtype=src_image.dtype)
        hit_flat.index_add_(0, flat_idx[mask_flat], one)
        hit = hit_flat.view(N, H, W, 1).permute(0, 3, 1, 2)

        self.out = out
        self.hit = hit
        return out, hit


# =========================
# Backward (nearest / gather)
# =========================
class BackwardWarpingNearest(WarpingBase):
    """
    flow 定義:target→source (對目標像素 (u,v) 給出位移 dx,dy 到來源）
    行為:對每個目標像素,從來源「抓取」最近鄰像素; hit=1 有效、0 越界。
    """
    def warp(self, src_image: torch.Tensor, flow_t2s: torch.Tensor):
        N, C, H, W = src_image.shape
        device = src_image.device

        # 目標網格 (u,v)
        u, v = self._make_uv_grid(N, H, W, device)

        xs = u + flow_t2s[:, 0]
        ys = v + flow_t2s[:, 1]

        xsn = xs.round().long()
        ysn = ys.round().long()

        mask = (xsn >= 0) & (xsn < W) & (ysn >= 0) & (ysn < H)

        xsn_c = xsn.clamp(0, W - 1)
        ysn_c = ysn.clamp(0, H - 1)
        flat_idx = self._flat_index(N, H, W, xsn_c, ysn_c)

        src_flat = src_image.permute(0, 2, 3, 1).reshape(-1, C)
        out = src_flat[flat_idx].view(N, H, W, C).permute(0, 3, 1, 2)

        # 越界清 0；hit 為 0/1
        out = out * mask[:, None, :, :]
        hit = mask[:, None, :, :].to(src_image.dtype)

        self.out = out
        self.hit = hit
        return out, hit