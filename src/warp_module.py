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
    
class ForwardWarpingNearestWithDepth(WarpingBase):
    """
    Forward warping with Z-buffer occlusion handling.
    flow 定義: source→target (來源像素 (x,y) + flow → target 位置)
    depth: [N,1,H,W] 深度圖 (越小表示越近)
    """
    def warp(self, src_image: torch.Tensor, flow_s2t: torch.Tensor, depth: torch.Tensor):
        N, C, H, W = src_image.shape
        device = src_image.device

        # 來源座標網格
        u, v = self._make_uv_grid(N, H, W, device)

        tx = u + flow_s2t[:, 0]
        ty = v + flow_s2t[:, 1]

        txn = tx.round().long()
        tyn = ty.round().long()

        # 有效範圍
        mask = (txn >= 0) & (txn < W) & (tyn >= 0) & (tyn < H)

        # ---- flatten ----
        linear_idx = tyn.clamp(0, H - 1) * W + txn.clamp(0, W - 1)
        base = (torch.arange(N, device=device) * (H * W)).view(N, 1, 1)
        flat_idx = (linear_idx + base).view(-1)

        src_flat = src_image.permute(0, 2, 3, 1).reshape(-1, C)
        depth_flat = depth.view(-1)

        mask_flat = mask.view(-1)
        idx_valid = flat_idx[mask_flat]
        depth_valid = depth_flat[mask_flat]
        color_valid = src_flat[mask_flat]

        # ---- Z-buffer with scatter_reduce ----
        # 每個 target 的最大 depth
        depth_buffer = torch.full((N * H * W,), float("-inf"), device=device)
        depth_buffer.scatter_reduce_(0, idx_valid, depth_valid, reduce="amax", include_self=True)

        # 選擇等於 depth_buffer 的 pixel
        front_mask = (depth_valid == depth_buffer[idx_valid])

        best_idx = idx_valid[front_mask]
        best_color = color_valid[front_mask]

        # ---- 回填到輸出 ----
        out_flat = torch.zeros((N * H * W, C), device=device, dtype=src_image.dtype)
        hit_flat = torch.zeros((N * H * W, 1), device=device, dtype=src_image.dtype)

        out_flat[best_idx] = best_color
        hit_flat[best_idx] = 1.0

        out = out_flat.view(N, H, W, C).permute(0, 3, 1, 2)
        hit = hit_flat.view(N, H, W, 1).permute(0, 3, 1, 2)

        self.out = out
        self.hit = hit
        return out, hit
    
class OcclusionMotionVector(WarpingBase):
    """
    重構版：依序做
      1. backward (取得 y 與 hit_y)
      2. forward (取得 z 與 hit_z)
      3. 計算 OMV: x^O_{t1} = y + (z - x_t2)
    """

    @torch.no_grad()
    def compute(self,
                flow_t1_to_t2: torch.Tensor,  # forward
                flow_t2_to_t1: torch.Tensor):  # backward
        N, _, H, W = flow_t1_to_t2.shape
        device = flow_t1_to_t2.device

        # =========================
        # Backward: t2→t1，取得 y 和 hit_y
        # =========================
        u2, v2 = self._make_uv_grid(N, H, W, device)  # grid of t2
        y_x = u2 + flow_t2_to_t1[:, 0]
        y_y = v2 + flow_t2_to_t1[:, 1]
        y_map = torch.stack((y_x, y_y), dim=1)  # [N,2,H,W]

        # hit_y: 合法範圍內 (非越界)
        hit_y = ((y_x >= 0) & (y_x < W) & (y_y >= 0) & (y_y < H)).float()[:, None, :, :]

        # =========================
        # Forward: 將 y 再通過 flow_t1_to_t2 投影成 z
        # =========================
        # 先在 t1 網格中建立座標（與 flow_t1_to_t2 對應）
        u1, v1 = self._make_uv_grid(N, H, W, device)

        # 將 flow_t1_to_t2 warp 到 y_map 的位置（雙線性取樣）
        gx = 2.0 * (y_x / (W - 1)) - 1.0
        gy = 2.0 * (y_y / (H - 1)) - 1.0
        grid_y = torch.stack((gx, gy), dim=-1)  # [N,H,W,2]

        flow_at_y = F.grid_sample(
            flow_t1_to_t2, grid_y,
            mode="bilinear", padding_mode="zeros", align_corners=True
        )  # [N,2,H,W]

        z_map = y_map + flow_at_y  # [N,2,H,W]

        # hit_z: 合法範圍內 (非越界)
        z_x, z_y = z_map[:, 0], z_map[:, 1]
        hit_z = ((z_x >= 0) & (z_x < W) & (z_y >= 0) & (z_y < H)).float()[:, None, :, :]

        # =========================
        # OMV 計算
        # =========================
        # OMV = (y + (z - x2)) - x2 = y + z - 2*x2
        x2 = torch.stack((u2, v2), dim=1)
        omv = y_map + (z_map - x2) - x2  # [N,2,H,W]

        # =========================
        # 輸出與 debug
        # =========================
        self.y_map = y_map
        self.z_map = z_map
        self.hit_y = hit_y
        self.hit_z = hit_z

        self.out = omv
        self.hit = (hit_y * hit_z)  # 同時合法才視為有效
        return omv, self.hit
