import torch
import numpy as np

class Warping:
    def __init__(self, warp_mode="nearest"):
        self.warp_mode = warp_mode
        self.valid_warp_modes = ["nearest"]

        # warping module variable
        self.out = None
        self.hit = None

    def set_warp_mode(self, mode):
        if mode in self.valid_warp_modes:
            self.warp_mode = mode
        else:
            raise ValueError(f"Invalid warp mode. Choose from {self.valid_warp_modes}.")

    def warp(self, image, flow):
        if self.warp_mode == "nearest":
            return self._warp_nearest(image, flow)
        else:
            raise ValueError("Unsupported warp mode.")
        
    def _warp_nearest(self, src_image, flow):
        # Implement nearest neighbor warping logic here
            N, C, H, W = src_image.shape
            device = src_image.device

            # 建 (y,x) 網格
            yy, xx = torch.meshgrid(
                torch.arange(H, device=device), torch.arange(W, device=device),
                indexing='ij'
            )  # [H,W]
            xx = xx.unsqueeze(0).expand(N, -1, -1).float()
            yy = yy.unsqueeze(0).expand(N, -1, -1).float()

            tx = xx + flow[:,0]  # [N,H,W]
            ty = yy + flow[:,1]  # [N,H,W]

            # 最近鄰
            txn = tx.round().long()
            tyn = ty.round().long()

            # 有效範圍
            mask = (txn >= 0) & (txn < W) & (tyn >= 0) & (tyn < H)

            out = torch.zeros_like(src_image)
            hit = torch.zeros((N,1,H,W), device=device, dtype=src_image.dtype)

            # 展平 index，做 scatter_add
            linear_idx = tyn.clamp(0, H-1) * W + txn.clamp(0, W-1)  # [N,H,W]
            base = (torch.arange(N, device=device) * (H*W)).view(N,1,1)
            flat_idx = (linear_idx + base).view(-1)  # [N*H*W]

            src_image_flat = src_image.permute(0,2,3,1).reshape(-1, C)  # [N*H*W, C]
            mask_flat = mask.view(-1)

            out_flat = torch.zeros((N*H*W, C), device=device, dtype=src_image.dtype)
            out_flat.index_add_(0, flat_idx[mask_flat], src_image_flat[mask_flat])
            out = out_flat.view(N, H, W, C).permute(0,3,1,2)

            hit_flat = torch.zeros((N*H*W, 1), device=device, dtype=src_image.dtype)
            one = torch.ones((mask_flat.sum(),1), device=device, dtype=src_image.dtype)
            hit_flat.index_add_(0, flat_idx[mask_flat], one)
            hit = hit_flat.view(N, H, W, 1).permute(0,3,1,2)

            self.out = out
            self.hit = hit

    def warp_backward_nearest(self, src_image, flow_t2s):
        """
        src_image: [N, C, H, W] 來源影像
        flow_t2s : [N, 2, H, W] 目標→來源 的位移（像素座標；dx, dy）
                對於目標(u,v)，來源座標 = (u + dx, v + dy)
        return:
        out : [N, C, H, W]  最近鄰取樣結果
        mask: [N, 1, H, W]  True 表示座標在合法邊界內
        """
        N, C, H, W = src_image.shape
        device = src_image.device

        # 目標網格 (u,v)
        v, u = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )
        u = u[None].float().expand(N, -1, -1)  # [N,H,W]
        v = v[None].float().expand(N, -1, -1)

        # 目標 -> 來源 的連續座標
        xs = u + flow_t2s[:, 0]   # [N,H,W]
        ys = v + flow_t2s[:, 1]   # [N,H,W]

        # 最近鄰（四捨五入）+ 邊界檢查
        xsn = xs.round().long()
        ysn = ys.round().long()
        mask = (xsn >= 0) & (xsn < W) & (ysn >= 0) & (ysn < H)  # [N,H,W]

        # 建立扁平索引，從來源抓取（gather）
        xsn_c = xsn.clamp(0, W - 1)
        ysn_c = ysn.clamp(0, H - 1)
        base = (torch.arange(N, device=device) * (H * W)).view(N, 1, 1)
        flat_idx = (ysn_c * W + xsn_c + base).view(-1)                  # [N*H*W]
        src_flat = src_image.permute(0, 2, 3, 1).reshape(-1, C)         # [N*H*W, C]

        out = src_flat[flat_idx].view(N, H, W, C).permute(0, 3, 1, 2)   # [N,C,H,W]
        # 對越界處清 0（可改成其他 padding 值）
        out = out * mask[:, None, :, :]
        
        self.out = out
        self.hit = mask[:, None, :, :].float()

    def get_warping_result(self, mode="average"):
        valid_modes = ["average", "raw"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode. Choose from {valid_modes}.")

        if self.out is None or self.hit is None:
            raise ValueError("No warping has been performed yet.")

        if mode == "average":
            warped_avg = self.out / (self.hit + 1e-6)
            warped_avg = torch.where(self.hit > 0, warped_avg, torch.zeros_like(warped_avg))  # 沒命中就置零/保持無值
            return warped_avg, self.hit
        elif mode == "raw":
            return self.out, self.hit
        
    def visualize_hit(self):
        if self.hit is None:
            raise ValueError("No warping has been performed yet.")
        hmap = self.hit[0,0].cpu().numpy().astype(np.int32)
        H, W = hmap.shape

        vis = np.zeros((H,W,3), dtype=np.uint8)

        vis[hmap == 0] = (255, 0, 0)   # none-to-one → 藍 (BGR)
        vis[hmap == 1] = (0, 255, 0)   # one-to-one  → 綠
        vis[hmap >= 2] = (0, 0, 255)   # many-to-one → 紅

        return vis