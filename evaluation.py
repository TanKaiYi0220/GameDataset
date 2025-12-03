# evaluation.py
from typing import Callable, Dict, Any, List
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch

# VFI metrics (VFI 專用)
def vfi_psnr_metric(*, img_gt: np.ndarray, img_pred: np.ndarray, **kwargs) -> float:
    return psnr(img_gt, img_pred, data_range=255.0)

def vfi_epe_1_to_0_metric(*, flow_1_to_0, bmv, **kwargs) -> float:
    # expect tensor [B, 2, H, W]
    return torch.norm(flow_1_to_0 - bmv, dim=1).mean().cpu().item()

def vfi_epe_1_to_2_metric(*, flow_1_to_2, fmv, **kwargs) -> float:
    return torch.norm(flow_1_to_2 - fmv, dim=1).mean().cpu().item()

VFI_METRICS = {
    "psnr": vfi_psnr_metric,
    "epe_1_to_0": vfi_epe_1_to_0_metric,
    "epe_1_to_2": vfi_epe_1_to_2_metric,
}

class TaskEvaluator:
    """
    Generic evaluator for different tasks (VFI, FlowEstimation, etc.)

    使用方式：
    - 建構時給 task_name + metric_fns(dict)
    - 每個 sample 呼叫 evaluate(meta=..., **data)
    - 之後可以 summary() / to_dataframe()
    """
    def __init__(self, task_name: str, metric_fns: Dict[str, Callable[..., float]]):
        self.task_name = task_name
        self.metric_fns = metric_fns  # e.g. {"psnr": fn, "epe": fn}
        self.rows: List[Dict[str, Any]] = []

    def evaluate(self, *, meta: Dict[str, Any], **data: Any) -> Dict[str, Any]:
        """
        meta: 一些識別用資訊 (record, mode, frame_range, inference_time, ...)
        data: metric 計算需要的原始資料 (img_gt, img_pred, flow_1_to_0, bmv, ...)

        回傳：meta + 所有 metrics 的 dict，並且 append 到內部列表
        """
        metrics = {}
        for name, fn in self.metric_fns.items():
            metrics[name] = fn(**data)

        row = {**meta, **metrics}
        self.rows.append(row)
        return row

    def summary(self) -> Dict[str, float]:
        """
        對所有 numeric 欄位取平均，當作全體 summary。
        （非數值欄位會略過）
        """
        if not self.rows:
            return {}

        # collect numeric keys
        numeric_keys = set()
        for row in self.rows:
            for k, v in row.items():
                if isinstance(v, (int, float)):
                    numeric_keys.add(k)

        summary = {}
        for k in numeric_keys:
            vals = [row[k] for row in self.rows if isinstance(row.get(k, None), (int, float))]
            if vals:
                summary[f"{k}_mean"] = float(np.mean(vals))

        summary["num_samples"] = len(self.rows)
        summary["task_name"] = self.task_name
        return summary

    def to_dataframe(self) -> pd.DataFrame:
        """
        直接把所有 per-sample rows 轉成 DataFrame，方便存 CSV 或做分析。
        """
        return pd.DataFrame(self.rows)
