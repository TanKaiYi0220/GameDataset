import os
from typing import Dict, Any

import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
from .dataset_config import MINOR_DATASET_CONFIGS, iter_dataset_configs
from .utils import load_backward_velocity

DEFAULT_MODALITY_CONFIG = {
    "colorNoScreenUI": {
        "prefix": "colorNoScreenUI_",
        "ext": ".png",
        "loader": "image",   # 用哪種 loader 讀檔
        "subdir": "",        # 如果每個 modality 有額外子資料夾，就填 "color_no_ui"
    },
    "colorScreenWithUI": {
        "prefix": "colorScreenWithUI_",
        "ext": ".png",
        "loader": "image",
        "subdir": "",
    },
    "backwardVel_Depth": {
        "prefix": "backwardVel_Depth_",
        "ext": ".exr",
        "loader": "flow",
        "subdir": "",
    },
    "forwardVel_Depth": {
        "prefix": "forwardVel_Depth_",
        "ext": ".exr",
        "loader": "flow",
        "subdir": "",
    },
}

class BaseDataset(Dataset):
    def __init__(
            self, 
            df: pd.DataFrame, 
            root_dir: str,
            record: str,
            mode: str,
            input_fps: int,
            modality_config = DEFAULT_MODALITY_CONFIG,
            transform=None
        ):
        self.df = df
        self.root_dir = root_dir
        self.record = record
        self.mode = mode
        self.df_fps = df.iloc[0]["fps"]
        self.input_fps = input_fps
        self.modality_config = modality_config

        self.transform = transform
        if self.input_fps > self.df_fps:
            raise ValueError("Input FPS cannot be greater than the dataframe FPS")
        elif self.df_fps % self.input_fps != 0:
            raise ValueError("Dataframe FPS must be divisible by input FPS")

        # loader registry
        self._loaders = {
            "image": self._load_image,
            "flow": self._load_flow,
            # 之後要加 npy / pfm 等可以再補
        }

    def __len__(self):
        return int(len(self.df) * (self.input_fps / self.df_fps))
    
    def _build_base_dir(self, record: str, mode: str) -> str:
        # 根據你的實際結構調整
        return os.path.join(self.root_dir, record, mode)
    
    def _build_modality_path(self, record, mode, frame_idx, modality_name):
        """
        根據 config 組出單一 modality + 單一 frame_idx 的檔案路徑
        """
        spec = self.modality_config[modality_name]
        base_dir = self._build_base_dir(record, mode)

        if spec.get("subdir"):
            base_dir = os.path.join(base_dir, spec["subdir"])

        frame_str = str(frame_idx)
        filename = f"{spec['prefix']}{frame_str}{spec['ext']}"

        return os.path.join(base_dir, filename)
    
    def _load_image(self, path: str) -> torch.Tensor:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # HxWxC
        return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().cuda()

    def _load_flow(self, path: str) -> torch.Tensor:
        # 你自己的 EXR loader 實作
        mv, depth = load_backward_velocity(path) 
        
        return torch.from_numpy(mv).permute(2, 0, 1).unsqueeze(0).float().cuda()
    
# ------------ 核心 __getitem__ ------------

    def __getitem__(self, index):
        raise NotImplementedError("BaseDataset is an abstract class. Please use a subclass that implements __getitem__.")

class FlowEstimationDataset(BaseDataset):
    def __getitem__(self, idx):
        if self.df_fps == self.input_fps: # 60 -> 60
            row = self.df.iloc[idx]
            frame_0_idx = row["img0"]
            frame_1_idx = row["img1"]
            mode = self.mode
        else: # 30 -> 60
            row = self.df.iloc[idx * 2]
            frame_0_idx = row["img0"] // 2
            frame_1_idx = row["img2"] // 2
            mode = self.mode.replace("fps_60", "fps_30")

        item = {
            "frame_range": f"frame_{frame_0_idx:04d}_{frame_1_idx:04d}",
            "input": {},
            "ground_truth": {}
        }

        img_0_path = self._build_modality_path(self.record, mode, frame_0_idx, "colorNoScreenUI")
        img_1_path = self._build_modality_path(self.record, mode, frame_1_idx, "colorNoScreenUI")
        
        bmv = self._build_modality_path(self.record, mode, frame_1_idx, "backwardVel_Depth")

        item["input"]["colorNoScreenUI"] = (img_0_path, img_1_path)
        item["ground_truth"]["backwardVel_Depth"] = bmv

        return item

class VFIDataset(BaseDataset):
    def __getitem__(self, idx):
        row = self.df.iloc[idx * 2]
        frame_0_idx = row["img0"]
        frame_1_idx = row["img1"]
        frame_2_idx = row["img2"]

        item = {
            "frame_range": f"frame_{frame_0_idx:04d}_{frame_2_idx:04d}",
            "input": {},
            "ground_truth": {}
        }

        img_0_path = self._build_modality_path(self.record, self.mode, frame_0_idx, "colorNoScreenUI")
        img_1_path = self._build_modality_path(self.record, self.mode, frame_1_idx, "colorNoScreenUI")
        img_2_path = self._build_modality_path(self.record, self.mode, frame_2_idx, "colorNoScreenUI")

        bmv = self._build_modality_path(self.record, self.mode, frame_1_idx, "backwardVel_Depth")
        fmv = self._build_modality_path(self.record, self.mode, frame_1_idx, "forwardVel_Depth")

        item["input"]["colorNoScreenUI"] = (img_0_path, img_2_path)
        item["ground_truth"]["backwardVel_Depth"] = bmv
        item["ground_truth"]["forwardVel_depth"] = fmv
        item["ground_truth"]["colorNoScreenUI"] = (img_1_path)

        return item

if __name__ == "__main__":
    # Flow Estimation Dataset
    for cfg in iter_dataset_configs(MINOR_DATASET_CONFIGS):
        if cfg.fps != 60:
            continue

        df = pd.read_csv(f"./data/AnimeFantasyRPG_3_60_preprocessed/{cfg.mode_index}_clipped_frame_index.csv")

        print(cfg.record, cfg.mode_name)

        print("Input FPS 60")
        dataset = FlowEstimationDataset(
            df=df,
            root_dir=MINOR_DATASET_CONFIGS["root_dir"],
            record=cfg.record,
            mode=cfg.mode_path,
            input_fps=60,
        )

        print(len(dataset))

        sample = dataset[0]
        print(sample)

        sample = dataset[4]
        print(sample)

        print("\nInput FPS 30")
        dataset = FlowEstimationDataset(
            df=df,
            root_dir=MINOR_DATASET_CONFIGS["root_dir"],
            record=cfg.record,
            mode=cfg.mode_path,
            input_fps=30,
        )

        print(len(dataset))

        sample = dataset[0]
        print(sample)

        sample = dataset[4]
        print(sample)

        break

    # VFI Dataset
    for cfg in iter_dataset_configs(MINOR_DATASET_CONFIGS):
        if cfg.fps != 60:
            continue

        df = pd.read_csv(f"./data/AnimeFantasyRPG_3_60_preprocessed/{cfg.mode_index}_clipped_frame_index.csv")

        print("\nInput FPS 30")

        dataset = VFIDataset(
            df=df,
            root_dir=MINOR_DATASET_CONFIGS["root_dir"],
            record=cfg.record,
            mode=cfg.mode_path,
            input_fps=30,
        )

        print(len(dataset))

        sample = dataset[0]
        print(sample)

        sample = dataset[4]
        print(sample)

        break