import os
from typing import Dict, Any

import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
from .dataset_config import MINOR_DATASET_CONFIGS, iter_dataset_configs
from .utils import load_backward_velocity


def random_resize(img0, imgt, img1, bmv, fmv, p=0.1):
    if random.uniform(0, 1) < p:
        img0 = cv2.resize(img0, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        imgt = cv2.resize(imgt, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        img1 = cv2.resize(img1, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        bmv = cv2.resize(bmv, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR) * 2.0
        fmv = cv2.resize(fmv, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR) * 2.0
    return img0, imgt, img1, bmv, fmv


def random_crop(img0, imgt, img1, bmv, fmv, crop_size=(224, 224)):
    h, w = crop_size[0], crop_size[1]
    ih, iw, _ = img0.shape
    x = np.random.randint(0, ih-h+1)
    y = np.random.randint(0, iw-w+1)
    img0 = img0[x:x+h, y:y+w, :]
    imgt = imgt[x:x+h, y:y+w, :]
    img1 = img1[x:x+h, y:y+w, :]
    bmv = bmv[x:x+h, y:y+w, :]
    fmv = fmv[x:x+h, y:y+w, :]
    return img0, imgt, img1, bmv, fmv


def random_reverse_channel(img0, imgt, img1, bmv, fmv, p=0.5):
    if random.uniform(0, 1) < p:
        img0 = img0[:, :, ::-1]
        imgt = imgt[:, :, ::-1]
        img1 = img1[:, :, ::-1]
    return img0, imgt, img1, bmv, fmv


def random_vertical_flip(img0, imgt, img1, bmv, fmv, p=0.3):
    if random.uniform(0, 1) < p:
        img0 = img0[::-1]
        imgt = imgt[::-1]
        img1 = img1[::-1]
        bmv = bmv[::-1]
        fmv = fmv[::-1]
        bmv = np.concatenate((bmv[:, :, 0:1], -bmv[:, :, 1:2], bmv[:, :, 2:3], -bmv[:, :, 3:4]), 2)
        fmv = np.concatenate((fmv[:, :, 0:1], -fmv[:, :, 1:2], fmv[:, :, 2:3], -fmv[:, :, 3:4]), 2)
    return img0, imgt, img1, bmv, fmv


def random_horizontal_flip(img0, imgt, img1, bmv, fmv, p=0.5):
    if random.uniform(0, 1) < p:
        img0 = img0[:, ::-1]
        imgt = imgt[:, ::-1]
        img1 = img1[:, ::-1]
        bmv = bmv[:, ::-1]
        fmv = fmv[:, ::-1]
        bmv = np.concatenate((-bmv[:, :, 0:1], bmv[:, :, 1:2], -bmv[:, :, 2:3], bmv[:, :, 3:4]), 2)
        fmv = np.concatenate((-fmv[:, :, 0:1], fmv[:, :, 1:2], -fmv[:, :, 2:3], fmv[:, :, 3:4]), 2)
    return img0, imgt, img1, bmv, fmv


def random_rotate(img0, imgt, img1, bmv, fmv, p=0.05):
    if random.uniform(0, 1) < p:
        img0 = img0.transpose((1, 0, 2))
        imgt = imgt.transpose((1, 0, 2))
        img1 = img1.transpose((1, 0, 2))
        bmv = bmv.transpose((1, 0, 2))
        fmv = fmv.transpose((1, 0, 2))
        bmv = np.concatenate((bmv[:, :, 1:2], bmv[:, :, 0:1], bmv[:, :, 3:4], bmv[:, :, 2:3]), 2)
        fmv = np.concatenate((fmv[:, :, 1:2], fmv[:, :, 0:1], fmv[:, :, 3:4], fmv[:, :, 2:3]), 2)
    return img0, imgt, img1, bmv, fmv


def random_reverse_time(img0, imgt, img1, bmv, fmv, p=0.5):
    if random.uniform(0, 1) < p:
        tmp = img1
        img1 = img0
        img0 = tmp
        bmv = np.concatenate((bmv[:, :, 2:4], bmv[:, :, 0:2]), 2)
        fmv = np.concatenate((fmv[:, :, 2:4], fmv[:, :, 0:2]), 2)
    return img0, imgt, img1, bmv, fmv

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
            input_fps: int,
            modality_config = DEFAULT_MODALITY_CONFIG,
            transform=None
        ):
        self.df = df
        self.root_dir = root_dir
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
        retries_count = 0
        while img is None:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # HxWxC
            retries_count += 1
            if retries_count > 5:
                raise FileNotFoundError(f"Image read failed: {path}")
        img = img[:, :, :3]

        # return torch.from_numpy(img.transpose(2,0,1)).float() / 255
        return img

    def _load_flow(self, path: str) -> torch.Tensor:
        # 你自己的 EXR loader 實作
        mv, depth = load_backward_velocity(path) 
        
        # return torch.from_numpy(mv).permute(2, 0, 1).float()
        return mv
    
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
            "ground_truth": {},
            "valid": {row["valid"]}
        }

        img_0_path = self._build_modality_path(self.record, mode, frame_0_idx, "colorNoScreenUI")
        img_1_path = self._build_modality_path(self.record, mode, frame_1_idx, "colorNoScreenUI")
        
        bmv = self._build_modality_path(self.record, mode, frame_1_idx, "backwardVel_Depth")

        item["input"]["colorNoScreenUI"] = (img_0_path, img_1_path)
        item["ground_truth"]["backwardVel_Depth"] = bmv

        return item

class VFIDataset(BaseDataset):
    def __init__(
        self,
        df: pd.DataFrame,
        root_dir: str,
        input_fps: int,
        modality_config = DEFAULT_MODALITY_CONFIG,
        transform=None,
    ):
        super().__init__(df=df, root_dir=root_dir, input_fps=input_fps, modality_config=modality_config, transform=transform)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        frame_0_idx = row["img0"]
        frame_1_idx = row["img1"]
        frame_2_idx = row["img2"]
        record = row["record"]
        mode = row["mode"]

        item = {
            "frame_range": f"frame_{frame_0_idx:04d}_{frame_2_idx:04d}",
            "input": {},
            "ground_truth": {},
            "valid": row["valid"],
            "distance_indexing": [row["D_index Mean"], row["D_index Median"]]
        }

        img_0_path = self._build_modality_path(record, mode, frame_0_idx, "colorNoScreenUI")
        img_1_path = self._build_modality_path(record, mode, frame_1_idx, "colorNoScreenUI")
        img_2_path = self._build_modality_path(record, mode, frame_2_idx, "colorNoScreenUI")

        bmv = self._build_modality_path(record, mode, frame_1_idx, "backwardVel_Depth")
        fmv = self._build_modality_path(record, mode, frame_1_idx, "forwardVel_Depth")

        item["input"]["colorNoScreenUI"] = (img_0_path, img_2_path)
        item["ground_truth"]["backwardVel_Depth"] = bmv
        item["ground_truth"]["forwardVel_Depth"] = fmv
        item["ground_truth"]["colorNoScreenUI"] = (img_1_path)

        return item

class VFITrainDataset(BaseDataset):
    def __init__(
        self, 
        df: pd.DataFrame, 
        root_dir: str,
        input_fps: int,
        augment: bool = True,
        modality_config = DEFAULT_MODALITY_CONFIG,
        transform=None
    ):
        super().__init__(df=df, root_dir=root_dir, input_fps=input_fps, modality_config=modality_config, transform=transform)
        self.augment = augment
        if self.input_fps != 30:
            raise ValueError("VFITrainDataset only supports input_fps=30 for now")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        frame_0_idx = row["img0"]
        frame_1_idx = row["img1"]
        frame_2_idx = row["img2"]
        record = row["record"]
        mode = row["mode"]

        info = {
            "frame_range": f"frame_{frame_0_idx:04d}_{frame_2_idx:04d}",
            "valid": row["valid"],
            "distance_indexing": [row["D_index Mean"], row["D_index Median"]]
        }

        img_0_path = self._build_modality_path(record, mode, frame_0_idx, "colorNoScreenUI")
        img_1_path = self._build_modality_path(record, mode, frame_1_idx, "colorNoScreenUI")
        img_2_path = self._build_modality_path(record, mode, frame_2_idx, "colorNoScreenUI")

        bmv_path = self._build_modality_path(record, mode, frame_1_idx, "backwardVel_Depth")
        fmv_path = self._build_modality_path(record, mode, frame_1_idx, "forwardVel_Depth")

        img0 = self._load_image(img_0_path)
        imgt = self._load_image(img_1_path)
        img1 = self._load_image(img_2_path)
        bmv = self._load_flow(bmv_path)
        fmv = self._load_flow(fmv_path)

        if self.augment:
            # img0, imgt, img1, bmv, fmv = random_resize(img0, imgt, img1, bmv, fmv, p=0.1)
            img0, imgt, img1, bmv, fmv = random_crop(img0, imgt, img1, bmv, fmv, crop_size=(224, 224))
            img0, imgt, img1, bmv, fmv = random_reverse_channel(img0, imgt, img1, bmv, fmv, p=0.5)
            img0, imgt, img1, bmv, fmv = random_vertical_flip(img0, imgt, img1, bmv, fmv, p=0.3)
            img0, imgt, img1, bmv, fmv = random_horizontal_flip(img0, imgt, img1, bmv, fmv, p=0.5)
            img0, imgt, img1, bmv, fmv = random_rotate(img0, imgt, img1, bmv, fmv, p=0.05)
            # img0, imgt, img1, bmv, fmv = random_reverse_time(img0, imgt, img1, bmv, fmv, p=0.5)

        img0 = torch.from_numpy(img0.transpose(2, 0, 1).astype(np.float32) / 255.0)
        imgt = torch.from_numpy(imgt.transpose(2, 0, 1).astype(np.float32) / 255.0)
        img1 = torch.from_numpy(img1.transpose(2, 0, 1).astype(np.float32) / 255.0)
        bmv = torch.from_numpy(bmv.transpose(2, 0, 1).astype(np.float32))
        fmv = torch.from_numpy(fmv.transpose(2, 0, 1).astype(np.float32))
        embt = torch.from_numpy(np.array(1/2).reshape(1, 1, 1).astype(np.float32))

        return img0, imgt, img1, bmv, fmv, embt, info
    
class FlowEstimationTrainDataset(BaseDataset):
    def __init__(
        self, 
        df: pd.DataFrame, 
        root_dir: str,
        input_fps: int,
        augment: bool = True,
        modality_config = DEFAULT_MODALITY_CONFIG,
        transform=None
    ):
        super().__init__(df=df, root_dir=root_dir, input_fps=input_fps, modality_config=modality_config, transform=transform)
        self.augment = augment
        if self.input_fps != 30:
            raise ValueError("VFITrainDataset only supports input_fps=30 for now")
        
        

    def __getitem__(self, idx):
        if self.df_fps == self.input_fps: # 60 -> 60
            raise NotImplementedError("120 fps still not ready")
        else: # 30 -> 60
            row = self.df.iloc[idx * 2]
            frame_30_0_idx = row["img0"] // 2
            frame_30_1_idx = row["img2"] // 2
            frame_60_0_idx = row["img0"]
            frame_60_1_idx = row["img1"]
            frame_60_2_idx = row["img2"]

        record = self.df.iloc[idx]["record"]
        mode = self.df.iloc[idx]["mode"]

        info = {
            "frame_range_60": f"frame_60_0_idx_{frame_60_0_idx:04d}_{frame_60_2_idx:04d}",
            "frame_range_30": f"frame_30_0_idx_{frame_30_0_idx:04d}_{frame_30_1_idx:04d}",
            "valid": row["valid"],
            "distance_indexing": [row["D_index Mean"], row["D_index Median"]]
        }

        img_60_0_path = self._build_modality_path(record, mode, frame_60_0_idx, "colorNoScreenUI")
        img_60_1_path = self._build_modality_path(record, mode, frame_60_1_idx, "colorNoScreenUI")
        img_60_2_path = self._build_modality_path(record, mode, frame_60_2_idx, "colorNoScreenUI")

        bmv_60_path = self._build_modality_path(record, mode, frame_60_1_idx, "backwardVel_Depth")
        fmv_60_path = self._build_modality_path(record, mode, frame_60_1_idx, "forwardVel_Depth")

        bmv_30_path = self._build_modality_path(record, mode.replace("fps_60", "fps_30"), frame_30_1_idx, "backwardVel_Depth")
        fmv_30_path = self._build_modality_path(record, mode.replace("fps_60", "fps_30"), frame_30_0_idx, "forwardVel_Depth")
        img_30_0_path = self._build_modality_path(record, mode.replace("fps_60", "fps_30"), frame_30_0_idx, "colorNoScreenUI")
        img_30_1_path = self._build_modality_path(record, mode.replace("fps_60", "fps_30"), frame_30_1_idx, "colorNoScreenUI")

        info["img_60_2_path"] = img_60_2_path
        info["img_30_1_path"] = img_30_1_path

        img0 = self._load_image(img_60_0_path)
        imgt = self._load_image(img_60_1_path)
        img1 = self._load_image(img_60_2_path)
        bmv_60 = self._load_flow(bmv_60_path)
        fmv_60 = self._load_flow(fmv_60_path)
        bmv_30 = self._load_flow(bmv_30_path)
        fmv_30 = self._load_flow(fmv_30_path)
        img0_30 = self._load_image(img_30_0_path)
        img1_30 = self._load_image(img_30_1_path)

        if self.augment:
            # img0, imgt, img1, bmv, fmv = random_resize(img0, imgt, img1, bmv, fmv, p=0.1)
            # img0, imgt, img1, bmv, fmv = random_crop(img0, imgt, img1, bmv, fmv, crop_size=(224, 224))
            # img0, imgt, img1, bmv, fmv = random_reverse_channel(img0, imgt, img1, bmv, fmv, p=0.5)
            # img0, imgt, img1, bmv, fmv = random_vertical_flip(img0, imgt, img1, bmv, fmv, p=0.3)
            # img0, imgt, img1, bmv, fmv = random_horizontal_flip(img0, imgt, img1, bmv, fmv, p=0.5)
            # img0, imgt, img1, bmv, fmv = random_rotate(img0, imgt, img1, bmv, fmv, p=0.05)
            # img0, imgt, img1, bmv, fmv = random_reverse_time(img0, imgt, img1, bmv, fmv, p=0.5)
            pass
            
        img0 = torch.from_numpy(img0.transpose(2, 0, 1).astype(np.float32) / 255.0)
        imgt = torch.from_numpy(imgt.transpose(2, 0, 1).astype(np.float32) / 255.0)
        img1 = torch.from_numpy(img1.transpose(2, 0, 1).astype(np.float32) / 255.0)
        bmv_60 = torch.from_numpy(bmv_60.transpose(2, 0, 1).astype(np.float32))
        fmv_60 = torch.from_numpy(fmv_60.transpose(2, 0, 1).astype(np.float32))
        bmv_30 = torch.from_numpy(bmv_30.transpose(2, 0, 1).astype(np.float32))
        fmv_30 = torch.from_numpy(fmv_30.transpose(2, 0, 1).astype(np.float32))
        img0_30 = torch.from_numpy(img0_30.transpose(2, 0, 1).astype(np.float32) / 255.0)
        img1_30 = torch.from_numpy(img1_30.transpose(2, 0, 1).astype(np.float32) / 255.0)
        embt = torch.from_numpy(np.array(1/2).reshape(1, 1, 1).astype(np.float32))

        return img0, imgt, img1, bmv_60, fmv_60, bmv_30, fmv_30, img0_30, img1_30, embt, info
    
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
