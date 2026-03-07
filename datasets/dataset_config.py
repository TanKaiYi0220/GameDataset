from dataclasses import dataclass
from itertools import product
from typing import Iterable, Dict, Any

@dataclass
class DatasetConfig:
    record: str       # AnimeFantasyRPG_3_60
    main_idx: str     # "0"
    difficulty: str   # "Easy", "Medium", "Hard"
    sub_idx: str      # "0"
    fps: int          # 30, 60, 120
    max_index: int    # e.g., 800

    @property
    def mode_path(self) -> str:
        return f"{self.main_idx}_{self.difficulty}/{self.main_idx}_{self.difficulty}_{self.sub_idx}/fps_{self.fps}"
    
    @property
    def mode_name(self) -> str:
        return f"{self.main_idx}_{self.difficulty}_{self.sub_idx}_fps_{self.fps}"
    
    @property
    def mode_index(self) -> str:
        return f"{self.main_idx}_fps_{self.fps}"
    
    @property
    def record_name(self) -> str:
        return self.record
    
# ------------------------------ USUAL CONFIG TO USED ------------------------------
MINOR_DATASET_CONFIGS = {
    "name": "AnimeFantasyRPG_3",
    "root_dir": "/datasets/VFI/datasets/AnimeFantasyRPG",
    "records": {
        "AnimeFantasyRPG_3_60": {
            "main_indices": ["0", "1"],
            "difficulties": ["Easy", "Medium"],
            "sub_index": ["1", "1"],
            "fps": [30, 60],
            "max_index": [400, 800],  # depending on fps
        }
    }
}

DATASET_CONFIGS = {
    "name": "AnimeFantasyRPG_3_Full",
    "root_dir": "/datasets/VFI/datasets/AnimeFantasyRPG",
    "records": {
        "AnimeFantasyRPG_3_60": {
            "main_indices": ["0", "1", "2", "3"],
            "difficulties": ["Easy", "Medium", "Difficult"],
            "sub_index": ["1", "1", "1", "1"],
            "fps": [30, 60],
            "max_index": [400, 800],  # depending on fps
        }
    }
}

TRAIN_DATASET_CONFIGS = {
    "name": "AnimeFantasyRPG_3_Full",
    "root_dir": "/datasets/VFI/datasets/AnimeFantasyRPG",
    "records": {
        "AnimeFantasyRPG_3_60": {
            "main_indices": ["0", "1", "2", "0", "1", "2"],
            "difficulties": ["Easy", "Medium", "Difficult"],
            "sub_index": ["2", "2", "2", "4", "4", "4"],
            "fps": [30, 60],
            "max_index": [400, 800],  # depending on fps
        },
        "AnimeFantasyRPG_2_60": {
            "main_indices": ["4"],
            "difficulties": ["Easy", "Medium", "Difficult"],
            "sub_index": ["2"],
            "fps": [30, 60],
            "max_index": [400, 800],  # depending on fps
        }
    }
}

VFX_DATASET_CONFIGS = {
    "name": "AnimeFantasyRPG_3_VFX",
    "root_dir": "/datasets/VFI/datasets/AnimeFantasyRPG",
    "records": {
        "AnimeFantasyRPG_3_60": {
            "main_indices": ["0", "1", "2", "3"],
            "difficulties": ["Difficult"],
            "sub_index": ["1", "1", "1", "1"],
            "fps": [30, 60],
            "max_index": [400, 800],  # depending on fps
        }
    }
}

STAIR_DATASET_CONFIG = {
    "name": "AnimeFantasyRPG_2_STAIR",
    "root_dir": "/datasets/VFI/datasets/AnimeFantasyRPG",
    "records": {
        "AnimeFantasyRPG_2_60": {
            "main_indices": ["4"],
            "difficulties": ["Easy", "Medium", "Difficult"],
            "sub_index": ["0"],
            "fps": [30, 60],
            "max_index": [400, 800],  # depending on fps
        }
    }
}

TEST_DATASET_CONFIGS = {
    "name": "AnimeFantasyRPG_2_STAIR",
    "root_dir": "/datasets/VFI/datasets/AnimeFantasyRPG",
    "records": {
        "AnimeFantasyRPG_3_60": {
            "main_indices": ["3"],
            "difficulties": ["Easy", "Medium", "Difficult"],
            "sub_index": ["2"],
            "fps": [30, 60],
            "max_index": [400, 800],  # depending on fps
        }
    }
}

TEST_VFX_DATASET_CONFIGS = {
    "name": "AnimeFantasyRPG_2_STAIR",
    "root_dir": "/datasets/VFI/datasets/AnimeFantasyRPG",
    "records": {
        "AnimeFantasyRPG_3_60": {
            "main_indices": ["3"],
            "difficulties": ["Difficult"],
            "sub_index": ["3"],
            "fps": [30, 60],
            "max_index": [400, 800],  # depending on fps
        },
        "AnimeFantasyRPG_5_60": {
            "main_indices": ["0", "3"],
            "difficulties": ["Medium", "Difficult"],
            "sub_index": ["2", "2"],
            "fps": [30, 60],
            "max_index": [400, 800],  # depending on fps
        }
    }
}

# iter function to yield DatasetConfig
def iter_dataset_configs(config_dict):
    records_cfg = config_dict["records"]

    for record_name, rec_cfg in records_cfg.items():

        main_indices  = rec_cfg["main_indices"]
        difficulties  = rec_cfg["difficulties"]
        sub_index_lst = rec_cfg["sub_index"]
        fps_list      = rec_cfg["fps"]
        max_index_lst = rec_cfg["max_index"]

        fps_to_max = dict(zip(fps_list, max_index_lst))

        # 保持 main_idx 與 sub_idx 的對應
        for main_idx, sub_idx in zip(main_indices, sub_index_lst):

            for difficulty, fps in product(difficulties, fps_list):

                max_index = fps_to_max[fps]

                yield DatasetConfig(
                    record=record_name,
                    main_idx=main_idx,
                    difficulty=difficulty,
                    sub_idx=sub_idx,
                    fps=fps,
                    max_index=max_index,
                )

if __name__ == "__main__":
    # get all dataset configs
    print("All Dataset Configs:")
    for cfg in iter_dataset_configs(TRAIN_DATASET_CONFIGS):
        print(cfg.mode_name)

    # get datasets config with filters
    print("\nFiltered Dataset Configs (fps=60, difficulty='Easy'):")
    for cfg in iter_dataset_configs(TRAIN_DATASET_CONFIGS):
        if cfg.fps == 60 and cfg.difficulty == "Easy":
            print(cfg.mode_name)