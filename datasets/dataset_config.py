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
    def record_name(self) -> str:
        return self.record
    
# ------------------------------ USUAL CONFIG TO USED ------------------------------
MINOR_DATASET_CONFIGS = {
    "root_dir": "/datasets/VFI/datasets/AnimeFantasyRPG",
    "records": {
        "AnimeFantasyRPG_3_60": {
            "main_indices": ["0", "1"],
            "difficulties": ["Easy", "Medium"],
            "sub_index": ["0", "0"],
            "fps": [30, 60],
            "max_index": [400, 800],  # depending on fps
        }
    }
}

DATASET_CONFIGS = {
    "root_dir": "/datasets/VFI/datasets/AnimeFantasyRPG",
    "records": {
        "AnimeFantasyRPG_3_60": {
            "main_indices": ["0", "1", "2", "3", "4"],
            "difficulties": ["Easy", "Medium"],
            "sub_index": ["0", "0", "0", "0", "0"],
            "fps": [30, 60],
            "max_index": [400, 800],  # depending on fps
        }
    }
}

# iter function to yield DatasetConfig
def iter_dataset_configs(config_dict: Dict[str, Any]) -> Iterable[DatasetConfig]:
    """
    給 MINOR_DATASET_CONFIGS 或 DATASET_CONFIGS 都可以。
    會 yield 出一堆 DatasetConfig。
    """
    records_cfg = config_dict["records"]

    for record_name, rec_cfg in records_cfg.items():
        main_indices  = rec_cfg["main_indices"]      # ["0", "1", ...]
        difficulties  = rec_cfg["difficulties"]      # ["Easy", "Medium", ...]
        sub_index_lst = rec_cfg["sub_index"]         # 對應 main_indices
        fps_list      = rec_cfg["fps"]               # [30, 60]
        max_index_lst = rec_cfg["max_index"]         # [400, 800]

        # main_idx -> sub_idx
        main_to_sub = dict(zip(main_indices, sub_index_lst))

        # fps -> max_index
        fps_to_max = dict(zip(fps_list, max_index_lst))

        # 做 main_idx × difficulty × fps 的組合
        for main_idx, difficulty, fps in product(main_indices, difficulties, fps_list):
            sub_idx   = main_to_sub[main_idx]
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
    for cfg in iter_dataset_configs(DATASET_CONFIGS):
        print(cfg.mode_name)

    # get datasets config with filters
    print("\nFiltered Dataset Configs (fps=60, difficulty='Easy'):")
    for cfg in iter_dataset_configs(DATASET_CONFIGS):
        if cfg.fps == 60 and cfg.difficulty == "Easy":
            print(cfg.mode_name)