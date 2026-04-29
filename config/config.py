import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
USER_CONFIG_PATH = REPO_ROOT / "config" / "user_config.json"


def load_user_config():
    if USER_CONFIG_PATH.exists():
        with USER_CONFIG_PATH.open("r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                raise RuntimeError(
                    f"Invalid JSON in user config: {USER_CONFIG_PATH}"
                )
    return {}


USER_CONFIG = load_user_config()

PROJECT_CONFIG = {
    "dataset_root": USER_CONFIG.get("dataset_root", "/workspace/datasets"),
    "model_root": USER_CONFIG.get("model_root", "models"),
    "output_root": USER_CONFIG.get("output_root", "output"),
    "data_root": USER_CONFIG.get("data_root", "./datasets/data"),
    "log_root": USER_CONFIG.get("log_root", "./logs"),
}


def get_config(key, default=None):
    return PROJECT_CONFIG.get(key, default)
