from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

DEFAULT_CONFIG: dict[str, Any] = {
    "image_size": 224,
    "batch_size": 32,
    "epochs": 15,
    "fine_tune_epochs": 10,
    "learning_rate": 1e-4,
    "fine_tune_learning_rate": 1e-5,
    "weight_decay": 1e-5,
    "fine_tune_at": 16,
    "class_names": ["mask", "no_mask", "incorrect_mask"],
    "dataset_dir": "dataset",
    "model_path": "outputs/best_model.keras",
    "final_model_path": "outputs/final_model.keras",
    "class_names_path": "outputs/class_names.json",
    "output_dir": "outputs",
    "prediction_output_dir": "outputs/predictions",
    "evaluation_output_dir": "outputs/evaluation",
    "training_output_dir": "outputs/training",
    "min_confidence": 0.9,
    "min_margin": 0.25,
}


def resolve_project_path(path_value: str | Path) -> Path:
    """Resolve relative project paths from the repository root."""
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load project configuration, falling back to safe defaults."""
    config = deepcopy(DEFAULT_CONFIG)
    path = resolve_project_path(config_path) if config_path else DEFAULT_CONFIG_PATH

    if path.exists():
        with open(path, "r", encoding="utf-8") as file:
            user_config = yaml.safe_load(file) or {}
        if not isinstance(user_config, dict):
            raise ValueError(f"Config file must contain a YAML object: {path}")
        config.update(user_config)
    elif config_path:
        raise FileNotFoundError(f"Config file not found: {path}")

    config["_config_path"] = str(path)
    config["_project_root"] = str(PROJECT_ROOT)
    return config


def get_config_value(config: dict[str, Any], key: str, override: Any = None) -> Any:
    """Return a CLI override when supplied, otherwise the config value."""
    return override if override is not None else config[key]


def display_class_name(class_name: str) -> str:
    """Convert a folder-safe class name into a presentation-friendly label."""
    return class_name.replace("_", " ").title()
