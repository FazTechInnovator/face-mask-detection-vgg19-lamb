from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import keras
import tensorflow as tf

from config_utils import IMAGE_EXTENSIONS


AUTOTUNE = tf.data.AUTOTUNE
STANDARD_CLASS_SETS = [
    {"mask", "no_mask"},
    {"mask", "no_mask", "incorrect_mask"},
]


@dataclass(frozen=True)
class DatasetValidation:
    class_names: list[str]
    train_counts: dict[str, int]
    val_counts: dict[str, int]
    test_counts: dict[str, int]
    test_available: bool


def image_files(class_dir: Path) -> list[Path]:
    return [
        path
        for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def count_images_by_class(split_dir: Path) -> dict[str, int]:
    if not split_dir.exists():
        return {}

    counts: dict[str, int] = {}
    for class_dir in sorted(split_dir.iterdir()):
        if class_dir.is_dir() and not class_dir.name.startswith("."):
            counts[class_dir.name] = len(image_files(class_dir))
    return counts


def order_class_names(detected_classes: list[str], preferred_order: list[str]) -> list[str]:
    preferred = [class_name for class_name in preferred_order if class_name in detected_classes]
    remaining = sorted(set(detected_classes) - set(preferred))
    return preferred + remaining


def expected_dataset_message(data_dir: Path) -> str:
    return (
        f"Expected dataset structure under {data_dir}:\n"
        "  dataset/train/mask\n"
        "  dataset/train/no_mask\n"
        "  dataset/train/incorrect_mask   (optional for 2-class datasets)\n"
        "  dataset/val/mask\n"
        "  dataset/val/no_mask\n"
        "  dataset/val/incorrect_mask     (optional for 2-class datasets)\n"
        "  dataset/test/...               (recommended for final evaluation)"
    )


def validate_training_dataset(data_dir: Path, preferred_class_names: list[str]) -> DatasetValidation:
    """Validate train/val folders before TensorFlow starts reading images."""
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir}\n{expected_dataset_message(data_dir)}"
        )

    missing_required = [split for split in ("train", "val") if not (data_dir / split).exists()]
    if missing_required:
        raise FileNotFoundError(
            "Missing required dataset split folder(s): "
            + ", ".join(str(data_dir / split) for split in missing_required)
            + "\n"
            + expected_dataset_message(data_dir)
        )

    train_counts = count_images_by_class(data_dir / "train")
    val_counts = count_images_by_class(data_dir / "val")
    test_counts = count_images_by_class(data_dir / "test")

    detected_set = set(train_counts)
    if detected_set not in STANDARD_CLASS_SETS:
        raise ValueError(
            "Training classes must be either {'mask', 'no_mask'} or "
            "{'mask', 'no_mask', 'incorrect_mask'}.\n"
            f"Detected classes: {sorted(detected_set)}\n"
            + expected_dataset_message(data_dir)
        )

    class_names = order_class_names(list(train_counts), preferred_class_names)

    _validate_split_counts("train", train_counts, class_names, required=True)
    _validate_split_counts("val", val_counts, class_names, required=True)

    test_available = False
    if (data_dir / "test").exists():
        try:
            _validate_split_counts("test", test_counts, class_names, required=True)
            test_available = True
        except ValueError as exc:
            print(f"Warning: test split is present but incomplete. Test evaluation will be skipped.\n{exc}")

    return DatasetValidation(
        class_names=class_names,
        train_counts=train_counts,
        val_counts=val_counts,
        test_counts=test_counts,
        test_available=test_available,
    )


def validate_evaluation_split(data_dir: Path, split: str, class_names: list[str]) -> dict[str, int]:
    split_dir = data_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Evaluation split not found: {split_dir}")

    counts = count_images_by_class(split_dir)
    _validate_split_counts(split, counts, class_names, required=True)
    return counts


def _validate_split_counts(
    split: str,
    counts: dict[str, int],
    class_names: list[str],
    required: bool,
) -> None:
    expected = set(class_names)
    detected = set(counts)
    missing = sorted(expected - detected)
    extra = sorted(detected - expected)
    empty = sorted(class_name for class_name in class_names if counts.get(class_name, 0) == 0)

    messages: list[str] = []
    if missing:
        messages.append(f"missing class folder(s): {missing}")
    if extra:
        messages.append(f"unexpected class folder(s): {extra}")
    if empty:
        messages.append(
            "no image files found in: "
            + ", ".join(f"{split}/{class_name}" for class_name in empty)
        )

    if required and messages:
        extensions = ", ".join(sorted(IMAGE_EXTENSIONS))
        raise ValueError(
            f"Invalid dataset split '{split}': " + "; ".join(messages) + f"\nSupported image extensions: {extensions}"
        )


def load_dataset_split(
    split_dir: Path,
    class_names: list[str],
    image_size: int,
    batch_size: int,
    shuffle: bool,
) -> tf.data.Dataset:
    dataset = keras.utils.image_dataset_from_directory(
        split_dir,
        labels="inferred",
        label_mode="int",
        class_names=class_names,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=shuffle,
        seed=42 if shuffle else None,
    )
    return dataset.prefetch(buffer_size=AUTOTUNE)
