from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from config_utils import IMAGE_EXTENSIONS, get_config_value, load_config, resolve_project_path


SOURCE_DATASET_NAME = "Face Mask Detection"
SOURCE_DATASET_URL = "https://www.kaggle.com/datasets/andrewmvd/face-mask-detection"
SOURCE_DATASET_VIEWER = "https://datasetninja.com/face-mask-detection"
SOURCE_DATASET_LICENSE = "CC0 1.0"

CLASS_MAP = {
    "with_mask": "mask",
    "without_mask": "no_mask",
    "mask_weared_incorrect": "incorrect_mask",
}

SPLIT_RATIOS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}


@dataclass(frozen=True)
class ObjectAnnotation:
    image_path: Path
    source_id: str
    source_label: str
    class_name: str
    box: tuple[int, int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download and prepare the public Face Mask Detection dataset into "
            "train/val/test classifier folders."
        )
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file.")
    parser.add_argument("--download", action="store_true", help="Download the public Kaggle dataset first.")
    parser.add_argument(
        "--download_source",
        choices=["kagglehub", "dataset-ninja"],
        default="kagglehub",
        help="Downloader backend. KaggleHub is the recommended default.",
    )
    parser.add_argument("--raw_dir", type=str, help="Raw dataset download/extraction directory.")
    parser.add_argument("--dataset_dir", type=str, help="Prepared classifier dataset directory.")
    parser.add_argument("--reset", action="store_true", help="Clear prepared dataset class folders before writing crops.")
    parser.add_argument("--seed", type=int, default=42, help="Random split seed.")
    parser.add_argument("--padding", type=float, default=0.18, help="Crop padding around each face box.")
    parser.add_argument("--min_crop_size", type=int, default=12, help="Skip boxes smaller than this size in pixels.")
    return parser.parse_args()


def download_dataset(raw_dir: Path, source: str) -> Path:
    """Download the public dataset and return the directory containing files."""
    raw_dir.mkdir(parents=True, exist_ok=True)

    if source == "kagglehub":
        try:
            import kagglehub
        except ImportError as exc:
            raise ImportError(
                "kagglehub is required for automatic download. Install requirements first:\n"
                "  pip install -r requirements.txt"
            ) from exc

        os.environ.setdefault("KAGGLEHUB_CACHE", str(raw_dir.parent / "kagglehub_cache"))
        print(f"Downloading Kaggle dataset {SOURCE_DATASET_URL} ...")
        return Path(kagglehub.dataset_download("andrewmvd/face-mask-detection")).resolve()

    try:
        import dataset_tools as dtools
    except ImportError as exc:
        raise ImportError(
            "dataset-tools is required for Dataset Ninja download. "
            "Install it with: pip install dataset-tools"
        ) from exc

    print(f"Downloading Dataset Ninja copy of {SOURCE_DATASET_NAME} to {raw_dir} ...")
    dtools.download(dataset=SOURCE_DATASET_NAME, dst_dir=str(raw_dir))
    return raw_dir


def count_prepared_images(split_dir: Path) -> dict[str, int]:
    if not split_dir.exists():
        return {}

    counts: dict[str, int] = {}
    for class_dir in sorted(split_dir.iterdir()):
        if class_dir.is_dir() and not class_dir.name.startswith("."):
            counts[class_dir.name] = sum(
                1
                for path in class_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
    return counts


def build_image_indexes(search_roots: list[Path]) -> tuple[dict[str, Path], dict[str, Path]]:
    by_name: dict[str, Path] = {}
    by_stem: dict[str, Path] = {}
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                by_name.setdefault(path.name.lower(), path)
                by_stem.setdefault(path.stem.lower(), path)
    return by_name, by_stem


def find_image_for_annotation(annotation_path: Path, by_name: dict[str, Path], by_stem: dict[str, Path]) -> Path | None:
    name_without_json = annotation_path.name[:-5] if annotation_path.name.lower().endswith(".json") else annotation_path.name
    candidate = by_name.get(name_without_json.lower())
    if candidate:
        return candidate

    candidate_stem = Path(name_without_json).stem.lower()
    return by_stem.get(candidate_stem) or by_stem.get(annotation_path.stem.lower())


def normalize_box(points: Any) -> tuple[int, int, int, int] | None:
    if not isinstance(points, list) or len(points) < 2:
        return None

    try:
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
    except (TypeError, ValueError, IndexError):
        return None

    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def parse_supervisely_json(annotation_path: Path, image_path: Path) -> list[ObjectAnnotation]:
    try:
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return []

    objects = data.get("objects", [])
    if not isinstance(objects, list):
        return []

    annotations: list[ObjectAnnotation] = []
    for index, item in enumerate(objects):
        if not isinstance(item, dict):
            continue

        source_label = str(item.get("classTitle", ""))
        class_name = CLASS_MAP.get(source_label)
        points = item.get("points", {}).get("exterior")
        box = normalize_box(points)
        if class_name is None or box is None:
            continue

        annotations.append(
            ObjectAnnotation(
                image_path=image_path,
                source_id=image_path.stem,
                source_label=source_label,
                class_name=class_name,
                box=box,
            )
        )

    return annotations


def parse_pascal_voc_xml(annotation_path: Path, by_name: dict[str, Path], by_stem: dict[str, Path]) -> list[ObjectAnnotation]:
    try:
        root = ET.parse(annotation_path).getroot()
    except ET.ParseError:
        return []

    filename_node = root.find("filename")
    image_path = None
    if filename_node is not None and filename_node.text:
        image_path = by_name.get(filename_node.text.lower()) or by_stem.get(Path(filename_node.text).stem.lower())
    if image_path is None:
        image_path = by_stem.get(annotation_path.stem.lower())
    if image_path is None:
        return []

    annotations: list[ObjectAnnotation] = []
    for item in root.findall("object"):
        label_node = item.find("name")
        box_node = item.find("bndbox")
        if label_node is None or box_node is None or not label_node.text:
            continue

        class_name = CLASS_MAP.get(label_node.text)
        if class_name is None:
            continue

        try:
            xmin = int(float(box_node.findtext("xmin", "0")))
            ymin = int(float(box_node.findtext("ymin", "0")))
            xmax = int(float(box_node.findtext("xmax", "0")))
            ymax = int(float(box_node.findtext("ymax", "0")))
        except ValueError:
            continue

        annotations.append(
            ObjectAnnotation(
                image_path=image_path,
                source_id=image_path.stem,
                source_label=label_node.text,
                class_name=class_name,
                box=(xmin, ymin, xmax, ymax),
            )
        )

    return annotations


def collect_annotations(raw_dir: Path) -> list[ObjectAnnotation]:
    search_roots = [raw_dir, raw_dir.parent]
    by_name, by_stem = build_image_indexes(search_roots)
    annotations: list[ObjectAnnotation] = []

    for annotation_path in raw_dir.rglob("*.xml"):
        annotations.extend(parse_pascal_voc_xml(annotation_path, by_name, by_stem))

    for annotation_path in raw_dir.rglob("*.json"):
        image_path = find_image_for_annotation(annotation_path, by_name, by_stem)
        if image_path is None:
            continue
        annotations.extend(parse_supervisely_json(annotation_path, image_path))

    unique: dict[tuple[Path, str, tuple[int, int, int, int]], ObjectAnnotation] = {}
    for annotation in annotations:
        unique[(annotation.image_path, annotation.class_name, annotation.box)] = annotation

    return list(unique.values())


def split_by_source(
    annotations: list[ObjectAnnotation],
    seed: int,
) -> dict[str, list[ObjectAnnotation]]:
    source_ids = sorted({annotation.source_id for annotation in annotations})
    random.Random(seed).shuffle(source_ids)

    train_end = int(len(source_ids) * SPLIT_RATIOS["train"])
    val_end = train_end + int(len(source_ids) * SPLIT_RATIOS["val"])

    split_by_source_id: dict[str, str] = {}
    for source_id in source_ids[:train_end]:
        split_by_source_id[source_id] = "train"
    for source_id in source_ids[train_end:val_end]:
        split_by_source_id[source_id] = "val"
    for source_id in source_ids[val_end:]:
        split_by_source_id[source_id] = "test"

    splits = {split: [] for split in SPLIT_RATIOS}
    for annotation in annotations:
        splits[split_by_source_id[annotation.source_id]].append(annotation)

    return splits


def reset_prepared_dataset(dataset_dir: Path, class_names: list[str]) -> None:
    dataset_dir_resolved = dataset_dir.resolve()
    for split in SPLIT_RATIOS:
        for class_name in class_names:
            class_dir = (dataset_dir / split / class_name).resolve()
            if not str(class_dir).startswith(str(dataset_dir_resolved)):
                raise RuntimeError(f"Refusing to clear path outside dataset directory: {class_dir}")
            if class_dir.exists():
                shutil.rmtree(class_dir)
            class_dir.mkdir(parents=True, exist_ok=True)


def ensure_empty_or_reset(dataset_dir: Path, class_names: list[str], reset: bool) -> None:
    existing_images = [
        path
        for split in SPLIT_RATIOS
        for class_name in class_names
        for path in (dataset_dir / split / class_name).glob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if existing_images and not reset:
        raise RuntimeError(
            f"Prepared dataset already contains {len(existing_images)} image(s). "
            "Re-run with --reset to rebuild it deterministically."
        )

    if reset:
        reset_prepared_dataset(dataset_dir, class_names)
    else:
        for split in SPLIT_RATIOS:
            for class_name in class_names:
                (dataset_dir / split / class_name).mkdir(parents=True, exist_ok=True)


def save_crop(
    annotation: ObjectAnnotation,
    destination: Path,
    padding: float,
    min_crop_size: int,
) -> bool:
    try:
        with Image.open(annotation.image_path) as image:
            image = image.convert("RGB")
            image_width, image_height = image.size
            xmin, ymin, xmax, ymax = annotation.box

            crop_width = xmax - xmin
            crop_height = ymax - ymin
            if crop_width < min_crop_size or crop_height < min_crop_size:
                return False

            pad_x = int(crop_width * padding)
            pad_y = int(crop_height * padding)
            left = max(0, xmin - pad_x)
            top = max(0, ymin - pad_y)
            right = min(image_width, xmax + pad_x)
            bottom = min(image_height, ymax + pad_y)

            if right <= left or bottom <= top:
                return False

            destination.parent.mkdir(parents=True, exist_ok=True)
            image.crop((left, top, right, bottom)).save(destination, quality=95)
            return True
    except OSError:
        return False


def write_prepared_dataset(
    splits: dict[str, list[ObjectAnnotation]],
    dataset_dir: Path,
    padding: float,
    min_crop_size: int,
) -> dict[str, dict[str, int]]:
    counts = {split: {class_name: 0 for class_name in CLASS_MAP.values()} for split in SPLIT_RATIOS}

    for split, annotations in splits.items():
        for annotation in annotations:
            index = counts[split][annotation.class_name]
            destination = dataset_dir / split / annotation.class_name / f"{annotation.source_id}_{index:05d}.jpg"
            if save_crop(annotation, destination, padding=padding, min_crop_size=min_crop_size):
                counts[split][annotation.class_name] += 1

    return counts


def save_metadata(
    metadata_path: Path,
    raw_dir: Path,
    dataset_dir: Path,
    counts: dict[str, dict[str, int]],
) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "source_dataset": SOURCE_DATASET_NAME,
        "source_url": SOURCE_DATASET_URL,
        "source_viewer": SOURCE_DATASET_VIEWER,
        "license": SOURCE_DATASET_LICENSE,
        "raw_dir": str(raw_dir),
        "prepared_dataset_dir": str(dataset_dir),
        "class_mapping": CLASS_MAP,
        "split_ratios": SPLIT_RATIOS,
        "prepared_counts": counts,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    raw_dir = resolve_project_path(get_config_value(config, "raw_dataset_dir", args.raw_dir))
    dataset_dir = resolve_project_path(get_config_value(config, "dataset_dir", args.dataset_dir))
    output_dir = resolve_project_path(config.get("dataset_preparation_output_dir", "outputs/dataset_preparation"))
    class_names = list(config.get("class_names", ["mask", "no_mask", "incorrect_mask"]))

    if args.download:
        raw_dir = download_dataset(raw_dir, source=args.download_source)

    annotations = collect_annotations(raw_dir)
    if not annotations:
        raise RuntimeError(
            "No supported face-mask annotations were found.\n"
            f"Checked raw dataset directory: {raw_dir}\n"
            "Use --download, or manually place the Kaggle images/annotations or "
            "Dataset Ninja Supervisely export in the raw dataset directory."
        )

    labels = sorted({annotation.class_name for annotation in annotations})
    missing = sorted(set(class_names) - set(labels))
    if missing:
        raise RuntimeError(f"Raw annotations are missing required prepared class(es): {missing}")

    ensure_empty_or_reset(dataset_dir, class_names, reset=args.reset)
    splits = split_by_source(annotations, seed=args.seed)
    counts = write_prepared_dataset(
        splits,
        dataset_dir,
        padding=args.padding,
        min_crop_size=args.min_crop_size,
    )
    save_metadata(output_dir / "dataset_source.json", raw_dir, dataset_dir, counts)

    print("\nDataset preparation complete.")
    print(f"Source: {SOURCE_DATASET_NAME} ({SOURCE_DATASET_LICENSE})")
    print(f"Prepared dataset: {dataset_dir}")
    print(f"Metadata: {output_dir / 'dataset_source.json'}")
    for split in SPLIT_RATIOS:
        print(f"{split}: {count_prepared_images(dataset_dir / split)}")


if __name__ == "__main__":
    main()
