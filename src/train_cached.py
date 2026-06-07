from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import keras
from keras import layers
from keras.applications import VGG19
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance
from sklearn.utils.class_weight import compute_class_weight

from config_utils import IMAGE_EXTENSIONS
from config_utils import get_config_value, load_config, resolve_project_path
from dataset_utils import DatasetValidation, load_dataset_split, validate_training_dataset
from evaluation_utils import print_metrics, save_evaluation_results
from model_utils import VGG19Preprocess, build_vgg19_lamb_model, compile_model, make_lamb_optimizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Laptop-friendly VGG19 + LAMB training with cached frozen VGG19 features."
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file.")
    parser.add_argument("--data_dir", type=str, help="Dataset directory.")
    parser.add_argument("--output_dir", type=str, help="Output directory.")
    parser.add_argument("--model_path", type=str, help="Where to save the selected best model.")
    parser.add_argument("--final_model_path", type=str, help="Where to save the final trained model.")
    parser.add_argument("--image_size", type=int, help="Image size.")
    parser.add_argument("--batch_size", type=int, help="Image feature extraction batch size.")
    parser.add_argument("--epochs", type=int, default=60, help="Classifier-head training epochs.")
    parser.add_argument("--lr", type=float, default=3e-4, help="LAMB learning rate for cached head training.")
    parser.add_argument("--weight_decay", type=float, help="LAMB weight decay.")
    parser.add_argument(
        "--class_weight_mode",
        choices=["balanced", "sqrt", "none"],
        default="sqrt",
        help="Class weighting mode. sqrt is a gentler imbalance correction for this dataset.",
    )
    parser.add_argument("--reuse_features", action="store_true", help="Reuse cached VGG19 feature arrays if present.")
    parser.add_argument(
        "--occlusion_negatives_per_image",
        type=int,
        default=0,
        help="Add synthetic lower-face occlusions to no_mask training images.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for synthetic live-demo augmentations.")
    return parser.parse_args()


def print_dataset_summary(validation: DatasetValidation) -> None:
    print("\nDataset Validation")
    print("------------------")
    print(f"Detected classes: {validation.class_names}")
    print(f"Train counts:     {validation.train_counts}")
    print(f"Validation counts:{validation.val_counts}")
    if validation.test_available:
        print(f"Test counts:      {validation.test_counts}")
    else:
        print("Test counts:      unavailable or incomplete")


def save_class_names(class_names: list[str], class_names_path: Path) -> None:
    class_names_path.parent.mkdir(parents=True, exist_ok=True)
    class_names_path.write_text(json.dumps(class_names, indent=2), encoding="utf-8")


def build_feature_extractor(image_size: int) -> keras.Model:
    inputs = keras.Input(shape=(image_size, image_size, 3), name="input_image")
    x = VGG19Preprocess(name="vgg19_preprocess")(inputs)
    base_model = VGG19(
        include_top=False,
        weights="imagenet",
        input_shape=(image_size, image_size, 3),
        name="vgg19_backbone",
    )
    base_model.trainable = False
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    return keras.Model(inputs=inputs, outputs=x, name="VGG19_Frozen_Feature_Extractor")


def extract_features(feature_model: keras.Model, dataset, split_name: str) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    print(f"\nExtracting frozen VGG19 features for {split_name}...")
    for images, batch_labels in dataset:
        features.append(feature_model.predict(images, verbose=0))
        labels.append(batch_labels.numpy().astype(np.int32))

    if not features:
        raise ValueError(f"No samples found in {split_name} dataset.")

    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def save_feature_cache(cache_path: Path, feature_data: dict[str, np.ndarray]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **feature_data)


def load_or_extract_features(
    cache_path: Path,
    reuse_features: bool,
    feature_model: keras.Model,
    datasets: dict[str, Any],
) -> dict[str, np.ndarray]:
    if reuse_features and cache_path.exists():
        print(f"Loading cached features from {cache_path}")
        cached = np.load(cache_path)
        return {key: cached[key] for key in cached.files}

    feature_data: dict[str, np.ndarray] = {}
    for split_name, dataset in datasets.items():
        split_features, split_labels = extract_features(feature_model, dataset, split_name)
        feature_data[f"{split_name}_features"] = split_features
        feature_data[f"{split_name}_labels"] = split_labels

    save_feature_cache(cache_path, feature_data)
    print(f"Feature cache saved to {cache_path}")
    return feature_data


def image_files(class_dir: Path) -> list[Path]:
    if not class_dir.exists():
        return []

    return [
        path
        for path in sorted(class_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def make_lower_face_occlusion(image_path: Path, image_size: int, rng: np.random.Generator) -> np.ndarray:
    image = Image.open(image_path).convert("RGB").resize((image_size, image_size))

    if rng.random() < 0.55:
        image = ImageEnhance.Brightness(image).enhance(float(rng.uniform(0.82, 1.18)))
    if rng.random() < 0.55:
        image = ImageEnhance.Contrast(image).enhance(float(rng.uniform(0.85, 1.20)))

    draw = ImageDraw.Draw(image, "RGBA")
    center_x = image_size * float(rng.uniform(0.42, 0.58))
    center_y = image_size * float(rng.uniform(0.58, 0.74))
    width = image_size * float(rng.uniform(0.28, 0.55))
    height = image_size * float(rng.uniform(0.18, 0.34))
    left = max(0, int(center_x - width / 2))
    top = max(0, int(center_y - height / 2))
    right = min(image_size, int(center_x + width / 2))
    bottom = min(image_size, int(center_y + height / 2))

    colors = [
        (190, 135, 105, 230),
        (215, 170, 135, 230),
        (145, 95, 70, 230),
        (55, 55, 55, 230),
        (95, 110, 125, 230),
        (225, 225, 225, 230),
    ]
    color = colors[int(rng.integers(0, len(colors)))]

    if rng.random() < 0.45:
        polygon = [
            (left + int(rng.integers(0, max(1, width // 5))), top),
            (right, top + int(rng.integers(0, max(1, height // 4)))),
            (right - int(rng.integers(0, max(1, width // 6))), bottom),
            (left, bottom - int(rng.integers(0, max(1, height // 4)))),
        ]
        draw.polygon(polygon, fill=color)
    else:
        draw.rounded_rectangle((left, top, right, bottom), radius=max(4, int(height // 4)), fill=color)

    return np.asarray(image, dtype=np.float32)


def extract_occlusion_negative_features(
    feature_model: keras.Model,
    no_mask_paths: list[Path],
    no_mask_label: int,
    image_size: int,
    per_image: int,
    cache_path: Path,
    seed: int,
    reuse_features: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if per_image <= 0 or not no_mask_paths:
        return np.empty((0, feature_model.output_shape[-1]), dtype=np.float32), np.empty((0,), dtype=np.int32)

    if reuse_features and cache_path.exists():
        print(f"Loading cached occlusion-negative features from {cache_path}")
        cached = np.load(cache_path)
        return cached["features"], cached["labels"]

    rng = np.random.default_rng(seed)
    batch: list[np.ndarray] = []
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    batch_size = 32

    print(f"\nExtracting {per_image} synthetic occlusion-negative(s) per no_mask image...")
    for image_path in no_mask_paths:
        for _ in range(per_image):
            batch.append(make_lower_face_occlusion(image_path, image_size, rng))
            if len(batch) == batch_size:
                batch_array = np.stack(batch, axis=0)
                features.append(feature_model.predict(batch_array, verbose=0))
                labels.append(np.full((len(batch),), no_mask_label, dtype=np.int32))
                batch = []

    if batch:
        batch_array = np.stack(batch, axis=0)
        features.append(feature_model.predict(batch_array, verbose=0))
        labels.append(np.full((len(batch),), no_mask_label, dtype=np.int32))

    if not features:
        return np.empty((0, feature_model.output_shape[-1]), dtype=np.float32), np.empty((0,), dtype=np.int32)

    occlusion_features = np.concatenate(features, axis=0)
    occlusion_labels = np.concatenate(labels, axis=0)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, features=occlusion_features, labels=occlusion_labels)
    print(f"Occlusion-negative feature cache saved to {cache_path}")
    return occlusion_features, occlusion_labels


def make_class_weights(labels: np.ndarray, mode: str) -> dict[int, float] | None:
    if mode == "none":
        return None

    classes = np.unique(labels)
    balanced = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    if mode == "sqrt":
        balanced = np.sqrt(balanced)

    return {int(class_id): float(weight) for class_id, weight in zip(classes, balanced)}


def build_feature_head(num_classes: int, feature_dim: int, learning_rate: float, weight_decay: float) -> keras.Model:
    inputs = keras.Input(shape=(feature_dim,), name="vgg19_features")
    x = layers.BatchNormalization(name="batch_norm")(inputs)
    x = layers.Dropout(0.40, name="dropout_1")(x)
    x = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(1e-4),
        name="dense_features",
    )(x)
    x = layers.Dropout(0.30, name="dropout_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="class_output")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="Cached_VGG19_LAMB_Head")
    model.compile(
        optimizer=make_lamb_optimizer(learning_rate=learning_rate, weight_decay=weight_decay),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def make_callbacks(checkpoint_path: Path, log_path: Path) -> list[keras.callbacks.Callback]:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return [
        keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.4,
            patience=4,
            min_lr=1e-7,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(log_path),
    ]


def plot_training_history(history: keras.callbacks.History, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", range(1, len(history_df) + 1))
    history_df.to_csv(output_dir / "training_history.csv", index=False)

    def save_plot(columns: list[str], title: str, ylabel: str, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(9, 5))
        for column in columns:
            if column in history_df:
                ax.plot(history_df["epoch"], history_df[column], marker="o", label=column)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=150)
        plt.close(fig)

    save_plot(["accuracy", "val_accuracy"], "Training and Validation Accuracy", "Accuracy", "accuracy.png")
    save_plot(["loss", "val_loss"], "Training and Validation Loss", "Loss", "loss.png")
    save_plot(["accuracy", "val_accuracy", "loss", "val_loss"], "Accuracy and Loss", "Value", "accuracy_loss.png")


def transfer_head_weights(head_model: keras.Model, full_model: keras.Model) -> None:
    for layer_name in ("batch_norm", "dense_features", "class_output"):
        full_model.get_layer(layer_name).set_weights(head_model.get_layer(layer_name).get_weights())


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    data_dir = resolve_project_path(get_config_value(config, "dataset_dir", args.data_dir))
    output_dir = resolve_project_path(get_config_value(config, "output_dir", args.output_dir))
    training_output_dir = resolve_project_path(config.get("training_output_dir", output_dir / "training"))
    evaluation_output_dir = resolve_project_path(config.get("evaluation_output_dir", output_dir / "evaluation"))
    feature_cache_dir = resolve_project_path(config.get("feature_cache_dir", output_dir / "feature_cache"))
    model_path = resolve_project_path(get_config_value(config, "model_path", args.model_path))
    final_model_path = resolve_project_path(get_config_value(config, "final_model_path", args.final_model_path))
    class_names_path = resolve_project_path(config.get("class_names_path", output_dir / "class_names.json"))

    image_size = int(get_config_value(config, "image_size", args.image_size))
    batch_size = int(get_config_value(config, "batch_size", args.batch_size))
    weight_decay = float(get_config_value(config, "weight_decay", args.weight_decay))
    preferred_class_names = list(config.get("class_names", []))

    output_dir.mkdir(parents=True, exist_ok=True)
    training_output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_output_dir.mkdir(parents=True, exist_ok=True)

    validation = validate_training_dataset(data_dir, preferred_class_names)
    class_names = validation.class_names
    print_dataset_summary(validation)
    save_class_names(class_names, class_names_path)

    train_ds = load_dataset_split(data_dir / "train", class_names, image_size, batch_size, shuffle=False)
    val_ds = load_dataset_split(data_dir / "val", class_names, image_size, batch_size, shuffle=False)
    test_ds = (
        load_dataset_split(data_dir / "test", class_names, image_size, batch_size, shuffle=False)
        if validation.test_available
        else None
    )

    feature_model = build_feature_extractor(image_size)
    datasets = {"train": train_ds, "val": val_ds}
    if test_ds is not None:
        datasets["test"] = test_ds

    cache_path = feature_cache_dir / f"vgg19_features_{image_size}.npz"
    feature_data = load_or_extract_features(cache_path, args.reuse_features, feature_model, datasets)

    train_features = feature_data["train_features"]
    train_labels = feature_data["train_labels"]
    val_features = feature_data["val_features"]
    val_labels = feature_data["val_labels"]

    if args.occlusion_negatives_per_image > 0 and "no_mask" in class_names:
        no_mask_label = class_names.index("no_mask")
        no_mask_paths = image_files(data_dir / "train" / "no_mask")
        occlusion_cache_path = (
            feature_cache_dir
            / f"vgg19_no_mask_occlusion_features_{image_size}_{args.occlusion_negatives_per_image}.npz"
        )
        occlusion_features, occlusion_labels = extract_occlusion_negative_features(
            feature_model,
            no_mask_paths=no_mask_paths,
            no_mask_label=no_mask_label,
            image_size=image_size,
            per_image=args.occlusion_negatives_per_image,
            cache_path=occlusion_cache_path,
            seed=args.seed,
            reuse_features=args.reuse_features,
        )
        if len(occlusion_labels) > 0:
            train_features = np.concatenate([train_features, occlusion_features], axis=0)
            train_labels = np.concatenate([train_labels, occlusion_labels], axis=0)
            print(f"Added synthetic no_mask occlusion negatives: {len(occlusion_labels)}")

    class_weights = make_class_weights(train_labels, args.class_weight_mode)
    print(f"Class weight mode: {args.class_weight_mode}")
    print(f"Class weights: {class_weights}")

    head_model = build_feature_head(
        num_classes=len(class_names),
        feature_dim=train_features.shape[1],
        learning_rate=args.lr,
        weight_decay=weight_decay,
    )

    head_checkpoint_path = training_output_dir / "cached_head_best.keras"
    print("\nTraining cached VGG19 classifier head...")
    history = head_model.fit(
        train_features,
        train_labels,
        validation_data=(val_features, val_labels),
        epochs=args.epochs,
        batch_size=64,
        class_weight=class_weights,
        callbacks=make_callbacks(head_checkpoint_path, training_output_dir / "cached_training_log.csv"),
        verbose=1,
    )
    plot_training_history(history, training_output_dir)

    best_head = keras.models.load_model(head_checkpoint_path)
    full_model = build_vgg19_lamb_model(
        num_classes=len(class_names),
        image_size=image_size,
        train_base=False,
    )
    compile_model(full_model, learning_rate=args.lr, weight_decay=weight_decay)
    transfer_head_weights(best_head, full_model)

    final_model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    full_model.save(final_model_path)
    shutil.copy2(final_model_path, model_path)

    if test_ds is not None:
        test_loss, test_accuracy = full_model.evaluate(test_ds, verbose=1)
        print(f"\nFinal Test Accuracy: {test_accuracy:.4f}")
        print(f"Final Test Loss:     {test_loss:.4f}")
        metrics = save_evaluation_results(full_model, test_ds, class_names, evaluation_output_dir, split_name="test")
        print_metrics(metrics, split_name="test")
    else:
        val_loss, val_accuracy = full_model.evaluate(val_ds, verbose=1)
        print("\nNo complete test split was found, so validation metrics were saved instead.")
        print(f"Final Validation Accuracy: {val_accuracy:.4f}")
        print(f"Final Validation Loss:     {val_loss:.4f}")
        metrics = save_evaluation_results(
            full_model,
            val_ds,
            class_names,
            evaluation_output_dir,
            split_name="validation",
        )
        print_metrics(metrics, split_name="validation")

    print("\nCached training complete.")
    print(f"Best model:       {model_path}")
    print(f"Final model:      {final_model_path}")
    print(f"Class names:      {class_names_path}")
    print(f"Feature cache:    {cache_path}")
    print(f"Training outputs: {training_output_dir}")
    print(f"Evaluation files: {evaluation_output_dir}")


if __name__ == "__main__":
    main()
