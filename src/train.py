from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

from config_utils import get_config_value, load_config, resolve_project_path
from dataset_utils import DatasetValidation, load_dataset_split, validate_training_dataset
from evaluation_utils import print_metrics, save_evaluation_results
from model_utils import (
    VGG19Preprocess,
    build_vgg19_lamb_model,
    compile_model,
    set_vgg19_backbone_trainable,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VGG19 + LAMB face mask detection model.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file.")
    parser.add_argument("--data_dir", type=str, help="Dataset directory.")
    parser.add_argument("--output_dir", type=str, help="Output directory.")
    parser.add_argument("--model_path", type=str, help="Where to save the selected best model.")
    parser.add_argument("--final_model_path", type=str, help="Where to save the final trained model.")
    parser.add_argument("--image_size", type=int, help="Image size.")
    parser.add_argument("--batch_size", type=int, help="Batch size.")
    parser.add_argument("--epochs", type=int, help="Classifier-head training epochs.")
    parser.add_argument("--fine_tune_epochs", type=int, help="Fine-tuning epochs.")
    parser.add_argument("--lr", type=float, help="Initial LAMB learning rate.")
    parser.add_argument("--fine_tune_lr", type=float, help="Fine-tuning learning rate.")
    parser.add_argument("--weight_decay", type=float, help="LAMB weight decay.")
    parser.add_argument("--fine_tune_at", type=int, help="VGG19 layer index where fine-tuning begins.")
    return parser.parse_args()


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
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(log_path),
    ]


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


def calculate_class_weights(train_ds) -> dict[int, float]:
    labels: list[int] = []
    for _, batch_y in train_ds:
        labels.extend(batch_y.numpy().astype(int).tolist())

    labels_array = np.array(labels, dtype=np.int32)
    classes = np.unique(labels_array)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels_array)
    return {int(class_id): float(weight) for class_id, weight in zip(classes, weights)}


def history_to_rows(history: keras.callbacks.History, start_epoch: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    history_frame = pd.DataFrame(history.history)
    for offset, row in enumerate(history_frame.to_dict("records"), start=0):
        row["epoch"] = start_epoch + offset + 1
        rows.append(row)
    return rows


def plot_training_history(history_rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    history_df = pd.DataFrame(history_rows)
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


def select_best_model(candidate_paths: list[Path], val_ds, destination_path: Path) -> tuple[keras.Model, Path, float]:
    best_model: keras.Model | None = None
    best_path: Path | None = None
    best_accuracy = -1.0

    print("\nSelecting best model using validation accuracy...")
    for candidate_path in candidate_paths:
        if not candidate_path.exists():
            continue

        candidate_model = keras.models.load_model(
            candidate_path,
            custom_objects={"VGG19Preprocess": VGG19Preprocess},
            compile=False,
        )
        compile_model(candidate_model)
        val_loss, val_accuracy = candidate_model.evaluate(val_ds, verbose=0)
        print(f"{candidate_path.name}: val_accuracy={val_accuracy:.4f}, val_loss={val_loss:.4f}")

        if val_accuracy > best_accuracy:
            best_accuracy = float(val_accuracy)
            best_path = candidate_path
            best_model = candidate_model

    if best_model is None or best_path is None:
        raise RuntimeError("No checkpoint model was saved. Check the training logs for earlier errors.")

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_path, destination_path)
    print(f"Selected best model: {best_path} -> {destination_path}")
    return best_model, best_path, best_accuracy


def save_class_names(class_names: list[str], class_names_path: Path) -> None:
    class_names_path.parent.mkdir(parents=True, exist_ok=True)
    with open(class_names_path, "w", encoding="utf-8") as file:
        json.dump(class_names, file, indent=2)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    data_dir = resolve_project_path(get_config_value(config, "dataset_dir", args.data_dir))
    output_dir = resolve_project_path(get_config_value(config, "output_dir", args.output_dir))
    training_output_dir = resolve_project_path(config.get("training_output_dir", output_dir / "training"))
    evaluation_output_dir = resolve_project_path(config.get("evaluation_output_dir", output_dir / "evaluation"))
    model_path = resolve_project_path(get_config_value(config, "model_path", args.model_path))
    final_model_path = resolve_project_path(get_config_value(config, "final_model_path", args.final_model_path))
    class_names_path = resolve_project_path(config.get("class_names_path", output_dir / "class_names.json"))

    image_size = int(get_config_value(config, "image_size", args.image_size))
    batch_size = int(get_config_value(config, "batch_size", args.batch_size))
    epochs = int(get_config_value(config, "epochs", args.epochs))
    fine_tune_epochs = int(get_config_value(config, "fine_tune_epochs", args.fine_tune_epochs))
    learning_rate = float(get_config_value(config, "learning_rate", args.lr))
    fine_tune_learning_rate = float(get_config_value(config, "fine_tune_learning_rate", args.fine_tune_lr))
    weight_decay = float(get_config_value(config, "weight_decay", args.weight_decay))
    fine_tune_at = int(get_config_value(config, "fine_tune_at", args.fine_tune_at))
    preferred_class_names = list(config.get("class_names", []))

    output_dir.mkdir(parents=True, exist_ok=True)
    training_output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_output_dir.mkdir(parents=True, exist_ok=True)

    validation = validate_training_dataset(data_dir, preferred_class_names)
    class_names = validation.class_names
    print_dataset_summary(validation)
    save_class_names(class_names, class_names_path)

    train_ds = load_dataset_split(data_dir / "train", class_names, image_size, batch_size, shuffle=True)
    val_ds = load_dataset_split(data_dir / "val", class_names, image_size, batch_size, shuffle=False)
    test_ds = (
        load_dataset_split(data_dir / "test", class_names, image_size, batch_size, shuffle=False)
        if validation.test_available
        else None
    )

    class_weights = calculate_class_weights(train_ds)
    print(f"Class weights: {class_weights}")

    model = build_vgg19_lamb_model(
        num_classes=len(class_names),
        image_size=image_size,
        train_base=False,
    )
    compile_model(model, learning_rate=learning_rate, weight_decay=weight_decay)

    stage1_best_path = output_dir / "stage1_best_model.keras"
    stage2_best_path = output_dir / "stage2_best_model.keras"
    history_rows: list[dict[str, Any]] = []

    print("\nStage 1: training the custom classifier head...")
    history_1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=make_callbacks(stage1_best_path, training_output_dir / "stage1_training_log.csv"),
    )
    history_rows.extend(history_to_rows(history_1, start_epoch=0))

    candidate_paths = [stage1_best_path]

    if fine_tune_epochs > 0:
        print("\nStage 2: fine-tuning upper VGG19 layers...")
        set_vgg19_backbone_trainable(model, fine_tune_at=fine_tune_at)
        compile_model(model, learning_rate=fine_tune_learning_rate, weight_decay=weight_decay)

        history_2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs + fine_tune_epochs,
            initial_epoch=epochs,
            class_weight=class_weights,
            callbacks=make_callbacks(stage2_best_path, training_output_dir / "stage2_training_log.csv"),
        )
        history_rows.extend(history_to_rows(history_2, start_epoch=epochs))
        candidate_paths.append(stage2_best_path)
    else:
        print("\nFine-tuning skipped because fine_tune_epochs is 0.")

    plot_training_history(history_rows, training_output_dir)

    final_model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(final_model_path)
    best_model, _, _ = select_best_model(candidate_paths, val_ds, model_path)

    if test_ds is not None:
        test_loss, test_accuracy = best_model.evaluate(test_ds, verbose=1)
        print(f"\nFinal Test Accuracy: {test_accuracy:.4f}")
        print(f"Final Test Loss:     {test_loss:.4f}")
        metrics = save_evaluation_results(best_model, test_ds, class_names, evaluation_output_dir, split_name="test")
        print_metrics(metrics, split_name="test")
    else:
        val_loss, val_accuracy = best_model.evaluate(val_ds, verbose=1)
        print("\nNo complete test split was found, so validation metrics were saved instead.")
        print(f"Final Validation Accuracy: {val_accuracy:.4f}")
        print(f"Final Validation Loss:     {val_loss:.4f}")
        metrics = save_evaluation_results(
            best_model,
            val_ds,
            class_names,
            evaluation_output_dir,
            split_name="validation",
        )
        print_metrics(metrics, split_name="validation")

    print("\nTraining complete.")
    print(f"Best model:       {model_path}")
    print(f"Final model:      {final_model_path}")
    print(f"Class names:      {class_names_path}")
    print(f"Training outputs: {training_output_dir}")
    print(f"Evaluation files: {evaluation_output_dir}")


if __name__ == "__main__":
    main()
