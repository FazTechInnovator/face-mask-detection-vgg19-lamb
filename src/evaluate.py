from __future__ import annotations

import argparse
import json
from pathlib import Path

import keras

from config_utils import get_config_value, load_config, resolve_project_path
from dataset_utils import load_dataset_split, validate_evaluation_split
from evaluation_utils import print_metrics, save_evaluation_results
from model_utils import VGG19Preprocess, compile_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained VGG19 + LAMB model.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file.")
    parser.add_argument("--data_dir", type=str, help="Dataset directory.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split.")
    parser.add_argument("--model", type=str, help="Path to trained model.")
    parser.add_argument("--classes", type=str, help="Path to class names JSON.")
    parser.add_argument("--output_dir", type=str, help="Directory for evaluation outputs.")
    parser.add_argument("--image_size", type=int, help="Image size.")
    parser.add_argument("--batch_size", type=int, help="Batch size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    data_dir = resolve_project_path(get_config_value(config, "dataset_dir", args.data_dir))
    model_path = resolve_project_path(get_config_value(config, "model_path", args.model))
    class_names_path = resolve_project_path(get_config_value(config, "class_names_path", args.classes))
    output_dir = resolve_project_path(get_config_value(config, "evaluation_output_dir", args.output_dir))
    image_size = int(get_config_value(config, "image_size", args.image_size))
    batch_size = int(get_config_value(config, "batch_size", args.batch_size))
    learning_rate = float(config.get("learning_rate", 1e-4))
    weight_decay = float(config.get("weight_decay", 1e-5))

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}\nTrain first with: python src/train.py")
    if not class_names_path.exists():
        raise FileNotFoundError(f"Class names file not found: {class_names_path}\nTrain first with: python src/train.py")

    with open(class_names_path, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    counts = validate_evaluation_split(data_dir, args.split, class_names)
    print(f"Evaluating split '{args.split}' with counts: {counts}")

    dataset = load_dataset_split(data_dir / args.split, class_names, image_size, batch_size, shuffle=False)
    model = keras.models.load_model(
        model_path,
        custom_objects={"VGG19Preprocess": VGG19Preprocess},
        compile=False,
    )
    compile_model(model, learning_rate=learning_rate, weight_decay=weight_decay)

    loss, accuracy = model.evaluate(dataset, verbose=1)
    print(f"\nLoss:     {loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    metrics = save_evaluation_results(model, dataset, class_names, output_dir, split_name=args.split)
    print_metrics(metrics, split_name=args.split)
    print(f"\nEvaluation files saved to: {output_dir}")


if __name__ == "__main__":
    main()
