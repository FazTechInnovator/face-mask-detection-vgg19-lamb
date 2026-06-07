from __future__ import annotations

import argparse
import json
from pathlib import Path

import keras
from PIL import Image

from config_utils import get_config_value, load_config, resolve_project_path
from image_prediction_utils import draw_predictions, load_face_cascades, predict_faces, prediction_label
from model_utils import VGG19Preprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict face mask class for a single image.")
    parser.add_argument("image_path", nargs="?", help="Path to image.")
    parser.add_argument("--image", type=str, help="Path to image. Kept for compatibility with older commands.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file.")
    parser.add_argument("--model", type=str, help="Path to trained model.")
    parser.add_argument("--classes", type=str, help="Path to class names JSON.")
    parser.add_argument("--output_dir", type=str, help="Directory for labeled prediction images.")
    parser.add_argument("--image_size", type=int, help="Image size.")
    parser.add_argument("--min_confidence", type=float, help="Minimum confidence for a final label.")
    parser.add_argument("--min_margin", type=float, default=0.25, help="Minimum gap between top two probabilities.")
    return parser.parse_args()


def save_labeled_image(image: Image.Image, output_dir: Path, source_path: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{source_path.stem}_prediction{source_path.suffix or '.jpg'}"
    image.save(output_path)
    return output_path


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    supplied_image = args.image or args.image_path
    if not supplied_image:
        raise ValueError("Please provide an image path, for example: python src/predict_image.py sample_images/test.jpg")

    model_path = resolve_project_path(get_config_value(config, "model_path", args.model))
    class_path = resolve_project_path(get_config_value(config, "class_names_path", args.classes))
    output_dir = resolve_project_path(get_config_value(config, "prediction_output_dir", args.output_dir))
    image_path = resolve_project_path(supplied_image)
    image_size = int(get_config_value(config, "image_size", args.image_size))
    min_confidence = float(get_config_value(config, "min_confidence", args.min_confidence))
    min_margin = float(config.get("min_margin", args.min_margin))

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}\nPlease train first using: python src/train.py")
    if not class_path.exists():
        raise FileNotFoundError(f"Class file not found: {class_path}\nPlease train first using: python src/train.py")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with open(class_path, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    model = keras.models.load_model(
        model_path,
        custom_objects={"VGG19Preprocess": VGG19Preprocess},
        compile=False,
    )

    original_image = Image.open(image_path).convert("RGB")
    cascades = load_face_cascades()
    predictions = predict_faces(
        model,
        original_image,
        image_size=image_size,
        min_confidence=min_confidence,
        min_margin=min_margin,
        class_names=class_names,
        cascades=cascades,
    )
    if not predictions:
        raise RuntimeError(
            "No clear face was detected. Use a front-facing photo, or crop the image around the face and try again."
        )

    labeled_image = draw_predictions(original_image, predictions, class_names)
    output_path = save_labeled_image(labeled_image, output_dir, image_path)

    print("\nPrediction Result")
    print("-----------------")
    print(f"Image:           {image_path}")
    print(f"Faces Detected:  {len(predictions)}")
    print(f"Saved Image:     {output_path}")

    for face_number, prediction in enumerate(predictions, start=1):
        predicted_class = class_names[prediction.predicted_index]
        print(f"\nFace {face_number}")
        print(f"Final Label:     {prediction_label(prediction, class_names)}")
        print(f"Top Class:       {predicted_class}")
        print(f"Confidence:      {prediction.confidence:.4f}")
        print(f"Margin:          {prediction.margin:.4f}")
        print("All Class Probabilities")
        for class_name, probability in zip(class_names, prediction.probabilities):
            print(f"{class_name}: {float(probability):.4f}")


if __name__ == "__main__":
    main()
