from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config_utils import display_class_name


CLASS_COLORS_RGB = {
    "mask": (60, 180, 75),
    "no_mask": (230, 60, 60),
    "incorrect_mask": (245, 150, 40),
    "uncertain": (150, 150, 150),
}


@dataclass(frozen=True)
class FaceBox:
    x: int
    y: int
    w: int
    h: int


@dataclass(frozen=True)
class FacePrediction:
    box: FaceBox
    predicted_index: int
    confidence: float
    margin: float
    probabilities: np.ndarray
    confident: bool
    skin_occlusion_like: bool


def load_face_cascades() -> list[cv2.CascadeClassifier]:
    cascade_names = [
        "haarcascade_frontalface_default.xml",
        "haarcascade_frontalface_alt2.xml",
    ]
    cascades: list[cv2.CascadeClassifier] = []
    for cascade_name in cascade_names:
        cascade_path = Path(cv2.data.haarcascades) / cascade_name
        cascade = cv2.CascadeClassifier(str(cascade_path))
        if not cascade.empty():
            cascades.append(cascade)

    if not cascades:
        raise RuntimeError("Could not load OpenCV face detection cascades.")
    return cascades


def _iou(first: FaceBox, second: FaceBox) -> float:
    x1 = max(first.x, second.x)
    y1 = max(first.y, second.y)
    x2 = min(first.x + first.w, second.x + second.w)
    y2 = min(first.y + first.h, second.y + second.h)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    first_area = first.w * first.h
    second_area = second.w * second.h
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _deduplicate_boxes(boxes: list[FaceBox], iou_threshold: float = 0.35) -> list[FaceBox]:
    selected: list[FaceBox] = []
    for box in sorted(boxes, key=lambda item: item.w * item.h, reverse=True):
        if all(_iou(box, kept) < iou_threshold for kept in selected):
            selected.append(box)
    return sorted(selected, key=lambda item: item.x)


def detect_faces(image: Image.Image, cascades: list[cv2.CascadeClassifier] | None = None) -> list[FaceBox]:
    cascades = cascades or load_face_cascades()
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)

    min_side = min(image.size)
    min_face_size = max(45, int(min_side * 0.08))
    boxes: list[FaceBox] = []

    for cascade in cascades:
        for min_neighbors in (5, 3):
            faces = cascade.detectMultiScale(
                gray,
                scaleFactor=1.08,
                minNeighbors=min_neighbors,
                minSize=(min_face_size, min_face_size),
            )
            boxes.extend(FaceBox(int(x), int(y), int(w), int(h)) for x, y, w, h in faces)
            if len(faces) > 0:
                break

    return _deduplicate_boxes(boxes)


def expand_box(box: FaceBox, image_size: tuple[int, int], padding_ratio: float = 0.20) -> FaceBox:
    image_w, image_h = image_size
    pad_x = int(box.w * padding_ratio)
    pad_y = int(box.h * padding_ratio)
    x1 = max(box.x - pad_x, 0)
    y1 = max(box.y - pad_y, 0)
    x2 = min(box.x + box.w + pad_x, image_w)
    y2 = min(box.y + box.h + pad_y, image_h)
    return FaceBox(x1, y1, max(1, x2 - x1), max(1, y2 - y1))


def crop_face(image: Image.Image, box: FaceBox, padding_ratio: float = 0.20) -> Image.Image:
    expanded = expand_box(box, image.size, padding_ratio=padding_ratio)
    return image.crop((expanded.x, expanded.y, expanded.x + expanded.w, expanded.y + expanded.h)).convert("RGB")


def prepare_face_batch(image: Image.Image, boxes: list[FaceBox], image_size: int) -> np.ndarray:
    crops = []
    for box in boxes:
        crop = crop_face(image, box).resize((image_size, image_size))
        crops.append(np.asarray(crop, dtype=np.float32))
    return np.stack(crops, axis=0)


def lower_face_skin_ratio(image: Image.Image) -> float:
    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    lower_face = rgb[int(height * 0.50) : int(height * 0.88), int(width * 0.18) : int(width * 0.82)]
    if lower_face.size == 0:
        return 0.0

    ycrcb = cv2.cvtColor(lower_face, cv2.COLOR_RGB2YCrCb)
    _, cr, cb = cv2.split(ycrcb)
    ycrcb_skin = (cr >= 133) & (cr <= 180) & (cb >= 75) & (cb <= 145)

    hsv = cv2.cvtColor(lower_face, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    hsv_skin = ((h <= 25) | (h >= 160)) & (s >= 20) & (s <= 190) & (v >= 45)

    return float(np.mean(ycrcb_skin | hsv_skin))


def summarize_probabilities(probabilities: np.ndarray) -> tuple[int, float, float]:
    order = np.argsort(probabilities)[::-1]
    predicted_index = int(order[0])
    confidence = float(probabilities[predicted_index])
    margin = float(probabilities[order[0]] - probabilities[order[1]]) if len(order) > 1 else confidence
    return predicted_index, confidence, margin


def predict_faces(
    model,
    image: Image.Image,
    image_size: int,
    min_confidence: float,
    min_margin: float,
    class_names: list[str] | None = None,
    cascades: list[cv2.CascadeClassifier] | None = None,
) -> list[FacePrediction]:
    boxes = detect_faces(image, cascades=cascades)
    if not boxes:
        return []

    crops = [crop_face(image, box) for box in boxes]
    batch = np.stack(
        [np.asarray(crop.resize((image_size, image_size)), dtype=np.float32) for crop in crops],
        axis=0,
    )
    probability_rows = model.predict(batch, verbose=0)
    predictions: list[FacePrediction] = []
    mask_index = class_names.index("mask") if class_names and "mask" in class_names else None
    for box, crop, probabilities in zip(boxes, crops, probability_rows):
        predicted_index, confidence, margin = summarize_probabilities(probabilities)
        skin_ratio = lower_face_skin_ratio(crop)
        skin_occlusion_like = mask_index is not None and predicted_index == mask_index and skin_ratio >= 0.48
        predictions.append(
            FacePrediction(
                box=box,
                predicted_index=predicted_index,
                confidence=confidence,
                margin=margin,
                probabilities=np.asarray(probabilities, dtype=np.float32),
                confident=confidence >= min_confidence and margin >= min_margin and not skin_occlusion_like,
                skin_occlusion_like=skin_occlusion_like,
            )
        )
    return predictions


def prediction_label(prediction: FacePrediction, class_names: list[str]) -> str:
    if not prediction.confident:
        return "Uncertain"
    return display_class_name(class_names[prediction.predicted_index])


def draw_predictions(image: Image.Image, predictions: list[FacePrediction], class_names: list[str]) -> Image.Image:
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)

    try:
        font = ImageFont.truetype("arial.ttf", max(18, annotated.width // 42))
    except OSError:
        font = ImageFont.load_default()

    for prediction in predictions:
        class_name = class_names[prediction.predicted_index]
        label_name = prediction_label(prediction, class_names)
        label = f"{label_name}: {prediction.confidence:.1%}"
        color = CLASS_COLORS_RGB[class_name] if prediction.confident else CLASS_COLORS_RGB["uncertain"]
        box = prediction.box
        x1, y1 = box.x, box.y
        x2, y2 = box.x + box.w, box.y + box.h

        draw.rectangle((x1, y1, x2, y2), outline=color, width=max(3, annotated.width // 240))
        text_box = draw.textbbox((0, 0), label, font=font)
        text_w = text_box[2] - text_box[0]
        text_h = text_box[3] - text_box[1]
        padding = 8
        label_y1 = max(0, y1 - text_h - padding * 2)
        label_y2 = label_y1 + text_h + padding * 2
        draw.rectangle((x1, label_y1, x1 + text_w + padding * 2, label_y2), fill=color)
        draw.text((x1 + padding, label_y1 + padding), label, fill=(255, 255, 255), font=font)

    return annotated
