from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
import json
import platform
import time
from pathlib import Path

import cv2
import keras
import numpy as np

from config_utils import display_class_name, get_config_value, load_config, resolve_project_path
from model_utils import VGG19Preprocess


CLASS_COLORS = {
    "mask": (60, 180, 75),
    "no_mask": (40, 40, 230),
    "incorrect_mask": (0, 165, 255),
}


WINDOWS_BACKENDS = [
    ("DirectShow", cv2.CAP_DSHOW),
    ("Media Foundation", cv2.CAP_MSMF),
    ("Default", cv2.CAP_ANY),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time face mask detection using webcam.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file.")
    parser.add_argument("--model", type=str, help="Path to trained model.")
    parser.add_argument("--classes", type=str, help="Path to class names JSON.")
    parser.add_argument("--image_size", type=int, help="Image size.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument("--auto_camera", action="store_true", help="Try camera indices 0 through 3 automatically.")
    parser.add_argument("--min_confidence", type=float, help="Minimum confidence threshold.")
    parser.add_argument("--min_margin", type=float, default=0.18, help="Minimum gap between top two probabilities.")
    parser.add_argument("--predict_every", type=float, default=0.45, help="Seconds between VGG19 predictions.")
    parser.add_argument("--smooth_window", type=int, default=5, help="Number of recent predictions to average.")
    parser.add_argument("--display_width", type=int, default=640, help="Resize webcam display width for better FPS.")
    parser.add_argument("--fallback_center_crop", action="store_true", help="Predict center crop when no face is detected.")
    return parser.parse_args()


def candidate_camera_indices(start_index: int, auto_camera: bool) -> list[int]:
    if not auto_camera:
        return [start_index]

    indices = [start_index]
    for index in range(4):
        if index not in indices:
            indices.append(index)
    return indices


def open_camera(camera_index: int, auto_camera: bool) -> cv2.VideoCapture:
    backend_candidates = WINDOWS_BACKENDS if platform.system() == "Windows" else [("Default", cv2.CAP_ANY)]

    for index in candidate_camera_indices(camera_index, auto_camera):
        for backend_name, backend_id in backend_candidates:
            print(f"Trying camera index {index} using {backend_name}...")
            camera = cv2.VideoCapture(index, backend_id)
            if not camera.isOpened():
                camera.release()
                continue

            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            time.sleep(0.4)

            for _ in range(10):
                success, frame = camera.read()
                if success and frame is not None and frame.size > 0:
                    print(f"Using camera index {index} with {backend_name}.")
                    return camera
                time.sleep(0.1)

            camera.release()

    raise RuntimeError(
        "Could not read from any webcam.\n"
        "Fixes to try:\n"
        "  1. Close Zoom/Teams/Camera app if it is using the webcam.\n"
        "  2. Enable Windows camera permission for desktop apps.\n"
        "  3. Try: python src\\webcam_detect.py --camera 1 --auto_camera\n"
        "  4. Restart the laptop if the camera light is stuck on."
    )


def predict_crop(model, crop_bgr: np.ndarray, image_size: int) -> tuple[int, float, np.ndarray]:
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    crop_rgb = cv2.resize(crop_rgb, (image_size, image_size))
    batch = np.expand_dims(crop_rgb.astype(np.float32), axis=0)
    probabilities = model.predict(batch, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index])
    return predicted_index, confidence, probabilities


def lower_face_skin_ratio_bgr(crop_bgr: np.ndarray) -> float:
    height, width = crop_bgr.shape[:2]
    lower_face = crop_bgr[int(height * 0.50) : int(height * 0.88), int(width * 0.18) : int(width * 0.82)]
    if lower_face.size == 0:
        return 0.0

    ycrcb = cv2.cvtColor(lower_face, cv2.COLOR_BGR2YCrCb)
    _, cr, cb = cv2.split(ycrcb)
    ycrcb_skin = (cr >= 133) & (cr <= 180) & (cb >= 75) & (cb <= 145)

    hsv = cv2.cvtColor(lower_face, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    hsv_skin = ((h <= 25) | (h >= 160)) & (s >= 20) & (s <= 190) & (v >= 45)

    return float(np.mean(ycrcb_skin | hsv_skin))


def predict_crop_with_guard(
    model,
    crop_bgr: np.ndarray,
    image_size: int,
    class_names: list[str],
) -> tuple[int, float, np.ndarray, bool]:
    predicted_index, confidence, probabilities = predict_crop(model, crop_bgr, image_size)
    skin_occlusion_like = (
        class_names[predicted_index] == "mask" and lower_face_skin_ratio_bgr(crop_bgr) >= 0.48
    )
    return predicted_index, confidence, probabilities, skin_occlusion_like


def resize_for_display(frame: np.ndarray, display_width: int) -> np.ndarray:
    if display_width <= 0 or frame.shape[1] <= display_width:
        return frame

    scale = display_width / frame.shape[1]
    height = int(frame.shape[0] * scale)
    return cv2.resize(frame, (display_width, height), interpolation=cv2.INTER_AREA)


def padded_face_crop(frame: np.ndarray, x: int, y: int, w: int, h: int, padding_ratio: float = 0.20) -> np.ndarray:
    frame_h, frame_w = frame.shape[:2]
    pad_x = int(w * padding_ratio)
    pad_y = int(h * padding_ratio)
    x1 = max(x - pad_x, 0)
    y1 = max(y - pad_y, 0)
    x2 = min(x + w + pad_x, frame_w)
    y2 = min(y + h + pad_y, frame_h)
    return frame[y1:y2, x1:x2]


def center_crop(frame: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = frame.shape[:2]
    size = int(min(h, w) * 0.70)
    x = (w - size) // 2
    y = (h - size) // 2
    return frame[y : y + size, x : x + size], (x, y, size, size)


def draw_prediction(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    class_name: str,
    confidence: float,
    min_confidence: float,
    min_margin: float,
    margin: float,
    force_uncertain: bool = False,
    fallback: bool = False,
) -> None:
    x, y, w, h = box
    is_confident = confidence >= min_confidence and margin >= min_margin and not force_uncertain
    label_class = class_name if is_confident else "uncertain"
    label = f"{display_class_name(label_class)}: {confidence:.2f}"
    if fallback:
        label = f"Center crop - {label}"

    color = CLASS_COLORS.get(class_name, (255, 255, 255)) if is_confident else (180, 180, 180)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    frame_h, frame_w = frame.shape[:2]
    label_width = min(frame_w - x, max(w, text_width + 18))
    label_height = text_height + baseline + 14
    if y - label_height < 0:
        label_top = y
        label_bottom = min(frame_h, y + label_height)
        text_y = min(frame_h - 6, y + text_height + 8)
    else:
        label_top = y - label_height
        label_bottom = y
        text_y = y - baseline - 7

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.rectangle(frame, (x, label_top), (x + label_width, label_bottom), color, -1)
    cv2.putText(
        frame,
        label,
        (x + 8, text_y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def draw_status(frame: np.ndarray, text: str) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness = 2
    padding = 10
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(
        frame,
        (10, 10),
        (text_width + padding * 2, text_height + baseline + padding * 2),
        (40, 40, 40),
        -1,
    )
    cv2.putText(
        frame,
        text,
        (padding + 10, text_height + padding + 10),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def summarize_probabilities(probabilities: np.ndarray) -> tuple[int, float, float]:
    averaged = np.mean(probabilities, axis=0) if probabilities.ndim == 2 else probabilities
    order = np.argsort(averaged)[::-1]
    predicted_index = int(order[0])
    confidence = float(averaged[predicted_index])
    margin = float(averaged[order[0]] - averaged[order[1]]) if len(order) > 1 else confidence
    return predicted_index, confidence, margin


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    model_path = resolve_project_path(get_config_value(config, "model_path", args.model))
    class_path = resolve_project_path(get_config_value(config, "class_names_path", args.classes))
    image_size = int(get_config_value(config, "image_size", args.image_size))
    min_confidence = float(get_config_value(config, "min_confidence", args.min_confidence))
    min_margin = float(get_config_value(config, "min_margin", args.min_margin))

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}\nPlease train first using: python src/train_cached.py")
    if not class_path.exists():
        raise FileNotFoundError(
            f"Class names file not found: {class_path}\nPlease train first using: python src/train_cached.py"
        )

    with open(class_path, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    model = keras.models.load_model(
        model_path,
        custom_objects={"VGG19Preprocess": VGG19Preprocess},
        compile=False,
    )

    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    face_detector = cv2.CascadeClassifier(str(cascade_path))
    if face_detector.empty():
        raise RuntimeError(f"Could not load Haar cascade: {cascade_path}")

    camera = open_camera(args.camera, auto_camera=args.auto_camera)

    print("Webcam started. Press 'q' to quit.")
    print("Tip: strong lighting and a front-facing pose give better predictions.")

    probability_history: deque[np.ndarray] = deque(maxlen=max(args.smooth_window, 1))
    pending_prediction: Future | None = None
    last_prediction_time = 0.0
    occlusion_history: deque[bool] = deque(maxlen=max(args.smooth_window, 1))
    last_prediction: tuple[int, float, float, bool] | None = None
    frame_count = 0
    fps_start = time.perf_counter()
    fps = 0.0
    executor = ThreadPoolExecutor(max_workers=1)

    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("Warning: could not read frame from webcam. Waiting briefly...")
                time.sleep(0.2)
                continue

            frame = resize_for_display(frame, args.display_width)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(70, 70),
            )

            active_box: tuple[int, int, int, int] | None = None
            active_crop: np.ndarray | None = None
            fallback = False

            if len(faces) == 0:
                probability_history.clear()
                occlusion_history.clear()
                last_prediction = None
                if args.fallback_center_crop:
                    active_crop, active_box = center_crop(frame)
                    fallback = True
                else:
                    draw_status(frame, "No face detected")
            else:
                x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
                active_box = (int(x), int(y), int(w), int(h))
                active_crop = padded_face_crop(frame, int(x), int(y), int(w), int(h))

            if pending_prediction is not None and pending_prediction.done():
                predicted_index, confidence, probabilities, skin_occlusion_like = pending_prediction.result()
                probability_history.append(probabilities)
                occlusion_history.append(skin_occlusion_like)
                smoothed_index, smoothed_confidence, smoothed_margin = summarize_probabilities(
                    np.array(probability_history)
                )
                force_uncertain = any(occlusion_history)
                last_prediction = (smoothed_index, smoothed_confidence, smoothed_margin, force_uncertain)
                pending_prediction = None

            now = time.perf_counter()
            if (
                active_crop is not None
                and pending_prediction is None
                and now - last_prediction_time >= args.predict_every
            ):
                pending_prediction = executor.submit(
                    predict_crop_with_guard,
                    model,
                    active_crop.copy(),
                    image_size,
                    class_names,
                )
                last_prediction_time = now

            if active_box is not None and last_prediction is not None:
                predicted_index, confidence, margin, force_uncertain = last_prediction
                draw_prediction(
                    frame,
                    active_box,
                    class_names[predicted_index],
                    confidence,
                    min_confidence,
                    min_margin=min_margin,
                    margin=margin,
                    force_uncertain=force_uncertain,
                    fallback=fallback,
                )

            frame_count += 1
            elapsed = now - fps_start
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_start = now
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Face Mask Detection - VGG19 + LAMB", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        if pending_prediction is not None:
            pending_prediction.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
