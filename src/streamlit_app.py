from __future__ import annotations

import json

import keras
import pandas as pd
import streamlit as st
from PIL import Image

from config_utils import display_class_name, load_config, resolve_project_path
from image_prediction_utils import (
    draw_predictions,
    load_face_cascades,
    predict_faces,
    prediction_label,
)
from model_utils import VGG19Preprocess


st.set_page_config(
    page_title="Face Mask Detection",
    layout="centered",
)

config = load_config("config.yaml")
model_path = resolve_project_path(config["model_path"])
classes_path = resolve_project_path(config["class_names_path"])
image_size = int(config["image_size"])
min_confidence = float(config.get("min_confidence", 0.9))
min_margin = float(config.get("min_margin", 0.25))
display_model_path = config["model_path"]


@st.cache_resource
def load_trained_model(path: str):
    return keras.models.load_model(
        path,
        custom_objects={"VGG19Preprocess": VGG19Preprocess},
        compile=False,
    )


@st.cache_data
def load_class_names(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_resource
def load_cascades():
    return load_face_cascades()


st.title("Real-Time Face Mask Detection")
st.caption("VGG19 transfer learning with the LAMB optimizer")

st.subheader("Project Overview")
st.markdown(
    """
This project classifies a face image into the trained mask categories:
`mask`, `no_mask`, and optionally `incorrect_mask`. It is designed for
single-image prediction, webcam demonstration, and local Streamlit deployment.
"""
)

st.markdown(
    """
**Supported classes:** `mask`, `no_mask`, `incorrect_mask`  
**Main workflow:** train the model, evaluate test performance, then run image,
webcam, or Streamlit prediction demos.
"""
)

with st.expander("Model details", expanded=True):
    details = pd.DataFrame(
        [
            {"Item": "Model", "Value": "VGG19"},
            {"Item": "Optimizer", "Value": "LAMB"},
            {"Item": "Input Size", "Value": f"{image_size}x{image_size}"},
            {"Item": "Best Model Path", "Value": display_model_path},
        ]
    )
    st.dataframe(details, hide_index=True, use_container_width=True)

if not model_path.exists() or not classes_path.exists():
    st.warning("Please train the model first using python src/train_cached.py")
    st.info("After training, this app expects outputs/best_model.keras and outputs/class_names.json.")
    st.code(
        "python src/train_cached.py --epochs 60 --class_weight_mode sqrt --reuse_features --occlusion_negatives_per_image 2",
        language="powershell",
    )
    st.stop()

model = load_trained_model(str(model_path))
class_names = load_class_names(str(classes_path))
cascades = load_cascades()

uploaded_file = st.file_uploader(
    "Upload a face image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Predict", type="primary", use_container_width=True):
        predictions = predict_faces(
            model,
            image,
            image_size=image_size,
            min_confidence=min_confidence,
            min_margin=min_margin,
            class_names=class_names,
            cascades=cascades,
        )

        if not predictions:
            st.warning("No clear face was detected. Please upload a front-facing face photo.")
            st.stop()

        annotated = draw_predictions(image, predictions, class_names)
        st.image(annotated, caption="Detected face result", use_container_width=True)

        st.subheader("Prediction Result")

        for face_number, prediction in enumerate(predictions, start=1):
            predicted_class = class_names[prediction.predicted_index]
            label = prediction_label(prediction, class_names)
            delta = f"top: {display_class_name(predicted_class)} {prediction.confidence:.2%}"
            st.metric(f"Face {face_number}", label, delta)

            probability_frame = pd.DataFrame(
                {
                    "Class": [display_class_name(class_name) for class_name in class_names],
                    "Probability": [float(probability) for probability in prediction.probabilities],
                }
            )
            st.bar_chart(probability_frame, x="Class", y="Probability", use_container_width=True)

            for class_name, probability in zip(class_names, prediction.probabilities):
                st.progress(float(probability), text=f"{display_class_name(class_name)}: {float(probability):.2%}")
else:
    st.info("Upload an image to start prediction.")
