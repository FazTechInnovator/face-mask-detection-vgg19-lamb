# Friend Handoff Guide

Use this when sharing the project with a friend for presentation or viva.

## GitHub Sharing

The repository should include source code, docs, requirements, batch files, and
the ready trained model:

```text
src/
config.yaml
requirements.txt
setup_windows.bat
run_webcam.bat
run_streamlit.bat
outputs/best_model.keras
outputs/final_model.keras
outputs/class_names.json
outputs/evaluation/
assets/screenshots/
README.md
HOW_TO_RUN_WINDOWS.md
```

Do not upload these folders:

```text
venv/
raw_datasets/
dataset/
outputs/feature_cache/
outputs/*_before_live_tuning.keras
```

The trained model is already available in `outputs/best_model.keras`. It is
about 77 MB, which is below GitHub's 100 MB single-file limit. If GitHub refuses
large files, upload `outputs/best_model.keras` to a GitHub Release or Google
Drive and tell your friend to place it back in `outputs/`.

## Friend Setup From GitHub

After downloading or cloning the project, your friend should install 64-bit
Python 3.11, then open Command Prompt or PowerShell inside the project folder
and run:

```powershell
setup_windows.bat
```

Then run either demo:

```powershell
run_webcam.bat
run_streamlit.bat
```

The webcam demo opens the laptop camera. Press `q` to quit. The Streamlit demo
opens in the browser, usually at `http://localhost:8501`.

If the webcam is slow:

```powershell
run_webcam.bat --predict_every 1.0 --display_width 420
```

If the wrong camera opens:

```powershell
run_webcam.bat --camera 1
```

## Retrain From Scratch

If your friend wants to rebuild the dataset and model:

```powershell
setup_windows.bat
prepare_dataset.bat
train_model.bat
```

`prepare_dataset.bat` downloads the public Face Mask Detection dataset through
KaggleHub and prepares cropped face images. `train_model.bat` uses the
laptop-friendly cached VGG19 training path.

## Current Trained Results

```text
Test accuracy: 94.55%
Weighted precision: 94.21%
Weighted recall: 94.55%
Weighted F1-score: 94.34%
```

## Notes

- The first setup can take time because TensorFlow is large.
- Native Windows TensorFlow uses CPU only for TensorFlow 2.11+.
- Webcam quality depends on lighting, face angle, and distance from the camera.
- The `incorrect_mask` class is harder because it has fewer training samples.
