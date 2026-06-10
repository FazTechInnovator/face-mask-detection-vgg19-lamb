# Face Mask Detection

A small computer vision project for detecting face mask status from images or a
laptop webcam.

The classifier uses VGG19 features and a LAMB optimizer. It predicts:

- `mask`
- `no_mask`
- `incorrect_mask`

## Run On Windows

Install Python 3.11, then open this folder in Command Prompt and run:

```bat
setup_windows.bat
```

Start the web app:

```bat
run_streamlit.bat
```

Start webcam detection:

```bat
run_webcam.bat
```

Press `q` to close the webcam window.

## Main Files

```text
src/streamlit_app.py      web interface
src/webcam_detect.py      live camera demo
src/predict_image.py      image prediction script
src/train_cached.py       training script
src/model_utils.py        VGG19 model and LAMB optimizer
```

The trained model is stored at:

```text
outputs/best_model.keras
```

Class labels are stored at:

```text
outputs/class_names.json
```

## Notes

If the webcam is slow:

```bat
run_webcam.bat --predict_every 1.0 --display_width 420
```

If the wrong camera opens:

```bat
run_webcam.bat --camera 1
```
