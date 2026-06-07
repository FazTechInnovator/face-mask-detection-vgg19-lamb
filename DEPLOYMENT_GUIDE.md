# Deployment Guide

## Local Streamlit Deployment

Install dependencies, prepare data, and train the model first:

```powershell
setup_windows.bat
prepare_dataset.bat
train_model.bat
```

Manual equivalent:

```powershell
python src\prepare_dataset.py --download --reset
python src\train_cached.py --epochs 60 --class_weight_mode sqrt --reuse_features --occlusion_negatives_per_image 2
```

Then run the Streamlit app:

```powershell
run_streamlit.bat
```

Manual equivalent:

```powershell
streamlit run src\streamlit_app.py
```

Open the local URL shown by Streamlit, usually:

```text
http://localhost:8501
```

## Optional LAN Deployment

To make the app visible on the same local network:

```powershell
streamlit run src\streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

Then open this URL from another device on the same network:

```text
http://YOUR_COMPUTER_IP:8501
```

## Required Files For Deployment

Keep these files and folders:

```text
config.yaml
requirements.txt
src/
outputs/best_model.keras
outputs/class_names.json
outputs/evaluation/
```

Optional but useful:

```text
outputs/evaluation/
assets/screenshots/
README.md
```

## Keeping Trained Model Files

The trained model files are ignored by Git because they can be large:

```text
outputs/best_model.keras
outputs/final_model.keras
```

For submission, include them in your ZIP file if your instructor expects a
ready-to-run demo. If the file is too large, provide the training instructions
and screenshots instead.

## Demo Mode

If the model is not trained, Streamlit will not crash. It shows:

```text
Please train the model first using python src/train_cached.py
```

## Common Errors And Fixes

### Model file missing

Run:

```powershell
python src\train_cached.py --epochs 60 --class_weight_mode sqrt --reuse_features --occlusion_negatives_per_image 2
```

### Class names file missing

The file `outputs/class_names.json` is created during training. Train the model
again if it is missing.

### Dataset folders missing

Prepare the real public dataset:

```powershell
python src\prepare_dataset.py --download --reset
```

Or manually check:

```text
dataset/train/mask
dataset/train/no_mask
dataset/val/mask
dataset/val/no_mask
dataset/test/mask
dataset/test/no_mask
```

For 3-class mode, also include:

```text
incorrect_mask
```

### Webcam not opening

Try:

```powershell
python src\webcam_detect.py --camera 1
```

### Streamlit port busy

Use another port:

```powershell
streamlit run src\streamlit_app.py --server.port 8502
```

### TensorFlow installation issue

Use Python 3.10 or 3.11, then reinstall:

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```
