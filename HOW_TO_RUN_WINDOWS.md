# How To Run On Windows

This guide assumes you are using PowerShell from the project root folder.

## 1. Create A Virtual Environment

Use 64-bit Python 3.11 for the smoothest TensorFlow setup on Windows.

```powershell
python -m venv venv
```

## 2. Activate The Virtual Environment

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this once and then activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## 3. Install Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Shortcut:

```powershell
setup_windows.bat
```

## 4. Download And Prepare The Real Dataset

Recommended automatic command:

```powershell
python src\prepare_dataset.py --download --reset
```

Shortcut:

```powershell
prepare_dataset.bat
```

The script uses the public Face Mask Detection dataset, crops annotated face
boxes, and creates this project structure automatically.

## 5. Manual Dataset Option

Use this folder structure:

```text
dataset/
  train/
    mask/
    no_mask/
    incorrect_mask/
  val/
    mask/
    no_mask/
    incorrect_mask/
  test/
    mask/
    no_mask/
    incorrect_mask/
```

For a 2-class dataset, keep only `mask` and `no_mask` inside each split.

## 6. Train The Model

Recommended for laptops:

```powershell
python src\train_cached.py --epochs 60 --class_weight_mode sqrt --reuse_features --occlusion_negatives_per_image 2
```

This still saves `outputs\best_model.keras` as a complete VGG19 model. It is
faster on CPU because frozen VGG19 features are cached once. The occlusion
option adds synthetic no-mask mouth-covering samples for a better live demo.

Full transfer-learning command:

```powershell
python src\train.py
```

Do not run training until real images have been added to the dataset folders.
The placeholder folders are intentionally empty.

Faster smoke test:

```powershell
python src\train.py --epochs 2 --fine_tune_epochs 1
```

Shortcut:

```powershell
train_model.bat
```

## 7. Evaluate The Model

```powershell
python src\evaluate.py --split test
```

Evaluation files are saved in `outputs\evaluation\`.

## 8. Predict A Single Image

Place a test image in `sample_images\`, then run:

```powershell
python src\predict_image.py sample_images\test.jpg
```

The labeled output image is saved in `outputs\predictions\`.

## 9. Run Webcam Demo

```powershell
python src\webcam_detect.py
```

Press `q` to close the webcam window.

If the wrong camera opens:

```powershell
python src\webcam_detect.py --camera 1
```

Shortcut:

```powershell
run_webcam.bat
```

## 10. Run Streamlit App

```powershell
streamlit run src\streamlit_app.py
```

If the model is not trained yet, the app shows:

```text
Please train the model first using python src/train_cached.py
```

Shortcut:

```powershell
run_streamlit.bat
```

## 11. Useful Config File

Most defaults are stored in `config.yaml`. You can change batch size, epochs,
learning rates, paths, and class names there instead of editing Python files.
