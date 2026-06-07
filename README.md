# Real-Time Face Mask Detection Using VGG19 With LAMB Optimizer

Final year / AI course project for classifying face images as:

- `mask`
- `no_mask`
- `incorrect_mask`

The project is deployment-ready for local Streamlit demos and presentation-ready
for viva, report submission, and classroom demonstration.

For GitHub upload and friend setup instructions, see
[`GITHUB_SHARE_GUIDE.md`](GITHUB_SHARE_GUIDE.md) and
[`FRIEND_HANDOFF.md`](FRIEND_HANDOFF.md).

## Problem Statement

Face mask compliance can be important in hospitals, laboratories, industrial
spaces, classrooms, and other controlled environments. Manual monitoring is slow
and inconsistent. This project builds a computer vision system that classifies
whether a person is wearing a mask correctly, not wearing a mask, or wearing a
mask incorrectly.

## Objectives

- Build a VGG19 transfer learning model for face mask classification.
- Use the LAMB optimizer for stable deep learning optimization.
- Support both 2-class and 3-class datasets.
- Validate dataset folders before training.
- Generate evaluation outputs for presentation and report writing.
- Provide single-image prediction, webcam demo, and Streamlit deployment app.

## Features

- Dynamic paths using `pathlib`
- Central configuration in `config.yaml`
- Automatic class detection
- 2-class support: `mask`, `no_mask`
- 3-class support: `mask`, `no_mask`, `incorrect_mask`
- Dataset validation before training
- Image augmentation inside the model
- VGG19 ImageNet transfer learning
- LAMB optimizer
- Class weights for imbalanced datasets
- ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
- Best model and final model saving
- Accuracy/loss plots
- Confusion matrix
- Classification report
- Standalone evaluation script
- Command-line image prediction
- Labeled prediction image output
- OpenCV webcam detection with Haar Cascade face detection
- Face-first prediction for uploaded photos instead of classifying the whole image
- Conservative `Uncertain` output for weak or ambiguous predictions
- Streamlit image upload app with demo-safe model checks
- Reproducible public dataset preparation script
- Windows batch files for setup, training, webcam, and Streamlit demos

## Folder Structure

```text
face_mask_vgg19_lamb_from_scratch/
  assets/
    screenshots/
  dataset/
    README.md
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
  outputs/
    evaluation/
    predictions/
    training/
  sample_images/
    README.md
  src/
    config_utils.py
    dataset_utils.py
    prepare_dataset.py
    evaluate.py
    evaluation_utils.py
    model_utils.py
    predict_image.py
    train_cached.py
    streamlit_app.py
    train.py
    webcam_detect.py
  config.yaml
  DEPLOYMENT_GUIDE.md
  FINAL_REPORT.md
  FRIEND_HANDOFF.md
  HOW_TO_RUN_WINDOWS.md
  PRESENTATION_CONTENT.md
  PROJECT_PROPOSAL.md
  README.md
  requirements.txt
  prepare_dataset.bat
  run_streamlit.bat
  run_webcam.bat
  RUN_COMMANDS.txt
  setup_windows.bat
  train_model.bat
  VIVA_QUESTIONS.md
```

## Dataset Structure

Use this structure for a 3-class dataset:

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

For a 2-class dataset, use only:

```text
mask/
no_mask/
```

inside `train`, `val`, and `test`.

Keep folder names exactly as shown. The scripts detect classes automatically but
expect the project class names.

## Public Dataset Used

The recommended real dataset is **Face Mask Detection** by Andrew Maranhao,
available on Kaggle and summarized/downloadable through Dataset Ninja.

- Dataset size: 853 images
- Annotation count: 4,072 objects
- Source labels: `with_mask`, `without_mask`, `mask_weared_incorrect`
- Project labels: `mask`, `no_mask`, `incorrect_mask`
- License listed by Dataset Ninja: CC0 1.0

The project prepares this detection dataset for VGG19 classification by cropping
each annotated face box into the correct class folder.

## Installation Guide

Recommended Python version: Python 3.10 or 3.11.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

The first training run may download the official VGG19 ImageNet weights through
Keras if they are not already cached on your computer.

For a one-click Windows setup, double-click or run:

```powershell
setup_windows.bat
```

## Dataset Preparation Guide

To download and prepare the public dataset automatically:

```powershell
python src\prepare_dataset.py --download --reset
```

Or with the batch file:

```powershell
prepare_dataset.bat
```

This creates cropped classifier images in:

```text
dataset/train/
dataset/val/
dataset/test/
```

and saves source metadata in:

```text
outputs/dataset_preparation/dataset_source.json
```

## Training Guide

Add dataset images first, then run the recommended laptop-friendly training
command:

```powershell
python src\train_cached.py --epochs 60 --class_weight_mode sqrt --reuse_features --occlusion_negatives_per_image 2
```

With the batch file, which uses cached-feature training:

```powershell
train_model.bat
```

`src/train.py` is the full transfer-learning pipeline with in-model
augmentation and optional fine-tuning. `src/train_cached.py` is recommended for
Windows laptops because it computes frozen VGG19 features once, then trains the
LAMB classifier head quickly. The `--occlusion_negatives_per_image 2` option
adds synthetic no-mask lower-face occlusions to reduce false mask predictions
when someone covers their mouth with a hand.

Optional fast smoke test:

```powershell
python src\train.py --epochs 2 --fine_tune_epochs 1
```

Important training outputs:

```text
outputs/best_model.keras
outputs/final_model.keras
outputs/class_names.json
outputs/training/training_history.csv
outputs/training/accuracy.png
outputs/training/loss.png
outputs/training/accuracy_loss.png
outputs/evaluation/test_confusion_matrix.png
outputs/evaluation/test_classification_report.csv
outputs/evaluation/test_metrics.json
```

## Evaluation Guide

```powershell
python src\evaluate.py --split test
```

Outputs are saved in:

```text
outputs/evaluation/
```

The evaluation script prints accuracy, precision, recall, and F1-score and saves
the confusion matrix, classification report, metrics JSON, and prediction table.

## Prediction Guide

Place a test image in `sample_images/`, then run:

```powershell
python src\predict_image.py sample_images\test.jpg
```

The script prints:

- Predicted class
- Confidence score
- Probability for each class
- Saved labeled image path

Prediction images are saved in:

```text
outputs/predictions/
```

## Webcam Demo Guide

```powershell
python src\webcam_detect.py
```

With the batch file:

```powershell
run_webcam.bat
```

Press `q` to quit.

If the wrong camera opens:

```powershell
python src\webcam_detect.py --camera 1
```

The webcam demo uses OpenCV Haar Cascade face detection. If no face is detected,
it shows `No face detected` instead of guessing from the whole frame.

## Streamlit Deployment Guide

```powershell
streamlit run src\streamlit_app.py
```

With the batch file:

```powershell
run_streamlit.bat
```

The app includes:

- Project overview
- Image upload
- Predict button
- Predicted class
- Confidence score
- Probability bars
- Model details
- Safe message if the model is not trained yet

If the model is missing, the app shows:

```text
Please train the model first using python src/train_cached.py
```

## Configuration

Main settings are in `config.yaml`:

```yaml
image_size: 224
batch_size: 32
epochs: 15
fine_tune_epochs: 10
learning_rate: 0.0001
fine_tune_learning_rate: 0.00001
weight_decay: 0.00001
class_names:
  - mask
  - no_mask
  - incorrect_mask
model_path: outputs/best_model.keras
output_dir: outputs
```

You can override common settings from the command line, for example:

```powershell
python src\train.py --batch_size 16 --epochs 5
```

## Model Explanation

VGG19 is a convolutional neural network with 19 weight layers. In this project,
VGG19 is used through transfer learning with ImageNet pretrained weights. The
original classification head is removed and replaced with a custom classifier for
mask-related classes.

Training is done in two stages:

1. Train the custom classifier head while VGG19 is frozen.
2. Fine-tune upper VGG19 layers using a smaller learning rate.

For laptop demos, the cached training script keeps VGG19 frozen and trains the
classifier head for more epochs. The final saved model is still a complete
VGG19-based `.keras` model for webcam and Streamlit inference.

## LAMB Optimizer Explanation

LAMB means Layer-wise Adaptive Moments optimizer for Batch training. It is related
to Adam-style adaptive optimization but scales updates layer by layer. This can
help stabilize training in deep networks and is a good choice to discuss in an AI
course project because it is less common than Adam in beginner projects.

## Results Section

Current trained model results:

```text
Dataset: Face Mask Detection by Andrew Maranhao
Prepared crop count: 3,374
Train / Val / Test: 2,355 / 524 / 495
Training command: python src\train_cached.py --epochs 60 --class_weight_mode sqrt --reuse_features --occlusion_negatives_per_image 2
Best Validation Accuracy: 90.46%
Test Accuracy: 94.55%
Weighted Precision: 94.21%
Weighted Recall: 94.55%
Weighted F1-score: 94.34%
Macro F1-score: 78.14%
```

Class-wise test report:

```text
mask:           precision 0.96, recall 0.98, F1 0.97
no_mask:        precision 0.89, recall 0.86, F1 0.87
incorrect_mask: precision 0.58, recall 0.44, F1 0.50
```

The `incorrect_mask` class is still the hardest class because it has fewer
training examples than the other two classes.

Use these generated files in your report and presentation:

```text
outputs/training/accuracy_loss.png
outputs/evaluation/test_confusion_matrix.png
outputs/evaluation/test_classification_report.csv
outputs/evaluation/test_metrics.json
```

## Screenshots Section

Add screenshots in:

```text
assets/screenshots/
```

Recommended screenshots:

- Dataset folder structure
- Training terminal output
- Accuracy/loss graph
- Confusion matrix
- Single-image prediction
- Webcam demo
- Streamlit app

## Future Improvements

- Train on a larger and more diverse dataset.
- Add deep learning face detection instead of Haar Cascade.
- Export the model to TensorFlow Lite for mobile deployment.
- Add Grad-CAM explainability for model interpretation.
- Add video file input support.
- Deploy the Streamlit app to a cloud service.

## Troubleshooting

### Model Not Found

Run training first:

```powershell
python src\train.py
```

### Dataset Folder Error

Make sure the folder names are exactly:

```text
mask
no_mask
incorrect_mask
```

For a 2-class dataset, remove `incorrect_mask` from all splits.

### Empty Dataset Error

The `.gitkeep` files are only placeholders. Add real image files before training.

### Webcam Not Opening

Try a different camera index:

```powershell
python src\webcam_detect.py --camera 1
```

### TensorFlow Install Issue

Use Python 3.10 or 3.11 and reinstall:

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### LAMB Optimizer Not Found

This project uses `keras.optimizers.Lamb`. Install the versions from
`requirements.txt`; older Keras versions may not include the LAMB optimizer.
