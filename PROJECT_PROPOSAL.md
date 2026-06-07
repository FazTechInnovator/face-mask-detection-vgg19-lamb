# Project Proposal

## Title

Real-Time Face Mask Detection Using VGG19 With LAMB Optimizer

## Abstract

This project proposes a deep learning-based face mask detection system that
classifies images into `mask`, `no_mask`, and optionally `incorrect_mask`. The
system uses VGG19 transfer learning for image feature extraction and the LAMB
optimizer for training. The final project includes training, evaluation,
single-image prediction, webcam detection, and a Streamlit deployment app.

## Problem Statement

In safety-sensitive environments, it may be necessary to identify whether people
are wearing masks correctly. Manual monitoring can be slow, inconsistent, and
difficult to scale. Many basic systems classify only `mask` and `no_mask`, which
does not handle incorrectly worn masks. This project addresses that gap by
supporting a third class, `incorrect_mask`.

## Objectives

- Develop a VGG19 transfer learning model for face mask classification.
- Train the model using the LAMB optimizer.
- Support both 2-class and 3-class datasets.
- Evaluate performance using accuracy, precision, recall, F1-score, confusion
  matrix, and classification report.
- Provide demo-ready prediction, webcam, and Streamlit interfaces.
- Prepare documentation suitable for academic submission and presentation.

## Scope

The project focuses on image classification and real-time webcam demonstration.
It does not enforce public health policy, identify individuals, or perform
biometric recognition. The output is limited to mask-related class prediction.

## Methodology

1. Organize image dataset into train, validation, and test splits.
2. Validate dataset folders and image availability.
3. Resize images to 224x224 pixels.
4. Apply data augmentation during training.
5. Load VGG19 with ImageNet pretrained weights.
6. Replace the top layer with a custom classifier.
7. Train the classifier head while freezing VGG19.
8. Fine-tune upper VGG19 layers with a smaller learning rate.
9. Evaluate on the test split.
10. Deploy locally using Streamlit and OpenCV webcam demo.

For Windows laptop demonstrations, a cached-feature training mode is also used
to compute frozen VGG19 features once and train the LAMB classifier head faster
on CPU.

## Dataset Description

The recommended dataset is the public Face Mask Detection dataset by Andrew
Maranhao. Dataset Ninja lists it as CC0 1.0 and describes it as 853 images with
4,072 annotated face-mask objects in three classes:

- `with_mask`
- `without_mask`
- `mask_weared_incorrect`

The preparation script maps these labels to the project labels `mask`,
`no_mask`, and `incorrect_mask`, crops the annotated face boxes, and creates
train, validation, and test folders for classification.

## Model Architecture

- Base model: VGG19
- Pretrained weights: ImageNet
- Input size: 224x224x3
- Pooling: GlobalAveragePooling2D
- Regularization: BatchNormalization, Dropout, L2 regularization
- Output layer: Softmax classifier

## Optimizer Explanation

LAMB stands for Layer-wise Adaptive Moments optimizer for Batch training. It is
an adaptive optimizer that computes layer-wise trust ratios. It is useful to
discuss in this project because it provides a stronger technical contribution
than using only common optimizers such as Adam or SGD.

## Expected Output

The system predicts one of the trained class labels for an image or webcam face
crop. It also produces training graphs, confusion matrix, classification report,
and metrics files for project documentation.

The current trained model reaches 94.55% test accuracy on the prepared test
split, with weighted F1-score of 94.34%.

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Classification report
- Test loss

## Tools And Technologies

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- Pillow
- Streamlit
- YAML configuration

## Conclusion

The project combines transfer learning, LAMB optimization, real-time computer
vision, and local web deployment. It is suitable for final year or AI course
submission because it includes both a working machine learning pipeline and
presentation-ready documentation.
