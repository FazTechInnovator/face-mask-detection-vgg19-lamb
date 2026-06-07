# Final Report

## Abstract

This project presents a real-time face mask detection system using VGG19 transfer
learning and the LAMB optimizer. The system classifies images into `mask`,
`no_mask`, and optionally `incorrect_mask`. It includes a complete machine
learning pipeline for dataset validation, training, evaluation, image prediction,
webcam demonstration, and Streamlit deployment.

## Introduction

Face mask detection is a practical computer vision task used in environments
where safety compliance is important. Traditional manual inspection is limited by
human attention and scalability. Deep learning can automate visual classification
by learning discriminative features from image data.

This project improves a basic face mask classifier by supporting an
incorrect-mask class and by using VGG19 with the LAMB optimizer. The final system
is prepared for training, evaluation, demonstration, and academic presentation.

## Literature Review Summary

Convolutional neural networks have been widely used for image classification.
Architectures such as VGG, ResNet, MobileNet, and EfficientNet are commonly used
for transfer learning. VGG19 is known for its simple stack of convolutional
layers and strong feature extraction capability. Transfer learning allows a
pretrained model to be adapted to a new task without training a large network
from scratch.

Face mask detection systems often use CNN-based classifiers with image datasets
containing masked and unmasked faces. Some systems also include real-time
detection with OpenCV. This project follows that direction while adding support
for incorrectly worn masks and using LAMB optimization.

## Methodology

The project methodology includes:

1. Organizing images into train, validation, and test folders.
2. Validating dataset structure and image availability.
3. Loading images at 224x224 resolution.
4. Applying image augmentation during training.
5. Building a VGG19 transfer learning model.
6. Training the custom classifier head.
7. Fine-tuning upper VGG19 layers.
8. Evaluating the best model on test data.
9. Deploying prediction through command line, webcam, and Streamlit.

## Dataset Preprocessing

The recommended dataset is the public Face Mask Detection dataset by Andrew
Maranhao. The dataset contains image-level files and bounding-box annotations
for three labels: `with_mask`, `without_mask`, and `mask_weared_incorrect`.

The project preparation script converts the detection dataset into a classifier
dataset by cropping each annotated face region, applying a small padding margin,
mapping labels to `mask`, `no_mask`, and `incorrect_mask`, and splitting the
prepared crops into train, validation, and test folders.

The model input size is 224x224x3. Images are loaded with Keras utilities, and
labels are inferred from class folder names. During training, augmentation
layers apply random horizontal flip, rotation, zoom, and contrast variation.
VGG19 preprocessing is included inside the model so prediction scripts can pass
normal RGB arrays.

## Model Architecture

The model uses VGG19 as a convolutional backbone with ImageNet pretrained
weights. The original VGG19 top classifier is removed. The custom classifier
includes:

- GlobalAveragePooling2D
- BatchNormalization
- Dropout
- Dense layer with ReLU
- Dropout
- Softmax output layer

The number of output neurons is automatically set from the detected class count.

## Optimizer

The optimizer used is LAMB. LAMB is an adaptive optimizer that uses layer-wise
trust ratios. It can be useful for training deep neural networks and provides a
more advanced optimizer choice than standard Adam in this project.

## Training Process

Training is divided into two stages:

1. The VGG19 backbone is frozen while the custom classifier head is trained.
2. The upper VGG19 layers are unfrozen and fine-tuned with a smaller learning
   rate.

For laptop-friendly training, the project also includes a cached-feature
training mode. In this mode, the frozen VGG19 backbone extracts features once,
and the LAMB classifier head is trained for more epochs on those cached
features. The final saved model remains a complete VGG19-based Keras model.
For the live demo, the cached training mode also adds synthetic no-mask
lower-face occlusion samples so that hand or object occlusion is less likely to
be classified as a real mask.

The training process includes:

- Class weights
- ModelCheckpoint
- EarlyStopping
- ReduceLROnPlateau
- CSVLogger

The best model and final model are saved separately.

## Evaluation Metrics

The project evaluates performance using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Classification report
- Test loss

These metrics help identify both overall performance and class-specific
weaknesses.

## Results

The trained model was evaluated on the prepared test split generated from the
public Face Mask Detection dataset. The dataset was converted from bounding-box
annotations into cropped face-classification images.

```text
Prepared crop count: 3,374
Synthetic no-mask occlusion negatives: 758
Number of classes: 3
Train / Validation / Test samples: 2,355 / 524 / 495
Best validation accuracy: 90.46%
Test accuracy: 94.55%
Weighted precision: 94.21%
Weighted recall: 94.55%
Weighted F1-score: 94.34%
Macro F1-score: 78.14%
Test loss: 0.2276
```

The class-wise test performance was strongest for `mask`, good for `no_mask`,
and weaker for `incorrect_mask`. This is expected because the incorrect-mask
class has fewer examples in the prepared dataset.

Generated result files:

```text
outputs/training/accuracy_loss.png
outputs/evaluation/test_confusion_matrix.png
outputs/evaluation/test_classification_report.csv
outputs/evaluation/test_metrics.json
```

## Conclusion

The project successfully provides a complete pipeline for face mask detection
using VGG19 and LAMB. It supports dataset validation, training, evaluation,
single-image prediction, webcam demonstration, and Streamlit deployment. The
project is suitable for academic submission because it includes working code,
clear documentation, and presentation material.

## Future Work

- Use a larger and more diverse dataset.
- Replace Haar Cascade with a modern face detector.
- Add Grad-CAM visualization.
- Export the model to TensorFlow Lite.
- Add video file processing.
- Deploy the Streamlit application online.
