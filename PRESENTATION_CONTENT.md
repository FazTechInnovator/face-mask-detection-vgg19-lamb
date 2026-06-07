# Presentation Content

## Slide 1: Title Slide

Content:

- Real-Time Face Mask Detection Using VGG19 With LAMB Optimizer
- Student name
- Course / department
- Supervisor name

Speaking notes:

- Introduce the project as an AI-based computer vision system.
- Mention that it supports mask, no mask, and incorrect mask classification.

## Slide 2: Introduction

Content:

- Face mask detection is a computer vision classification problem.
- It can support safety monitoring in controlled environments.
- The system works with image upload and webcam input.

Speaking notes:

- Explain why automatic visual classification can reduce manual checking.
- Clarify that the system predicts mask status, not identity.

## Slide 3: Problem Statement

Content:

- Manual monitoring is time-consuming and inconsistent.
- Many existing beginner projects only support two classes.
- Incorrectly worn masks should be detected separately.

Speaking notes:

- Emphasize the limitation of basic `mask` vs `no_mask` systems.
- Explain the value of adding the `incorrect_mask` class.

## Slide 4: Objectives

Content:

- Build a VGG19 transfer learning model.
- Use LAMB optimizer.
- Support 2-class and 3-class datasets.
- Generate evaluation reports and demo interfaces.

Speaking notes:

- Explain that the project covers training, evaluation, prediction, webcam demo,
  and Streamlit deployment.

## Slide 5: Dataset

Content:

- Classes: mask, no_mask, incorrect_mask
- Splits: train, validation, test
- Images resized to 224x224
- Augmentation applied during training

Speaking notes:

- Mention that class folders are automatically detected.
- Explain that class weights are used for imbalanced data.

## Slide 6: Proposed Methodology

Content:

- Validate dataset folders.
- Load images using Keras.
- Train classifier head.
- Fine-tune upper VGG19 layers.
- Evaluate on test data.
- Deploy with webcam and Streamlit.

Speaking notes:

- Walk through the pipeline from dataset to final output.
- Mention that outputs are saved for report and presentation.

## Slide 7: Model Architecture: VGG19

Content:

- VGG19 pretrained on ImageNet
- Convolutional feature extractor
- Global average pooling
- Dense classifier
- Softmax output

Speaking notes:

- Explain transfer learning.
- Mention that VGG19 is strong for visual feature extraction.

## Slide 8: Optimizer: LAMB

Content:

- Layer-wise Adaptive Moments optimizer
- Uses adaptive learning behavior
- Applies layer-wise trust ratio
- Less common than Adam in beginner projects

Speaking notes:

- Explain that LAMB is the optimization method used for training.
- Compare briefly with Adam without going too deep into equations.

## Slide 9: Results And Evaluation

Content:

- Prepared dataset: 3,374 cropped face images
- Test accuracy: 94.55%
- Weighted precision: 94.21%
- Weighted recall: 94.55%
- Weighted F1-score: 94.34%
- Added synthetic no-mask occlusion negatives for live demo robustness
- Hardest class: incorrect_mask

Speaking notes:

- Explain that the trained model performs well overall on the test split.
- Mention that `incorrect_mask` is harder because it has fewer examples.
- Show `outputs/evaluation/test_confusion_matrix.png` and the accuracy/loss graph.

## Slide 10: Conclusion And Future Work

Content:

- Completed VGG19 + LAMB mask detection pipeline
- Supports image, webcam, and Streamlit demo
- Future work: larger dataset, better face detector, mobile deployment

Speaking notes:

- Summarize the complete system.
- End by discussing realistic limitations and improvements.
