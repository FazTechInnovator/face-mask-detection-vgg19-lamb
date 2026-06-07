# Viva Questions And Answers

## 1. What is face mask detection?

Face mask detection is a computer vision task where a model predicts whether a
person is wearing a mask, not wearing a mask, or wearing a mask incorrectly.

## 2. What are the classes in this project?

The project supports `mask`, `no_mask`, and `incorrect_mask`.

## 3. Why did you use VGG19?

VGG19 is a well-known convolutional neural network with strong feature extraction
ability. It is also easy to explain in an academic presentation.

## 4. What is transfer learning?

Transfer learning uses a model trained on a large dataset, such as ImageNet, and
adapts it to a new task with a smaller dataset.

## 5. Why is transfer learning useful here?

It reduces training time and helps performance when the available face mask
dataset is limited.

## 6. What is the input size of VGG19?

This project uses 224x224 RGB images.

## 7. What is LAMB optimizer?

LAMB is the Layer-wise Adaptive Moments optimizer for Batch training. It applies
adaptive updates with layer-wise scaling.

## 8. Why use LAMB instead of Adam?

Adam is common and effective, but LAMB gives a stronger technical contribution
because it uses layer-wise trust ratios and is less common in beginner projects.

## 9. What loss function is used?

Sparse categorical crossentropy is used because labels are integer class IDs and
the output layer uses softmax.

## 10. What is softmax?

Softmax converts raw model outputs into probabilities for each class.

## 11. What is data augmentation?

Data augmentation applies random transformations such as flip, rotation, zoom,
and contrast changes to improve generalization.

## 12. What is overfitting?

Overfitting happens when a model performs well on training data but poorly on new
unseen data.

## 13. How does this project reduce overfitting?

It uses augmentation, dropout, L2 regularization, early stopping, and transfer
learning.

## 14. What is class imbalance?

Class imbalance occurs when one class has many more images than another class.

## 15. How does the project handle class imbalance?

It computes class weights from the training labels and passes them to model
training.

## 16. What is ModelCheckpoint?

ModelCheckpoint saves the best model during training based on validation
performance.

## 17. What is EarlyStopping?

EarlyStopping stops training when validation performance stops improving.

## 18. What is ReduceLROnPlateau?

It reduces the learning rate when validation loss stops improving.

## 19. What is CSVLogger?

CSVLogger saves epoch-by-epoch training metrics to a CSV file.

## 20. What is a confusion matrix?

A confusion matrix shows correct and incorrect predictions for each class.

## 21. What is precision?

Precision measures how many predicted positives are actually correct.

## 22. What is recall?

Recall measures how many actual positives the model correctly finds.

## 23. What is F1-score?

F1-score is the harmonic mean of precision and recall.

## 24. What does accuracy mean?

Accuracy is the fraction of total predictions that are correct.

## 25. How does webcam detection work?

OpenCV reads frames from the camera, Haar Cascade detects faces, each face crop
is resized to 224x224, and the trained model predicts the class.

## 26. Why use Haar Cascade?

Haar Cascade is lightweight, available in OpenCV, and easy to use for a live demo.

## 27. What happens if no face is detected?

The webcam script uses a center-crop fallback so the demo can still show a
prediction instead of crashing.

## 28. What does the Streamlit app do?

It provides a local web interface where a user uploads an image and receives the
predicted class, confidence score, and probability bars.

## 29. What files are saved after training?

The project saves the best model, final model, class names, training logs,
accuracy/loss graphs, confusion matrix, classification report, and metrics JSON.

## 30. What are the limitations of this project?

The model depends on dataset quality, lighting, camera angle, face visibility,
and whether the training dataset is diverse.

## 31. Can this project work with two classes?

Yes. It supports a 2-class dataset with `mask` and `no_mask`.

## 32. Can this project work with three classes?

Yes. It supports `mask`, `no_mask`, and `incorrect_mask`.

## 33. Why save class names?

Class names ensure the prediction output matches the class order used during
training.

## 34. Why use validation data?

Validation data helps monitor generalization during training and is used for
checkpoint selection.

## 35. What future improvements can be made?

Future improvements include a larger dataset, better face detector, TensorFlow
Lite export, cloud deployment, and explainability using Grad-CAM.
