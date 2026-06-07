from __future__ import annotations

import keras
from keras import layers
from keras.applications import VGG19


@keras.saving.register_keras_serializable(package="FaceMaskProject")
class VGG19Preprocess(layers.Layer):
    """VGG19 preprocessing layer kept inside the saved model."""

    def call(self, inputs):
        return keras.applications.vgg19.preprocess_input(inputs)

    def get_config(self):
        return super().get_config()


def build_vgg19_lamb_model(
    num_classes: int,
    image_size: int = 224,
    dropout_rate: float = 0.40,
    dense_units: int = 256,
    l2_reg: float = 1e-4,
    train_base: bool = False,
    fine_tune_at: int | None = None,
) -> keras.Model:
    """Build a VGG19 transfer learning classifier for mask detection."""
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2.")

    inputs = keras.Input(shape=(image_size, image_size, 3), name="input_image")

    x = layers.RandomFlip("horizontal", name="random_flip")(inputs)
    x = layers.RandomRotation(0.08, name="random_rotation")(x)
    x = layers.RandomZoom(0.12, name="random_zoom")(x)
    x = layers.RandomContrast(0.10, name="random_contrast")(x)
    x = VGG19Preprocess(name="vgg19_preprocess")(x)

    base_model = VGG19(
        include_top=False,
        weights="imagenet",
        input_shape=(image_size, image_size, 3),
        name="vgg19_backbone",
    )
    base_model.trainable = train_base

    if train_base and fine_tune_at is not None:
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

    x = base_model(x, training=train_base)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = layers.BatchNormalization(name="batch_norm")(x)
    x = layers.Dropout(dropout_rate, name="dropout_1")(x)
    x = layers.Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        name="dense_features",
    )(x)
    x = layers.Dropout(0.30, name="dropout_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="class_output")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="VGG19_LAMB_Face_Mask")


def set_vgg19_backbone_trainable(model: keras.Model, fine_tune_at: int | None = 16) -> keras.Model:
    """Unfreeze the upper VGG19 layers for fine-tuning."""
    base_model = model.get_layer("vgg19_backbone")
    base_model.trainable = True

    for index, layer in enumerate(base_model.layers):
        layer.trainable = fine_tune_at is None or index >= fine_tune_at

    return model


def make_lamb_optimizer(learning_rate: float = 1e-4, weight_decay: float = 1e-5):
    """Create the LAMB optimizer with a clear error if Keras is too old."""
    lamb_cls = getattr(keras.optimizers, "Lamb", None)
    if lamb_cls is None:
        raise ImportError("keras.optimizers.Lamb was not found. Install keras>=3.3.0 and tensorflow>=2.16.0.")

    return lamb_cls(learning_rate=learning_rate, weight_decay=weight_decay)


def compile_model(
    model: keras.Model,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
) -> keras.Model:
    """Compile the model with LAMB and sparse categorical crossentropy."""
    model.compile(
        optimizer=make_lamb_optimizer(learning_rate=learning_rate, weight_decay=weight_decay),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model
