from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight


PROJECT_DIR = Path(__file__).parent

TRAIN_DIR = (
    PROJECT_DIR
    / "dataset"
    / "train"
)

VALIDATION_DIR = (
    PROJECT_DIR
    / "dataset"
    / "validation"
)

MODEL_DIR = (
    PROJECT_DIR
    / "model"
)

MODEL_PATH = (
    MODEL_DIR
    / "crop_disease_model_38classes.keras"
)

CLASS_NAMES_PATH = (
    MODEL_DIR
    / "class_names.txt"
)


IMAGE_SIZE = (
    224,
    224
)

BATCH_SIZE = 32

EPOCHS = 15

SEED = 42


print(
    "========================================"
)

print(
    "38-CLASS CROP DISEASE MODEL TRAINING"
)

print(
    "========================================"
)


if not TRAIN_DIR.exists():

    raise FileNotFoundError(
        f"Training directory not found: {TRAIN_DIR}"
    )


if not VALIDATION_DIR.exists():

    raise FileNotFoundError(
        f"Validation directory not found: {VALIDATION_DIR}"
    )


MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print(
    "\nLoading training dataset..."
)


train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)


print(
    "\nLoading validation dataset..."
)


validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VALIDATION_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


class_names = (
    train_dataset.class_names
)

number_of_classes = (
    len(class_names)
)


print(
    f"\nNumber of classes: {number_of_classes}"
)


print(
    "\nClasses:"
)


for index, class_name in enumerate(
    class_names
):

    print(
        f"{index}: {class_name}"
    )


with open(
    CLASS_NAMES_PATH,
    "w",
    encoding="utf-8"
) as file:

    for class_name in class_names:

        file.write(
            class_name + "\n"
        )


print(
    f"\nClass names saved to:"
)

print(
    CLASS_NAMES_PATH
)


print(
    "\nCalculating class weights..."
)


class_counts = np.zeros(
    number_of_classes,
    dtype=np.int64
)


for _, labels in train_dataset:

    labels_numpy = (
        labels.numpy()
    )

    for label in labels_numpy:

        class_counts[label] += 1


class_indices = np.arange(
    number_of_classes
)


class_weights_array = (
    compute_class_weight(
        class_weight="balanced",
        classes=class_indices,
        y=np.repeat(
            class_indices,
            class_counts
        )
    )
)


class_weights = {
    index: float(weight)
    for index, weight in enumerate(
        class_weights_array
    )
}


print(
    "\nClass weights:"
)


for index, class_name in enumerate(
    class_names
):

    print(
        f"{class_name}: "
        f"{class_weights[index]:.4f}"
    )


AUTOTUNE = (
    tf.data.AUTOTUNE
)


train_dataset = (
    train_dataset
    .prefetch(AUTOTUNE)
)

validation_dataset = (
    validation_dataset
    .prefetch(AUTOTUNE)
)


data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip(
            "horizontal"
        ),

        tf.keras.layers.RandomRotation(
            0.15
        ),

        tf.keras.layers.RandomZoom(
            0.15
        ),

        tf.keras.layers.RandomContrast(
            0.10
        ),
    ]
)


print(
    "\nLoading MobileNetV2..."
)


base_model = (
    tf.keras.applications.MobileNetV2(
        input_shape=(
            IMAGE_SIZE[0],
            IMAGE_SIZE[1],
            3
        ),
        include_top=False,
        weights="imagenet"
    )
)


base_model.trainable = False


inputs = tf.keras.Input(
    shape=(
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        3
    )
)


x = data_augmentation(
    inputs
)


x = tf.keras.applications.mobilenet_v2.preprocess_input(
    x
)


x = base_model(
    x,
    training=False
)


x = tf.keras.layers.GlobalAveragePooling2D()(
    x
)


x = tf.keras.layers.Dropout(
    0.30
)(
    x
)


outputs = tf.keras.layers.Dense(
    number_of_classes,
    activation="softmax"
)(
    x
)


model = tf.keras.Model(
    inputs,
    outputs
)


model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),

    loss=(
        "sparse_categorical_crossentropy"
    ),

    metrics=[
        "accuracy"
    ]
)


model.summary()


callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=0.000001,
        verbose=1
    )
]


print(
    "\n========================================"
)

print(
    "STARTING TRAINING"
)

print(
    "========================================\n"
)


history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)


print(
    "\n========================================"
)

print(
    "TRAINING COMPLETED"
)

print(
    "========================================"
)


best_model = tf.keras.models.load_model(
    MODEL_PATH
)


validation_loss, validation_accuracy = (
    best_model.evaluate(
        validation_dataset,
        verbose=1
    )
)


print(
    f"\nFinal validation accuracy: "
    f"{validation_accuracy * 100:.2f}%"
)


print(
    f"Final validation loss: "
    f"{validation_loss:.4f}"
)


print(
    "\nBest model saved at:"
)

print(
    MODEL_PATH
)