from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


project_dir = Path(__file__).parent

model_path = (
    project_dir
    / "model"
    / "crop_disease_model.keras"
)

class_names_path = (
    project_dir
    / "model"
    / "class_names.txt"
)


# Load model
model = tf.keras.models.load_model(
    model_path
)


# Load class names
with open(
    class_names_path,
    "r",
    encoding="utf-8"
) as file:

    class_names = [
        line.strip()
        for line in file.readlines()
    ]


def predict_disease(image):

    image = image.convert("RGB")

    image = image.resize(
        (224, 224)
    )

    image_array = np.array(
        image
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    image_array = (
        tf.keras.applications
        .mobilenet_v2
        .preprocess_input(
            image_array
        )
    )

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]


    # Get top 3 predictions
    top_indices = np.argsort(
        predictions
    )[-3:][::-1]


    top_predictions = []

    for index in top_indices:

        predicted_class = (
            class_names[index]
        )

        confidence = float(
            predictions[index]
        )

        top_predictions.append(
            (
                predicted_class,
                confidence
            )
        )


    # Top prediction
    predicted_class = (
        top_predictions[0][0]
    )

    confidence = (
        top_predictions[0][1]
    )


    return (
        predicted_class,
        confidence,
        top_predictions
    )