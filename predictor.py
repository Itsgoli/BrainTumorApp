import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import logging
from pathlib import Path

import numpy as np
import tensorflow as tf

tf.get_logger().setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

MODEL_PATH = (
    Path(__file__).resolve().parent / "models" / "brain_tumor_pad_finetuned.keras"
)


class Predictor:
    def __init__(self):
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

        self.model = tf.keras.models.load_model(
            str(MODEL_PATH),
            compile=False,
        )

    def predict(self, image_path):
        image_bytes = Path(image_path).read_bytes()

        image = tf.io.decode_image(
            image_bytes,
            channels=3,
            expand_animations=False,
        )

        height, width = tf.unstack(tf.shape(image)[:2])
        side = tf.maximum(height, width)

        square = tf.image.pad_to_bounding_box(
            image,
            offset_height=(side - height) // 2,
            offset_width=(side - width) // 2,
            target_height=side,
            target_width=side,
        )

        resized = tf.image.resize(square, (224, 224))
        processed = tf.keras.applications.mobilenet_v2.preprocess_input(resized)
        batch = tf.expand_dims(processed, axis=0)

        probabilities = np.asarray(self.model(batch, training=False))[0]

        if probabilities.shape != (4,) or not np.all(np.isfinite(probabilities)):
            raise ValueError("Invalid model output.")

        predicted_index = int(np.argmax(probabilities))

        return {
            "class_name": CLASS_NAMES[predicted_index],
            "confidence": float(probabilities[predicted_index]),
            "probabilities": probabilities.tolist(),
        }
