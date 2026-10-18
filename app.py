"""Research-only Gradio inference for the MobileNetV2 + XGBoost hybrid.

Requires the exact saved XGBoost model from Hybrid_Model.ipynb.
Uses the training notebook's 1/255 image scaling, NOT MobileNetV2
preprocess_input (which the original Gradio notebook used inconsistently).
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from xgboost import XGBClassifier

CLASS_NAMES = ("lung_aca", "lung_n", "lung_scc")
MODEL_FILE = Path(os.getenv("LUNG_XGBOOST_MODEL", "xgboost_hybrid_model.json"))


@lru_cache(maxsize=1)
def load_models():
    if not MODEL_FILE.is_file():
        raise FileNotFoundError(
            f"Saved XGBoost model not found: {MODEL_FILE}. "
            "Run Hybrid_Model.ipynb and provide its xgboost_hybrid_model.json."
        )
    extractor = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
        pooling="avg",
    )
    extractor.trainable = False
    classifier = XGBClassifier()
    classifier.load_model(str(MODEL_FILE))
    classes = getattr(classifier, "classes_", np.arange(3))
    if len(classes) != 3 or sorted(int(x) for x in classes) != [0, 1, 2]:
        raise ValueError(
            "Unexpected class identifiers. Review the saved training "
            "generator's class_indices before using this demo."
        )
    return extractor, classifier


def classify(image: Image.Image | None) -> dict[str, float]:
    if image is None:
        raise gr.Error("Please upload a histopathology image.")
    extractor, classifier = load_models()
    # ImageDataGenerator(rescale=1./255) in Hybrid_Model.ipynb.
    # Do not replace with MobileNetV2 preprocess_input without retraining
    # and independently validating the resulting inference pipeline.
    rgb = image.convert("RGB").resize((224, 224))
    array = np.asarray(rgb, dtype=np.float32)[None, ...] / 255.0
    features = extractor.predict(array, verbose=0)
    probabilities = classifier.predict_proba(features)[0]
    if len(probabilities) != len(CLASS_NAMES):
        raise gr.Error("Model output dimension does not match the three training classes.")
    return {
        name: float(value)
        for name, value in zip(CLASS_NAMES, probabilities)
    }


if __name__ == "__main__":
    demo = gr.Interface(
        fn=classify,
        inputs=gr.Image(type="pil", label="Histopathology image"),
        outputs=gr.Label(num_top_classes=3, label="Research-model probabilities"),
        title="Lung Cancer Subtype Classification — Research Demo",
        description=(
            "MobileNetV2 + XGBoost proof of concept (LC25000). "
            "Dataset-specific outputs; NOT for medical diagnosis. "
            "Training and inference use identical image scaling. "
            "A saved XGBoost checkpoint is required."
        ),
        flagging_mode="never",
    )
    demo.launch()
