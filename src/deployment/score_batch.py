import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras


def init():
    global model, threshold

    model_dir = Path(os.getenv("AZUREML_MODEL_DIR"))
    model_files = list(model_dir.rglob("*.keras"))
    if not model_files:
        raise FileNotFoundError(f"No .keras model file found under {model_dir}")
    model = keras.models.load_model(model_files[0])

    threshold = float(os.getenv("ANOMALY_THRESHOLD", "0.05"))
    print(f"Model loaded from {model_files[0]}, using threshold={threshold}")


def run(mini_batch):
   
    results = []

    for file_path in mini_batch:
        windows = np.load(file_path)
        predictions = model.predict(windows, verbose=0).flatten()

        for i, pred in enumerate(predictions):
            results.append({
                "source_file": os.path.basename(file_path),
                "window_index": i,
                "predicted_next_value": float(pred),
            })

    return pd.DataFrame(results)