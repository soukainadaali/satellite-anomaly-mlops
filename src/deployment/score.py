import os
import json
import numpy as np
from pathlib import Path
from tensorflow import keras


def init():
    global model, threshold

    # AZUREML_MODEL_DIR points at the mounted model folder at deployment time
    model_dir = Path(os.getenv("AZUREML_MODEL_DIR"))
    model_files = list(model_dir.rglob("*.keras"))
    if not model_files:
        raise FileNotFoundError(f"No .keras model file found under {model_dir}")
    model = keras.models.load_model(model_files[0])

    # Threshold is passed in as an environment variable set on the deployment (see deploy script)
    threshold = float(os.getenv("ANOMALY_THRESHOLD", "0.05"))
    print(f"Model loaded from {model_files[0]}, using threshold={threshold}")


def run(raw_data):
    """
    Expects JSON body: {"data": [[[...25 features...], [...], ... 100 timesteps]]}
    i.e. a single windowed sequence of shape (1, window_size, n_features).
    """
    try:
        body = json.loads(raw_data)
        window = np.array(body["data"], dtype=np.float32)  # shape (1, window_size, n_features)

        prediction = model.predict(window, verbose=0).flatten()[0]
        actual_next_value = body.get("actual_next_value")  # optional, if caller has ground truth to compare

        result = {"predicted_next_value": float(prediction)}

        if actual_next_value is not None:
            error = abs(prediction - float(actual_next_value))
            result["error"] = float(error)
            result["is_anomaly"] = bool(error > threshold)
            result["threshold"] = threshold

        return result

    except Exception as e:
        return {"error": str(e)}