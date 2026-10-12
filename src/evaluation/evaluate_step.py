import argparse
import ast
import csv
import json
import os
from pathlib import Path

import numpy as np
import mlflow
from tensorflow import keras
from sklearn.metrics import precision_score, recall_score, f1_score

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential


SUBSCRIPTION_ID = ""
RESOURCE_GROUP = "rg-satellite-anomaly-mlops"
WORKSPACE_NAME = "satellite-anomaly-ml"
TENANT_ID = ""


def configure_mlflow():
    """Same fix as train.py: skip the MLClient/workspace lookup when running as an
    Azure ML job, since MLFLOW_TRACKING_URI is already auto-injected there, and the
    manual lookup would authenticate as the compute's managed identity (which lacks
    workspace-read rights)."""
    if "MLFLOW_TRACKING_URI" in os.environ:
        mlflow.set_experiment("satellite-anomaly-detection")
        return

    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
        tenant_id=TENANT_ID,
    )
    tracking_uri = ml_client.workspaces.get(WORKSPACE_NAME).mlflow_tracking_uri
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("satellite-anomaly-detection")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained channel model and write metrics.json for the next pipeline step.")
    parser.add_argument("--channel_name", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True,
                         help="Folder containing model_<channel>.keras (output of the train step)")
    parser.add_argument("--labeled_anomalies_path", type=str, required=True)
    parser.add_argument("--std_multiplier", type=float, default=3.0)
    parser.add_argument("--metrics_output", type=str, required=True,
                         help="Folder to write metrics.json into, for the register step to read")
    return parser.parse_args()


def load_channel_data(data_path: str, channel_name: str):
    channel_dir = Path(data_path) / channel_name
    X_train = np.load(channel_dir / "X_train.npy")
    y_train = np.load(channel_dir / "y_train.npy")
    X_test = np.load(channel_dir / "X_test.npy")
    y_test = np.load(channel_dir / "y_test.npy")
    return X_train, y_train, X_test, y_test


def load_anomaly_sequences(labeled_anomalies_path: str, channel_name: str):
    with open(labeled_anomalies_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["chan_id"] == channel_name:
                return ast.literal_eval(row["anomaly_sequences"]), int(row["num_values"])
    raise ValueError(f"Channel '{channel_name}' not found in {labeled_anomalies_path}")


def build_ground_truth_mask(sequences, num_values: int) -> np.ndarray:
    mask = np.zeros(num_values, dtype=bool)
    for start, end in sequences:
        mask[start:end + 1] = True
    return mask


def main():
    args = parse_args()
    configure_mlflow()

    print(f"[{args.channel_name}] loading data and model for evaluation")
    X_train, y_train, X_test, y_test = load_channel_data(args.data_path, args.channel_name)

    model_file = Path(args.model_path) / f"model_{args.channel_name}.keras"
    model = keras.models.load_model(model_file)

    train_preds = model.predict(X_train, verbose=0).flatten()
    train_errors = np.abs(train_preds - y_train.flatten())
    threshold = train_errors.mean() + args.std_multiplier * train_errors.std()

    test_preds = model.predict(X_test, verbose=0).flatten()
    test_errors = np.abs(test_preds - y_test.flatten())
    predicted_anomaly = test_errors > threshold

    sequences, num_values = load_anomaly_sequences(args.labeled_anomalies_path, args.channel_name)
    gt_mask_full = build_ground_truth_mask(sequences, num_values)
    window_size = X_test.shape[1]
    gt_aligned = gt_mask_full[window_size:]

    n = min(len(gt_aligned), len(predicted_anomaly))
    if len(gt_aligned) != len(predicted_anomaly):
        print(f"[{args.channel_name}] WARNING: length mismatch — trimming to {n}.")
    gt_aligned = gt_aligned[:n]
    predicted_anomaly = predicted_anomaly[:n]

    precision = precision_score(gt_aligned, predicted_anomaly, zero_division=0)
    recall = recall_score(gt_aligned, predicted_anomaly, zero_division=0)
    f1 = f1_score(gt_aligned, predicted_anomaly, zero_division=0)

    print(f"[{args.channel_name}] precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}")

    with mlflow.start_run(run_name=f"{args.channel_name}-eval"):
        mlflow.log_params({"channel_name": args.channel_name, "std_multiplier": args.std_multiplier})
        mlflow.log_metric("threshold", threshold)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

    output_dir = Path(args.metrics_output)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "channel_name": args.channel_name,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": threshold,
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[{args.channel_name}] metrics written to {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()