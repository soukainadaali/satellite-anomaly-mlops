import argparse
import ast
from pathlib import Path

import numpy as np
import mlflow
from tensorflow import keras
from sklearn.metrics import precision_score, recall_score, f1_score

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

import csv


SUBSCRIPTION_ID = ""
RESOURCE_GROUP = "rg-satellite-anomaly-mlops"
WORKSPACE_NAME = "satellite-anomaly-ml"
TENANT_ID = ""


def configure_mlflow():
    """Point MLflow at this Azure ML workspace's built-in tracking server."""
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
    parser = argparse.ArgumentParser(description="Evaluate a trained channel model against labeled anomalies.")

    parser.add_argument("--channel_name", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True,
                         help="Path to the processed data folder (contains per-channel .npy files)")
    parser.add_argument("--model_path", type=str, default="models",
                         help="Folder containing the trained model file (model_<channel>.keras)")
    parser.add_argument("--labeled_anomalies_path", type=str, default="raw/labeled_anomalies.csv")
    parser.add_argument("--std_multiplier", type=float, default=3.0,
                         help="Threshold = mean(train_errors) + std_multiplier * std(train_errors)")

    return parser.parse_args()


def load_channel_data(data_path: str, channel_name: str):
    channel_dir = Path(data_path) / channel_name

    X_train = np.load(channel_dir / "X_train.npy")
    y_train = np.load(channel_dir / "y_train.npy")
    X_test = np.load(channel_dir / "X_test.npy")
    y_test = np.load(channel_dir / "y_test.npy")

    return X_train, y_train, X_test, y_test


def load_anomaly_sequences(labeled_anomalies_path: str, channel_name: str):
    """Returns (list of [start, end] index pairs, num_values) for one channel."""
    with open(labeled_anomalies_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["chan_id"] == channel_name:
                sequences = ast.literal_eval(row["anomaly_sequences"])
                num_values = int(row["num_values"])
                return sequences, num_values

    raise ValueError(f"Channel '{channel_name}' not found in {labeled_anomalies_path}")


def build_ground_truth_mask(sequences, num_values: int) -> np.ndarray:
    """Boolean array of length num_values, True at indices inside any labeled anomaly range (inclusive)."""
    mask = np.zeros(num_values, dtype=bool)
    for start, end in sequences:
        mask[start:end + 1] = True
    return mask


def main():
    args = parse_args()
    configure_mlflow()

    print(f"[{args.channel_name}] loading data and model")
    X_train, y_train, X_test, y_test = load_channel_data(args.data_path, args.channel_name)

    model_path = Path(args.model_path) / f"model_{args.channel_name}.keras"
    model = keras.models.load_model(model_path)

    # Threshold derived from TRAINING errors only (train is assumed anomaly-free)
    train_preds = model.predict(X_train, verbose=0).flatten()
    train_errors = np.abs(train_preds - y_train.flatten())
    threshold = train_errors.mean() + args.std_multiplier * train_errors.std()
    print(f"[{args.channel_name}] threshold={threshold:.6f} "
          f"(train_error_mean={train_errors.mean():.6f}, train_error_std={train_errors.std():.6f})")

    # Apply threshold to TEST errors to get predicted anomalies
    test_preds = model.predict(X_test, verbose=0).flatten()
    test_errors = np.abs(test_preds - y_test.flatten())
    predicted_anomaly = test_errors > threshold

    # Ground truth, aligned to the same windowed positions
    sequences, num_values = load_anomaly_sequences(args.labeled_anomalies_path, args.channel_name)
    gt_mask_full = build_ground_truth_mask(sequences, num_values)

    window_size = X_test.shape[1]
    gt_aligned = gt_mask_full[window_size:]  # first window_size positions have no prediction

    # Trim to the shorter of the two, in case of an off-by-a-few mismatch, and warn if so
    n = min(len(gt_aligned), len(predicted_anomaly))
    if len(gt_aligned) != len(predicted_anomaly):
        print(f"[{args.channel_name}] WARNING: length mismatch — "
              f"ground truth={len(gt_aligned)}, predictions={len(predicted_anomaly)}. Trimming to {n}.")
    gt_aligned = gt_aligned[:n]
    predicted_anomaly = predicted_anomaly[:n]

    precision = precision_score(gt_aligned, predicted_anomaly, zero_division=0)
    recall = recall_score(gt_aligned, predicted_anomaly, zero_division=0)
    f1 = f1_score(gt_aligned, predicted_anomaly, zero_division=0)

    print(f"[{args.channel_name}] precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}")
    print(f"[{args.channel_name}] true_anomaly_points={gt_aligned.sum()}, predicted_anomaly_points={predicted_anomaly.sum()}")

    with mlflow.start_run(run_name=f"{args.channel_name}-eval"):
        mlflow.log_params({
            "channel_name": args.channel_name,
            "std_multiplier": args.std_multiplier,
        })
        mlflow.log_metric("threshold", threshold)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("true_anomaly_points", int(gt_aligned.sum()))
        mlflow.log_metric("predicted_anomaly_points", int(predicted_anomaly.sum()))


if __name__ == "__main__":
    main()