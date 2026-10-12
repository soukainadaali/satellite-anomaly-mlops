import argparse
import os
from pathlib import Path
import numpy as np
import mlflow
from tensorflow import keras
from tensorflow.keras import layers

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential


SUBSCRIPTION_ID = ""
RESOURCE_GROUP = "rg-satellite-anomaly-mlops"
WORKSPACE_NAME = "satellite-anomaly-ml"
TENANT_ID = ""


def configure_mlflow():
    
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
    parser = argparse.ArgumentParser(description="Train an LSTM anomaly-detection model for a single telemetry channel.")

    parser.add_argument("--channel_name", type=str, required=True,
                         help="Channel identifier, e.g. P-1, E-13, F-7")
    parser.add_argument("--data_path", type=str, required=True,
                         help="Path to the mounted processed data folder (contains per-channel .npy files)")
    parser.add_argument("--model_output", type=str, required=True,
                         help="Folder to write the trained model artifact to")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lstm_units", type=int, default=64)

    return parser.parse_args()


def load_channel_data(data_path: str, channel_name: str):
    channel_dir = Path(data_path) / channel_name

    X_train = np.load(channel_dir / "X_train.npy")
    y_train = np.load(channel_dir / "y_train.npy")
    X_test = np.load(channel_dir / "X_test.npy")
    y_test = np.load(channel_dir / "y_test.npy")

    return X_train, y_train, X_test, y_test


def build_model(window_size: int, n_features: int, lstm_units: int):
    model = keras.Sequential([
        layers.Input(shape=(window_size, n_features)),
        layers.LSTM(lstm_units),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


class MLflowMetricsCallback(keras.callbacks.Callback):
    """Logs loss/val_loss to MLflow at the end of each epoch, without relying on autolog."""

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for metric_name, value in logs.items():
            mlflow.log_metric(metric_name, value, step=epoch)


def main():
    args = parse_args()
    configure_mlflow()

    print(f"[{args.channel_name}] loading data from {args.data_path}")
    X_train, y_train, X_test, y_test = load_channel_data(args.data_path, args.channel_name)

    window_size = X_train.shape[1]
    n_features = X_train.shape[2]
    print(f"[{args.channel_name}] window_size={window_size}, n_features={n_features}, "
          f"train_samples={X_train.shape[0]}, test_samples={X_test.shape[0]}")

    model = build_model(window_size, n_features, args.lstm_units)
    print(f"[{args.channel_name}] training: epochs={args.epochs}, batch_size={args.batch_size}, lstm_units={args.lstm_units}")

    with mlflow.start_run(run_name=args.channel_name):
        mlflow.log_params({
            "channel_name": args.channel_name,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lstm_units": args.lstm_units,
            "optimizer": "adam",
            "loss_function": "mse",
        })

        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=2,
            callbacks=[MLflowMetricsCallback()],
        )

        final_train_loss = history.history["loss"][-1]
        final_val_loss = history.history["val_loss"][-1]
        print(f"[{args.channel_name}] final train_loss={final_train_loss:.6f}, val_loss={final_val_loss:.6f}")

        mlflow.log_metric("final_train_loss", final_train_loss)
        mlflow.log_metric("final_val_loss", final_val_loss)

        output_dir = Path(args.model_output)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / f"model_{args.channel_name}.keras"
        model.save(model_path)
        print(f"[{args.channel_name}] model saved to {model_path}")


if __name__ == "__main__":
    main()