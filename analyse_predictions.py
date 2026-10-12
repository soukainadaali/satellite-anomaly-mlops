import ast
import csv
from pathlib import Path

import numpy as np


CHANNEL_NAME = "P-1"
THRESHOLD = 0.4416313119485826  # from the model's "threshold" tag

PREDICTIONS_CSV_PATH = "predictions.csv"  # path to the predictions.csv downloaded from the batch job
Y_TEST_PATH = Path("src/data/processed") / CHANNEL_NAME / "y_test.npy"
LABELED_ANOMALIES_PATH = Path("src/data/raw/labeled_anomalies.csv")


def load_predictions(csv_path):
    # Parses predictions.csv
    predictions = []
    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            if len(parts) != 3:
                continue
            source_file, window_index, predicted_value = parts
            predictions.append((int(window_index), float(predicted_value)))
    predictions.sort(key=lambda x: x[0])
    return predictions


def load_anomaly_sequences(labeled_anomalies_path, channel_name):
    with open(labeled_anomalies_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["chan_id"] == channel_name:
                return ast.literal_eval(row["anomaly_sequences"]), int(row["num_values"])
    raise ValueError(f"Channel '{channel_name}' not found")


def main():
    predictions = load_predictions(PREDICTIONS_CSV_PATH)
    y_test = np.load(Y_TEST_PATH).flatten()

    sequences, num_values = load_anomaly_sequences(LABELED_ANOMALIES_PATH, CHANNEL_NAME)
    gt_mask_full = np.zeros(num_values, dtype=bool)
    for start, end in sequences:
        gt_mask_full[start:end + 1] = True

    window_size = 100  # matches your project's window size
    gt_aligned = gt_mask_full[window_size:]

    print(f"{'window':>6} {'predicted':>12} {'actual':>10} {'error':>8} {'flagged':>8} {'truth':>7}")
    flagged_count = 0
    true_positive = 0
    false_positive = 0

    for window_index, predicted in predictions:
        if window_index >= len(y_test):
            continue
        actual = float(y_test[window_index])
        error = abs(predicted - actual)
        is_anomaly = error > THRESHOLD
        is_true_anomaly = bool(gt_aligned[window_index]) if window_index < len(gt_aligned) else None

        if is_anomaly:
            flagged_count += 1
            if is_true_anomaly:
                true_positive += 1
            else:
                false_positive += 1

        flag_str = "ANOMALY" if is_anomaly else "-"
        truth_str = "real" if is_true_anomaly else ("normal" if is_true_anomaly is not None else "?")
        print(f"{window_index:>6} {predicted:>12.4f} {actual:>10.4f} {error:>8.4f} {flag_str:>8} {truth_str:>7}")

    print(f"\nFlagged {flagged_count} of {len(predictions)} windows as anomalies "
          f"(threshold={THRESHOLD:.4f})")
    print(f"  True positives (correctly flagged real anomalies): {true_positive}")
    print(f"  False positives (flagged but not a real anomaly): {false_positive}")


if __name__ == "__main__":
    main()