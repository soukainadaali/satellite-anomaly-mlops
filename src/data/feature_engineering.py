#Transforms raw NASA SMAP/MSL .npy files into LSTM-ready windowed sequences.

import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

CHANNELS = ["P-1", "E-13", "F-7"]
WINDOW_SIZE = 100

RAW_TRAIN_DIR = Path("src/data/raw/train")
RAW_TEST_DIR = Path("src/data/raw/test")
PROCESSED_DIR = Path("src/data/processed")


def create_windows(data: np.ndarray, window_size: int):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size])
        y.append(data[i + window_size, 0])  
    return np.array(X), np.array(y)


def process_channel(channel: str):
    train_raw = np.load(RAW_TRAIN_DIR / f"{channel}.npy")
    test_raw = np.load(RAW_TEST_DIR / f"{channel}.npy")

    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_scaled = scaler.fit_transform(train_raw)
    test_scaled = scaler.transform(test_raw)

    X_train, y_train = create_windows(train_scaled, WINDOW_SIZE)
    X_test, y_test = create_windows(test_scaled, WINDOW_SIZE)

    out_dir = PROCESSED_DIR / channel
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X_train.npy", X_train)
    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "X_test.npy", X_test)
    np.save(out_dir / "y_test.npy", y_test)

    print(f"{channel}: X_train {X_train.shape}, X_test {X_test.shape}")


if __name__ == "__main__":
    for ch in CHANNELS:
        process_channel(ch)