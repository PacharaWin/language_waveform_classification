import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import dct, rfft
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

SEED = 42
SR = 16000
N_MFCC = 20
N_FFT = 512
HOP_LENGTH = 160
FRAME_LENGTH = 400
N_MELS = 26


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_data(train_path: str = "train_set.pkl", test_path: str = "test_set.pkl"):
    train_df = pd.read_pickle(train_path).copy()
    test_df = pd.read_pickle(test_path).copy()
    label_names = sorted(train_df["language"].unique())
    return train_df, test_df, label_names


def _normalize_audio(y: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    return y.astype(np.float32, copy=False)


def _pre_emphasis(y: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    if y.size == 0:
        return y
    return np.append(y[0], y[1:] - coeff * y[:-1]).astype(np.float32, copy=False)


def _frame_signal(y: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if y.size <= frame_length:
        pad_width = frame_length - y.size
        y = np.pad(y, (0, pad_width), mode="constant")

    num_frames = 1 + int(np.ceil((len(y) - frame_length) / hop_length))
    pad_amount = max(0, (num_frames - 1) * hop_length + frame_length - len(y))
    if pad_amount:
        y = np.pad(y, (0, pad_amount), mode="constant")

    indices = (
        np.arange(frame_length)[None, :]
        + hop_length * np.arange(num_frames)[:, None]
    )
    return y[indices]


def _mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float = 0.0, fmax: float | None = None) -> np.ndarray:
    if fmax is None:
        fmax = sr / 2

    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10 ** (m / 2595.0) - 1.0)

    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fbanks = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = bin_points[m - 1], bin_points[m], bin_points[m + 1]
        left = max(left, 0)
        center = max(center, left + 1)
        right = max(right, center + 1)
        right = min(right, n_fft // 2 + 1)

        for k in range(left, min(center, fbanks.shape[1])):
            fbanks[m - 1, k] = (k - left) / max(center - left, 1)
        for k in range(center, right):
            fbanks[m - 1, k] = (right - k) / max(right - center, 1)

    return fbanks


def _delta(features: np.ndarray, width: int = 2) -> np.ndarray:
    if features.shape[1] < 2:
        return np.zeros_like(features)

    denom = 2 * sum(i * i for i in range(1, width + 1))
    padded = np.pad(features, ((0, 0), (width, width)), mode="edge")
    delta_feat = np.zeros_like(features)
    for t in range(features.shape[1]):
        numerator = sum(
            i * (padded[:, t + width + i] - padded[:, t + width - i])
            for i in range(1, width + 1)
        )
        delta_feat[:, t] = numerator / denom
    return delta_feat


def mfcc_matrix(signal, sr: int = SR, n_mfcc: int = N_MFCC) -> np.ndarray:
    y = np.asarray(signal, dtype=np.float32)
    if y.size == 0 or not np.any(np.isfinite(y)):
        return np.zeros((n_mfcc, 1), dtype=np.float32)

    y = np.nan_to_num(y)
    y = _normalize_audio(y)
    y = _pre_emphasis(y)

    frames = _frame_signal(y, FRAME_LENGTH, HOP_LENGTH)
    window = np.hamming(FRAME_LENGTH).astype(np.float32)
    frames = frames * window[None, :]

    spectrum = np.abs(rfft(frames, n=N_FFT, axis=1)) ** 2
    spectrum /= float(N_FFT)

    mel_fbanks = _mel_filterbank(sr, N_FFT, N_MELS)
    mel_spectrogram = np.dot(spectrum, mel_fbanks.T)
    mel_spectrogram = np.where(mel_spectrogram <= 0, np.finfo(np.float32).eps, mel_spectrogram)
    log_mel = np.log(mel_spectrogram)

    coeffs = dct(log_mel, type=2, axis=1, norm="ortho")[:, :n_mfcc]
    return coeffs.T.astype(np.float32)


def extract_mfcc_features(signal, sr: int = SR, n_mfcc: int = N_MFCC) -> np.ndarray:
    """Convert one waveform into a fixed-length MFCC feature vector."""
    mfcc = mfcc_matrix(signal, sr=sr, n_mfcc=n_mfcc)
    if mfcc.size == 0:
        return np.zeros(n_mfcc * 6, dtype=np.float32)

    delta = _delta(mfcc)
    delta2 = _delta(delta)

    stats = [
        mfcc.mean(axis=1),
        mfcc.std(axis=1),
        delta.mean(axis=1),
        delta.std(axis=1),
        delta2.mean(axis=1),
        delta2.std(axis=1),
    ]
    return np.concatenate(stats).astype(np.float32)


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    features = [extract_mfcc_features(sig) for sig in df["signal_data"]]
    return np.vstack(features)


def plot_sample_mfcc(train_df: pd.DataFrame, show: bool = True) -> None:
    sample = train_df.iloc[0]
    mfcc = mfcc_matrix(sample["signal_data"], sr=SR, n_mfcc=N_MFCC)

    plt.figure(figsize=(10, 4))
    duration = len(sample["signal_data"]) / SR
    extent = [0, duration, 0, N_MFCC]
    plt.imshow(mfcc, origin="lower", aspect="auto", cmap="magma", extent=extent)
    plt.colorbar(label="MFCC value")
    plt.xlabel("Time (s)")
    plt.ylabel("MFCC coefficient")
    plt.title(f"Sample MFCCs for {sample['language']} speaker {sample['speaker']}")
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig("sample_mfcc.png", dpi=150, bbox_inches="tight")
        plt.close()


def train_and_evaluate(train_df: pd.DataFrame, test_df: pd.DataFrame, show_plots: bool = True):
    label_names = sorted(train_df["language"].unique())
    label2id = {label: idx for idx, label in enumerate(label_names)}
    id2label = {idx: label for label, idx in label2id.items()}

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["target"] = train_df["language"].map(label2id)
    test_df["target"] = test_df["language"].map(label2id)

    train_X = build_feature_matrix(train_df)
    test_X = build_feature_matrix(test_df)
    y_train = train_df["target"].to_numpy()
    y_test = test_df["target"].to_numpy()

    model = make_pipeline(
        StandardScaler(),
        SVC(kernel="linear", class_weight="balanced"),
    )
    model.fit(train_X, y_train)

    train_pred = model.predict(train_X)
    test_pred = model.predict(test_X)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    print("Languages:", label_names)
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Feature shape: {train_X.shape[1]} MFCC-based features per sample")
    print(f"Training accuracy: {train_acc:.3f}")
    print(f"Test accuracy: {test_acc:.3f}")
    print("\nClassification report on test set:")
    print(
        classification_report(
            y_test,
            test_pred,
            target_names=[id2label[i] for i in range(len(label_names))],
            zero_division=0,
        )
    )

    cm = confusion_matrix(y_test, test_pred, labels=range(len(label_names)))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[id2label[i] for i in range(len(label_names))],
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    plt.title("MFCC Language Classification - Confusion Matrix")
    plt.tight_layout()
    if show_plots:
        plt.show()
    else:
        plt.savefig("mfcc_confusion_matrix.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    preview = test_df[["language", "speaker"]].copy()
    preview["predicted_language"] = [id2label[i] for i in test_pred]
    print("\nSample predictions:")
    print(preview.head(10).to_string(index=False))

    return {
        "model": model,
        "label_names": label_names,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "predictions": test_pred,
        "preview": preview,
    }


def main(show_plots: bool = True):
    set_seed(SEED)
    train_df, test_df, _ = load_data()
    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
    print(train_df[["language", "speaker", "length"]].head().to_string(index=False))
    plot_sample_mfcc(train_df, show=show_plots)
    return train_and_evaluate(train_df, test_df, show_plots=show_plots)


if __name__ == "__main__":
    main(show_plots=False)
