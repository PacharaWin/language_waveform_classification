import os
import random
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import dct, rfft
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    import xgboost as xgb
except Exception as exc:  # pragma: no cover - handled at runtime
    xgb = None
    XGBOOST_IMPORT_ERROR = exc
else:
    XGBOOST_IMPORT_ERROR = None


os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

SEED = 42
SR = 16000
FRAME_LENGTH = 400


@dataclass(frozen=True)
class MFCCConfig:
    n_mfcc: int
    n_fft: int
    hop_length: int
    n_mels: int
    pre_emphasis: float


SVM_FEATURE_GRID = [
    MFCCConfig(n_mfcc=13, n_fft=256, hop_length=80, n_mels=26, pre_emphasis=0.97),
    MFCCConfig(n_mfcc=20, n_fft=256, hop_length=160, n_mels=26, pre_emphasis=0.97),
    MFCCConfig(n_mfcc=20, n_fft=512, hop_length=160, n_mels=40, pre_emphasis=0.97),
    MFCCConfig(n_mfcc=26, n_fft=512, hop_length=80, n_mels=40, pre_emphasis=0.97),
]

XGB_FEATURE_GRID = [
    MFCCConfig(n_mfcc=13, n_fft=256, hop_length=80, n_mels=26, pre_emphasis=0.95),
    MFCCConfig(n_mfcc=20, n_fft=512, hop_length=160, n_mels=40, pre_emphasis=0.97),
    MFCCConfig(n_mfcc=26, n_fft=512, hop_length=80, n_mels=40, pre_emphasis=0.97),
]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_data(train_path: str = "train_set.pkl", test_path: str = "test_set.pkl"):
    train_df = pd.read_pickle(train_path).copy()
    test_df = pd.read_pickle(test_path).copy()
    label_names = sorted(train_df["language"].unique())
    label2id = {label: idx for idx, label in enumerate(label_names)}
    id2label = {idx: label for label, idx in label2id.items()}

    train_df["target"] = train_df["language"].map(label2id)
    test_df["target"] = test_df["language"].map(label2id)
    return train_df, test_df, label_names, label2id, id2label


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
        y = np.pad(y, (0, frame_length - y.size), mode="constant")

    num_frames = 1 + int(np.ceil((len(y) - frame_length) / hop_length))
    pad_amount = max(0, (num_frames - 1) * hop_length + frame_length - len(y))
    if pad_amount:
        y = np.pad(y, (0, pad_amount), mode="constant")

    idx = np.arange(frame_length)[None, :] + hop_length * np.arange(num_frames)[:, None]
    return y[idx]


def _mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10 ** (m / 2595.0) - 1.0)

    mel_points = np.linspace(hz_to_mel(0.0), hz_to_mel(sr / 2), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fbanks = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
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


def mfcc_matrix(signal, config: MFCCConfig) -> np.ndarray:
    y = np.asarray(signal, dtype=np.float32)
    if y.size == 0 or not np.any(np.isfinite(y)):
        return np.zeros((config.n_mfcc, 1), dtype=np.float32)

    y = np.nan_to_num(y)
    y = _normalize_audio(y)
    y = _pre_emphasis(y, coeff=config.pre_emphasis)

    frames = _frame_signal(y, FRAME_LENGTH, config.hop_length)
    frames = frames * np.hamming(FRAME_LENGTH).astype(np.float32)[None, :]

    spectrum = np.abs(rfft(frames, n=config.n_fft, axis=1)) ** 2
    spectrum /= float(config.n_fft)

    mel_fbanks = _mel_filterbank(SR, config.n_fft, config.n_mels)
    mel_spectrogram = np.dot(spectrum, mel_fbanks.T)
    mel_spectrogram = np.where(
        mel_spectrogram <= 0, np.finfo(np.float32).eps, mel_spectrogram
    )
    log_mel = np.log(mel_spectrogram)

    coeffs = dct(log_mel, type=2, axis=1, norm="ortho")[:, : config.n_mfcc]
    return coeffs.T.astype(np.float32)


def extract_mfcc_features(signal, config: MFCCConfig) -> np.ndarray:
    mfcc = mfcc_matrix(signal, config)
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


def build_feature_matrix(df: pd.DataFrame, config: MFCCConfig) -> np.ndarray:
    features = [extract_mfcc_features(sig, config) for sig in df["signal_data"]]
    return np.vstack(features)


def evaluate_predictions(y_true, y_pred, label_names, title: str):
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n{title}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced accuracy: {balanced_acc:.4f}")
    print(f"Macro F1: {f1_macro:.4f}")
    print(f"Weighted F1: {f1_weighted:.4f}")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=label_names,
            digits=4,
            zero_division=0,
        )
    )
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }


def plot_confusion(y_true, y_pred, label_names, title: str, path: str | None = None):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(label_names)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    plt.title(title)
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def pick_best_config(train_part, val_part, grid, model_name: str, label_names):
    best = None
    best_val = -np.inf
    best_train_x = None
    best_val_x = None

    for config in grid:
        print(f"Testing {model_name} MFCC config: {config}")
        x_train = build_feature_matrix(train_part, config)
        x_val = build_feature_matrix(val_part, config)
        y_train = train_part["target"].to_numpy()
        y_val = val_part["target"].to_numpy()

        if model_name == "SVM":
            model = make_pipeline(
                StandardScaler(),
                SVC(kernel="linear", class_weight="balanced", C=1.0),
            )
        else:
            if xgb is None:
                raise RuntimeError(
                    "xgboost is not available. Install libomp and xgboost first."
                ) from XGBOOST_IMPORT_ERROR
            class_counts = train_part["target"].value_counts().sort_index()
            sample_weight = train_part["target"].map(
                lambda t: 1.0 / class_counts.loc[t]
            ).to_numpy(dtype=np.float32)
            model = xgb.XGBClassifier(
                objective="multi:softprob",
                num_class=len(label_names),
                n_estimators=250,
                learning_rate=0.06,
                max_depth=5,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                min_child_weight=1.0,
                tree_method="hist",
                random_state=SEED,
                eval_metric="mlogloss",
            )

        if model_name == "SVM":
            model.fit(x_train, y_train)
        else:
            model.fit(x_train, y_train, sample_weight=sample_weight)

        pred = model.predict(x_val)
        val_f1 = f1_score(y_val, pred, average="macro", zero_division=0)
        print(f"Validation macro F1: {val_f1:.4f}")

        if val_f1 > best_val:
            best_val = val_f1
            best = config
            best_train_x = x_train
            best_val_x = x_val

    print(f"Best {model_name} config: {best} with val macro F1={best_val:.4f}")
    return best, best_train_x, best_val_x


def train_final_model(model_name: str, train_df: pd.DataFrame, config: MFCCConfig, label_names):
    x_train = build_feature_matrix(train_df, config)
    y_train = train_df["target"].to_numpy()

    if model_name == "SVM":
        model = make_pipeline(
            StandardScaler(),
            SVC(kernel="linear", class_weight="balanced", C=1.0),
        )
        model.fit(x_train, y_train)
        return model

    if xgb is None:
        raise RuntimeError("xgboost is not available.") from XGBOOST_IMPORT_ERROR

    class_counts = train_df["target"].value_counts().sort_index()
    sample_weight = train_df["target"].map(lambda t: 1.0 / class_counts.loc[t]).to_numpy(
        dtype=np.float32
    )
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=len(label_names),
        n_estimators=350,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        min_child_weight=1.0,
        tree_method="hist",
        random_state=SEED,
        eval_metric="mlogloss",
    )
    model.fit(x_train, y_train, sample_weight=sample_weight)
    return model


def main2():
    set_seed(SEED)
    train_df, test_df, label_names, _, _ = load_data()

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
    print("Languages:", label_names)
    print("Train language counts:")
    print(train_df["language"].value_counts().to_string())

    train_part, val_part = train_test_split(
        train_df,
        test_size=0.2,
        random_state=SEED,
        stratify=train_df["target"],
    )
    print("Train/validation split:", train_part.shape, val_part.shape)

    best_svm_cfg, _, _ = pick_best_config(
        train_part, val_part, SVM_FEATURE_GRID, "SVM", label_names
    )
    best_xgb_cfg, _, _ = pick_best_config(
        train_part, val_part, XGB_FEATURE_GRID, "XGBoost", label_names
    )

    svm_model = train_final_model("SVM", train_df, best_svm_cfg, label_names)
    xgb_model = train_final_model("XGBoost", train_df, best_xgb_cfg, label_names)

    svm_train_x = build_feature_matrix(train_df, best_svm_cfg)
    xgb_train_x = build_feature_matrix(train_df, best_xgb_cfg)
    svm_test_x = build_feature_matrix(test_df, best_svm_cfg)
    xgb_test_x = build_feature_matrix(test_df, best_xgb_cfg)
    y_train = train_df["target"].to_numpy()
    y_test = test_df["target"].to_numpy()

    svm_train_pred = svm_model.predict(svm_train_x)
    svm_pred = svm_model.predict(svm_test_x)
    xgb_train_pred = xgb_model.predict(xgb_train_x)
    xgb_pred = xgb_model.predict(xgb_test_x)

    svm_train_scores = evaluate_predictions(
        y_train, svm_train_pred, label_names, "SVM training results"
    )
    svm_scores = evaluate_predictions(
        y_test, svm_pred, label_names, "SVM test results"
    )
    xgb_train_scores = evaluate_predictions(
        y_train, xgb_train_pred, label_names, "XGBoost training results"
    )
    xgb_scores = evaluate_predictions(
        y_test, xgb_pred, label_names, "XGBoost test results"
    )

    plot_confusion(
        y_test,
        svm_pred,
        label_names,
        "SVM confusion matrix",
        path="mfcc_svm_confusion_matrix.png",
    )
    plot_confusion(
        y_test,
        xgb_pred,
        label_names,
        "XGBoost confusion matrix",
        path="mfcc_xgboost_confusion_matrix.png",
    )

    svm_preview = test_df[["language", "speaker"]].copy()
    svm_preview["predicted_language"] = [label_names[i] for i in svm_pred]
    xgb_preview = test_df[["language", "speaker"]].copy()
    xgb_preview["predicted_language"] = [label_names[i] for i in xgb_pred]

    print("\nSVM sample predictions:")
    print(svm_preview.head(10).to_string(index=False))
    print("\nXGBoost sample predictions:")
    print(xgb_preview.head(10).to_string(index=False))

    return {
        "svm": {
            "config": best_svm_cfg,
            "train_scores": svm_train_scores,
            "scores": svm_scores,
            "predictions": svm_pred,
        },
        "xgboost": {
            "config": best_xgb_cfg,
            "train_scores": xgb_train_scores,
            "scores": xgb_scores,
            "predictions": xgb_pred,
        },
    }


if __name__ == "__main__":
    main2()
