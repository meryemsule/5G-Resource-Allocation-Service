from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler


CATEGORIES = [
    "Game_Streaming",
    "Live_Streaming",
    "Metaverse",
    "Online_Game",
    "Stored_Streaming",
    "Video_Conferencing",
]


def find_csv_files(dataset_root: str) -> list[str]:
    csv_files: list[str] = []
    for root, _dirs, files in os.walk(dataset_root):
        if any(category.lower() in root.lower() for category in CATEGORIES):
            for file_name in files:
                if file_name.lower().endswith(".csv"):
                    csv_files.append(os.path.join(root, file_name))
    return csv_files


def read_csv_with_fallback(file_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(file_path, encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return pd.read_csv(file_path, encoding="latin1")
        except UnicodeDecodeError:
            return pd.read_csv(file_path, encoding="cp1254")


def load_raw_traffic_series(dataset_root: str) -> np.ndarray:
    csv_files = find_csv_files(dataset_root)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {dataset_root}")

    all_series: list[np.ndarray] = []
    for file_path in csv_files:
        try:
            df = read_csv_with_fallback(file_path)
            df.columns = [str(column).lower() for column in df.columns]

            timestamp_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
            length_col = "length" if "length" in df.columns else ("size" if "size" in df.columns else df.columns[1])

            df[timestamp_col] = pd.to_datetime(df[timestamp_col], format="mixed", errors="coerce")
            df = df.dropna(subset=[timestamp_col])
            df[length_col] = pd.to_numeric(df[length_col], errors="coerce").fillna(0.0)
            df = df[[timestamp_col, length_col]].copy().set_index(timestamp_col)

            resampled = df[length_col].resample("1ms").sum().fillna(0.0)
            if len(resampled) > 10:
                all_series.append(resampled.values.astype(np.float32))
        except Exception:
            continue

    if not all_series:
        raise RuntimeError("Dataset was found, but no CSV file could be converted into 1 ms TTI series.")

    return np.concatenate(all_series).astype(np.float32)


def build_sequences(scaled_series: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for index in range(len(scaled_series) - window_size):
        X.append(scaled_series[index:index + window_size])
        y.append(scaled_series[index + window_size])

    X_array = np.array(X, dtype=np.float32).reshape(-1, window_size, 1)
    y_array = np.array(y, dtype=np.float32).reshape(-1, 1)
    return X_array, y_array


def contiguous_time_series_folds(total_length: int, n_splits: int, min_train_fraction: float = 0.5) -> list[tuple[slice, slice]]:
    if total_length < n_splits + 2:
        raise ValueError("Not enough samples for the requested number of folds.")

    train_start = int(total_length * min_train_fraction)
    remaining = total_length - train_start
    fold_size = max(1, remaining // n_splits)

    folds: list[tuple[slice, slice]] = []
    for fold_index in range(n_splits):
        train_end = min(train_start + fold_index * fold_size, total_length - 2)
        valid_start = train_end
        valid_end = min(valid_start + fold_size, total_length - 1)
        if valid_end <= valid_start or train_end <= 0:
            continue
        folds.append((slice(0, train_end), slice(valid_start, valid_end)))

    if not folds:
        raise ValueError("Could not construct valid cross-validation folds.")
    return folds


class TrafficLSTM(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 128, num_layers: int = 2) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _state = self.lstm(x)
        return self.head(out[:, -1, :])


class TrafficGRU(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 128, num_layers: int = 2) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _state = self.gru(x)
        return self.head(out[:, -1, :])


class TrafficTransformer(nn.Module):
    def __init__(self, input_size: int = 1, d_model: int = 64, nhead: int = 4, num_layers: int = 2) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            batch_first=True,
            dropout=0.0,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.input_projection(x)
        encoded = self.encoder(projected)
        return self.head(encoded[:, -1, :])


MODEL_BUILDERS: dict[str, callable] = {
    "LSTM": lambda: TrafficLSTM(),
    "GRU": lambda: TrafficGRU(),
    "Transformer": lambda: TrafficTransformer(),
}


@dataclass
class FoldMetrics:
    model_name: str
    fold_index: int
    mse: float
    mae: float
    rmse: float
    average_inference_latency_ms: float
    rb_allocation_success_rate: float
    tti_compliance_rate: float
    average_allocation_margin: float


def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    learning_rate: float,
    device: torch.device,
) -> nn.Module:
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for _epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(X_train_t)
        loss = criterion(predictions, y_train_t)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def evaluate_fold(
    model_name: str,
    model: nn.Module,
    scaler: StandardScaler,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    raw_valid_targets: np.ndarray,
    peak_reference: float,
    device: torch.device,
    fold_index: int,
) -> FoldMetrics:
    X_valid_t = torch.tensor(X_valid, dtype=torch.float32, device=device)

    with torch.inference_mode():
        _ = model(X_valid_t[:1])

    started_at = time.perf_counter()
    with torch.inference_mode():
        predictions = model(X_valid_t)
    elapsed_ms = ((time.perf_counter() - started_at) * 1000.0) / max(len(X_valid), 1)

    predicted_scaled = predictions.cpu().numpy()
    predicted_raw = scaler.inverse_transform(predicted_scaled).reshape(-1)
    actual_raw = raw_valid_targets.reshape(-1)

    mse = float(np.mean((predicted_raw - actual_raw) ** 2))
    mae = float(np.mean(np.abs(predicted_raw - actual_raw)))
    rmse = float(np.sqrt(np.mean((predicted_raw - actual_raw) ** 2)))

    predicted_ratio = np.clip(predicted_raw / peak_reference, 0.0, 1.0)
    actual_ratio = np.clip(actual_raw / peak_reference, 0.0, 1.0)
    success_flags = predicted_ratio >= actual_ratio
    margins = predicted_ratio - actual_ratio
    compliance_flags = np.array([elapsed_ms < 1.0] * len(predicted_ratio), dtype=bool)

    return FoldMetrics(
        model_name=model_name,
        fold_index=fold_index,
        mse=mse,
        mae=mae,
        rmse=rmse,
        average_inference_latency_ms=float(elapsed_ms),
        rb_allocation_success_rate=float(np.mean(success_flags)),
        tti_compliance_rate=float(np.mean(compliance_flags)),
        average_allocation_margin=float(np.mean(margins)),
    )


def aggregate_metrics(metrics: list[FoldMetrics]) -> list[dict[str, Any]]:
    grouped: dict[str, list[FoldMetrics]] = {}
    for metric in metrics:
        grouped.setdefault(metric.model_name, []).append(metric)

    rows: list[dict[str, Any]] = []
    for model_name, model_metrics in grouped.items():
        rows.append(
            {
                "model_name": model_name,
                "fold_count": len(model_metrics),
                "mse_mean": float(np.mean([item.mse for item in model_metrics])),
                "mae_mean": float(np.mean([item.mae for item in model_metrics])),
                "rmse_mean": float(np.mean([item.rmse for item in model_metrics])),
                "rmse_std": float(np.std([item.rmse for item in model_metrics])),
                "average_inference_latency_ms": float(np.mean([item.average_inference_latency_ms for item in model_metrics])),
                "rb_allocation_success_rate": float(np.mean([item.rb_allocation_success_rate for item in model_metrics])),
                "tti_compliance_rate": float(np.mean([item.tti_compliance_rate for item in model_metrics])),
                "average_allocation_margin": float(np.mean([item.average_allocation_margin for item in model_metrics])),
            }
        )
    return rows


def pick_best_model(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}

    def score(row: dict[str, Any]) -> float:
        return (
            row["rb_allocation_success_rate"] * 0.35
            + row["tti_compliance_rate"] * 0.25
            - row["mae_mean"] * 0.15
            - row["rmse_mean"] * 0.15
            - row["average_inference_latency_ms"] * 0.10
        )

    ranked = sorted(rows, key=score, reverse=True)
    best = dict(ranked[0])
    best["selection_score"] = score(best)
    return best


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_rows = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_row, separator_row] + body_rows)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_models(
    dataset_root: str,
    output_dir: str,
    window_size: int,
    n_splits: int,
    epochs: int,
    learning_rate: float,
) -> None:
    raw_series = load_raw_traffic_series(dataset_root)
    folds = contiguous_time_series_folds(len(raw_series), n_splits=n_splits)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_fold_metrics: list[FoldMetrics] = []

    for model_name, builder in MODEL_BUILDERS.items():
        print(f"Evaluating {model_name}...")
        for fold_index, (train_slice, valid_slice) in enumerate(folds, start=1):
            train_series = raw_series[train_slice]
            valid_series = raw_series[valid_slice]

            scaler = StandardScaler()
            scaled_train = scaler.fit_transform(train_series.reshape(-1, 1)).reshape(-1)
            scaled_valid = scaler.transform(valid_series.reshape(-1, 1)).reshape(-1)

            X_train, y_train = build_sequences(scaled_train, window_size)
            X_valid, y_valid = build_sequences(scaled_valid, window_size)
            raw_valid_targets = valid_series[window_size:].reshape(-1, 1)

            if len(X_train) == 0 or len(X_valid) == 0:
                continue

            model = builder().to(device)
            trained_model = train_model(
                model=model,
                X_train=X_train,
                y_train=y_train,
                epochs=epochs,
                learning_rate=learning_rate,
                device=device,
            )

            peak_reference = float(max(np.max(train_series), 1.0))
            fold_metrics = evaluate_fold(
                model_name=model_name,
                model=trained_model,
                scaler=scaler,
                X_valid=X_valid,
                y_valid=y_valid,
                raw_valid_targets=raw_valid_targets,
                peak_reference=peak_reference,
                device=device,
                fold_index=fold_index,
            )
            print(
                f"  Fold {fold_index}: RMSE={fold_metrics.rmse:.4f}, "
                f"Latency={fold_metrics.average_inference_latency_ms:.4f} ms, "
                f"RB Success={fold_metrics.rb_allocation_success_rate:.2%}, "
                f"TTI={fold_metrics.tti_compliance_rate:.2%}"
            )
            all_fold_metrics.append(fold_metrics)

    aggregated_rows = aggregate_metrics(all_fold_metrics)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    folds_json_path = output_path / "model_reliability_folds.json"
    summary_json_path = output_path / "model_reliability_summary.json"
    summary_csv_path = output_path / "model_reliability_summary.csv"
    report_md_path = output_path / "model_reliability_report.md"
    best_model_path = output_path / "best_model_selection.json"

    folds_json_path.write_text(
        json.dumps([metric.__dict__ for metric in all_fold_metrics], indent=2),
        encoding="utf-8",
    )
    summary_json_path.write_text(json.dumps(aggregated_rows, indent=2), encoding="utf-8")
    write_csv(summary_csv_path, aggregated_rows)
    best_model = pick_best_model(aggregated_rows)
    best_model_path.write_text(json.dumps(best_model, indent=2), encoding="utf-8")

    table_rows = [
        [
            row["model_name"],
            str(row["fold_count"]),
            f"{row['mse_mean']:.4f}",
            f"{row['mae_mean']:.4f}",
            f"{row['rmse_mean']:.4f}",
            f"{row['rmse_std']:.4f}",
            f"{row['average_inference_latency_ms']:.4f}",
            f"{row['rb_allocation_success_rate'] * 100:.2f}%",
            f"{row['tti_compliance_rate'] * 100:.2f}%",
            f"{row['average_allocation_margin']:.4f}",
        ]
        for row in aggregated_rows
    ]

    report_md = "\n".join(
        [
            "# Model Reliability Report",
            "",
            f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Cross-Validation Comparison",
            "",
            markdown_table(
                [
                    "Model",
                    "Folds",
                    "MSE Mean",
                    "MAE Mean",
                    "RMSE Mean",
                    "RMSE Std",
                    "Avg Inference Latency (ms)",
                    "RB Allocation Success",
                    "TTI Compliance",
                    "Avg Allocation Margin",
                ],
                table_rows,
            ),
            "",
            "## Interpretation",
            "",
            "- `RMSE Mean` lower is better for load prediction accuracy.",
            "- `Avg Inference Latency (ms)` should ideally stay below 1 ms for TTI compliance.",
            "- `RB Allocation Success` measures how often predicted allocation covered actual next-step demand.",
            "- `Avg Allocation Margin` shows whether the model tends to over-allocate or under-allocate.",
            "",
            "## Auto-Selected Best Model",
            "",
            json.dumps(best_model, indent=2),
            "",
        ]
    )
    report_md_path.write_text(report_md, encoding="utf-8")

    print("Reliability evaluation files generated:")
    print(f"- {folds_json_path}")
    print(f"- {summary_json_path}")
    print(f"- {summary_csv_path}")
    print(f"- {report_md_path}")
    print(f"- {best_model_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LSTM, GRU and Transformer reliability with cross-validation.")
    parser.add_argument(
        "--dataset-root",
        default="/kaggle/input/datasets/kimdaegyeom/5g-traffic-datasets",
        help="Root folder of the Kaggle dataset",
    )
    parser.add_argument("--output-dir", default="reliability_output", help="Output folder for evaluation artifacts")
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()

    evaluate_models(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        window_size=args.window_size,
        n_splits=args.n_splits,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()
