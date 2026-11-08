from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, r2_score

from app.model_runtime import TemporalAttention
from distill_fast_predictor import build_windows


def build_conv1d(window_size: int, feature_count: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(window_size, feature_count), name="traffic_window")
    x = tf.keras.layers.Conv1D(32, kernel_size=3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Conv1D(32, kernel_size=3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(inputs, outputs, name="conv1d")


def build_lstm(window_size: int, feature_count: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(window_size, feature_count), name="traffic_window")
    x = tf.keras.layers.LSTM(48, return_sequences=False)(inputs)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(inputs, outputs, name="lstm")


def build_conv1d_gru(window_size: int, feature_count: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(window_size, feature_count), name="traffic_window")
    x = tf.keras.layers.Conv1D(32, kernel_size=3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.GRU(48, return_sequences=False)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(inputs, outputs, name="conv1d_gru")


def build_conv1d_gru_attention(window_size: int, feature_count: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(window_size, feature_count), name="traffic_window")
    x = tf.keras.layers.Conv1D(32, kernel_size=3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.GRU(48, return_sequences=True)(x)
    context = TemporalAttention(name="temporal_attention")(x)
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = tf.keras.layers.Concatenate()([context, avg_pool, max_pool])
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(inputs, outputs, name="conv1d_gru_attention")


def measure_latency(model: tf.keras.Model, x_test: np.ndarray, repeats: int) -> float:
    sample = x_test[:1]
    for _ in range(5):
        model(sample, training=False)
    started = time.perf_counter()
    for _ in range(repeats):
        model(sample, training=False)
    return ((time.perf_counter() - started) * 1000.0) / repeats


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Conv1D, Conv1D-GRU, and Conv1D-GRU-Attention traffic predictors.")
    parser.add_argument("--samples", type=int, default=12000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-load", type=float, default=29127.712890625)
    parser.add_argument("--output-dir", default="report_output")
    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    window_size = 10
    feature_count = 7

    teacher = tf.keras.models.load_model(
        "models/sayzek_traffic_predictor.keras",
        custom_objects={"TemporalAttention": TemporalAttention},
        compile=False,
    )
    x_scaler = joblib.load("models/x_scaler.pkl")
    y_scaler = joblib.load("models/y_scaler.pkl")

    windows = build_windows(args.samples, window_size, args.seed, args.max_load)
    scaled = x_scaler.transform(windows.reshape(-1, feature_count)).reshape(args.samples, window_size, feature_count)
    teacher_scaled = teacher.predict(scaled, batch_size=512, verbose=0).astype(np.float32)
    y = y_scaler.inverse_transform(teacher_scaled).reshape(-1).astype(np.float32)

    split = int(args.samples * 0.8)
    x_train, x_test = scaled[:split], scaled[split:]
    y_train_scaled = teacher_scaled[:split]
    y_train, y_test = y[:split], y[split:]

    builders = [
        ("Conv1D", build_conv1d),
        ("LSTM", build_lstm),
        ("Conv1D-GRU", build_conv1d_gru),
        ("Conv1D-GRU-Attention", build_conv1d_gru_attention),
    ]
    rows: list[dict[str, Any]] = []
    for name, builder in builders:
        model = builder(window_size, feature_count)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
        started = time.perf_counter()
        history = model.fit(
            x_train,
            y_train_scaled,
            validation_split=0.15,
            epochs=args.epochs,
            batch_size=256,
            verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)],
        )
        train_seconds = time.perf_counter() - started
        predictions_scaled = model.predict(x_test, batch_size=512, verbose=0)
        predictions = y_scaler.inverse_transform(predictions_scaled).reshape(-1)
        rows.append(
            {
                "architecture": name,
                "samples": args.samples,
                "epochs_ran": len(history.history["loss"]),
                "mae": float(mean_absolute_error(y_test, predictions)),
                "r2": float(r2_score(y_test, predictions)),
                "avg_latency_ms": measure_latency(model, x_test, repeats=120),
                "train_seconds": train_seconds,
                "parameters": int(model.count_params()),
            }
        )

    output_dir = Path(args.output_dir)
    write_csv(output_dir / "architecture_ablation.csv", rows)
    (output_dir / "architecture_ablation.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
