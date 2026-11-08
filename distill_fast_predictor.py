from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app.model_runtime import TemporalAttention


def build_windows(sample_count: int, window_size: int, seed: int, max_load: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    windows = np.zeros((sample_count, window_size, 7), dtype=np.float32)
    service_scales = rng.choice([0.08, 0.18, 0.35, 0.55], size=sample_count, p=[0.25, 0.35, 0.25, 0.15])

    for index in range(sample_count):
        base = rng.uniform(0.01, service_scales[index]) * max_load
        trend = rng.uniform(-0.025, 0.04) * max_load
        noise = rng.normal(0.0, max_load * rng.uniform(0.003, 0.025), size=window_size)
        wave = np.sin(np.linspace(0, rng.uniform(np.pi, 3 * np.pi), window_size)) * rng.uniform(0, 0.05) * max_load
        tti_load = np.clip(base + np.linspace(0, trend, window_size) + noise + wave, 0.0, max_load)
        avg_packet_size = rng.uniform(350.0, 1500.0)
        packet_count = np.maximum(tti_load / avg_packet_size, np.where(tti_load > 0, 1.0, 0.0))
        mean_packet_size = np.divide(tti_load, packet_count, out=np.full_like(tti_load, avg_packet_size), where=packet_count > 0)
        std_packet_size = np.full(window_size, rng.uniform(0.0, 250.0), dtype=np.float32)
        max_packet_size = np.maximum(mean_packet_size, avg_packet_size)
        inter_arrival = np.full(window_size, rng.uniform(0.05, 5.0), dtype=np.float32)

        windows[index, :, 0] = tti_load
        windows[index, :, 1] = packet_count
        windows[index, :, 2] = mean_packet_size
        windows[index, :, 3] = std_packet_size
        windows[index, :, 4] = max_packet_size
        windows[index, :, 5] = inter_arrival
        windows[index, :, 6] = rng.uniform(0.0, 1.0)

    return windows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="models/sayzek_traffic_predictor.keras")
    parser.add_argument("--x-scaler-path", default="models/x_scaler.pkl")
    parser.add_argument("--y-scaler-path", default="models/y_scaler.pkl")
    parser.add_argument("--output-path", default="models/sayzek_fast_predictor.pkl")
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-load", type=float, default=29127.712890625)
    parser.add_argument("--student", choices=["hgb", "mlp"], default="mlp")
    parser.add_argument("--hidden-layers", default="48,24")
    parser.add_argument("--max-iter", type=int, default=350)
    args = parser.parse_args()

    window_size = 10
    feature_count = 7
    teacher = tf.keras.models.load_model(
        args.model_path,
        custom_objects={"TemporalAttention": TemporalAttention},
        compile=False,
    )
    x_scaler = joblib.load(args.x_scaler_path)
    y_scaler = joblib.load(args.y_scaler_path)

    windows = build_windows(args.samples, window_size, args.seed, args.max_load)
    scaled = x_scaler.transform(windows.reshape(-1, feature_count)).reshape(args.samples, window_size, feature_count)
    teacher_scaled = teacher.predict(scaled, batch_size=512, verbose=1)
    y = y_scaler.inverse_transform(teacher_scaled).reshape(-1)
    x = windows.reshape(args.samples, window_size * feature_count)

    split = int(args.samples * 0.85)
    if args.student == "hgb":
        model = HistGradientBoostingRegressor(
            max_iter=160,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=0.01,
            random_state=args.seed,
        )
    else:
        hidden_layers = tuple(int(value.strip()) for value in args.hidden_layers.split(",") if value.strip())
        model = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=hidden_layers,
                activation="relu",
                solver="adam",
                alpha=0.0001,
                learning_rate_init=0.001,
                max_iter=args.max_iter,
                early_stopping=True,
                random_state=args.seed,
            ),
        )
    model.fit(x[:split], y[:split])
    predictions = model.predict(x[split:])

    artifact_model = model
    if args.student == "mlp":
        scaler = model.named_steps["standardscaler"]
        mlp = model.named_steps["mlpregressor"]
        artifact_model = {
            "kind": "mlp_numpy",
            "x_mean": scaler.mean_.astype(np.float32),
            "x_scale": scaler.scale_.astype(np.float32),
            "coefs": [coef.astype(np.float32) for coef in mlp.coefs_],
            "intercepts": [intercept.astype(np.float32) for intercept in mlp.intercepts_],
        }

    artifact = {
        "model": artifact_model,
        "student": args.student,
        "window_size": window_size,
        "feature_count": feature_count,
        "teacher_model_path": args.model_path,
        "sample_count": args.samples,
        "hidden_layers": args.hidden_layers if args.student == "mlp" else None,
        "max_iter": args.max_iter if args.student == "mlp" else None,
        "mae": float(mean_absolute_error(y[split:], predictions)),
        "r2": float(r2_score(y[split:], predictions)),
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    print(artifact)


if __name__ == "__main__":
    main()
