from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from app.model_runtime import TemporalAttention, runtime_model
from app.schemas import TrafficFeatures


SCENARIOS = {
    "URLLC": {
        "tti_loads": [900, 980, 1120, 1350, 1600, 1900, 2200, 2450, 2700, 2950],
        "packet_count": 10,
        "avg_packet_size": 720,
        "avg_inter_arrival_ms": 0.45,
    },
    "eMBB": {
        "tti_loads": [1800, 2100, 2450, 2850, 3300, 3900, 4550, 5200, 5900, 6600],
        "packet_count": 10,
        "avg_packet_size": 1100,
        "avg_inter_arrival_ms": 1.8,
    },
    "mMTC": {
        "tti_loads": [350, 390, 430, 470, 520, 570, 620, 680, 740, 800],
        "packet_count": 10,
        "avg_packet_size": 256,
        "avg_inter_arrival_ms": 8.0,
    },
}


def softmax(values: np.ndarray, axis: int) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def build_feature_window(service_type: str, spec: dict[str, Any]) -> tuple[np.ndarray, TrafficFeatures]:
    features = TrafficFeatures(
        service_type=service_type,  # type: ignore[arg-type]
        tti_loads=spec["tti_loads"],
        packet_count=spec["packet_count"],
        avg_packet_size=spec["avg_packet_size"],
        avg_inter_arrival_ms=spec["avg_inter_arrival_ms"],
    )
    feature_window, _trace = runtime_model._build_feature_window(features)
    return feature_window, features


def attention_weights(model: tf.keras.Model, scaled_window: np.ndarray) -> np.ndarray:
    attention_layer = model.get_layer("temporal_attention")
    attention_input_layer = model.get_layer("dropout_17")
    intermediate = tf.keras.Model(inputs=model.input, outputs=attention_input_layer.output)
    sequence = intermediate(scaled_window, training=False).numpy()

    weight, bias = attention_layer.get_weights()
    logits = np.tanh(np.matmul(sequence, weight) + bias)
    return softmax(logits, axis=1).reshape(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate temporal attention weights for representative SAYZEK traffic windows.")
    parser.add_argument("--output-dir", default="report_output")
    args = parser.parse_args()

    model = tf.keras.models.load_model(
        "models/sayzek_traffic_predictor.keras",
        custom_objects={"TemporalAttention": TemporalAttention},
        compile=False,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for service_type, spec in SCENARIOS.items():
        feature_window, features = build_feature_window(service_type, spec)
        scaled = runtime_model.x_scaler.transform(feature_window).reshape(
            1,
            runtime_model.window_size,
            len(runtime_model.feature_columns),
        )
        scaled_prediction = model(scaled, training=False).numpy()
        predicted_load = float(runtime_model.y_scaler.inverse_transform(scaled_prediction).reshape(-1)[0])
        weights = attention_weights(model, scaled)
        top_index = int(np.argmax(weights))

        summary.append(
            {
                "service_type": service_type,
                "predicted_load_raw": predicted_load,
                "predicted_load_ratio": predicted_load / max(runtime_model.max_load_reference, 1.0),
                "top_timestep": top_index,
                "top_attention_weight": float(weights[top_index]),
            }
        )
        for timestep, weight in enumerate(weights):
            rows.append(
                {
                    "service_type": service_type,
                    "timestep": timestep,
                    "tti_load": float(features.tti_loads[timestep]),
                    "attention_weight": float(weight),
                    "is_top_timestep": timestep == top_index,
                }
            )

    csv_path = output_dir / "attention_weights.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = output_dir / "attention_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
