from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import PPO

from training.feature_engineering import FEATURE_COLUMNS, engineer_sample, sample_to_feature_vector
from training.service_aware_dataset import TraceWindow, generate_synthetic_mmtc_windows, load_trace_windows


class ServiceAwareAllocationEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        observations: np.ndarray,
        predicted_loads: np.ndarray,
        actual_loads: np.ndarray,
        service_types: list[str],
    ) -> None:
        super().__init__()
        self.observations = observations.astype(np.float32)
        self.predicted_loads = np.clip(predicted_loads.astype(np.float32), 0.0, 1.0)
        self.actual_loads = np.clip(actual_loads.astype(np.float32), 0.0, 1.0)
        self.service_types = service_types
        self.index = 0

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observations.shape[1] + 2,),
            dtype=np.float32,
        )

    def _service_bias(self, service_type: str) -> float:
        return {"URLLC": 0.15, "eMBB": 0.08, "mMTC": 0.04}.get(service_type, 0.08)

    def _augmented_observation(self) -> np.ndarray:
        vector = self.observations[self.index]
        predicted = self.predicted_loads[self.index]
        bias = self._service_bias(self.service_types[self.index])
        return np.concatenate([vector, np.array([predicted, bias], dtype=np.float32)]).astype(np.float32)

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.index = 0
        return self._augmented_observation(), {}

    def step(self, action: np.ndarray):
        allocation = float(np.clip(action[0], 0.0, 1.0))
        predicted = float(self.predicted_loads[self.index])
        actual = float(self.actual_loads[self.index])
        service_bias = self._service_bias(self.service_types[self.index])

        throughput_score = 1.0 - abs(allocation - actual)
        sla_satisfaction = 1.0 if allocation >= actual else -1.0
        latency_bonus = 0.4 if service_bias >= 0.15 and allocation >= predicted else 0.0
        fairness_penalty = abs(allocation - predicted) * 0.2
        over_allocation_penalty = max(0.0, allocation - actual) * 0.35
        starvation_penalty = 0.8 if allocation < actual * 0.75 else 0.0

        reward = (
            throughput_score
            + sla_satisfaction
            + latency_bonus
            - fairness_penalty
            - over_allocation_penalty
            - starvation_penalty
        )

        self.index += 1
        done = self.index >= len(self.observations)
        if done:
            next_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            next_obs = self._augmented_observation()

        return next_obs, float(reward), done, False, {
            "actual_load": actual,
            "predicted_load": predicted,
            "allocation": allocation,
        }


def build_engineered_dataset(
    dataset_root: str,
    window_size: int,
    synthetic_mmtc_count: int,
) -> list[dict[str, Any]]:
    trace_windows = load_trace_windows(dataset_root=dataset_root, window_size=window_size + 1)
    trace_windows.extend(generate_synthetic_mmtc_windows(count=synthetic_mmtc_count, window_size=window_size + 1))

    engineered: list[dict[str, Any]] = []
    for trace in trace_windows:
        base_window = TraceWindow(
            service_type=trace.service_type,
            app_name=trace.app_name,
            source_file=trace.source_file,
            tti_loads=trace.tti_loads[:window_size],
            packet_sizes=trace.packet_sizes[:window_size],
            inter_arrivals_ms=trace.inter_arrivals_ms[:window_size],
            uplink_ratio=trace.uplink_ratio,
            downlink_ratio=trace.downlink_ratio,
        )
        sample = engineer_sample(base_window, next_load=float(trace.tti_loads[window_size]))
        record = asdict(sample)
        record["window_values"] = base_window.tti_loads.tolist()
        engineered.append(record)
    return engineered


def train_predictor(
    records: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    groups = [record["source_file"] for record in records]
    feature_matrix = np.array(
        [[float(record[column]) for column in FEATURE_COLUMNS] for record in records],
        dtype=np.float32,
    )
    targets = np.array([float(record["target_next_load"]) for record in records], dtype=np.float32)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_indices, test_indices = next(splitter.split(feature_matrix, targets, groups=groups))

    X_train_raw, X_test_raw = feature_matrix[train_indices], feature_matrix[test_indices]
    y_train, y_test = targets[train_indices], targets[test_indices]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    predictor = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=6,
        max_iter=250,
        random_state=42,
    )
    predictor.fit(X_train, y_train)

    y_pred = predictor.predict(X_test)
    mse = float(mean_squared_error(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))

    predictor_path = output_dir / "service_aware_predictor.json"
    metadata_path = output_dir / "service_aware_metadata.json"

    predictor_blob = {
        "feature_columns": FEATURE_COLUMNS,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "model_params": predictor.get_params(),
        "train_size": int(len(train_indices)),
        "test_size": int(len(test_indices)),
        "metrics": {"mse": mse, "mae": mae, "rmse": rmse},
    }
    predictor_path.write_text(json.dumps(predictor_blob, indent=2), encoding="utf-8")

    return {
        "predictor": predictor,
        "scaler": scaler,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "train_indices": train_indices,
        "test_indices": test_indices,
        "metadata_path": metadata_path,
        "predictor_path": predictor_path,
        "metrics": predictor_blob["metrics"],
    }


def train_allocator(
    records: list[dict[str, Any]],
    predictor_bundle: dict[str, Any],
    output_dir: Path,
    ppo_timesteps: int,
) -> dict[str, Any]:
    scaler: StandardScaler = predictor_bundle["scaler"]
    predictor: HistGradientBoostingRegressor = predictor_bundle["predictor"]

    feature_matrix = np.array(
        [[float(record[column]) for column in FEATURE_COLUMNS] for record in records],
        dtype=np.float32,
    )
    service_types = [str(record["service_type"]) for record in records]
    actual_loads = np.array([float(record["target_next_load"]) for record in records], dtype=np.float32)
    max_reference = max(float(actual_loads.max()), 1.0)

    scaled_features = scaler.transform(feature_matrix)
    predicted_loads_raw = predictor.predict(scaled_features)
    predicted_ratios = np.clip(predicted_loads_raw / max_reference, 0.0, 1.0)
    actual_ratios = np.clip(actual_loads / max_reference, 0.0, 1.0)

    env = ServiceAwareAllocationEnv(
        observations=scaled_features.astype(np.float32),
        predicted_loads=predicted_ratios.astype(np.float32),
        actual_loads=actual_ratios.astype(np.float32),
        service_types=service_types,
    )
    ppo_model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        gamma=0.98,
        clip_range=0.2,
        device="cpu",
    )
    ppo_model.learn(total_timesteps=ppo_timesteps)

    ppo_path = output_dir / "ppo_5g_agent_v2.zip"
    ppo_model.save(str(ppo_path))

    reward_metadata = {
        "max_reference_load": max_reference,
        "ppo_timesteps": ppo_timesteps,
        "feature_columns": FEATURE_COLUMNS,
    }
    return {
        "ppo_path": ppo_path,
        "reward_metadata": reward_metadata,
        "predicted_ratios": predicted_ratios,
        "actual_ratios": actual_ratios,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train service-aware predictor + PPO allocator for SAYZEK.")
    parser.add_argument(
        "--dataset-root",
        default="/kaggle/input/datasets/kimdaegyeom/5g-traffic-datasets",
        help="Kaggle dataset root",
    )
    parser.add_argument("--output-dir", default="/kaggle/working/service_aware_output")
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--synthetic-mmtc-count", type=int, default=600)
    parser.add_argument("--ppo-timesteps", type=int, default=30000)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = build_engineered_dataset(
        dataset_root=args.dataset_root,
        window_size=args.window_size,
        synthetic_mmtc_count=args.synthetic_mmtc_count,
    )
    (output_dir / "engineered_dataset_preview.json").write_text(
        json.dumps(records[:50], indent=2),
        encoding="utf-8",
    )

    predictor_bundle = train_predictor(records, output_dir)
    allocator_bundle = train_allocator(records, predictor_bundle, output_dir, args.ppo_timesteps)

    summary = {
        "predictor_metrics": predictor_bundle["metrics"],
        "predictor_artifact": str(predictor_bundle["predictor_path"]),
        "allocator_artifact": str(allocator_bundle["ppo_path"]),
        "reward_metadata": allocator_bundle["reward_metadata"],
        "dataset_size": len(records),
        "services": {
            service_type: sum(1 for record in records if record["service_type"] == service_type)
            for service_type in ("URLLC", "eMBB", "mMTC")
        },
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Service-aware pipeline completed:")
    print(f"- {output_dir / 'engineered_dataset_preview.json'}")
    print(f"- {predictor_bundle['predictor_path']}")
    print(f"- {allocator_bundle['ppo_path']}")
    print(f"- {output_dir / 'training_summary.json'}")


if __name__ == "__main__":
    main()
