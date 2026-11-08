from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import PPO


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


def load_and_preprocess_5g_data(dataset_root: str, window_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    csv_files = find_csv_files(dataset_root)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {dataset_root}")

    all_data: list[np.ndarray] = []
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            df.columns = [str(column).lower() for column in df.columns]

            timestamp_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
            length_col = "length" if "length" in df.columns else ("size" if "size" in df.columns else df.columns[1])

            df[timestamp_col] = pd.to_datetime(df[timestamp_col], format="mixed", errors="coerce")
            df = df.dropna(subset=[timestamp_col])
            df = df[[timestamp_col, length_col]].copy()
            df[length_col] = pd.to_numeric(df[length_col], errors="coerce").fillna(0.0)
            df = df.set_index(timestamp_col)

            resampled = df[length_col].resample("1ms").sum().fillna(0.0)
            if len(resampled) > window_size:
                all_data.append(resampled.values.astype(np.float32))
        except Exception:
            continue

    if not all_data:
        raise RuntimeError("Dataset was found, but no CSV file could be converted into 1 ms TTI series.")

    flat_data = np.concatenate(all_data).reshape(-1, 1)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(flat_data).astype(np.float32)

    X, y = [], []
    for index in range(len(scaled_data) - window_size):
        X.append(scaled_data[index:index + window_size])
        y.append(scaled_data[index + window_size])

    X_array = np.array(X, dtype=np.float32)
    y_array = np.array(y, dtype=np.float32)
    return X_array, y_array, scaled_data.flatten(), scaler


class TrafficLSTM(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 128, num_layers: int = 2) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        device = x.device
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=device)
        out, (hn, _cn) = self.lstm(x, (h0, c0))
        prediction = self.fc(out[:, -1, :])
        return prediction, hn


class FiveGResourceEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, traffic_series: np.ndarray, lstm_model: TrafficLSTM, window_size: int, device: torch.device) -> None:
        super().__init__()
        self.traffic_series = traffic_series.astype(np.float32)
        self.lstm_model = lstm_model
        self.window_size = window_size
        self.device = device
        self.current_step = window_size

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(130,), dtype=np.float32)

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.current_step = self.window_size
        observation = self._build_observation()
        return observation, {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        predicted_load = float(self._predict_load())
        reward = float(1.0 - abs(float(action[0]) - predicted_load))

        self.current_step += 1
        done = self.current_step >= len(self.traffic_series) - 1
        observation = self._build_observation()
        return observation, reward, done, False, {"predicted_load": predicted_load}

    def _current_window_tensor(self) -> torch.Tensor:
        raw_window = self.traffic_series[self.current_step - self.window_size:self.current_step]
        return torch.tensor(raw_window, dtype=torch.float32, device=self.device).view(1, self.window_size, 1)

    def _predict_load(self) -> float:
        with torch.no_grad():
            x_input = self._current_window_tensor()
            prediction, _hn = self.lstm_model(x_input)
            return float(prediction.cpu().item())

    def _build_observation(self) -> np.ndarray:
        with torch.no_grad():
            x_input = self._current_window_tensor()
            prediction, hn = self.lstm_model(x_input)
            l_pred = float(prediction.cpu().item())
            hidden = hn[-1].cpu().numpy().flatten()
        return np.concatenate([[l_pred], hidden, [0.0]]).astype(np.float32)


@dataclass
class TrainResult:
    lstm_model: TrafficLSTM
    ppo_model: PPO
    scaler: StandardScaler


def train_models(
    dataset_root: str,
    output_dir: str,
    window_size: int,
    hidden_size: int,
    num_layers: int,
    lstm_epochs: int,
    ppo_timesteps: int,
    learning_rate: float,
) -> TrainResult:
    X_train, y_train, traffic_series, scaler = load_and_preprocess_5g_data(dataset_root, window_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)

    lstm_model = TrafficLSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(lstm_model.parameters(), lr=learning_rate)

    print(f"LSTM training started on {device} with {len(X_train_t)} samples")
    lstm_model.train()
    for epoch in range(lstm_epochs):
        optimizer.zero_grad()
        predictions, _ = lstm_model(X_train_t)
        loss = criterion(predictions, y_train_t)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{lstm_epochs} - loss: {loss.item():.6f}")

    lstm_model.eval()

    env = FiveGResourceEnv(
        traffic_series=traffic_series,
        lstm_model=lstm_model,
        window_size=window_size,
        device=device,
    )

    print(f"PPO training started for {ppo_timesteps} timesteps")
    ppo_model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        clip_range=0.2,
        gamma=0.95,
        device="cpu",
    )
    ppo_model.learn(total_timesteps=ppo_timesteps)

    export_models(
        lstm_model=lstm_model,
        ppo_model=ppo_model,
        scaler=scaler,
        output_dir=output_dir,
        window_size=window_size,
    )
    return TrainResult(lstm_model=lstm_model, ppo_model=ppo_model, scaler=scaler)


def export_models(
    lstm_model: TrafficLSTM,
    ppo_model: PPO,
    scaler: StandardScaler,
    output_dir: str,
    window_size: int,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    lstm_path = output_path / "lstm_traffic_model.pth"
    ppo_path = output_path / "ppo_5g_agent.zip"
    metadata_path = output_path / "model_metadata.json"

    torch.save(lstm_model.state_dict(), lstm_path)
    ppo_model.save(str(ppo_path))

    metadata = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "hidden_size": lstm_model.hidden_size,
        "num_layers": lstm_model.num_layers,
        "window_size": window_size,
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    print("Export completed:")
    print(f"- {lstm_path}")
    print(f"- {ppo_path}")
    print(f"- {metadata_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and export 5G TrafficLSTM + PPO models.")
    parser.add_argument(
        "--dataset-root",
        default="/kaggle/input/datasets/kimdaegyeom/5g-traffic-datasets",
        help="Root folder of the Kaggle dataset",
    )
    parser.add_argument("--output-dir", default="/kaggle/working", help="Export directory")
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--lstm-epochs", type=int, default=20)
    parser.add_argument("--ppo-timesteps", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()

    train_models(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        window_size=args.window_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        lstm_epochs=args.lstm_epochs,
        ppo_timesteps=args.ppo_timesteps,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()
