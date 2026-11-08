import json
import os

import torch


EXPORT_DIR = "/kaggle/working"


def export_models() -> None:
    if "traffic_scaler" not in globals():
        raise RuntimeError("traffic_scaler bulunamadi. Once veri hazirlama hucrelerini calistirin.")
    if "model" not in globals():
        raise RuntimeError("model bulunamadi. Once LSTM egitim/yukleme hucrelerini calistirin.")
    if "ppo_model" not in globals():
        raise RuntimeError("ppo_model bulunamadi. Once PPO egitim/yukleme hucrelerini calistirin.")

    lstm_path = os.path.join(EXPORT_DIR, "lstm_traffic_model.pth")
    ppo_path = os.path.join(EXPORT_DIR, "ppo_5g_agent.zip")
    metadata_path = os.path.join(EXPORT_DIR, "model_metadata.json")

    torch.save(model.state_dict(), lstm_path)
    ppo_model.save(ppo_path)

    mean_value = traffic_scaler.mean_.tolist() if hasattr(traffic_scaler.mean_, "tolist") else [float(traffic_scaler.mean_)]
    scale_value = traffic_scaler.scale_.tolist() if hasattr(traffic_scaler.scale_, "tolist") else [float(traffic_scaler.scale_)]

    metadata = {
        "mean": mean_value,
        "scale": scale_value,
        "hidden_size": getattr(model, "hidden_size", 128),
        "num_layers": getattr(model, "num_layers", 2),
        "window_size": 10,
    }

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file)

    print("Kritik 3 dosya hazir:")
    print("-", lstm_path)
    print("-", ppo_path)
    print("-", metadata_path)


export_models()
