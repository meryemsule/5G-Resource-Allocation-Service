from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("report_output")
COLORS = {
    "URLLC": (210, 67, 67),
    "eMBB": (44, 116, 179),
    "mMTC": (63, 145, 92),
    "loss": (44, 116, 179),
    "grid": (220, 224, 230),
    "text": (36, 42, 52),
}


def math_ceil(value: float, step: float) -> float:
    return math.ceil(value / step) * step


def font(size: int = 14) -> ImageFont.ImageFont:
    del size
    return ImageFont.load_default()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def draw_axes(
    draw: ImageDraw.ImageDraw,
    left: int,
    top: int,
    right: int,
    bottom: int,
    y_ticks: int = 5,
    y_min: float = 0.0,
    y_max: float = 1.0,
) -> None:
    draw.line((left, top, left, bottom), fill=(70, 76, 86), width=2)
    draw.line((left, bottom, right, bottom), fill=(70, 76, 86), width=2)
    for idx in range(y_ticks + 1):
        y = bottom - int((bottom - top) * idx / y_ticks)
        draw.line((left, y, right, y), fill=COLORS["grid"], width=1)
        label = y_min + ((idx / y_ticks) * (y_max - y_min))
        draw.text((left - 44, y - 5), f"{label:.2f}", fill=COLORS["text"], font=font())


def draw_attention_plot() -> None:
    rows = read_csv(ROOT / "attention_weights.csv")
    grouped: dict[str, list[tuple[int, float]]] = {}
    for row in rows:
        grouped.setdefault(row["service_type"], []).append(
            (int(row["timestep"]), float(row["attention_weight"]))
        )

    width, height = 980, 520
    left, top, right, bottom = 86, 64, 930, 430
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((left, 24), "Attention Agirliklari - TTI Zaman Adimlari", fill=COLORS["text"], font=font(18))
    attention_axis_max = 0.25
    draw_axes(draw, left, top, right, bottom, y_min=0.0, y_max=attention_axis_max)

    for service, points in grouped.items():
        points = sorted(points)
        xy: list[tuple[int, int]] = []
        for timestep, weight in points:
            x = left + int((right - left) * timestep / 9)
            y = bottom - int((bottom - top) * min(weight / attention_axis_max, 1.0))
            xy.append((x, y))
        color = COLORS.get(service, (80, 80, 80))
        if len(xy) > 1:
            draw.line(xy, fill=color, width=3)
        for x, y in xy:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)

    for timestep in range(10):
        x = left + int((right - left) * timestep / 9)
        draw.text((x - 5, bottom + 14), str(timestep), fill=COLORS["text"], font=font())
    draw.text((left + 340, bottom + 42), "TTI zaman adimi", fill=COLORS["text"], font=font())
    draw.text((18, top + 150), "Agirlik", fill=COLORS["text"], font=font())

    legend_x = left
    for service in ("URLLC", "eMBB", "mMTC"):
        color = COLORS[service]
        draw.rectangle((legend_x, 458, legend_x + 18, 476), fill=color)
        draw.text((legend_x + 26, 459), service, fill=COLORS["text"], font=font())
        legend_x += 140

    image.save(ROOT / "attention_weights_plot.png")


def extract_loss_rows(notebook_path: Path) -> list[dict[str, Any]]:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8-sig"))
    pattern = re.compile(r"Epoch \[(\d+)/(\d+)\].*?([0-9]+(?:\.[0-9]+)?)")
    rows: list[dict[str, Any]] = []
    for cell in notebook.get("cells", []):
        for output in cell.get("outputs", []):
            text = output.get("text", "")
            if isinstance(text, list):
                text = "".join(text)
            for line in str(text).splitlines():
                match = pattern.search(line)
                if match:
                    rows.append(
                        {
                            "epoch": int(match.group(1)),
                            "total_epochs": int(match.group(2)),
                            "training_loss": float(match.group(3)),
                        }
                    )
    rows.sort(key=lambda item: int(item["epoch"]))
    return rows


def draw_loss_curve(loss_rows: list[dict[str, Any]]) -> None:
    width, height = 860, 500
    left, top, right, bottom = 88, 60, 810, 420
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((left, 24), "Egitim Loss Egrisi", fill=COLORS["text"], font=font(18))
    max_loss = max(float(row["training_loss"]) for row in loss_rows)
    y_axis_max = math_ceil(max_loss, 0.1)
    draw_axes(draw, left, top, right, bottom, y_min=0.0, y_max=y_axis_max)

    max_epoch = max(int(row["epoch"]) for row in loss_rows)
    points: list[tuple[int, int]] = []
    for row in loss_rows:
        x = left + int((right - left) * int(row["epoch"]) / max_epoch)
        normalized = float(row["training_loss"]) / max(y_axis_max, 1e-9)
        y = bottom - int((bottom - top) * normalized)
        points.append((x, y))
    draw.line(points, fill=COLORS["loss"], width=3)
    for row, (x, y) in zip(loss_rows, points):
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=COLORS["loss"])
        draw.text((x - 16, bottom + 14), str(row["epoch"]), fill=COLORS["text"], font=font())
        draw.text((x - 22, y - 22), f"{row['training_loss']:.3f}", fill=COLORS["text"], font=font())
    draw.text((left + 310, bottom + 42), "Epoch", fill=COLORS["text"], font=font())
    draw.text((18, top + 150), "Loss", fill=COLORS["text"], font=font())
    image.save(ROOT / "training_loss_curve.png")


def draw_load_level_plot() -> None:
    path = ROOT / "load_level_summary.csv"
    if not path.exists():
        return
    rows = [
        row for row in read_csv(path)
        if row["method"] in {"logged_runtime", "load_plus_service_bias", "guarded_hybrid_proxy"}
    ]
    levels = ["low", "medium", "high"]
    methods = ["logged_runtime", "load_plus_service_bias", "guarded_hybrid_proxy"]
    values = {
        (row["load_level"], row["method"]): float(row["avg_allocation_error"])
        for row in rows
    }
    max_value = max(values.values() or [1.0])

    width, height = 980, 520
    left, top, right, bottom = 92, 64, 930, 430
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((left, 24), "Yuk Seviyesine Gore Tahsis Hatasi", fill=COLORS["text"], font=font(18))
    draw_axes(draw, left, top, right, bottom)

    bar_area = right - left
    group_width = bar_area // len(levels)
    bar_width = 52
    method_colors = {
        "logged_runtime": (44, 116, 179),
        "load_plus_service_bias": (63, 145, 92),
        "guarded_hybrid_proxy": (210, 67, 67),
    }
    for group_index, level in enumerate(levels):
        center = left + group_width * group_index + group_width // 2
        for method_index, method in enumerate(methods):
            value = values.get((level, method), 0.0)
            bar_height = int((bottom - top) * value / max_value)
            x0 = center + (method_index - 1) * (bar_width + 8)
            y0 = bottom - bar_height
            draw.rectangle((x0, y0, x0 + bar_width, bottom), fill=method_colors[method])
        draw.text((center - 20, bottom + 18), level, fill=COLORS["text"], font=font())

    legend_x = left
    for method in methods:
        draw.rectangle((legend_x, 458, legend_x + 18, 476), fill=method_colors[method])
        draw.text((legend_x + 26, 459), method, fill=COLORS["text"], font=font())
        legend_x += 245
    image.save(ROOT / "load_level_allocation_error.png")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    draw_attention_plot()
    loss_rows = extract_loss_rows(Path("sayzek.ipynb"))
    if loss_rows:
        write_csv(ROOT / "training_loss_history.csv", loss_rows)
        draw_loss_curve(loss_rows)
    draw_load_level_plot()
    print(
        json.dumps(
            {
                "attention_plot": str(ROOT / "attention_weights_plot.png"),
                "loss_csv": str(ROOT / "training_loss_history.csv"),
                "loss_plot": str(ROOT / "training_loss_curve.png"),
                "load_level_plot": str(ROOT / "load_level_allocation_error.png"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
