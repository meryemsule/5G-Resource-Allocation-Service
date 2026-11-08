from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


def fetch_json(url: str) -> Any:
    headers = {}
    api_key = os.getenv("SAYZEK_API_KEY", "")
    bearer_token = os.getenv("SAYZEK_BEARER_TOKEN", "")
    if api_key:
        headers["X-API-Key"] = api_key
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"
    response = requests.get(url, timeout=15, headers=headers)
    response.raise_for_status()
    return response.json()


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def group_by_service(logs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for log in logs:
        grouped[str(log.get("service_type", "unknown"))].append(log)
    return dict(grouped)


def compute_service_metrics(logs: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(logs)
    if count == 0:
        return {
            "sample_count": 0,
            "average_latency_ms": 0.0,
            "average_margin": 0.0,
            "allocation_success_rate": 0.0,
            "tti_compliance_rate": 0.0,
            "average_rb_allocation": 0.0,
            "average_predicted_load": 0.0,
        }

    avg_latency = sum(safe_float(log.get("inference_latency_ms")) for log in logs) / count
    avg_margin = sum(safe_float(log.get("allocation_margin")) for log in logs) / count
    allocation_success = sum(safe_int(log.get("is_allocation_sufficient")) for log in logs) / count
    tti_success = sum(safe_int(log.get("is_tti_compliant")) for log in logs) / count
    avg_rb = sum(safe_float(log.get("allocated_rb_ratio")) for log in logs) / count
    avg_predicted = sum(safe_float(log.get("predicted_load")) for log in logs) / count

    return {
        "sample_count": count,
        "average_latency_ms": avg_latency,
        "average_margin": avg_margin,
        "allocation_success_rate": allocation_success,
        "tti_compliance_rate": tti_success,
        "average_rb_allocation": avg_rb,
        "average_predicted_load": avg_predicted,
    }


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_rows = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_row, separator_row] + body_rows)


def svg_bar_chart(title: str, values: dict[str, float], width: int = 900, height: int = 360, color: str = "#2274a5") -> str:
    if not values:
        return f"<svg width='{width}' height='{height}' xmlns='http://www.w3.org/2000/svg'><text x='20' y='40'>{title}: no data</text></svg>"

    margin_left = 70
    margin_bottom = 70
    margin_top = 50
    plot_width = width - margin_left - 30
    plot_height = height - margin_top - margin_bottom
    max_value = max(values.values()) or 1.0
    bar_count = len(values)
    gap = 20
    bar_width = max(30, int((plot_width - (bar_count - 1) * gap) / max(bar_count, 1)))

    svg_parts = [
        f"<svg width='{width}' height='{height}' xmlns='http://www.w3.org/2000/svg'>",
        f"<rect width='{width}' height='{height}' fill='white' />",
        f"<text x='{width / 2}' y='28' text-anchor='middle' font-size='20' font-family='Segoe UI'>{title}</text>",
        f"<line x1='{margin_left}' y1='{margin_top + plot_height}' x2='{width - 20}' y2='{margin_top + plot_height}' stroke='#333' />",
        f"<line x1='{margin_left}' y1='{margin_top}' x2='{margin_left}' y2='{margin_top + plot_height}' stroke='#333' />",
    ]

    for index, (label, value) in enumerate(values.items()):
        bar_height = 0 if max_value == 0 else (value / max_value) * plot_height
        x = margin_left + index * (bar_width + gap)
        y = margin_top + plot_height - bar_height
        svg_parts.append(f"<rect x='{x}' y='{y}' width='{bar_width}' height='{bar_height}' fill='{color}' rx='4' />")
        svg_parts.append(f"<text x='{x + bar_width / 2}' y='{margin_top + plot_height + 22}' text-anchor='middle' font-size='12' font-family='Segoe UI'>{label}</text>")
        svg_parts.append(f"<text x='{x + bar_width / 2}' y='{max(18, y - 8)}' text-anchor='middle' font-size='12' font-family='Segoe UI'>{value:.3f}</text>")

    for tick in range(5):
        tick_value = max_value * tick / 4
        y = margin_top + plot_height - (plot_height * tick / 4)
        svg_parts.append(f"<line x1='{margin_left - 5}' y1='{y}' x2='{margin_left}' y2='{y}' stroke='#333' />")
        svg_parts.append(f"<text x='{margin_left - 10}' y='{y + 4}' text-anchor='end' font-size='11' font-family='Segoe UI'>{tick_value:.2f}</text>")

    svg_parts.append("</svg>")
    return "".join(svg_parts)


def svg_line_chart(title: str, series: list[float], width: int = 900, height: int = 360, color: str = "#d1495b") -> str:
    if not series:
        return f"<svg width='{width}' height='{height}' xmlns='http://www.w3.org/2000/svg'><text x='20' y='40'>{title}: no data</text></svg>"

    margin_left = 70
    margin_bottom = 60
    margin_top = 50
    plot_width = width - margin_left - 30
    plot_height = height - margin_top - margin_bottom
    max_value = max(series) or 1.0
    min_value = min(series)
    span = max(max_value - min_value, 1e-9)

    points = []
    for index, value in enumerate(series):
        x = margin_left + (plot_width * index / max(len(series) - 1, 1))
        normalized = (value - min_value) / span
        y = margin_top + plot_height - (normalized * plot_height)
        points.append((x, y, value))

    path = " ".join(f"{x},{y}" for x, y, _ in points)
    svg_parts = [
        f"<svg width='{width}' height='{height}' xmlns='http://www.w3.org/2000/svg'>",
        f"<rect width='{width}' height='{height}' fill='white' />",
        f"<text x='{width / 2}' y='28' text-anchor='middle' font-size='20' font-family='Segoe UI'>{title}</text>",
        f"<line x1='{margin_left}' y1='{margin_top + plot_height}' x2='{width - 20}' y2='{margin_top + plot_height}' stroke='#333' />",
        f"<line x1='{margin_left}' y1='{margin_top}' x2='{margin_left}' y2='{margin_top + plot_height}' stroke='#333' />",
        f"<polyline fill='none' stroke='{color}' stroke-width='3' points='{path}' />",
    ]

    for x, y, value in points:
        svg_parts.append(f"<circle cx='{x}' cy='{y}' r='3' fill='{color}' />")

    svg_parts.append("</svg>")
    return "".join(svg_parts)


def write_csv(logs: list[dict[str, Any]], output_path: Path) -> None:
    if not logs:
        output_path.write_text("", encoding="utf-8")
        return

    fieldnames = sorted({key for log in logs for key in log.keys()})
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(logs)


def generate_report(base_url: str, metrics_limit: int, logs_limit: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = fetch_json(f"{base_url.rstrip('/')}/metrics?limit={metrics_limit}")
    logs = fetch_json(f"{base_url.rstrip('/')}/decision-logs?limit={logs_limit}")
    service_groups = group_by_service(logs)
    service_metrics = {service: compute_service_metrics(service_logs) for service, service_logs in service_groups.items()}

    metrics_path = output_dir / "metrics_summary.json"
    logs_csv_path = output_dir / "decision_logs.csv"
    report_md_path = output_dir / "sayzek_report.md"
    report_html_path = output_dir / "sayzek_charts.html"

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_csv(logs, logs_csv_path)

    global_table = markdown_table(
        ["Metric", "Value"],
        [
            ["Average Inference Latency (ms)", f"{safe_float(metrics.get('average_latency_ms')):.6f}"],
            ["TTI Compliance Rate", f"{safe_float(metrics.get('tti_compliance_rate')) * 100:.2f}%"],
            ["Allocation Success Rate", f"{safe_float(metrics.get('allocation_success_rate')) * 100:.2f}%"],
            ["Total Allocated RB Ratio", f"{safe_float(metrics.get('total_allocated_rb_ratio')):.6f}"],
            ["Average Predicted Load", f"{safe_float(metrics.get('average_predicted_load')):.6f}"],
            ["Sample Count", str(safe_int(metrics.get('sample_count')))],
        ],
    )

    service_rows = []
    for service, values in sorted(service_metrics.items()):
        service_rows.append(
            [
                service,
                str(values["sample_count"]),
                f"{values['average_latency_ms']:.6f}",
                f"{values['average_margin']:.6f}",
                f"{values['allocation_success_rate'] * 100:.2f}%",
                f"{values['tti_compliance_rate'] * 100:.2f}%",
                f"{values['average_rb_allocation']:.6f}",
            ]
        )

    service_table = markdown_table(
        [
            "Service",
            "Samples",
            "Avg Latency (ms)",
            "Avg Allocation Margin",
            "Allocation Success",
            "TTI Compliance",
            "Avg RB Allocation",
        ],
        service_rows,
    )

    report_md = "\n".join(
        [
            "# SAYZEK Performance Report",
            "",
            f"Generated at: {datetime.now().isoformat()}",
            "",
            "## Global Metrics",
            "",
            global_table,
            "",
            "## Service-Based Metrics",
            "",
            service_table,
            "",
            "## Notes",
            "",
            "- `average_latency_ms` shows the end-to-end inference time seen by the API.",
            "- `allocation_margin` indicates how much RB ratio was left above the predicted load.",
            "- `allocation_success_rate` shows the ratio of requests where allocation covered predicted load.",
            "",
        ]
    )
    report_md_path.write_text(report_md, encoding="utf-8")

    latency_by_service = {service: values["average_latency_ms"] for service, values in sorted(service_metrics.items())}
    margin_by_service = {service: values["average_margin"] for service, values in sorted(service_metrics.items())}
    recent_latency_series = [safe_float(log.get("inference_latency_ms")) for log in reversed(logs)]

    html_parts = [
        "<html><head><meta charset='utf-8'><title>SAYZEK Charts</title></head><body style='font-family:Segoe UI, sans-serif'>",
        "<h1>SAYZEK Performance Charts</h1>",
        "<h2>Average Inference Latency by Service</h2>",
        svg_bar_chart("Average Inference Latency by Service", latency_by_service, color="#1f77b4"),
        "<h2>Average Allocation Margin by Service</h2>",
        svg_bar_chart("Average Allocation Margin by Service", margin_by_service, color="#2ca02c"),
        "<h2>Recent Inference Latency Trend</h2>",
        svg_line_chart("Recent Inference Latency Trend", recent_latency_series, color="#d62728"),
        "</body></html>",
    ]
    report_html_path.write_text("".join(html_parts), encoding="utf-8")

    print("Report files generated:")
    print("-", metrics_path)
    print("-", logs_csv_path)
    print("-", report_md_path)
    print("-", report_html_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze SAYZEK API metrics and decision logs.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="FastAPI base URL")
    parser.add_argument("--metrics-limit", type=int, default=50, help="Window size for /metrics")
    parser.add_argument("--logs-limit", type=int, default=200, help="How many recent decision logs to analyze")
    parser.add_argument("--output-dir", default="report_output", help="Output folder for tables and charts")
    args = parser.parse_args()

    generate_report(
        base_url=args.base_url,
        metrics_limit=args.metrics_limit,
        logs_limit=args.logs_limit,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
