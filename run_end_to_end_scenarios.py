from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import requests


SCENARIOS = [
    {
        "scenario": "normal_embb_streaming",
        "description": "Stable eMBB video/streaming-like throughput window.",
        "features": {
            "service_type": "eMBB",
            "tti_loads": [1800, 1920, 2050, 2180, 2310, 2450, 2520, 2640, 2760, 2890],
            "packet_count": 10,
            "avg_packet_size": 1100,
            "avg_inter_arrival_ms": 1.8,
        },
    },
    {
        "scenario": "urllc_burst_attack",
        "description": "URLLC latency-critical burst window representing sudden tactical/control-plane pressure.",
        "features": {
            "service_type": "URLLC",
            "tti_loads": [900, 980, 1060, 1190, 1600, 2400, 4200, 6200, 7600, 8800],
            "packet_count": 10,
            "avg_packet_size": 720,
            "avg_inter_arrival_ms": 0.45,
        },
    },
    {
        "scenario": "mmtc_iot_telemetry",
        "description": "mMTC IoT telemetry window with many small low-rate packets.",
        "features": {
            "service_type": "mMTC",
            "tti_loads": [240, 260, 235, 310, 280, 295, 330, 305, 340, 360],
            "packet_count": 10,
            "avg_packet_size": 256,
            "avg_inter_arrival_ms": 8.0,
        },
    },
]


def post_predict(base_url: str, features: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(
        f"{base_url.rstrip('/')}/predict",
        json={"features": features},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def flatten_result(scenario: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    result = results[-1]
    trace = result.get("decision_trace", {})
    latencies = [float(item.get("inference_latency_ms", 0.0)) for item in results]
    sorted_latencies = sorted(latencies)
    p95_index = int(0.95 * (len(sorted_latencies) - 1))
    return {
        "scenario": scenario["scenario"],
        "description": scenario["description"],
        "service_type": result.get("service_type"),
        "model_source": result.get("model_source"),
        "policy_source": trace.get("policy_source"),
        "service_profile": trace.get("service_profile"),
        "latency_ms": latencies[-1],
        "avg_latency_ms": sum(latencies) / len(latencies),
        "p95_latency_ms": sorted_latencies[p95_index],
        "tti_compliant": all(item.get("is_tti_compliant", False) for item in results),
        "predicted_load": result.get("predicted_load"),
        "rb_allocation_ratio": result.get("rb_allocation_ratio"),
        "allocation_margin": result.get("allocation_margin"),
        "risk_score": trace.get("risk_score"),
        "safety_floor": trace.get("safety_floor"),
        "dynamic_cap": trace.get("dynamic_cap"),
        "service_adjustment": trace.get("service_adjustment"),
    }


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# SAYZEK End-to-End Scenario Results",
        "",
        "These scenarios were executed through the running FastAPI service, not by directly calling Python internals.",
        "",
        "| Scenario | Service | Profile | Latency ms | TTI compliant | Predicted load | RB allocation | Risk score |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {scenario} | {service_type} | {service_profile} | {avg_latency_ms:.4f} | {tti_compliant} | {predicted_load:.4f} | {rb_allocation_ratio:.4f} | {risk_score:.4f} |".format(
                **row
            )
        )
    lines.extend(["", "Scenario notes:"])
    for row in rows:
        lines.append(f"- `{row['scenario']}`: {row['description']}")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output-dir", default="report_output")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_results = []
    rows = []
    for scenario in SCENARIOS:
        for _ in range(args.warmup):
            post_predict(args.base_url, scenario["features"])
        results = [post_predict(args.base_url, scenario["features"]) for _ in range(args.repeats)]
        raw_results.append({"scenario": scenario, "results": results})
        rows.append(flatten_result(scenario, results))

    with (output_dir / "end_to_end_scenarios.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    (output_dir / "end_to_end_scenarios.json").write_text(
        json.dumps(raw_results, indent=2),
        encoding="utf-8",
    )
    write_markdown(output_dir / "end_to_end_scenarios.md", rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
