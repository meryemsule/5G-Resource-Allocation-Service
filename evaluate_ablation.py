from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from app.model_runtime import runtime_model
from app.schemas import TrafficFeatures


SERVICES = ("URLLC", "eMBB", "mMTC")


@dataclass(frozen=True)
class Scenario:
    service_type: str
    tti_loads: list[float]
    actual_next_load_ratio: float
    packet_count: int
    avg_packet_size: float
    avg_inter_arrival_ms: float


def build_scenarios(count_per_service: int, max_reference: float) -> list[Scenario]:
    scenarios: list[Scenario] = []
    service_base = {"URLLC": 900.0, "eMBB": 1800.0, "mMTC": 350.0}
    service_step = {"URLLC": 135.0, "eMBB": 260.0, "mMTC": 42.0}
    packet_size = {"URLLC": 720.0, "eMBB": 1100.0, "mMTC": 256.0}
    inter_arrival = {"URLLC": 0.45, "eMBB": 1.8, "mMTC": 8.0}

    for service in SERVICES:
        for index in range(count_per_service):
            base = service_base[service] + (index * service_step[service])
            wave = (index % 7) * service_step[service] * 0.18
            loads = [
                max(0.0, base + (slot * service_step[service] * 0.55) + wave)
                for slot in range(10)
            ]
            next_load = loads[-1] + service_step[service] * (0.4 + ((index % 5) * 0.08))
            scenarios.append(
                Scenario(
                    service_type=service,
                    tti_loads=loads,
                    actual_next_load_ratio=min(next_load / max_reference, 1.0),
                    packet_count=10,
                    avg_packet_size=packet_size[service],
                    avg_inter_arrival_ms=inter_arrival[service],
                )
            )
    return scenarios


def to_features(scenario: Scenario) -> TrafficFeatures:
    return TrafficFeatures(
        service_type=scenario.service_type,  # type: ignore[arg-type]
        tti_loads=scenario.tti_loads,
        packet_count=scenario.packet_count,
        avg_packet_size=scenario.avg_packet_size,
        avg_inter_arrival_ms=scenario.avg_inter_arrival_ms,
    )


def round_robin_baseline(scenario: Scenario) -> dict[str, Any]:
    return {
        "rb_allocation_ratio": 1.0 / 3.0,
        "predicted_load": scenario.tti_loads[-1] / runtime_model.max_load_reference,
        "model_source": "round-robin-equal-slice",
    }


def heuristic_baseline(scenario: Scenario) -> dict[str, Any]:
    predicted = scenario.tti_loads[-1] / runtime_model.max_load_reference
    service_bias = {"URLLC": 0.06, "eMBB": 0.035, "mMTC": 0.015}[scenario.service_type]
    return {
        "rb_allocation_ratio": min(1.0, max(0.0, predicted + service_bias)),
        "predicted_load": predicted,
        "model_source": "load-plus-service-bias",
    }


def teacher_only(scenario: Scenario) -> dict[str, Any]:
    features = to_features(scenario)
    original_fast_predictor = runtime_model.fast_predictor
    original_fast_policy = runtime_model.fast_policy_correction
    try:
        runtime_model.fast_predictor = None
        runtime_model.fast_policy_correction = None
        runtime_model._last_signature = None
        runtime_model._last_result = None
        result = runtime_model._allocate_sayzek_v2(features)
        result["model_source"] = "teacher-keras-ppo"
        return result
    finally:
        runtime_model.fast_predictor = original_fast_predictor
        runtime_model.fast_policy_correction = original_fast_policy


def fast_distilled(scenario: Scenario) -> dict[str, Any]:
    runtime_model._last_signature = None
    runtime_model._last_result = None
    return runtime_model._allocate_sayzek_v2(to_features(scenario))


def measure(method_name: str, scenario: Scenario, fn: Callable[[Scenario], dict[str, Any]]) -> dict[str, Any]:
    started_at = time.perf_counter()
    result = fn(scenario)
    latency_ms = (time.perf_counter() - started_at) * 1000.0
    allocation = float(result["rb_allocation_ratio"])
    actual = scenario.actual_next_load_ratio
    return {
        "method": method_name,
        "service_type": scenario.service_type,
        "latency_ms": latency_ms,
        "tti_compliant": latency_ms < 1.0,
        "predicted_load": float(result.get("predicted_load", 0.0)),
        "actual_next_load_ratio": actual,
        "allocated_rb_ratio": allocation,
        "allocation_margin": allocation - actual,
        "allocation_success": allocation >= actual,
        "over_allocation": max(0.0, allocation - actual),
        "under_allocation": max(0.0, actual - allocation),
        "model_source": result.get("model_source", method_name),
    }


def warmup_methods(methods: list[tuple[str, Callable[[Scenario], dict[str, Any]]]], scenarios: list[Scenario]) -> None:
    for _method_name, fn in methods:
        for scenario in scenarios[: min(6, len(scenarios))]:
            fn(scenario)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    methods = sorted({str(row["method"]) for row in rows})
    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        latencies = [float(row["latency_ms"]) for row in subset]
        summary_rows.append(
            {
                "method": method,
                "samples": len(subset),
                "avg_latency_ms": statistics.fmean(latencies),
                "p95_latency_ms": sorted(latencies)[int(0.95 * (len(latencies) - 1))],
                "tti_compliance_rate": statistics.fmean(float(row["tti_compliant"]) for row in subset),
                "allocation_success_rate": statistics.fmean(float(row["allocation_success"]) for row in subset),
                "avg_over_allocation": statistics.fmean(float(row["over_allocation"]) for row in subset),
                "avg_under_allocation": statistics.fmean(float(row["under_allocation"]) for row in subset),
                "avg_allocation_margin": statistics.fmean(float(row["allocation_margin"]) for row in subset),
            }
        )
    return summary_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# SAYZEK Ablation and Baseline Results",
        "",
        "Scenario set: synthetic but fixed URLLC/eMBB/mMTC TTI windows generated from the same max-load reference used by the runtime.",
        "",
        "| Method | Avg latency ms | P95 latency ms | TTI compliance | Allocation success | Avg over-allocation | Avg under-allocation |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {method} | {avg_latency_ms:.4f} | {p95_latency_ms:.4f} | {tti_compliance_rate:.2%} | {allocation_success_rate:.2%} | {avg_over_allocation:.4f} | {avg_under_allocation:.4f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "Interpretation notes:",
            "- `round-robin-equal-slice` is a static equal-share baseline and is intentionally load-agnostic.",
            "- `load-plus-service-bias` is a lightweight heuristic baseline.",
            "- `teacher-keras-ppo` uses the original Keras teacher predictor and PPO policy.",
            "- `fast-distilled-hybrid` uses the distilled NumPy MLP predictor and distilled constant PPO correction.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count-per-service", type=int, default=40)
    parser.add_argument("--output-dir", default="report_output")
    args = parser.parse_args()

    scenarios = build_scenarios(args.count_per_service, runtime_model.max_load_reference)
    methods: list[tuple[str, Callable[[Scenario], dict[str, Any]]]] = [
        ("round-robin-equal-slice", round_robin_baseline),
        ("load-plus-service-bias", heuristic_baseline),
        ("teacher-keras-ppo", teacher_only),
        ("fast-distilled-hybrid", fast_distilled),
    ]

    warmup_methods(methods, scenarios)
    rows = [measure(method_name, scenario, fn) for scenario in scenarios for method_name, fn in methods]
    summary_rows = summarize(rows)

    output_dir = Path(args.output_dir)
    write_csv(output_dir / "ablation_detailed.csv", rows)
    write_csv(output_dir / "ablation_summary.csv", summary_rows)
    write_markdown(output_dir / "ablation_report.md", summary_rows)
    (output_dir / "ablation_summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    print(json.dumps(summary_rows, indent=2))


if __name__ == "__main__":
    main()
