from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


SERVICES = ("URLLC", "eMBB", "mMTC")
SERVICE_BIAS = {"URLLC": 0.06, "eMBB": 0.035, "mMTC": 0.015}


@dataclass(frozen=True)
class Record:
    service_type: str
    window: list[float]
    predicted_load: float
    logged_allocation: float
    latency_ms: float
    model_source: str
    features: dict[str, Any]


def clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def load_records(db_path: Path, limit: int | None = None) -> list[Record]:
    query = """
        select service_type, input_traffic_window, predicted_load, allocated_rb_ratio,
               inference_latency_ms, model_source, full_features_json
        from decision_logs
        order by id
    """
    if limit:
        query += f" limit {int(limit)}"

    records: list[Record] = []
    with sqlite3.connect(db_path) as conn:
        for row in conn.execute(query):
            service_type, raw_window, predicted, allocation, latency, model_source, raw_features = row
            try:
                window = [float(value) for value in json.loads(raw_window)]
                features = json.loads(raw_features)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if not window:
                continue
            records.append(
                Record(
                    service_type=service_type if service_type in SERVICES else "eMBB",
                    window=window,
                    predicted_load=float(predicted),
                    logged_allocation=float(allocation),
                    latency_ms=float(latency),
                    model_source=str(model_source),
                    features=features if isinstance(features, dict) else {},
                )
            )
    return records


def infer_reference(records: list[Record]) -> float:
    peak = max(max(record.window) for record in records)
    return max(peak, 1.0)


def demand_ratio(record: Record, reference: float) -> float:
    return clip(max(float(value) for value in record.window) / reference)


def window_ratios(record: Record, reference: float) -> list[float]:
    return [clip(value / reference) for value in record.window]


def fixed_half(record: Record, reference: float) -> float:
    del record, reference
    return 0.50


def persistence(record: Record, reference: float) -> float:
    return clip(float(record.window[-1]) / reference)


def mean_load(record: Record, reference: float) -> float:
    ratios = window_ratios(record, reference)
    return clip(statistics.fmean(ratios))


def load_plus_service_bias(record: Record, reference: float) -> float:
    return clip(demand_ratio(record, reference) + SERVICE_BIAS[record.service_type])


def logged_runtime(record: Record, reference: float) -> float:
    del reference
    return clip(record.logged_allocation)


def guarded_proxy(record: Record, reference: float) -> float:
    ratios = window_ratios(record, reference)
    current = ratios[-1]
    mean_value = statistics.fmean(ratios)
    peak = max(ratios)
    burstiness = clip((peak - mean_value) / max(mean_value, 1e-9) / 10.0)
    predicted = record.predicted_load if 0.0 <= record.predicted_load <= 1.0 else current

    risk = clip(0.45 * peak + 0.35 * burstiness + 0.20 * max(predicted, current))
    residual = clip(predicted + 0.25)
    alpha = 0.25 + 0.65 * risk
    hybrid = predicted + alpha * (residual - predicted)
    safety_floor = 0.05 if peak > 0.30 or burstiness > 0.50 else 0.02
    dynamic_cap = clip(max(predicted, current) + 0.12 + 0.20 * risk, safety_floor, 1.0)
    allocation = clip(hybrid, safety_floor, dynamic_cap)

    if record.service_type == "URLLC":
        allocation = clip(allocation + 0.066 + 0.03 * risk)
    elif record.service_type == "mMTC":
        allocation = clip(max(predicted + 0.015, allocation - 0.024, 0.01))
    else:
        allocation = clip(allocation + 0.075)
    return allocation


METHODS: dict[str, Callable[[Record, float], float]] = {
    "fixed_0_50": fixed_half,
    "persistence_current_load": persistence,
    "mean_load_baseline": mean_load,
    "load_plus_service_bias": load_plus_service_bias,
    "guarded_hybrid_proxy": guarded_proxy,
    "logged_runtime": logged_runtime,
}


def score_row(record: Record, method: str, allocation: float, reference: float, latency_ms: float) -> dict[str, Any]:
    demand = demand_ratio(record, reference)
    under = max(0.0, demand - allocation)
    over = max(0.0, allocation - demand)
    sla_score = 1.0 if demand <= 0 else clip(allocation / demand)
    return {
        "scenario": "real_log",
        "load_level": load_level(demand),
        "method": method,
        "service_type": record.service_type,
        "model_source": record.model_source,
        "demand_ratio": demand,
        "allocated_ratio": allocation,
        "allocation_error": abs(allocation - demand),
        "under_allocation": under,
        "over_allocation": over,
        "sla_score": sla_score,
        "sla_success": allocation >= demand,
        "latency_ms": latency_ms,
    }


def load_level(demand: float) -> str:
    if demand < 0.33:
        return "low"
    if demand < 0.66:
        return "medium"
    return "high"


def evaluate_records(records: list[Record], reference: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        for method, fn in METHODS.items():
            started = time.perf_counter()
            allocation = fn(record, reference)
            measured_latency = (time.perf_counter() - started) * 1000.0
            latency = record.latency_ms if method == "logged_runtime" else measured_latency
            rows.append(score_row(record, method, allocation, reference, latency))
    return rows


def mutate_record(record: Record, scenario: str) -> Record:
    window = list(record.window)
    if scenario == "burst_x2_last3":
        window[-3:] = [value * 2.0 for value in window[-3:]]
    elif scenario == "burst_x3_last3":
        window[-3:] = [value * 3.0 for value in window[-3:]]
    elif scenario == "load_stress_x5_last2":
        window[-2:] = [value * 5.0 for value in window[-2:]]
    elif scenario == "packet_loss_10":
        window = [value * 0.90 for value in window]
    elif scenario == "packet_loss_20":
        window = [value * 0.80 for value in window]
    elif scenario == "jamming_like_mixed":
        window = [
            value * (1.70 if index % 3 == 0 else 0.82 if index % 3 == 1 else 1.15)
            for index, value in enumerate(window)
        ]
    return Record(
        service_type=record.service_type,
        window=window,
        predicted_load=record.predicted_load,
        logged_allocation=record.logged_allocation,
        latency_ms=record.latency_ms,
        model_source=record.model_source,
        features=record.features,
    )


def evaluate_stress(records: list[Record], reference: float) -> list[dict[str, Any]]:
    stress_methods = {
        key: value for key, value in METHODS.items() if key != "logged_runtime"
    }
    rows: list[dict[str, Any]] = []
    scenarios = [
        "burst_x2_last3",
        "burst_x3_last3",
        "load_stress_x5_last2",
        "packet_loss_10",
        "packet_loss_20",
        "jamming_like_mixed",
    ]
    for scenario in scenarios:
        for record in records:
            mutated = mutate_record(record, scenario)
            for method, fn in stress_methods.items():
                started = time.perf_counter()
                allocation = fn(mutated, reference)
                latency = (time.perf_counter() - started) * 1000.0
                row = score_row(mutated, method, allocation, reference, latency)
                row["scenario"] = scenario
                rows.append(row)
    return rows


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int((pct / 100.0) * (len(ordered) - 1))
    return ordered[index]


def summarize(rows: list[dict[str, Any]], group_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[item] for item in group_keys)
        groups.setdefault(key, []).append(row)

    summary: list[dict[str, Any]] = []
    for key, subset in sorted(groups.items()):
        latencies = [float(row["latency_ms"]) for row in subset]
        item = {name: value for name, value in zip(group_keys, key)}
        item.update(
            {
                "samples": len(subset),
                "avg_allocation_error": statistics.fmean(float(row["allocation_error"]) for row in subset),
                "avg_sla_score": statistics.fmean(float(row["sla_score"]) for row in subset),
                "sla_success_rate": statistics.fmean(float(row["sla_success"]) for row in subset),
                "avg_under_allocation": statistics.fmean(float(row["under_allocation"]) for row in subset),
                "avg_over_allocation": statistics.fmean(float(row["over_allocation"]) for row in subset),
                "avg_latency_ms": statistics.fmean(latencies),
                "p95_latency_ms": percentile(latencies, 95),
                "p99_latency_ms": percentile(latencies, 99),
            }
        )
        summary.append(item)
    return summary


def slice_weighted_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    weights = {
        "URLLC": {"under": 2.5, "over": 0.2, "latency": 0.2},
        "eMBB": {"under": 1.2, "over": 0.4, "latency": 0.05},
        "mMTC": {"under": 0.8, "over": 1.4, "latency": 0.05},
    }
    output: list[dict[str, Any]] = []
    for row in rows:
        service_weights = weights[str(row["service_type"])]
        score = 1.0
        score -= service_weights["under"] * float(row["under_allocation"])
        score -= service_weights["over"] * float(row["over_allocation"])
        score -= service_weights["latency"] * max(0.0, float(row["latency_ms"]) - 1.0)
        item = dict(row)
        item["slice_weighted_score"] = clip(score)
        output.append(item)
    return summarize(output, ("method", "service_type"))


def api_latency_probe(base_url: str, records: list[Record], repeats: int) -> dict[str, Any]:
    if not base_url:
        return {"enabled": False}

    latencies: list[float] = []
    errors = 0
    sample_records = records[: max(1, min(len(records), repeats))]
    for record in sample_records:
        payload = {
            "features": {
                "service_type": record.service_type,
                "tti_loads": record.window,
                "packet_count": int(record.features.get("packet_count") or len(record.window)),
                "avg_packet_size": float(record.features.get("avg_packet_size") or 1.0),
                "avg_inter_arrival_ms": float(record.features.get("avg_inter_arrival_ms") or 1.0),
            }
        }
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/predict",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                response.read()
            latencies.append((time.perf_counter() - started) * 1000.0)
        except (urllib.error.URLError, TimeoutError, OSError):
            errors += 1

    if not latencies:
        return {"enabled": True, "samples": 0, "errors": errors}
    return {
        "enabled": True,
        "samples": len(latencies),
        "errors": errors,
        "avg_api_latency_ms": statistics.fmean(latencies),
        "p95_api_latency_ms": percentile(latencies, 95),
        "p99_api_latency_ms": percentile(latencies, 99),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_float(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return "0.0000"
        return f"{value:.4f}"
    return str(value)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_float(row.get(column, "")) for column in columns) + " |")
    return lines


def write_markdown(
    path: Path,
    metadata: dict[str, Any],
    real_summary: list[dict[str, Any]],
    load_level_summary: list[dict[str, Any]],
    stress_summary: list[dict[str, Any]],
    slice_summary: list[dict[str, Any]],
    api_summary: dict[str, Any],
) -> None:
    top_real = sorted(real_summary, key=lambda row: float(row["avg_allocation_error"]))[:8]
    top_stress = [
        row for row in stress_summary
        if row["method"] in {"fixed_0_50", "load_plus_service_bias", "guarded_hybrid_proxy"}
    ]

    lines = [
        "# SAYZEK Real-Log, Baseline, Latency, Stress, and Slicing Experiments",
        "",
        f"Input records: {metadata['record_count']}",
        f"Normalization reference: {metadata['normalization_reference']:.4f}",
        "",
        "## Real-log baseline summary",
        "",
    ]
    lines.extend(
        markdown_table(
            top_real,
            [
                "method",
                "samples",
                "avg_allocation_error",
                "avg_sla_score",
                "sla_success_rate",
                "avg_under_allocation",
                "avg_over_allocation",
                "avg_latency_ms",
                "p95_latency_ms",
            ],
        )
    )
    lines.extend(["", "## Load-level summary", ""])
    lines.extend(
        markdown_table(
            load_level_summary,
            [
                "load_level",
                "method",
                "samples",
                "avg_allocation_error",
                "avg_sla_score",
                "sla_success_rate",
                "avg_under_allocation",
                "avg_over_allocation",
            ],
        )
    )
    lines.extend(["", "## Stress-test summary", ""])
    lines.extend(
        markdown_table(
            top_stress,
            [
                "scenario",
                "method",
                "samples",
                "avg_allocation_error",
                "avg_sla_score",
                "avg_under_allocation",
                "avg_over_allocation",
            ],
        )
    )
    lines.extend(["", "## Network slicing summary", ""])
    lines.extend(
        markdown_table(
            slice_summary,
            [
                "method",
                "service_type",
                "samples",
                "avg_allocation_error",
                "avg_sla_score",
                "sla_success_rate",
                "avg_under_allocation",
                "avg_over_allocation",
            ],
        )
    )
    lines.extend(["", "## API latency probe", "", "```json", json.dumps(api_summary, indent=2), "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SAYZEK against real decision logs, baselines, stress tests, and slicing views.")
    parser.add_argument("--db-path", default="data/decision_logs.sqlite3")
    parser.add_argument("--output-dir", default="report_output")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--normalization-reference", type=float, default=None)
    parser.add_argument("--api-url", default="")
    parser.add_argument("--api-repeats", type=int, default=30)
    args = parser.parse_args()

    records = load_records(Path(args.db_path), args.limit)
    if not records:
        raise SystemExit(f"No usable records found in {args.db_path}")

    reference = float(args.normalization_reference or infer_reference(records))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    real_rows = evaluate_records(records, reference)
    stress_rows = evaluate_stress(records, reference)
    real_summary = summarize(real_rows, ("method",))
    load_level_summary = summarize(real_rows, ("load_level", "method"))
    stress_summary = summarize(stress_rows, ("scenario", "method"))
    slice_summary = slice_weighted_summary(real_rows)
    latency_by_model = summarize([row for row in real_rows if row["method"] == "logged_runtime"], ("model_source", "service_type"))
    api_summary = api_latency_probe(args.api_url, records, args.api_repeats)

    metadata = {
        "record_count": len(records),
        "normalization_reference": reference,
        "services": sorted({record.service_type for record in records}),
        "model_sources": sorted({record.model_source for record in records}),
        "api_latency": api_summary,
    }

    write_csv(output_dir / "realworld_experiment_detailed.csv", real_rows)
    write_csv(output_dir / "realworld_experiment_summary.csv", real_summary)
    write_csv(output_dir / "load_level_summary.csv", load_level_summary)
    write_csv(output_dir / "stress_experiment_detailed.csv", stress_rows)
    write_csv(output_dir / "stress_experiment_summary.csv", stress_summary)
    write_csv(output_dir / "network_slicing_summary.csv", slice_summary)
    write_csv(output_dir / "latency_by_model_source.csv", latency_by_model)
    (output_dir / "realworld_experiment_summary.json").write_text(
        json.dumps(
            {
                "metadata": metadata,
                "real_log_summary": real_summary,
                "load_level_summary": load_level_summary,
                "stress_summary": stress_summary,
                "network_slicing_summary": slice_summary,
                "latency_by_model_source": latency_by_model,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_markdown(
        output_dir / "realworld_experiment_report.md",
        metadata,
        real_summary,
        load_level_summary,
        stress_summary,
        slice_summary,
        api_summary,
    )
    print(json.dumps({"metadata": metadata, "real_log_summary": real_summary}, indent=2))


if __name__ == "__main__":
    main()
