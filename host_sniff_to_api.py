from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import requests

try:
    from scapy.all import ICMP, IP, TCP, UDP, conf, sniff  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Scapy could not be imported. Install dependencies and Npcap, then run from an elevated terminal."
    ) from exc

TTI_WINDOW_SIZE = 10
TTI_DURATION_MS = 1


def list_interfaces() -> None:
    print("Available capture interfaces:")
    for iface in conf.ifaces.values():
        print(f"- {iface.name}")
        if getattr(iface, "description", None):
            print(f"  {iface.description}")


def protocol_name(packet: Any) -> str:
    if TCP and packet.haslayer(TCP):
        return "tcp"
    if UDP and packet.haslayer(UDP):
        return "udp"
    if ICMP and packet.haslayer(ICMP):
        return "icmp"
    return "unknown"


def packet_to_dict(packet: Any) -> dict[str, Any]:
    ip_layer = packet.getlayer(IP) if IP else None
    tcp_layer = packet.getlayer(TCP) if TCP else None
    udp_layer = packet.getlayer(UDP) if UDP else None
    return {
        "timestamp": float(getattr(packet, "time", time.time())),
        "length": int(len(packet)),
        "protocol": protocol_name(packet),
        "src_ip": getattr(ip_layer, "src", None),
        "dst_ip": getattr(ip_layer, "dst", None),
        "src_port": getattr(tcp_layer or udp_layer, "sport", None),
        "dst_port": getattr(tcp_layer or udp_layer, "dport", None),
        "tcp_flags": str(getattr(tcp_layer, "flags", "")),
    }


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return (sum((value - avg) ** 2 for value in values) / len(values)) ** 0.5


def extract_features_from_packets(
    packets: list[dict[str, Any]],
    window_seconds: float,
    service_type: str,
) -> dict[str, Any]:
    if not packets:
        tti_loads = [0.0] * TTI_WINDOW_SIZE
        return build_features(tti_loads, [], 0, 0, service_type)

    timestamps_ms = [float(packet.get("timestamp", 0.0)) * 1000.0 for packet in packets]
    first_ts = min(timestamps_ms)
    arrival_times_ms = [timestamp - first_ts for timestamp in timestamps_ms]
    total_bytes = sum(int(packet.get("length", 0)) for packet in packets)
    bins = max(TTI_WINDOW_SIZE, int((window_seconds * 1000) / TTI_DURATION_MS))
    tti_loads = [0.0 for _ in range(bins)]

    for packet, relative_ms in zip(packets, arrival_times_ms):
        idx = min(int(relative_ms // TTI_DURATION_MS), bins - 1)
        tti_loads[idx] += float(packet.get("length", 0))

    return build_features(tti_loads[-TTI_WINDOW_SIZE:], arrival_times_ms, total_bytes, len(packets), service_type)


def build_features(
    tti_loads: list[float],
    arrival_times_ms: list[float],
    total_bytes: int,
    packet_count: int,
    service_type: str,
) -> dict[str, Any]:
    inter_arrivals = [
        arrival_times_ms[index] - arrival_times_ms[index - 1]
        for index in range(1, len(arrival_times_ms))
        if arrival_times_ms[index] >= arrival_times_ms[index - 1]
    ]
    avg_packet_size = (total_bytes / packet_count) if packet_count else 0.0
    return {
        "service_type": service_type,
        "tti_loads": tti_loads,
        "arrival_times_ms": arrival_times_ms,
        "total_bytes": total_bytes,
        "packet_count": packet_count,
        "avg_packet_size": avg_packet_size,
        "avg_inter_arrival_ms": mean(inter_arrivals),
        "peak_tti_load": max(tti_loads) if tti_loads else 0.0,
        "mean_tti_load": mean(tti_loads),
        "load_std": std(tti_loads),
        "window_seconds": (len(tti_loads) * TTI_DURATION_MS) / 1000.0,
        "tti_duration_ms": TTI_DURATION_MS,
        "metadata": {"captured_packet_count": packet_count},
    }


def predict(api_url: str, features: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(
        f"{api_url.rstrip('/')}/predict",
        json={"features": features},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def warmup_api(api_url: str, service_type: str) -> None:
    features = build_features(
        tti_loads=[1000.0, 1100.0, 1200.0, 1250.0, 1300.0, 1350.0, 1400.0, 1450.0, 1500.0, 1550.0],
        arrival_times_ms=[float(index) for index in range(10)],
        total_bytes=12750,
        packet_count=10,
        service_type=service_type,
    )
    try:
        predict(api_url, features)
    except Exception as exc:
        print(f"Warmup failed, continuing anyway: {exc}")


def append_result(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture packets from the host PC and send SAYZEK features to the API.")
    parser.add_argument("--list-interfaces", action="store_true")
    parser.add_argument("--interface", default="")
    parser.add_argument("--duration-seconds", type=int, default=5)
    parser.add_argument("--filter", default="ip")
    parser.add_argument("--no-filter", action="store_true")
    parser.add_argument("--service-type", choices=["URLLC", "eMBB", "mMTC"], default="eMBB")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", default="report_output/physical_sniff_results.csv")
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--scenario-label", default="ad-hoc")
    args = parser.parse_args()

    if args.list_interfaces:
        list_interfaces()
        return

    if not args.interface:
        raise SystemExit("Pass --interface with one of the names shown by --list-interfaces.")

    if not args.skip_warmup:
        print("Warming up API inference path...")
        warmup_api(args.api_url, args.service_type)

    capture_filter = "" if args.no_filter else args.filter
    print(f"Capturing {args.duration_seconds}s on: {args.interface}")
    print(f"Filter: {capture_filter or '(none)'} | Service: {args.service_type}")
    packets = sniff(
        iface=args.interface,
        filter=capture_filter or None,
        timeout=args.duration_seconds,
        store=True,
    )
    normalized_packets = [packet_to_dict(packet) for packet in packets]
    features = extract_features_from_packets(
        normalized_packets,
        float(args.duration_seconds),
        args.service_type,
    )
    result = predict(args.api_url, features)

    row = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "interface": args.interface,
        "scenario_label": args.scenario_label,
        "service_type": args.service_type,
        "packet_count": features["packet_count"],
        "total_bytes": features["total_bytes"],
        "avg_packet_size": features["avg_packet_size"],
        "avg_inter_arrival_ms": features["avg_inter_arrival_ms"],
        "rb_allocation_ratio": result["rb_allocation_ratio"],
        "predicted_load": result["predicted_load"],
        "inference_latency_ms": result["inference_latency_ms"],
        "is_tti_compliant": result["is_tti_compliant"],
        "is_allocation_sufficient": result["is_allocation_sufficient"],
        "allocation_margin": result["allocation_margin"],
        "model_source": result["model_source"],
    }
    append_result(Path(args.output), row)

    print(json.dumps(row, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
