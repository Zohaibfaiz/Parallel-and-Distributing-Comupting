"""
================================================================================
CSC-334: Parallel and Distributed Computing
Lab 4: Custom Distributed Task Offloading & Remote GPU Rendering System
Module: client/benchmark.py
Description: Task 5 Benchmarking Suite (Local vs Remote GPU Offloading Analysis).
================================================================================
"""

import os
import sys
import time
import json
import argparse
import subprocess
from typing import Dict, Any, List, Optional, Callable

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from network_client import OffloadingClient
from utils import get_ffmpeg_binary, generate_sample_video, format_time, format_bytes


def execute_local_transcode(input_file: str, output_file: str, resolution: str, bitrate: str = "3000k") -> float:
    """
    Simulates resource-constrained client execution locally (using single thread or constrained CPU).
    Measures baseline T_local.
    """
    ffmpeg_exe = get_ffmpeg_binary()
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    vf_arg = ["-vf", f"scale={resolution}"] if resolution != "original" else []
    cmd = [
        ffmpeg_exe, "-y",
        "-i", input_file,
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-b:v", bitrate,
        "-threads", "1",
        *vf_arg,
        "-c:a", "copy",
        output_file
    ]

    t_start = time.time()
    res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if res.returncode != 0:
        cmd_fallback = [ffmpeg_exe, "-y", "-i", input_file, *vf_arg, output_file]
        subprocess.run(cmd_fallback, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return time.time() - t_start


def generate_plots_from_data(data: dict, output_dir: str):
    """
    Renders two high-resolution analytical plots:
    1. Execution Time Breakdown (Local vs Remote Stacks)
    2. Speedup Factor & Network Overhead vs Resolution
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[!] Matplotlib not installed, skipping chart rendering.")
        return None, None

    os.makedirs(output_dir, exist_ok=True)
    records = data["records"]

    labels = [r["configuration"] for r in records]
    t_local = [r["t_local_sec"] for r in records]
    t_upload = [r["t_upload_sec"] for r in records]
    t_compute = [r["t_remote_compute_sec"] for r in records]
    t_download = [r["t_download_sec"] for r in records]
    t_remote_total = [r["t_remote_total_sec"] for r in records]
    speedups = [r["speedup_factor"] for r in records]
    overheads = [r["network_overhead_pct"] for r in records]

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 10

    # CHART 1: Comparative Execution Time Breakdown
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, t_local, width, label="Local Client Execution (T_local)", color="#ef4444", edgecolor="#991b1b")
    bars_up = ax.bar(x + width/2, t_upload, width, label="Network Upload (T_upload)", color="#3b82f6", edgecolor="#1d4ed8")
    bars_comp = ax.bar(x + width/2, t_compute, width, bottom=t_upload, label="Remote Engine Compute (T_remote)", color="#10b981", edgecolor="#047857")
    bottom_dl = [u + c for u, c in zip(t_upload, t_compute)]
    bars_dl = ax.bar(x + width/2, t_download, width, bottom=bottom_dl, label="Network Download (T_download)", color="#f59e0b", edgecolor="#b45309")

    for i, v in enumerate(t_local):
        ax.text(i - width/2, v + 0.1, f"{v:.2f}s", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#991b1b")
    for i, v in enumerate(t_remote_total):
        ax.text(i + width/2, v + 0.1, f"{v:.2f}s", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#065f46")

    ax.set_title("CSC-334: Local Client vs Remote Offloading Turnaround Time", fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Workload Resolution / Configuration", fontweight="bold", labelpad=8)
    ax.set_ylabel("Turnaround Time (Seconds) - Lower is Better", fontweight="bold", labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight="bold")
    ax.legend(frameon=True, facecolor="#f8fafc", loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plot1_path = os.path.join(output_dir, "benchmark_execution_time.png")
    plt.tight_layout()
    plt.savefig(plot1_path)
    plt.close()

    # CHART 2: Speedup Factor & Network Overhead Dual Plot
    fig, ax1 = plt.subplots(figsize=(9, 5.5), dpi=300)

    color_speedup = "#8b5cf6"
    color_overhead = "#f97316"

    ax1.set_title("CSC-334: Speedup Factor vs Network Transfer Overhead", fontsize=13, fontweight="bold", pad=14)
    ax1.set_xlabel("Workload Resolution / Configuration", fontweight="bold", labelpad=8)
    ax1.set_ylabel("Speedup Factor (S = T_local / T_remote)", color=color_speedup, fontweight="bold", labelpad=8)
    
    line1 = ax1.plot(labels, speedups, marker="o", markersize=8, linewidth=2.5, color=color_speedup, label="Speedup Factor (x)")
    for i, txt in enumerate(speedups):
        ax1.annotate(f"{txt:.2f}x", (labels[i], speedups[i]), textcoords="offset points", xytext=(0, 10), ha="center", fontweight="bold", color=color_speedup)
    
    ax1.tick_params(axis="y", labelcolor=color_speedup)
    ax1.set_ylim(0, max(speedups) * 1.35 if speedups else 5)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Network Overhead (%)", color=color_overhead, fontweight="bold", labelpad=8)
    line2 = ax2.plot(labels, overheads, marker="s", markersize=8, linewidth=2.5, linestyle="--", color=color_overhead, label="Network Overhead (%)")
    for i, txt in enumerate(overheads):
        ax2.annotate(f"{txt:.1f}%", (labels[i], overheads[i]), textcoords="offset points", xytext=(0, -18), ha="center", fontweight="bold", color=color_overhead)
    
    ax2.tick_params(axis="y", labelcolor=color_overhead)
    ax2.set_ylim(0, 100)

    lines = line1 + line2
    labels_legend = [l.get_label() for l in lines]
    ax1.legend(lines, labels_legend, loc="upper right", frameon=True, facecolor="#f8fafc")

    plot2_path = os.path.join(output_dir, "benchmark_speedup_overhead.png")
    plt.tight_layout()
    plt.savefig(plot2_path)
    plt.close()

    return plot1_path, plot2_path


def run_comparative_benchmark(
    worker_host: str = "127.0.0.1",
    worker_port: int = 5000,
    sample_video_path: Optional[str] = None,
    log_fn: Optional[Callable[[str], None]] = None
) -> Dict[str, Any]:
    """
    Executes comprehensive comparative benchmarking across multiple resolutions:
    - 480p (854x480)
    - 720p (1280x720)
    - 1080p (1920x1080)
    Calculates: T_local, T_remote, Speedup Factor, and Network Transfer Overhead.
    """
    def log(msg: str):
        if log_fn:
            log_fn(msg)
        print(msg)

    # Use self-contained benchmark directory inside client folder
    benchmark_dir = os.path.join(current_dir, "benchmark_results")
    os.makedirs(benchmark_dir, exist_ok=True)

    # 1. Ensure test asset exists
    if not sample_video_path or not os.path.exists(sample_video_path):
        sample_video_path = os.path.join(benchmark_dir, "bench_input_sample.mp4")
        log(f"[*] Generating standard benchmark video asset: {sample_video_path}...")
        generate_sample_video(sample_video_path, duration_sec=5, resolution=(1280, 720))

    asset_size = os.path.getsize(sample_video_path)
    log(f"[*] Benchmark Asset: {os.path.basename(sample_video_path)} ({format_bytes(asset_size)})")

    # Connect to remote worker
    client = OffloadingClient(host=worker_host, port=worker_port)
    client.connect()
    worker_info = client.perform_handshake()
    latency_ms = client.measure_ping_latency(num_samples=3)

    log(f"[*] Connected to Worker: {worker_info.get('hostname')} | Acceleration: {worker_info.get('nvenc_status')[:35]} | Ping: {latency_ms} ms")

    test_configs = [
        {"name": "480p SD", "res": "854x480", "bitrate": "1500k"},
        {"name": "720p HD", "res": "1280x720", "bitrate": "3000k"},
        {"name": "1080p FHD", "res": "1920x1080", "bitrate": "6000k"},
    ]

    benchmark_records = []
    local_scratch_dir = os.path.join(benchmark_dir, "scratch")
    os.makedirs(local_scratch_dir, exist_ok=True)

    for cfg in test_configs:
        log("-" * 65)
        log(f"[*] Running Benchmark Configuration: {cfg['name']} ({cfg['res']} @ {cfg['bitrate']})")

        # Step A: Local Execution
        local_out = os.path.join(local_scratch_dir, f"local_{cfg['res']}.mp4")
        log(f"  [1/2] Benchmarking Local Client Execution...")
        t_local = execute_local_transcode(sample_video_path, local_out, cfg["res"], cfg["bitrate"])
        log(f"        -> Local Execution Time (T_local): {t_local:.3f} s")

        # Step B: Remote Offloaded Execution
        log(f"  [2/2] Benchmarking Remote Worker Offloading...")
        remote_res = client.offload_video_transcode(
            input_file=sample_video_path,
            output_dir=local_scratch_dir,
            render_config={"resolution": cfg["res"], "bitrate": cfg["bitrate"], "preset": "fast", "mode": "auto"}
        )

        t_upload = remote_res["upload_duration_sec"]
        t_remote_compute = remote_res["remote_compute_duration_sec"]
        t_download = remote_res["download_duration_sec"]
        t_remote_total = remote_res["total_turnaround_sec"]
        t_net_overhead = remote_res["network_overhead_sec"]

        # Calculate metrics
        speedup = t_local / t_remote_total if t_remote_total > 0 else 0.0
        compute_speedup = t_local / t_remote_compute if t_remote_compute > 0 else 0.0
        overhead_pct = (t_net_overhead / t_remote_total) * 100.0 if t_remote_total > 0 else 0.0

        log(f"        -> Network Upload Time   : {t_upload:.3f} s")
        log(f"        -> Remote Compute Time   : {t_remote_compute:.3f} s ({remote_res.get('engine')})")
        log(f"        -> Network Download Time : {t_download:.3f} s")
        log(f"        -> Total Turnaround Time : {t_remote_total:.3f} s")
        log(f"        -> Speedup Factor (S)    : {speedup:.2f}x")
        log(f"        -> Compute Speedup       : {compute_speedup:.2f}x")
        log(f"        -> Network Overhead      : {overhead_pct:.1f}% ({t_net_overhead:.3f} s)")

        benchmark_records.append({
            "configuration": cfg["name"],
            "resolution": cfg["res"],
            "bitrate": cfg["bitrate"],
            "t_local_sec": round(t_local, 3),
            "t_upload_sec": round(t_upload, 3),
            "t_remote_compute_sec": round(t_remote_compute, 3),
            "t_download_sec": round(t_download, 3),
            "t_remote_total_sec": round(t_remote_total, 3),
            "network_overhead_sec": round(t_net_overhead, 3),
            "network_overhead_pct": round(overhead_pct, 1),
            "speedup_factor": round(speedup, 2),
            "compute_speedup": round(compute_speedup, 2),
            "checksum_verified": remote_res["checksum_verified"],
            "engine": remote_res.get("engine", "Remote Node")
        })

    client.disconnect()

    avg_speedup = sum(r["speedup_factor"] for r in benchmark_records) / len(benchmark_records)
    avg_overhead = sum(r["network_overhead_pct"] for r in benchmark_records) / len(benchmark_records)

    summary_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "worker_hostname": worker_info.get("hostname"),
        "worker_acceleration": worker_info.get("nvenc_status"),
        "latency_rtt_ms": latency_ms,
        "average_speedup": round(avg_speedup, 2),
        "average_network_overhead_pct": round(avg_overhead, 1),
        "records": benchmark_records
    }

    # Save to JSON
    json_path = os.path.join(benchmark_dir, "benchmark_data.json")
    with open(json_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    log(f"[*] Benchmark data saved to: {json_path}")

    # Generate charts
    try:
        generate_plots_from_data(summary_data, output_dir=benchmark_dir)
        log(f"[*] High-resolution comparison plots saved to {benchmark_dir}")
    except Exception as e:
        log(f"[!] Plot generation warning: {e}")

    return summary_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSC-334: Local vs Remote GPU Offloading Benchmark")
    parser.add_argument("--host", default="127.0.0.1", help="Remote Worker IP")
    parser.add_argument("--port", type=int, default=5000, help="Remote Worker Port")
    args = parser.parse_args()

    run_comparative_benchmark(worker_host=args.host, worker_port=args.port)
