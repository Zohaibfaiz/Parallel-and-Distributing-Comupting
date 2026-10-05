#!/usr/bin/env python3
"""Performance benchmark (Task 5): local render vs. remote GPU offload.

Example (client laptop, worker already running):

    python benchmark/benchmark.py --host 192.168.1.1 --token mysecret --matrix full --runs 3 \
        --client-desc "Laptop, i5-8250U, MX150 500MB" --server-desc "Desktop, Ryzen 5, GTX 1650 4GB"

For every test clip it measures
  * local render time on this machine (baseline),
  * remote offload time, split into hash / upload / queue / render / download / verify,
  * CPU utilisation of the client in both cases and CPU/GPU utilisation of the worker,
then writes results.json / results.csv / charts and docs-ready REPORT.md
(see benchmark/report.py).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from client.local_render import local_nvenc_available, render_local  # noqa: E402
from client.net_client import OffloadClient  # noqa: E402
from common.ffmpeg_utils import gen_test_video, normalize_settings, probe, query_gpus  # noqa: E402
from common.protocol import DEFAULT_PORT, DEFAULT_TOKEN  # noqa: E402

try:
    import psutil
except ImportError:
    psutil = None

MATRICES = {
    "quick": [("360p-10s", 640, 360, 10), ("720p-15s", 1280, 720, 15), ("1080p-20s", 1920, 1080, 20)],
    "full": [("360p-30s", 640, 360, 30), ("720p-30s", 1280, 720, 30), ("1080p-30s", 1920, 1080, 30),
             ("1080p-90s", 1920, 1080, 90), ("1440p-30s", 2560, 1440, 30), ("2160p-20s", 3840, 2160, 20)],
}


def collect_env(args, chk: dict, bw: dict | None) -> dict:
    mem = round(psutil.virtual_memory().total / 1024 ** 3, 1) if psutil else None
    srv = chk.get("server", {})
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "client": {"description": args.client_desc, "platform": platform.platform(),
                   "cpu": platform.processor() or platform.machine(),
                   "cpu_threads": os.cpu_count(), "ram_gb": mem, "gpus": query_gpus(),
                   "local_nvenc": local_nvenc_available(), "python": platform.python_version()},
        "server": {"description": args.server_desc, "hostname": srv.get("server"), "gpus": srv.get("gpus", []),
                   "nvenc": srv.get("nvenc"), "ffmpeg": srv.get("ffmpeg"), "hwdecode": srv.get("hwdecode"),
                   "torch": srv.get("torch")},
        "link": {"host": args.host, "rtt_avg_ms": chk.get("rtt_avg_ms"), "rtt_min_ms": chk.get("rtt_min_ms"),
                 "rtt_max_ms": chk.get("rtt_max_ms"), "jitter_ms": chk.get("jitter_ms"),
                 "iperf_like_up_mbps": (bw or {}).get("upload_mbps"),
                 "iperf_like_down_mbps": (bw or {}).get("download_mbps"), "description": args.link_desc},
        "settings": {"resolution": args.res, "bitrate": args.bitrate, "preset": args.preset, "codec": args.codec},
        "runs": args.runs,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Local vs remote GPU benchmark")
    p.add_argument("--host", default="192.168.1.1")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--token", default=os.environ.get("OFFLOAD_TOKEN", DEFAULT_TOKEN))
    p.add_argument("--matrix", choices=["quick", "full", "none"], default="quick",
                   help="synthetic clips to generate (none = only --inputs)")
    p.add_argument("--inputs", nargs="*", default=[], help="your own video files to benchmark as well")
    p.add_argument("--runs", type=int, default=3, help="repetitions per clip (averaged)")
    p.add_argument("--res", default="720", help="output resolution (height) used on BOTH sides")
    p.add_argument("--bitrate", default="5M")
    p.add_argument("--preset", default="p4")
    p.add_argument("--codec", default="h264")
    p.add_argument("--local-engine", choices=["auto", "cpu", "gpu"], default="auto",
                   help="auto = use the laptop's own NVENC if it has one, else CPU")
    p.add_argument("--skip-local", action="store_true", help="only measure the remote side")
    p.add_argument("--workdir", default=str(ROOT / "benchmark" / "work"))
    p.add_argument("--out", default=str(ROOT / "benchmark" / "results"))
    p.add_argument("--client-desc", default="Client laptop (describe CPU / GPU / RAM)")
    p.add_argument("--server-desc", default="Worker PC (describe CPU / GPU / RAM)")
    p.add_argument("--link-desc", default="Direct CAT6 Ethernet, static IPs 192.168.1.1/24")
    a = p.parse_args(argv)

    settings = normalize_settings({"resolution": a.res, "bitrate": a.bitrate, "preset": a.preset,
                                   "codec": a.codec})
    work = Path(a.workdir)
    work.mkdir(parents=True, exist_ok=True)
    out_dir = Path(a.out) / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    client = OffloadClient(a.host, a.port, a.token, log=lambda m: None)
    print(f"→ checking worker {a.host}:{a.port} ...")
    chk = client.check_worker(10)
    if not chk["reachable"]:
        print(f"✖ worker not reachable: {chk['error']}")
        return 1
    print(f"✔ RTT {chk['rtt_avg_ms']} ms | NVENC on worker: {chk['server'].get('nvenc')}")
    if not chk["server"].get("nvenc"):
        print("⚠ the worker has NO working NVENC - 'remote' numbers will be CPU numbers!")
    bw = client.bandwidth_test(64)
    print(f"✔ link throughput: ↑ {bw['upload_mbps']} / ↓ {bw['download_mbps']} Mbit/s")

    # ---- test inputs
    cases = []
    for name, w, h, dur in MATRICES.get(a.matrix, []):
        path = work / f"{name}.mp4"
        if not path.exists():
            print(f"→ generating test clip {name} ({w}x{h}, {dur}s) ...")
            gen_test_video(str(path), w, h, dur)
        cases.append((name, str(path)))
    for f in a.inputs:
        cases.append((Path(f).stem, f))
    if not cases:
        print("✖ nothing to benchmark (use --matrix or --inputs)")
        return 1

    env = collect_env(a, chk, bw)
    rows = []
    total = len(cases) * a.runs
    n = 0
    for name, path in cases:
        info = probe(path)
        for run in range(1, a.runs + 1):
            n += 1
            print(f"\n[{n}/{total}] {name}  ({info['width']}x{info['height']}, {info['duration']:.0f}s, "
                  f"{os.path.getsize(path) / 1e6:.1f} MB)  run {run}/{a.runs}")
            row = {"case": name, "run": run, "width": info["width"], "height": info["height"],
                   "duration_s": round(info["duration"], 2), "fps": info["fps"],
                   "input_mb": round(os.path.getsize(path) / 1e6, 3)}
            if not a.skip_local:
                lo = str(work / f"local_{name}.mp4")
                r = render_local(path, lo, settings, a.local_engine)
                row.update(local_s=round(r["render_s"], 3), local_engine=r["engine"],
                           local_cpu_avg=r["resource"].get("cpu_avg"), local_gpu_avg=r["resource"].get("gpu_util_avg"))
                print(f"   local  : {r['render_s']:.1f}s ({r['engine']})")
            ro = str(work / f"remote_{name}.mp4")
            r = client.run_transcode(path, settings, ro)
            sr, cr = r["server_resource"] or {}, r["client_resource"] or {}
            row.update(remote_total_s=round(r["total_s"], 3), hash_s=round(r["hash_s"], 3),
                       upload_s=round(r["upload_s"], 3), queue_s=round(r["queue_wait_s"], 3),
                       render_s=round(r["render_s"], 3), download_s=round(r["download_s"], 3),
                       verify_s=round(r["verify_s"], 3), upload_mbps=round(r["upload_mbps"] or 0, 1),
                       download_mbps=round(r["download_mbps"] or 0, 1), output_mb=round(r["output_size"] / 1e6, 3),
                       remote_engine=r["engine"], server_gpu_avg=sr.get("gpu_util_avg"),
                       server_gpu_max=sr.get("gpu_util_max"), server_vram_mb=sr.get("gpu_mem_max_mb"),
                       server_cpu_avg=sr.get("cpu_avg"), server_gpu_power_w=sr.get("gpu_power_avg_w"),
                       client_cpu_avg_offload=cr.get("cpu_avg"), retries=r["retries"], avg_fps=r.get("avg_fps"))
            print(f"   remote : {r['total_s']:.1f}s total = hash {r['hash_s']:.1f} + up {r['upload_s']:.1f} + "
                  f"render {r['render_s']:.1f} + down {r['download_s']:.1f} + verify {r['verify_s']:.1f} "
                  f"({r['engine']})")
            if "local_s" in row:
                print(f"   speedup: x{row['local_s'] / row['remote_total_s']:.2f}")
            rows.append(row)
            (out_dir / "results.json").write_text(json.dumps({"meta": env, "rows": rows}, indent=2))

    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with open(out_dir / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    from benchmark.report import generate  # noqa: E402
    md = generate(out_dir / "results.json", out_dir)
    print(f"\n✔ results : {out_dir / 'results.json'}\n✔ CSV     : {out_dir / 'results.csv'}\n✔ report  : {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
