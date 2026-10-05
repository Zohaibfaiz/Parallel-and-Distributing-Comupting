#!/usr/bin/env python3
"""Headless command line client (same engine as the GUI).

    python client/cli.py check     --host 192.168.1.1
    python client/cli.py bandwidth --host 192.168.1.1
    python client/cli.py transcode --host 192.168.1.1 input.mp4 --resolution 1080 --bitrate 8M --preset p5
    python client/cli.py local     input.mp4 --resolution 1080 --bitrate 8M --preset p5
    python client/cli.py torch     --host 192.168.1.1 --n 4096 --iters 100
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from client.local_render import render_local  # noqa: E402
from client.net_client import JobFailed, OffloadClient, TransferError  # noqa: E402
from common.ffmpeg_utils import CODECS, PRESETS, RESOLUTIONS  # noqa: E402
from common.protocol import DEFAULT_PORT, DEFAULT_TOKEN, AuthError, Cancelled, RemoteError  # noqa: E402


def _bar(p: dict) -> None:
    pct = p["overall"]
    filled = int(pct / 100 * 30)
    sys.stdout.write(f"\r[{'#' * filled}{'.' * (30 - filled)}] {pct:5.1f}%  {p['stage']:<11} {p.get('text', '')[:60]:<60}")
    sys.stdout.flush()
    if p["stage"] == "done":
        sys.stdout.write("\n")


def _log(msg: str) -> None:
    sys.stdout.write("\r" + " " * 110 + "\r" + msg + "\n")
    sys.stdout.flush()


def _client(a) -> OffloadClient:
    return OffloadClient(a.host, a.port, a.token, log=_log, on_progress=None if a.quiet else _bar)


def _settings(a) -> dict:
    return {"resolution": a.resolution, "bitrate": a.bitrate, "preset": a.preset, "codec": a.codec,
            "engine": a.engine}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="GPU offload client (CLI)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def net(sp):
        sp.add_argument("--host", default="192.168.1.1")
        sp.add_argument("--port", type=int, default=DEFAULT_PORT)
        sp.add_argument("--token", default=os.environ.get("OFFLOAD_TOKEN", DEFAULT_TOKEN))
        sp.add_argument("--quiet", action="store_true", help="no progress bar")

    def render_opts(sp):
        sp.add_argument("--resolution", default="original", choices=RESOLUTIONS)
        sp.add_argument("--bitrate", default="5M")
        sp.add_argument("--preset", default="p4", choices=PRESETS)
        sp.add_argument("--codec", default="h264", choices=CODECS)
        sp.add_argument("--engine", default="auto", choices=["auto", "gpu", "cpu"])

    sp = sub.add_parser("check", help="handshake + latency ping test"); net(sp)
    sp.add_argument("--pings", type=int, default=10)
    sp = sub.add_parser("bandwidth", help="TCP throughput test"); net(sp)
    sp.add_argument("--mb", type=int, default=64)
    sp = sub.add_parser("transcode", help="offload a video transcode to the GPU worker"); net(sp); render_opts(sp)
    sp.add_argument("input"); sp.add_argument("-o", "--output")
    sp = sub.add_parser("local", help="render locally (baseline)"); render_opts(sp)
    sp.add_argument("input"); sp.add_argument("-o", "--output"); sp.add_argument("--quiet", action="store_true")
    sp = sub.add_parser("torch", help="run a PyTorch CUDA matmul benchmark on the worker"); net(sp)
    sp.add_argument("--n", type=int, default=4096); sp.add_argument("--iters", type=int, default=50)
    sp.add_argument("--dtype", default="fp32", choices=["fp32", "fp16"])
    a = p.parse_args(argv)

    try:
        if a.cmd == "check":
            r = _client(a).check_worker(a.pings)
            if not r["reachable"]:
                print(f"✖ worker NOT reachable: {r['error']}" + (f"\n  hint: {r['hint']}" if r["hint"] else ""))
                return 1
            s = r["server"]
            print(f"✔ worker {s['server']} reachable at {a.host}:{a.port}")
            print(f"  connect {r['connect_ms']} ms | RTT avg {r['rtt_avg_ms']} ms (min {r['rtt_min_ms']}, "
                  f"max {r['rtt_max_ms']}, jitter {r['jitter_ms']})")
            print(f"  NVENC: {s['nvenc']} | GPUs: {', '.join(g['name'] for g in s['gpus']) or 'none'} | "
                  f"queue: {s['queue_length']} | free disk: {s['free_disk_gb']} GB")
            if r["hint"]:
                print(f"  ⚠ {r['hint']}")
        elif a.cmd == "bandwidth":
            r = _client(a).bandwidth_test(a.mb)
            print(f"upload   : {r['upload_mbps']} Mbit/s\ndownload : {r['download_mbps']} Mbit/s")
        elif a.cmd == "transcode":
            r = _client(a).run_transcode(a.input, _settings(a), a.output)
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
                              if k not in ("server_resource", "client_resource")}, indent=2))
        elif a.cmd == "local":
            out = a.output or str(Path(a.input).with_suffix("")) + "_local.mp4"
            t = time.perf_counter()
            r = render_local(a.input, out, _settings(a), a.engine,
                             None if a.quiet else (lambda pr: sys.stdout.write(
                                 f"\r{pr['percent']:5.1f}%  {pr['fps']:.0f} fps  {pr['speed']:.2f}x   ")))
            print(f"\n✔ local render ({r['engine']}) took {r['render_s']:.1f}s -> {out}")
        elif a.cmd == "torch":
            r = _client(a).run_torch({"n": a.n, "iters": a.iters, "dtype": a.dtype})
            print(f"\n✔ {r['engine']}: {r['tflops']} TFLOPS in {r['render_s']}s")
        return 0
    except AuthError as exc:
        print(f"\n✖ authentication failed: {exc}")
    except (TransferError, JobFailed, RemoteError) as exc:
        print(f"\n✖ {exc}")
    except Cancelled:
        print("\n✖ cancelled")
    except KeyboardInterrupt:
        print("\n✖ interrupted")
    return 1


if __name__ == "__main__":
    sys.exit(main())
