"""GPU execution engine of the worker node (Task 2).

* Video transcoding through FFmpeg + NVENC (``-c:v h264_nvenc`` / ``hevc_nvenc``),
  optionally with CUDA hardware decoding (``-hwaccel cuda``).  If CUDA decoding
  fails for an odd input the encode is transparently retried with software
  decoding (NVENC still does the encoding).
* PyTorch CUDA tensor workload (matrix-multiplication benchmark).
* Graceful CPU fallback (libx264/libx265) when NVENC is unavailable, so the
  whole system can still be developed/tested on machines without an NVIDIA GPU.
  The log always states which engine was really used.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.ffmpeg_utils import (  # noqa: E402
    EncodeCancelled, EncodeError, ResourceSampler, build_encode_cmd, detect_nvenc,
    encoder_name, ffmpeg_version, find_binary, normalize_settings, probe, query_gpus, run_encode,
)
from common.protocol import Cancelled  # noqa: E402


class EngineError(Exception):
    pass


def normalize_torch_settings(raw: dict | None) -> dict:
    raw = raw or {}
    try:
        n = int(raw.get("n", 4096))
        iters = int(raw.get("iters", 50))
    except (TypeError, ValueError) as exc:
        raise ValueError("n and iters must be integers") from exc
    dtype = str(raw.get("dtype", "fp32")).lower()
    if not 256 <= n <= 16384:
        raise ValueError("n must be between 256 and 16384")
    if not 1 <= iters <= 5000:
        raise ValueError("iters must be between 1 and 5000")
    if dtype not in ("fp32", "fp16"):
        raise ValueError("dtype must be fp32 or fp16")
    return {"n": n, "iters": iters, "dtype": dtype}


def _probe_torch() -> dict:
    try:
        import torch  # noqa: WPS433
        info = {"available": True, "version": torch.__version__, "cuda": bool(torch.cuda.is_available())}
        if info["cuda"]:
            info["device"] = torch.cuda.get_device_name(0)
        return info
    except Exception:  # ImportError or a broken CUDA install
        return {"available": False, "cuda": False}


class GpuEngine:
    def __init__(self, mode: str = "auto", hwdecode: bool = True, ffmpeg: str | None = None):
        self.mode = mode
        self.hwdecode = hwdecode
        self.ffmpeg = ffmpeg or find_binary("ffmpeg")
        self.ffprobe = find_binary("ffprobe")
        if not self.ffmpeg or not self.ffprobe:
            raise EngineError("ffmpeg/ffprobe not found in PATH - install FFmpeg first")
        self.ffmpeg_version = ffmpeg_version(self.ffmpeg)
        self.nvenc = False if mode == "cpu" else detect_nvenc(self.ffmpeg)
        if mode == "gpu" and not self.nvenc:
            raise EngineError("--engine gpu requested but a test NVENC encode failed "
                              "(check NVIDIA driver, GPU and that your FFmpeg build has NVENC)")
        self.gpus = query_gpus()
        self.torch = _probe_torch()

    def describe(self) -> dict:
        return {
            "engine_mode": self.mode,
            "nvenc": self.nvenc,
            "hwdecode": self.hwdecode and self.nvenc,
            "gpus": self.gpus,
            "ffmpeg": self.ffmpeg_version,
            "torch": self.torch,
        }

    # ------------------------------------------------------------------ dispatch
    def run(self, job) -> dict:
        try:
            if job.kind == "transcode":
                return self._transcode(job)
            if job.kind == "torch":
                return self._torch(job)
            raise EngineError(f"unknown job kind {job.kind!r}")
        except EncodeCancelled as exc:
            raise Cancelled() from exc

    # ------------------------------------------------------------------ video
    def _transcode(self, job) -> dict:
        s = normalize_settings(job.settings)
        if s["engine"] == "gpu" and not self.nvenc:
            raise EngineError("GPU engine requested but NVENC is not available on this worker")
        use_gpu = self.nvenc and s["engine"] != "cpu"
        if not use_gpu and s["engine"] == "auto" and self.mode != "cpu":
            job.log("WARNING: NVENC unavailable - falling back to CPU encoder (libx264/libx265)")
        info = probe(str(job.input_path), self.ffprobe)
        enc = encoder_name(s, use_gpu)
        job.log(f"Input : {info['width']}x{info['height']} {info['vcodec']} "
                f"{info['fps']:.2f} fps {info['duration']:.1f}s")
        job.log(f"Engine: {enc} ({'GPU / NVENC' if use_gpu else 'CPU'}) preset={s['preset']} "
                f"bitrate={s['bitrate']} resolution={s['resolution']}")

        last_decile = {"v": -1}

        def on_progress(p: dict) -> None:
            job.set_progress(stage="rendering", **p)
            dec = int(p["percent"] // 10)
            if dec > last_decile["v"] and p["percent"] < 100:
                last_decile["v"] = dec
                job.log(f"Rendering {p['percent']:.0f}%  ({p['fps']:.0f} fps, {p['speed']:.2f}x)")

        attempts = [True, False] if (use_gpu and self.hwdecode) else [False]
        sampler = ResourceSampler(gpu=True).start()
        t0 = time.perf_counter()
        used_hwdecode = False
        frames = 0
        try:
            for i, hw in enumerate(attempts):
                cmd = build_encode_cmd(str(job.input_path), str(job.output_path), s, use_gpu, hw, self.ffmpeg)
                if hw:
                    job.log("Decoding on GPU (CUDA) + encoding on GPU (NVENC)")
                res = run_encode(cmd, info["duration"], on_progress, job.cancel_event)
                frames = res["frames"]
                if res["rc"] == 0:
                    used_hwdecode = hw
                    break
                err = " | ".join(res["tail"][-3:])
                if i + 1 < len(attempts):
                    job.log(f"CUDA decode path failed ({err[:160]}) - retrying with software decode")
                    continue
                raise EngineError(f"ffmpeg failed (exit {res['rc']}): {err[:300]}")
        finally:
            render_s = time.perf_counter() - t0
            resource = sampler.stop()

        if not job.output_path.exists() or job.output_path.stat().st_size == 0:
            raise EngineError("ffmpeg produced no output")
        job.set_progress(stage="rendering", percent=100.0, fps=0.0, speed=0.0, eta_s=0.0)
        out_info = probe(str(job.output_path), self.ffprobe)
        job.log(f"Done in {render_s:.1f}s -> {out_info['width']}x{out_info['height']} "
                f"{out_info['size'] / 1e6:.1f} MB")
        return {
            "engine": enc, "gpu": use_gpu, "hwdecode": used_hwdecode,
            "render_s": round(render_s, 3), "frames": frames,
            "avg_fps": round(frames / render_s, 2) if render_s > 0 else 0.0,
            "input": info, "output": out_info, "resource": resource,
        }

    # ------------------------------------------------------------------ tensors
    def _torch(self, job) -> dict:
        if not self.torch["available"]:
            raise EngineError("PyTorch is not installed on the worker (pip install torch)")
        import torch  # noqa: WPS433

        s = normalize_torch_settings(job.settings)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        dt = torch.float16 if (s["dtype"] == "fp16" and dev == "cuda") else torch.float32
        job.log(f"PyTorch {torch.__version__} on {dev}"
                + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else " (CUDA not available!)")
                + f" - {s['n']}x{s['n']} matmul x{s['iters']} ({dt})")
        a = torch.randn(s["n"], s["n"], device=dev, dtype=dt)
        b = torch.randn(s["n"], s["n"], device=dev, dtype=dt)
        for _ in range(3):  # warm-up
            a @ b
        if dev == "cuda":
            torch.cuda.synchronize()
        sampler = ResourceSampler(gpu=True).start()
        t0 = time.perf_counter()
        try:
            for i in range(s["iters"]):
                if job.cancel_event.is_set():
                    raise Cancelled()
                a @ b
                if dev == "cuda":
                    torch.cuda.synchronize()
                pct = (i + 1) / s["iters"] * 100
                job.set_progress(stage="rendering", percent=round(pct, 2), fps=0.0, speed=0.0, eta_s=None)
        finally:
            seconds = time.perf_counter() - t0
            resource = sampler.stop()
        tflops = 2 * s["n"] ** 3 * s["iters"] / seconds / 1e12
        job.log(f"{tflops:.2f} TFLOPS in {seconds:.2f}s")
        return {"engine": f"torch-{dev}", "gpu": dev == "cuda", "render_s": round(seconds, 3),
                "tflops": round(tflops, 3), "device": dev, "torch": torch.__version__,
                "settings": s, "resource": resource}
