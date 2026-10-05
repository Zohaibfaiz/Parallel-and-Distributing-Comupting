"""FFmpeg helpers shared by the worker daemon (NVENC) and the client (local render).

* settings validation (never pass raw user strings to a shell: we always use
  argument lists and a strict whitelist),
* ffprobe wrapper,
* NVENC detection,
* command builder for h264_nvenc / hevc_nvenc with a libx264 / libx265 fallback,
* progress parser (``-progress pipe:1``),
* resource sampler (CPU via psutil, GPU via nvidia-smi),
* synthetic test-video generator for benchmarks / tests.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
from collections import deque
from typing import Callable, Optional

try:
    import psutil
except ImportError:  # psutil is optional but recommended
    psutil = None

RESOLUTIONS = ("original", "2160", "1440", "1080", "720", "480", "360", "240")
PRESETS = ("p1", "p2", "p3", "p4", "p5", "p6", "p7")  # p1 fastest ... p7 best quality
CODECS = ("h264", "hevc")
ENGINES = ("auto", "gpu", "cpu")
BITRATE_RE = re.compile(r"^(\d+(?:\.\d+)?)([kKmM])$")
CPU_PRESET_MAP = {
    "p1": "ultrafast", "p2": "superfast", "p3": "veryfast", "p4": "faster",
    "p5": "fast", "p6": "medium", "p7": "slow",
}
DEFAULT_SETTINGS = {
    "resolution": "original",
    "bitrate": "5M",
    "preset": "p4",
    "codec": "h264",
    "audio_bitrate": "128k",
    "engine": "auto",
}
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class EncodeError(Exception):
    pass


class EncodeCancelled(Exception):
    pass


# --------------------------------------------------------------------------- settings
def normalize_settings(raw: dict | None) -> dict:
    """Validate user-supplied render settings against a strict whitelist."""
    s = dict(DEFAULT_SETTINGS)
    s.update({k: v for k, v in (raw or {}).items() if k in DEFAULT_SETTINGS and v is not None})
    s["resolution"] = str(s["resolution"])
    s["codec"] = str(s["codec"]).lower()
    s["preset"] = str(s["preset"]).lower()
    s["engine"] = str(s["engine"]).lower()
    s["bitrate"] = str(s["bitrate"]).strip()
    s["audio_bitrate"] = str(s["audio_bitrate"]).strip()
    if s["resolution"] not in RESOLUTIONS:
        raise ValueError(f"resolution must be one of {RESOLUTIONS}")
    if s["codec"] not in CODECS:
        raise ValueError(f"codec must be one of {CODECS}")
    if s["preset"] not in PRESETS:
        raise ValueError(f"preset must be one of {PRESETS}")
    if s["engine"] not in ENGINES:
        raise ValueError(f"engine must be one of {ENGINES}")
    for key in ("bitrate", "audio_bitrate"):
        m = BITRATE_RE.match(s[key])
        if not m or float(m.group(1)) <= 0:
            raise ValueError(f"{key} must look like 5M or 800k")
    return s


def _scale_bitrate(br: str, factor: float) -> str:
    m = BITRATE_RE.match(br)
    val = float(m.group(1)) * factor
    unit = m.group(2)
    return f"{int(round(val))}{unit}" if unit in "kK" else f"{val:g}{unit}"


def bitrate_to_bps(br: str) -> float:
    m = BITRATE_RE.match(br)
    mult = 1e3 if m.group(2) in "kK" else 1e6
    return float(m.group(1)) * mult


# --------------------------------------------------------------------------- binaries / probing
def find_binary(name: str) -> Optional[str]:
    return shutil.which(name)


def ffmpeg_version(ffmpeg: str = "ffmpeg") -> str:
    try:
        out = subprocess.run([ffmpeg, "-version"], capture_output=True, text=True, timeout=10,
                             creationflags=_NO_WINDOW).stdout
        return out.splitlines()[0] if out else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def probe(path: str, ffprobe: str = "ffprobe") -> dict:
    try:
        proc = subprocess.run(
            [ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path],
            capture_output=True, text=True, timeout=60, creationflags=_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise EncodeError(f"ffprobe failed: {exc}") from exc
    if proc.returncode != 0:
        raise EncodeError(f"ffprobe could not read the file: {proc.stderr.strip()[:200]}")
    data = json.loads(proc.stdout or "{}")
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    if video is None:
        raise EncodeError("input contains no video stream")
    fmt = data.get("format", {})

    def _f(v, default=0.0):
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    fps = 0.0
    try:
        num, den = str(video.get("r_frame_rate", "0/1")).split("/")
        fps = float(num) / float(den) if float(den) else 0.0
    except (ValueError, ZeroDivisionError):
        pass
    duration = _f(fmt.get("duration")) or _f(video.get("duration"))
    return {
        "duration": duration,
        "width": int(video.get("width", 0)),
        "height": int(video.get("height", 0)),
        "fps": round(fps, 3),
        "vcodec": video.get("codec_name", "?"),
        "bit_rate": int(_f(fmt.get("bit_rate"))),
        "size": int(_f(fmt.get("size"))) or (os.path.getsize(path) if os.path.exists(path) else 0),
    }


def detect_nvenc(ffmpeg: str = "ffmpeg") -> bool:
    """True only if a real 1-frame NVENC encode succeeds (driver + GPU + ffmpeg build)."""
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
           "color=c=black:s=256x256:r=10:d=0.2", "-c:v", "h264_nvenc", "-f", "null", "-"]
    try:
        return subprocess.run(cmd, capture_output=True, timeout=30, creationflags=_NO_WINDOW).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def query_gpus() -> list[dict]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, creationflags=_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            gpus.append({"name": parts[0], "memory_mb": parts[1], "driver": parts[2]})
    return gpus


# --------------------------------------------------------------------------- command builder
def encoder_name(settings: dict, use_gpu: bool) -> str:
    if use_gpu:
        return "h264_nvenc" if settings["codec"] == "h264" else "hevc_nvenc"
    return "libx264" if settings["codec"] == "h264" else "libx265"


def build_encode_cmd(src: str, dst: str, settings: dict, use_gpu: bool,
                     hwdecode: bool = False, ffmpeg: str = "ffmpeg") -> list[str]:
    s = settings
    br = s["bitrate"]
    cmd = [ffmpeg, "-hide_banner", "-nostdin", "-y", "-loglevel", "warning",
           "-progress", "pipe:1", "-nostats"]
    if use_gpu and hwdecode:
        cmd += ["-hwaccel", "cuda"]
    cmd += ["-i", src, "-map", "0:v:0", "-map", "0:a:0?"]
    if s["resolution"] != "original":
        cmd += ["-vf", f"scale=-2:'min({int(s['resolution'])},ih)'"]
    cmd += ["-pix_fmt", "yuv420p"]
    cmd += ["-c:v", encoder_name(s, use_gpu)]
    if use_gpu:
        cmd += ["-preset", s["preset"], "-rc", "vbr"]
    else:
        cmd += ["-preset", CPU_PRESET_MAP[s["preset"]]]
        if s["codec"] == "hevc":
            cmd += ["-x265-params", "log-level=error"]
    cmd += ["-b:v", br, "-maxrate", _scale_bitrate(br, 1.5), "-bufsize", _scale_bitrate(br, 2)]
    if s["codec"] == "hevc":
        cmd += ["-tag:v", "hvc1"]
    cmd += ["-c:a", "aac", "-b:a", s["audio_bitrate"], "-movflags", "+faststart", dst]
    return cmd


# --------------------------------------------------------------------------- running ffmpeg
def run_encode(cmd: list[str], duration: float,
               on_progress: Optional[Callable[[dict], None]] = None,
               cancel_event: Optional[threading.Event] = None) -> dict:
    """Run ffmpeg, parse ``-progress`` output, honour cancellation.

    Returns {"rc", "tail", "frames"}.  Raises EncodeCancelled if cancelled.
    """
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                encoding="utf-8", errors="replace", bufsize=1,
                                creationflags=_NO_WINDOW)
    except OSError as exc:
        raise EncodeError(f"cannot start ffmpeg: {exc}") from exc

    tail: deque[str] = deque(maxlen=40)
    cancelled = threading.Event()

    def _read_err():
        for line in proc.stderr:
            line = line.rstrip()
            if line:
                tail.append(line)

    def _watch_cancel():
        while proc.poll() is None:
            if cancel_event.wait(0.3):
                cancelled.set()
                proc.terminate()
                return

    t_err = threading.Thread(target=_read_err, daemon=True)
    t_err.start()
    if cancel_event is not None:
        threading.Thread(target=_watch_cancel, daemon=True).start()

    block: dict[str, str] = {}
    frames = 0
    for raw in proc.stdout:
        key, _, val = raw.strip().partition("=")
        block[key] = val
        if key != "progress":
            continue
        try:
            frames = int(block.get("frame", frames))
        except ValueError:
            pass
        out_us = None
        for k in ("out_time_us", "out_time_ms"):  # out_time_ms is (confusingly) microseconds too
            try:
                out_us = int(block[k])
                break
            except (KeyError, ValueError):
                continue
        t = (out_us or 0) / 1e6
        percent = min(99.9, t / duration * 100) if duration > 0 else 0.0
        if val == "end":
            percent = 100.0
        try:
            fps = float(block.get("fps", 0) or 0)
        except ValueError:
            fps = 0.0
        speed = 0.0
        sm = re.match(r"^\s*([\d.]+)x", block.get("speed", ""))
        if sm:
            speed = float(sm.group(1))
        eta = (duration - t) / speed if speed > 0 and duration > 0 else None
        if on_progress:
            on_progress({"percent": round(percent, 2), "fps": fps, "speed": speed,
                         "out_time_s": round(t, 2), "eta_s": None if eta is None else round(max(eta, 0), 1),
                         "frames": frames})
        block = {}

    rc = proc.wait()
    t_err.join(timeout=2)
    for stream in (proc.stdout, proc.stderr):
        try:
            stream.close()
        except OSError:
            pass
    if cancelled.is_set():
        raise EncodeCancelled()
    return {"rc": rc, "tail": list(tail), "frames": frames}


# --------------------------------------------------------------------------- resource sampling
class ResourceSampler:
    """Samples system CPU % (psutil) and NVIDIA GPU stats (nvidia-smi) in a thread."""

    def __init__(self, interval: float = 1.0, gpu: bool = True):
        self.interval = interval
        self._gpu_enabled = gpu and shutil.which("nvidia-smi") is not None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.cpu: list[float] = []
        self.gpu_util: list[float] = []
        self.gpu_mem: list[float] = []
        self.gpu_temp: list[float] = []
        self.gpu_power: list[float] = []

    def start(self) -> "ResourceSampler":
        if psutil:
            psutil.cpu_percent(None)  # prime
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _sample(self) -> None:
        if psutil:
            self.cpu.append(psutil.cpu_percent(None))
        if self._gpu_enabled:
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=4, creationflags=_NO_WINDOW,
                )
                if out.returncode != 0:
                    self._gpu_enabled = False
                    return
                best = None
                for line in out.stdout.strip().splitlines():
                    vals = []
                    for p in line.split(","):
                        try:
                            vals.append(float(p.strip()))
                        except ValueError:
                            vals.append(float("nan"))
                    if len(vals) >= 4 and (best is None or vals[0] > best[0]):
                        best = vals
                if best:
                    self.gpu_util.append(best[0])
                    self.gpu_mem.append(best[1])
                    self.gpu_temp.append(best[2])
                    if best[3] == best[3]:  # not NaN
                        self.gpu_power.append(best[3])
            except (OSError, subprocess.SubprocessError):
                self._gpu_enabled = False

    def _loop(self) -> None:
        wait = 0.5
        while not self._stop.wait(wait):
            self._sample()
            wait = self.interval

    def stop(self) -> dict:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        if not self.cpu and not self.gpu_util:
            self._sample()

        def avg(v):
            return round(sum(v) / len(v), 1) if v else None

        def mx(v):
            return round(max(v), 1) if v else None

        return {
            "cpu_avg": avg(self.cpu), "cpu_max": mx(self.cpu),
            "gpu_util_avg": avg(self.gpu_util), "gpu_util_max": mx(self.gpu_util),
            "gpu_mem_max_mb": mx(self.gpu_mem), "gpu_temp_max_c": mx(self.gpu_temp),
            "gpu_power_avg_w": avg(self.gpu_power), "samples": max(len(self.cpu), len(self.gpu_util)),
        }


# --------------------------------------------------------------------------- test media
def gen_test_video(path: str, width: int, height: int, duration: int, fps: int = 30,
                   ffmpeg: str = "ffmpeg") -> str:
    """Synthetic but incompressible-ish clip (noise) with realistic source bitrate + audio."""
    bps = int(10_000_000 * (width * height) / (1920 * 1080))
    bps = max(bps, 800_000)
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", f"testsrc2=size={width}x{height}:rate={fps}:duration={duration}",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
        "-vf", "noise=alls=25:allf=t+u,format=yuv420p",
        "-c:v", "libx264", "-preset", "ultrafast", "-b:v", str(bps), "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart", path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, creationflags=_NO_WINDOW)
    if proc.returncode != 0:
        raise EncodeError(f"could not generate test video: {proc.stderr.strip()[:300]}")
    return path
