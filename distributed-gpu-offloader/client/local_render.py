"""Render on the *client* machine - used for the GUI's local-render button and as
the baseline in the benchmark (Task 5)."""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.ffmpeg_utils import (  # noqa: E402
    EncodeError, ResourceSampler, build_encode_cmd, detect_nvenc, encoder_name, find_binary,
    normalize_settings, probe, run_encode,
)

_nvenc_cache: dict[str, bool] = {}


def local_nvenc_available() -> bool:
    ffmpeg = find_binary("ffmpeg")
    if not ffmpeg:
        return False
    if ffmpeg not in _nvenc_cache:
        _nvenc_cache[ffmpeg] = detect_nvenc(ffmpeg)
    return _nvenc_cache[ffmpeg]


def render_local(input_path: str, output_path: str, settings: dict, engine: str = "auto",
                 on_progress: Optional[Callable[[dict], None]] = None,
                 cancel_event: Optional[threading.Event] = None) -> dict:
    """Transcode locally. engine: auto (NVENC if the laptop has one) | cpu | gpu."""
    ffmpeg, ffprobe = find_binary("ffmpeg"), find_binary("ffprobe")
    if not ffmpeg or not ffprobe:
        raise EncodeError("ffmpeg/ffprobe not found in PATH on this machine")
    s = normalize_settings(settings)
    nvenc = local_nvenc_available() if engine != "cpu" else False
    if engine == "gpu" and not nvenc:
        raise EncodeError("local NVENC is not available on this machine")
    use_gpu = nvenc
    info = probe(input_path, ffprobe)
    cmd = build_encode_cmd(input_path, output_path, s, use_gpu, hwdecode=False, ffmpeg=ffmpeg)
    sampler = ResourceSampler(gpu=True).start()
    t0 = time.perf_counter()
    try:
        res = run_encode(cmd, info["duration"], on_progress, cancel_event)
    finally:
        render_s = time.perf_counter() - t0
        resource = sampler.stop()
    if res["rc"] != 0:
        raise EncodeError(f"local ffmpeg failed: {' | '.join(res['tail'][-3:])[:300]}")
    return {"engine": encoder_name(s, use_gpu), "gpu": use_gpu, "render_s": render_s,
            "frames": res["frames"], "resource": resource, "input": info}
