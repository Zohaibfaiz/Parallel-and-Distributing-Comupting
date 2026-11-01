"""
================================================================================
CSC-334: Parallel and Distributed Computing
Lab 4: Custom Distributed Task Offloading & Remote GPU Rendering System
Module: client/utils.py
Description: Checksum integrity, system discovery, formatting, and asset generation.
================================================================================
"""

import os
import sys
import time
import hashlib
import platform
import subprocess
from typing import Dict, Any, Tuple

def compute_sha256(filepath: str, chunk_size: int = 128 * 1024) -> str:
    """
    Computes SHA-256 checksum of a file in chunks.
    Ensures end-to-end data integrity validation across the network.
    """
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            sha.update(chunk)
    return sha.hexdigest()


def compute_bytes_sha256(data: bytes) -> str:
    """Computes SHA-256 checksum of raw in-memory bytes."""
    return hashlib.sha256(data).hexdigest()


def format_bytes(size: float) -> str:
    """Human-readable byte size formatter (e.g., 14.5 MB)."""
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(size) < 1024.0:
            return f"{size:3.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def format_time(seconds: float) -> str:
    """Formats duration into human-readable string."""
    if seconds < 1.0:
        return f"{seconds * 1000:.1f} ms"
    elif seconds < 60.0:
        return f"{seconds:.2f} s"
    else:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {s:.1f}s"


def get_ffmpeg_binary() -> str:
    """
    Locates the best available FFmpeg binary:
    Checks system PATH, imageio_ffmpeg, or environment variable.
    """
    # 1. Environment variable override
    env_ffmpeg = os.environ.get("FFMPEG_PATH")
    if env_ffmpeg and os.path.exists(env_ffmpeg):
        return env_ffmpeg
        
    # 2. imageio_ffmpeg package
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.exists(exe):
            return exe
    except Exception:
        pass

    # 3. System PATH fallback
    return "ffmpeg"


def detect_system_capabilities() -> Dict[str, Any]:
    """
    Probes system hardware:
    - OS details & Architecture
    - CPU cores
    - FFmpeg presence
    - NVIDIA NVENC hardware encoder availability
    - CUDA / PyTorch availability
    """
    ffmpeg_exe = get_ffmpeg_binary()
    ffmpeg_available = False
    nvenc_available = False
    nvenc_name = "None"
    
    try:
        # Probe FFmpeg version
        res = subprocess.run([ffmpeg_exe, "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
        if res.returncode == 0:
            ffmpeg_available = True
            
        # Probe NVENC hardware encoder functionality
        test_nvenc = subprocess.run(
            [ffmpeg_exe, "-f", "lavfi", "-i", "nullsrc=s=64x64:d=0.1", "-c:v", "h264_nvenc", "-f", "null", "-"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5
        )
        if test_nvenc.returncode == 0:
            nvenc_available = True
            nvenc_name = "NVIDIA NVENC (Hardware Acceleration Active)"
        else:
            list_enc = subprocess.run([ffmpeg_exe, "-encoders"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
            if "h264_nvenc" in list_enc.stdout:
                nvenc_name = "NVIDIA NVENC Supported (Fallback to CPU Multi-Thread on non-GPU hosts)"
            else:
                nvenc_name = "Not Available (CPU libx264 Multi-Thread Active)"
    except Exception as e:
        nvenc_name = f"Probe Error: {e}"

    cuda_available = False
    try:
        import importlib
        torch_mod = importlib.import_module("torch")
        cuda_available = hasattr(torch_mod, "cuda") and torch_mod.cuda.is_available()
    except Exception:
        cuda_available = False

    return {
        "hostname": platform.node(),
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "cpu_count": os.cpu_count() or 4,
        "python_version": sys.version.split()[0],
        "ffmpeg_available": ffmpeg_available,
        "ffmpeg_binary": ffmpeg_exe,
        "nvenc_available": nvenc_available,
        "nvenc_status": nvenc_name,
        "cuda_available": cuda_available,
    }


def generate_sample_video(
    output_path: str,
    duration_sec: int = 5,
    resolution: Tuple[int, int] = (1280, 720),
    fps: int = 30
) -> str:
    """
    Generates a synthetic high-definition MP4 test video with visual animation,
    timecode overlay, and dynamic geometric graphics for testing distributed rendering.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    width, height = resolution
    total_frames = duration_sec * fps

    try:
        import cv2
        import numpy as np

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))

        for frame_idx in range(total_frames):
            t = frame_idx / total_frames
            bg = np.zeros((height, width, 3), dtype=np.uint8)
            
            r_val = int(255 * (0.5 + 0.5 * np.sin(2 * np.pi * t)))
            g_val = int(255 * (0.5 + 0.5 * np.sin(2 * np.pi * t + 2.0)))
            b_val = int(255 * (0.5 + 0.5 * np.sin(2 * np.pi * t + 4.0)))
            
            bg[:, :, 0] = b_val // 3
            bg[:, :, 1] = g_val // 3
            bg[:, :, 2] = r_val // 3

            cx = int(width * (0.5 + 0.4 * np.cos(4 * np.pi * t)))
            cy = int(height * (0.5 + 0.3 * np.sin(4 * np.pi * t)))
            cv2.circle(bg, (cx, cy), int(min(width, height) * 0.15), (b_val, g_val, r_val), -1)
            cv2.circle(bg, (cx, cy), int(min(width, height) * 0.16), (255, 255, 255), 3)

            cv2.putText(bg, "CSC-334: Distributed Task Offloading Test Video", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(bg, f"Frame: {frame_idx + 1}/{total_frames}  |  FPS: {fps}  |  Res: {width}x{height}", 
                        (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2, cv2.LINE_AA)
            cv2.putText(bg, f"Time: {frame_idx / fps:.2f}s / {duration_sec:.2f}s", 
                        (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2, cv2.LINE_AA)
            cv2.putText(bg, "Target: Remote GPU Rendering Engine (NVENC / CUDA)", 
                        (30, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 255, 180), 2, cv2.LINE_AA)

            out.write(bg)

        out.release()
        return output_path
    except Exception as e:
        ffmpeg_exe = get_ffmpeg_binary()
        cmd = [
            ffmpeg_exe, "-y", "-f", "lavfi",
            "-i", f"testsrc=duration={duration_sec}:size={width}x{height}:rate={fps}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", output_path
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return output_path


def get_all_local_ips() -> list:
    """
    Detects all active IPv4 addresses for network adapters on this machine.
    """
    import socket
    ip_list = []
    
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.5)
        s.connect(("10.255.255.255", 1))
        primary_ip = s.getsockname()[0]
        if primary_ip and primary_ip != "127.0.0.1" and primary_ip not in ip_list:
            ip_list.append(primary_ip)
        s.close()
    except Exception:
        pass

    try:
        host_info = socket.gethostbyname_ex(socket.gethostname())
        for ip in host_info[2]:
            if not ip.startswith("127.") and ip not in ip_list:
                ip_list.append(ip)
    except Exception:
        pass

    if "127.0.0.1" not in ip_list:
        ip_list.append("127.0.0.1")

    return ip_list
