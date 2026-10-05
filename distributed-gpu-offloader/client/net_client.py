"""Client side network layer (Tasks 1 & 4).

Features
--------
* Handshake with HMAC authentication, protocol-version check, latency ping test
  and friendly error diagnosis (``check_worker``).
* Throughput probe (``bandwidth_test``).
* Upload with SHA-256 integrity check and **resume** after a broken connection.
* Asynchronous progress streaming (``watch``) with automatic reconnect and event replay.
* Download with checksum verification and **resume**.
* Exponential back-off retries for timeouts, resets and corrupted transfers.
* Cooperative cancellation.
"""
from __future__ import annotations

import os
import platform
import socket
import statistics
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.ffmpeg_utils import ResourceSampler  # noqa: E402
from common.protocol import (  # noqa: E402
    DEFAULT_PORT, DEFAULT_TOKEN, PROTOCOL_VERSION, AuthError, BusyError, Cancelled, ChecksumError,
    ConnectionClosed, NullSink, ProtocolError, RemoteError, auth_response, expect, recv_file_range,
    recv_json, send_data, send_file_range, send_json, sha256_file, tune_socket,
)

RETRYABLE = (OSError, ProtocolError)  # socket.timeout/ConnectionError are OSError subclasses


class TransferError(Exception):
    """Gave up after the maximum number of retries."""


class JobFailed(Exception):
    """The worker reported that the job failed."""


def describe_exception(exc: BaseException) -> str:
    """Human readable explanation + hint for common network failures."""
    if isinstance(exc, ConnectionRefusedError):
        return "connection refused - is the worker daemon running and the port open in the firewall?"
    if isinstance(exc, socket.timeout):
        return "timed out - check the cable / Wi-Fi, the IP address and the firewall"
    if isinstance(exc, socket.gaierror):
        return "invalid host name / IP address"
    if isinstance(exc, (ConnectionResetError, BrokenPipeError, ConnectionClosed)):
        return "connection lost"
    if isinstance(exc, ChecksumError):
        return "checksum mismatch (data corrupted in transit)"
    if isinstance(exc, OSError):
        return f"network error: {exc.strerror or exc}"
    return str(exc)


class OffloadClient:
    def __init__(self, host: str, port: int = DEFAULT_PORT, token: str = DEFAULT_TOKEN,
                 timeout: float = 15.0, max_retries: int = 6,
                 log: Optional[Callable[[str], None]] = None,
                 on_progress: Optional[Callable[[dict], None]] = None,
                 cancel_event: Optional[threading.Event] = None):
        self.host, self.port, self.token = host, int(port), token
        self.timeout = timeout
        self.max_retries = max_retries
        self._log = log or (lambda m: print(m, flush=True))
        self._on_progress = on_progress or (lambda p: None)
        self.cancel_event = cancel_event or threading.Event()
        self.retries = 0
        self.fault_hook: Optional[Callable[[str], None]] = None  # test hook: raise to simulate failures
        self.server_info: dict = {}

    # ------------------------------------------------------------------ helpers
    def log(self, msg: str) -> None:
        self._log(msg)

    def _progress(self, stage: str, stage_pct: float, text: str = "", **extra) -> None:
        weights = {"hashing": (0, 3), "uploading": (3, 30), "queued": (30, 32),
                   "rendering": (32, 88), "downloading": (88, 98), "verifying": (98, 100), "done": (100, 100)}
        lo, hi = weights.get(stage, (0, 100))
        overall = lo + (hi - lo) * max(0.0, min(stage_pct, 100.0)) / 100.0
        self._on_progress({"stage": stage, "stage_percent": stage_pct, "overall": overall,
                           "text": text, **extra})

    def _check_cancel(self) -> None:
        if self.cancel_event.is_set():
            raise Cancelled()

    def _sleep(self, seconds: float) -> None:
        if self.cancel_event.wait(seconds):
            raise Cancelled()

    def _hook(self, point: str) -> None:
        if self.fault_hook:
            self.fault_hook(point)

    def _connect(self) -> tuple[socket.socket, dict]:
        sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        try:
            tune_socket(sock)
            sock.settimeout(self.timeout)
            welcome = expect(sock, "WELCOME")
            if welcome.get("version") != PROTOCOL_VERSION:
                raise AuthError(f"protocol version mismatch (worker {welcome.get('version')}, "
                                f"client {PROTOCOL_VERSION})")
            send_json(sock, {"type": "HELLO", "version": PROTOCOL_VERSION, "client": platform.node(),
                             "auth": auth_response(self.token, welcome["nonce"])})
            info = expect(sock, "OK")
            self.server_info = info
            return sock, info
        except BaseException:
            sock.close()
            raise

    def _retry(self, label: str, fn):
        attempt = 0
        while True:
            self._check_cancel()
            try:
                return fn()
            except (AuthError, RemoteError, Cancelled, JobFailed):
                raise
            except RETRYABLE as exc:
                attempt += 1
                self.retries += 1
                if attempt > self.max_retries:
                    raise TransferError(f"{label} failed after {self.max_retries} retries: "
                                        f"{describe_exception(exc)}") from exc
                delay = min(2 ** (attempt - 1), 10)
                self.log(f"⚠ {label}: {describe_exception(exc)} - retry {attempt}/{self.max_retries} in {delay}s")
                self._sleep(delay)

    # ------------------------------------------------------------------ Task 1: handshake / latency
    def check_worker(self, pings: int = 5) -> dict:
        """Connect, authenticate and measure latency. Never raises: returns a result dict."""
        try:
            t0 = time.perf_counter()
            sock, info = self._connect()
            connect_ms = (time.perf_counter() - t0) * 1000
            rtts = []
            try:
                for i in range(pings):
                    t = time.perf_counter()
                    send_json(sock, {"type": "PING", "seq": i, "t": t})
                    pong = expect(sock, "PONG")
                    if pong.get("seq") != i:
                        raise ProtocolError("out-of-order PONG")
                    rtts.append((time.perf_counter() - t) * 1000)
                send_json(sock, {"type": "BYE"})
            finally:
                sock.close()
            res = {"reachable": True, "connect_ms": round(connect_ms, 2),
                   "rtt_avg_ms": round(statistics.mean(rtts), 3), "rtt_min_ms": round(min(rtts), 3),
                   "rtt_max_ms": round(max(rtts), 3),
                   "jitter_ms": round(statistics.pstdev(rtts), 3) if len(rtts) > 1 else 0.0,
                   "server": info, "error": None, "hint": None}
            if res["rtt_avg_ms"] > 50:
                res["hint"] = "High latency - is this really a direct Ethernet/LAN link?"
            if not info.get("nvenc"):
                res["hint"] = (res["hint"] + " | " if res["hint"] else "") + \
                    "Worker has NO working NVENC - jobs will run on its CPU."
            return res
        except AuthError as exc:
            return {"reachable": False, "error": f"authentication rejected: {exc}",
                    "hint": "Use the same --token / token field on client and worker."}
        except (OSError, ProtocolError) as exc:
            return {"reachable": False, "error": describe_exception(exc), "hint": None}

    def bandwidth_test(self, megabytes: int = 32) -> dict:
        """Measure raw TCP throughput in both directions (Mbit/s)."""
        size = megabytes * 1024 * 1024
        block = os.urandom(1024 * 1024)
        sock, _ = self._connect()
        try:
            send_json(sock, {"type": "BWTEST", "direction": "up", "size": size})
            expect(sock, "BW_READY")
            t0 = time.perf_counter()
            sent = 0
            while sent < size:
                n = min(len(block), size - sent)
                send_data(sock, block[:n])
                sent += n
            res = expect(sock, "BW_RESULT")
            up_s = max(time.perf_counter() - t0, 1e-6)
            send_json(sock, {"type": "BWTEST", "direction": "down", "size": size})
            expect(sock, "BW_READY")
            t0 = time.perf_counter()
            recv_file_range(sock, NullSink(), size)
            down_s = max(time.perf_counter() - t0, 1e-6)
            send_json(sock, {"type": "BYE"})
        finally:
            sock.close()
        return {"megabytes": megabytes, "upload_mbps": round(size * 8 / up_s / 1e6, 1),
                "download_mbps": round(size * 8 / down_s / 1e6, 1), "server_recv_s": res.get("seconds")}

    # ------------------------------------------------------------------ control helpers
    def cancel_job(self, job_id: str) -> None:
        try:
            sock, _ = self._connect()
            try:
                send_json(sock, {"type": "CANCEL", "job_id": job_id})
                recv_json(sock)
            finally:
                sock.close()
        except (OSError, ProtocolError) as exc:
            self.log(f"⚠ could not send cancel request: {describe_exception(exc)}")

    def cleanup_job(self, job_id: str) -> None:
        try:
            sock, _ = self._connect()
            try:
                send_json(sock, {"type": "CLEANUP", "job_id": job_id})
                recv_json(sock)
            finally:
                sock.close()
        except (OSError, ProtocolError):
            pass  # worker janitor removes it later

    # ------------------------------------------------------------------ upload
    def _submit_and_upload(self, job_id: str, kind: str, path: Optional[str], size: int, sha: str,
                           settings: dict, stats: dict) -> None:
        def attempt():
            sock, _ = self._connect()
            t_start = time.perf_counter()
            try:
                send_json(sock, {"type": "SUBMIT", "job_id": job_id, "kind": kind,
                                 "filename": os.path.basename(path) if path else "tensor-task",
                                 "size": size, "sha256": sha, "settings": settings})
                acc = expect(sock, "ACCEPT")
                if not acc.get("upload_required"):
                    return
                offset = int(acc["offset"])
                if offset:
                    self.log(f"↻ resuming upload at {offset / 1e6:.1f} MB of {size / 1e6:.1f} MB")
                last = {"t": time.perf_counter(), "b": offset}

                def cb(done: int) -> None:
                    self._hook("upload")
                    now = time.perf_counter()
                    if now - last["t"] >= 0.25 or done == size:
                        rate = (done - last["b"]) * 8 / max(now - last["t"], 1e-6) / 1e6
                        last.update(t=now, b=done)
                        pct = done / size * 100
                        self._progress("uploading", pct, f"Uploading {done / 1e6:.1f}/{size / 1e6:.1f} MB "
                                       f"@ {rate:.0f} Mbit/s", mbps=rate)

                send_file_range(sock, path, offset, size, cb, self.cancel_event.is_set)
                msg = expect(sock, "UPLOAD_OK", "UPLOAD_BAD")
                if msg["type"] == "UPLOAD_BAD":
                    raise ChecksumError("worker reports corrupted upload")
                expect(sock, "QUEUED")
            finally:
                stats["upload_s"] += time.perf_counter() - t_start
                sock.close()

        self._retry("upload", attempt)

    # ------------------------------------------------------------------ watch (progress streaming)
    def _watch(self, job_id: str) -> dict:
        state = {"since": 0, "final": None, "result": None, "error": None, "cancel_sent": False}

        def attempt():
            sock, _ = self._connect()
            try:
                send_json(sock, {"type": "WATCH", "job_id": job_id, "since": state["since"]})
                expect(sock, "WATCH_OK")
                while True:
                    self._hook("watch")
                    if self.cancel_event.is_set() and not state["cancel_sent"]:
                        state["cancel_sent"] = True
                        threading.Thread(target=self.cancel_job, args=(job_id,), daemon=True).start()
                    msg = expect(sock, "EVENT", "PROGRESS", "HEARTBEAT", "WATCH_END")
                    t = msg["type"]
                    if t == "EVENT":
                        state["since"] = msg["seq"] + 1
                        if msg["kind"] == "log":
                            self.log(f"[worker] {msg['message']}")
                        elif msg["kind"] == "state":
                            st = msg["state"]
                            if st == "done":
                                state.update(final="done", result=msg.get("result"))
                            elif st == "failed":
                                state.update(final="failed", error=msg.get("error"))
                            elif st == "cancelled":
                                state.update(final="cancelled")
                    elif t == "PROGRESS":
                        if msg.get("stage") == "queued":
                            self._progress("queued", 0, f"Queued on worker (position {msg.get('position', '?')})")
                        else:
                            eta = msg.get("eta_s")
                            txt = (f"Rendering {msg.get('percent', 0):.1f}%  {msg.get('fps', 0):.0f} fps  "
                                   f"{msg.get('speed', 0):.2f}x" + (f"  ETA {eta:.0f}s" if eta is not None else ""))
                            self._progress("rendering", msg.get("percent", 0), txt, fps=msg.get("fps"),
                                           speed=msg.get("speed"), eta_s=eta)
                    elif t == "WATCH_END":
                        return
            finally:
                sock.close()

        self._retry("progress stream", attempt)
        if state["final"] == "failed":
            raise JobFailed(state["error"] or "job failed on the worker")
        if state["final"] == "cancelled":
            raise Cancelled()
        return state["result"] or {}

    # ------------------------------------------------------------------ download
    def _download(self, job_id: str, dest: str, stats: dict) -> None:
        part = dest + ".part"

        def attempt():
            offset = os.path.getsize(part) if os.path.exists(part) else 0
            sock, _ = self._connect()
            t_start = time.perf_counter()
            try:
                send_json(sock, {"type": "FETCH", "job_id": job_id, "offset": offset})
                res = expect(sock, "RESULT")
                size, want_sha = int(res["size"]), res["sha256"]
                if offset > size:
                    offset = 0
                    os.remove(part)
                if offset:
                    self.log(f"↻ resuming download at {offset / 1e6:.1f} MB of {size / 1e6:.1f} MB")
                last = {"t": time.perf_counter(), "b": offset}

                def cb(done: int) -> None:
                    self._hook("download")
                    got = offset + done
                    now = time.perf_counter()
                    if now - last["t"] >= 0.25 or got == size:
                        rate = (got - last["b"]) * 8 / max(now - last["t"], 1e-6) / 1e6
                        last.update(t=now, b=got)
                        self._progress("downloading", got / size * 100,
                                       f"Downloading {got / 1e6:.1f}/{size / 1e6:.1f} MB @ {rate:.0f} Mbit/s",
                                       mbps=rate)

                with open(part, "ab" if offset else "wb") as fh:
                    recv_file_range(sock, fh, size - offset, cb, self.cancel_event.is_set)
            finally:
                stats["download_s"] += time.perf_counter() - t_start
                sock.close()
            t_v = time.perf_counter()
            self._progress("verifying", 0, "Verifying SHA-256 of the result ...")
            got_sha = sha256_file(part)
            stats["verify_s"] += time.perf_counter() - t_v
            if got_sha != want_sha:
                os.remove(part)  # force a clean re-download
                raise ChecksumError("downloaded file is corrupted")
            os.replace(part, dest)
            self._progress("verifying", 100, "Checksum OK")

        self._retry("download", attempt)

    # ------------------------------------------------------------------ high level API
    def run_transcode(self, input_path: str, settings: dict, output_path: Optional[str] = None,
                      job_id: Optional[str] = None) -> dict:
        """Full offload pipeline. Returns timing / resource statistics."""
        job_id = job_id or uuid.uuid4().hex
        size = os.path.getsize(input_path)
        output_path = output_path or str(Path(input_path).with_suffix("")) + "_offloaded.mp4"
        stats = {"upload_s": 0.0, "download_s": 0.0, "verify_s": 0.0}
        sampler = ResourceSampler(gpu=False).start()
        t_all = time.perf_counter()
        self.retries = 0
        try:
            self.log(f"Offloading {os.path.basename(input_path)} ({size / 1e6:.1f} MB) to {self.host}:{self.port}")
            t = time.perf_counter()
            self._progress("hashing", 0, "Computing SHA-256 of the input ...")
            sha = sha256_file(input_path, lambda d: self._progress("hashing", d / size * 100,
                                                                    "Computing SHA-256 of the input ..."))
            hash_s = time.perf_counter() - t
            self.log(f"Input SHA-256 {sha[:16]}... ({hash_s:.1f}s)")
            self._submit_and_upload(job_id, "transcode", input_path, size, sha, settings, stats)
            self.log("✔ Upload verified by the worker")
            result = self._watch(job_id)
            self.log("✔ Render finished on the worker - downloading result")
            self._download(job_id, output_path, stats)
            self.cleanup_job(job_id)
            total = time.perf_counter() - t_all
            out_size = os.path.getsize(output_path)
            self._progress("done", 100, f"Done in {total:.1f}s")
            self.log(f"✔ Result saved to {output_path} ({out_size / 1e6:.1f} MB) - total {total:.1f}s")
            return {
                "job_id": job_id, "output_path": output_path, "input_size": size, "output_size": out_size,
                "hash_s": hash_s, **stats, "queue_wait_s": result.get("queue_wait_s", 0.0),
                "render_s": result.get("render_s", 0.0), "total_s": total, "engine": result.get("engine"),
                "gpu": result.get("gpu"), "avg_fps": result.get("avg_fps"),
                "server_resource": result.get("resource", {}), "client_resource": sampler.stop(),
                "retries": self.retries,
                "upload_mbps": size * 8 / stats["upload_s"] / 1e6 if stats["upload_s"] > 0 else None,
                "download_mbps": out_size * 8 / stats["download_s"] / 1e6 if stats["download_s"] > 0 else None,
            }
        except BaseException:
            sampler.stop()
            raise

    def run_torch(self, settings: dict, job_id: Optional[str] = None) -> dict:
        job_id = job_id or uuid.uuid4().hex
        stats = {"upload_s": 0.0, "download_s": 0.0, "verify_s": 0.0}
        self._submit_and_upload(job_id, "torch", None, 0, "", settings, stats)
        result = self._watch(job_id)
        self.cleanup_job(job_id)
        return result
