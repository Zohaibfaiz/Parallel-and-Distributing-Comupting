#!/usr/bin/env python3
"""GPU worker daemon (server side).

Run:   python server/daemon.py --host 0.0.0.0 --port 5050 --token <secret>

Design
------
* One thread per TCP connection (control + file transfer), a FIFO job queue and
  N worker threads that run jobs on the GPU (default N=1).
* Jobs are identified by a client generated id and are *independent of the TCP
  connection*: if the socket dies the job keeps running, and the client can
  reconnect and WATCH (progress replay), resume an interrupted upload, or
  resume an interrupted download.
* Integrity: SHA-256 is verified for the upload (server side) and for the
  rendered result (client side).
* Security: HMAC-SHA256 challenge/response with a shared token, strict
  whitelist of render parameters, no shell is ever invoked with user data.
"""
from __future__ import annotations

import argparse
import logging
import logging.handlers
import os
import queue
import re
import shutil
import signal
import socket
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.ffmpeg_utils import normalize_settings  # noqa: E402
from common.protocol import (  # noqa: E402
    DEFAULT_PORT, DEFAULT_TOKEN, PROTOCOL_VERSION, Cancelled, NullSink, ProtocolError,
    make_nonce, recv_file_range, recv_json, send_data, send_error, send_file_range, send_json,
    sha256_file, tune_socket, verify_auth,
)
from common.util import human_bytes, local_ips  # noqa: E402
from server.gpu_engine import EngineError, GpuEngine, normalize_torch_settings  # noqa: E402

log = logging.getLogger("gpu-worker")
JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
TERMINAL = ("done", "failed", "cancelled")
IDLE_TIMEOUT = 30.0  # seconds without a byte from the client during a request


# =========================================================================== jobs
class Job:
    def __init__(self, job_id: str, kind: str, filename: str, size: int, sha256: str,
                 settings: dict, root: Path):
        self.id = job_id
        self.kind = kind
        self.filename = filename
        self.size = size
        self.sha256 = sha256
        self.settings = settings
        self.dir = root / job_id
        self.dir.mkdir(parents=True, exist_ok=True)
        ext = Path(filename).suffix.lower()
        ext = ext if re.fullmatch(r"\.[a-z0-9]{1,6}", ext) else ".bin"
        self.input_path = self.dir / f"input{ext}"
        self.output_path = self.dir / "output.mp4"
        self.state = "uploading" if kind == "transcode" else "created"
        self.events: list[dict] = []
        self.progress: dict = {}
        self.version = 0
        self.cond = threading.Condition()
        self.cancel_event = threading.Event()
        self.upload_lock = threading.Lock()
        self.upload_conn: socket.socket | None = None
        self.created = time.time()
        self.enqueued_at: float | None = None
        self.finished_at: float | None = None
        self.result: dict = {}
        self.error: str | None = None

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL

    def emit(self, kind: str, **data) -> None:
        with self.cond:
            self.events.append({"type": "EVENT", "seq": len(self.events), "kind": kind,
                                "ts": round(time.time(), 3), **data})
            self.version += 1
            self.cond.notify_all()

    def log(self, msg: str) -> None:
        log.info("[%s] %s", self.id[:8], msg)
        self.emit("log", message=msg)

    def set_progress(self, **p) -> None:
        with self.cond:
            self.progress = p
            self.version += 1
            self.cond.notify_all()

    def set_state(self, state: str, **extra) -> None:
        with self.cond:
            self.state = state
            self.emit("state", state=state, **extra)


class JobManager:
    def __init__(self, workdir: Path, engine: GpuEngine, workers: int, retention_h: float):
        self.root = workdir / "jobs"
        self.root.mkdir(parents=True, exist_ok=True)
        self.engine = engine
        self.retention = retention_h * 3600
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()
        self.queue: "queue.Queue[str]" = queue.Queue()
        self._stop = threading.Event()
        self._threads = [threading.Thread(target=self._worker, name=f"worker-{i}", daemon=True)
                         for i in range(max(1, workers))]
        self._threads.append(threading.Thread(target=self._janitor, name="janitor", daemon=True))
        for t in self._threads:
            t.start()

    def get(self, job_id: str) -> Job | None:
        with self.lock:
            return self.jobs.get(job_id)

    def create(self, *args) -> Job:
        job = Job(*args, root=self.root)
        with self.lock:
            self.jobs[job.id] = job
        return job

    def enqueue(self, job: Job) -> int:
        job.enqueued_at = time.time()
        job.set_state("queued", position=self.queue.qsize() + 1)
        self.queue.put(job.id)
        job.log(f"Queued (position {self.queue_position(job)})")
        return self.queue_position(job)

    def queue_position(self, job: Job) -> int:
        with self.queue.mutex:
            try:
                return list(self.queue.queue).index(job.id) + 1
            except ValueError:
                return 0

    def cancel(self, job: Job) -> None:
        job.cancel_event.set()
        if job.state in ("uploading", "created", "queued"):
            job.finished_at = time.time()
            job.set_state("cancelled")

    def remove(self, job_id: str) -> bool:
        with self.lock:
            job = self.jobs.get(job_id)
            if not job or not job.terminal:
                return False
            del self.jobs[job_id]
        shutil.rmtree(job.dir, ignore_errors=True)
        return True

    def active_count(self) -> int:
        with self.lock:
            return sum(1 for j in self.jobs.values() if not j.terminal)

    def shutdown(self) -> None:
        self._stop.set()
        with self.lock:
            for j in self.jobs.values():
                j.cancel_event.set()

    # ------------------------------------------------------------------ threads
    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                job_id = self.queue.get(timeout=1.0)
            except queue.Empty:
                continue
            job = self.get(job_id)
            if job is None or job.terminal:
                continue
            if job.cancel_event.is_set():
                job.finished_at = time.time()
                job.set_state("cancelled")
                continue
            self._run(job)

    def _run(self, job: Job) -> None:
        started = time.time()
        wait = started - (job.enqueued_at or started)
        job.set_state("running", queue_wait_s=round(wait, 3))
        job.log(f"Job started (waited {wait:.1f}s in queue)")
        try:
            result = self.engine.run(job)
            if job.kind == "transcode":
                job.log("Hashing result for integrity check ...")
                result["output_size"] = job.output_path.stat().st_size
                result["output_sha256"] = sha256_file(str(job.output_path))
            result["queue_wait_s"] = round(wait, 3)
            result["kind"] = job.kind
            job.result = result
            job.finished_at = time.time()
            job.log("Job finished successfully")
            job.set_state("done", result=result)
        except Cancelled:
            job.finished_at = time.time()
            job.log("Job cancelled")
            job.set_state("cancelled")
        except Exception as exc:  # noqa: BLE001 - report every failure to the client
            log.exception("job %s failed", job.id)
            job.error = str(exc)
            job.finished_at = time.time()
            job.log(f"ERROR: {exc}")
            job.set_state("failed", error=str(exc))

    def _janitor(self) -> None:
        while not self._stop.wait(300):
            now = time.time()
            with self.lock:
                stale = [j for j in self.jobs.values()
                         if (j.terminal and now - (j.finished_at or j.created) > self.retention)
                         or (not j.terminal and j.state == "uploading" and now - j.created > self.retention)]
            for job in stale:
                job.cancel_event.set()
                with self.lock:
                    self.jobs.pop(job.id, None)
                shutil.rmtree(job.dir, ignore_errors=True)
                log.info("janitor removed job %s", job.id[:8])


# =========================================================================== server
class WorkerServer:
    def __init__(self, host: str, port: int, token: str, workdir: Path, engine: GpuEngine,
                 workers: int = 1, max_gb: float = 20.0, retention_h: float = 24.0):
        self.host, self.port, self.token = host, port, token
        self.workdir = workdir
        self.engine = engine
        self.max_bytes = int(max_gb * 1024 ** 3)
        self.mgr = JobManager(workdir, engine, workers, retention_h)
        self.workers = workers
        self._sock: socket.socket | None = None
        self._stop = threading.Event()
        self._accept_thread: threading.Thread | None = None

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> "WorkerServer":
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        self.port = self._sock.getsockname()[1]
        self._sock.listen(16)
        self._sock.settimeout(1.0)
        self._accept_thread = threading.Thread(target=self._accept_loop, name="acceptor", daemon=True)
        self._accept_thread.start()
        return self

    def serve_forever(self) -> None:
        self.start()
        try:
            while not self._stop.wait(0.5):
                pass
        finally:
            self.stop()

    def stop(self) -> None:
        self._stop.set()
        self.mgr.shutdown()
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                conn, addr = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=self._serve_conn, args=(conn, addr), daemon=True).start()

    # ------------------------------------------------------------------ info
    def info(self) -> dict:
        free = shutil.disk_usage(self.workdir).free
        return {"type": "OK", "version": PROTOCOL_VERSION, "server": socket.gethostname(),
                "workers": self.workers, "queue_length": self.mgr.queue.qsize(),
                "active_jobs": self.mgr.active_count(), "free_disk_gb": round(free / 1024 ** 3, 1),
                **self.engine.describe()}

    # ------------------------------------------------------------------ connection
    def _serve_conn(self, conn: socket.socket, addr) -> None:
        tag = f"{addr[0]}:{addr[1]}"
        conn.settimeout(IDLE_TIMEOUT)
        tune_socket(conn)
        try:
            if not self._handshake(conn, tag):
                return
            while not self._stop.is_set():
                msg = recv_json(conn)
                mtype = msg.get("type")
                if mtype == "PING":
                    send_json(conn, {"type": "PONG", "seq": msg.get("seq"), "t": msg.get("t"),
                                     "server_time": time.time()})
                elif mtype == "INFO":
                    send_json(conn, self.info())
                elif mtype == "BWTEST":
                    self._bwtest(conn, msg)
                elif mtype == "SUBMIT":
                    self._submit(conn, msg, tag)
                elif mtype == "WATCH":
                    self._watch(conn, msg)
                elif mtype == "FETCH":
                    self._fetch(conn, msg, tag)
                elif mtype == "CANCEL":
                    job = self._job_or_error(conn, msg)
                    if job:
                        self.mgr.cancel(job)
                        send_json(conn, {"type": "OK", "state": job.state})
                elif mtype == "CLEANUP":
                    send_json(conn, {"type": "OK", "removed": self.mgr.remove(str(msg.get("job_id")))})
                elif mtype == "BYE":
                    break
                else:
                    send_error(conn, "bad_request", f"unknown message type {mtype!r}")
        except (ProtocolError, OSError) as exc:
            log.info("connection %s closed: %s", tag, exc)
        except Exception:  # noqa: BLE001
            log.exception("unexpected error on connection %s", tag)
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def _handshake(self, conn: socket.socket, tag: str) -> bool:
        nonce = make_nonce()
        send_json(conn, {"type": "WELCOME", "version": PROTOCOL_VERSION, "nonce": nonce,
                         "server": socket.gethostname()})
        hello = recv_json(conn)
        if hello.get("type") != "HELLO":
            send_error(conn, "bad_request", "expected HELLO")
            return False
        if hello.get("version") != PROTOCOL_VERSION:
            send_error(conn, "version", f"protocol version mismatch (worker speaks {PROTOCOL_VERSION})")
            return False
        if not verify_auth(self.token, nonce, hello.get("auth", "")):
            log.warning("authentication failed from %s", tag)
            time.sleep(0.5)  # slow down brute force
            send_error(conn, "auth", "authentication failed (wrong token)")
            return False
        log.info("client %s authenticated (%s)", tag, hello.get("client", "?"))
        send_json(conn, self.info())
        return True

    # ------------------------------------------------------------------ handlers
    def _job_or_error(self, conn, msg):
        job = self.mgr.get(str(msg.get("job_id", "")))
        if job is None:
            send_error(conn, "unknown_job", "no such job on this worker")
        return job

    def _bwtest(self, conn: socket.socket, msg: dict) -> None:
        size = max(1, min(int(msg.get("size", 16 * 1024 * 1024)), 256 * 1024 * 1024))
        direction = msg.get("direction", "up")
        send_json(conn, {"type": "BW_READY", "size": size})
        t0 = time.perf_counter()
        if direction == "up":  # client -> worker
            recv_file_range(conn, NullSink(), size)
            send_json(conn, {"type": "BW_RESULT", "bytes": size,
                             "seconds": round(time.perf_counter() - t0, 6)})
        else:  # worker -> client
            block = os.urandom(1024 * 1024)
            sent = 0
            while sent < size:
                n = min(len(block), size - sent)
                send_data(conn, block[:n])
                sent += n

    def _submit(self, conn: socket.socket, msg: dict, tag: str) -> None:
        job_id = str(msg.get("job_id", ""))
        kind = msg.get("kind", "transcode")
        if not JOB_ID_RE.match(job_id):
            return send_error(conn, "bad_job_id", "job_id must be 32 hex characters")
        if kind not in ("transcode", "torch"):
            return send_error(conn, "bad_kind", "kind must be 'transcode' or 'torch'")
        try:
            settings = (normalize_settings(msg.get("settings")) if kind == "transcode"
                        else normalize_torch_settings(msg.get("settings")))
        except ValueError as exc:
            return send_error(conn, "bad_settings", str(exc))

        job = self.mgr.get(job_id)
        if kind == "torch":
            if job is None:
                job = self.mgr.create(job_id, kind, "tensor-task", 0, "", settings)
                self.mgr.enqueue(job)
            return send_json(conn, {"type": "ACCEPT", "job_id": job_id, "offset": 0,
                                    "upload_required": False, "state": job.state})

        sha = str(msg.get("sha256", "")).lower()
        try:
            size = int(msg.get("size", 0))
        except (TypeError, ValueError):
            size = 0
        if not SHA_RE.match(sha) or size <= 0:
            return send_error(conn, "bad_request", "size and sha256 are required")
        if size > self.max_bytes:
            return send_error(conn, "too_large", f"file exceeds the {human_bytes(self.max_bytes)} limit")
        filename = os.path.basename(str(msg.get("filename", "input.bin")))[:120]

        if job is None:
            if shutil.disk_usage(self.workdir).free < size * 3:
                return send_error(conn, "no_space", "not enough free disk space on the worker")
            job = self.mgr.create(job_id, kind, filename, size, sha, settings)
        elif job.sha256 != sha or job.size != size:
            return send_error(conn, "job_conflict", "job id already used for a different file")

        if job.state != "uploading":  # upload already finished earlier (client lost our answer)
            return send_json(conn, {"type": "ACCEPT", "job_id": job_id, "offset": size,
                                    "upload_required": False, "state": job.state})

        if not job.upload_lock.acquire(blocking=False):
            # A stale connection of the same job still holds the lock: kick it out.
            old = job.upload_conn
            if old is not None:
                try:
                    old.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            if not job.upload_lock.acquire(timeout=10):
                return send_error(conn, "busy", "another upload of this job is still in progress")
        try:
            job.upload_conn = conn
            have = job.input_path.stat().st_size if job.input_path.exists() else 0
            if have > size:
                job.input_path.unlink()
                have = 0
            send_json(conn, {"type": "ACCEPT", "job_id": job_id, "offset": have,
                             "upload_required": True, "state": job.state})
            job.log(f"Receiving {job.filename} ({human_bytes(size)}) from {tag}, resuming at {human_bytes(have)}"
                    if have else f"Receiving {job.filename} ({human_bytes(size)}) from {tag}")
            with open(job.input_path, "ab" if have else "wb") as fh:
                recv_file_range(conn, fh, size - have)
            job.log("Upload complete - verifying SHA-256 ...")
            actual = sha256_file(str(job.input_path))
            if actual != sha:
                job.input_path.unlink(missing_ok=True)
                job.log("Checksum mismatch - upload discarded, client must resend")
                return send_json(conn, {"type": "UPLOAD_BAD", "reason": "checksum mismatch"})
            job.log("Checksum OK")
            send_json(conn, {"type": "UPLOAD_OK"})
            pos = self.mgr.enqueue(job)
            send_json(conn, {"type": "QUEUED", "job_id": job_id, "position": pos})
        finally:
            job.upload_conn = None
            job.upload_lock.release()

    def _watch(self, conn: socket.socket, msg: dict) -> None:
        job = self._job_or_error(conn, msg)
        if job is None:
            return
        idx = max(0, int(msg.get("since", 0)))
        send_json(conn, {"type": "WATCH_OK", "state": job.state, "events": len(job.events)})
        last_version = -1
        while not self._stop.is_set():
            with job.cond:
                if job.version == last_version and not job.terminal:
                    job.cond.wait(timeout=2.0)
                last_version = job.version
                pending = job.events[idx:]
                prog = dict(job.progress)
                state = job.state
                terminal = job.terminal
            for ev in pending:
                send_json(conn, ev)
            idx += len(pending)
            if state == "queued":
                prog = {"stage": "queued", "position": self.mgr.queue_position(job)}
            if prog and not terminal:
                send_json(conn, {"type": "PROGRESS", **prog})
            elif not pending:
                send_json(conn, {"type": "HEARTBEAT"})
            if terminal:
                break
        send_json(conn, {"type": "WATCH_END", "state": job.state})

    def _fetch(self, conn: socket.socket, msg: dict, tag: str) -> None:
        job = self._job_or_error(conn, msg)
        if job is None:
            return
        if job.state != "done" or job.kind != "transcode":
            return send_error(conn, "not_ready", f"job is {job.state}, no result to fetch")
        size = job.result["output_size"]
        offset = max(0, min(int(msg.get("offset", 0)), size))
        send_json(conn, {"type": "RESULT", "size": size, "sha256": job.result["output_sha256"],
                         "offset": offset, "filename": "output.mp4"})
        job.log(f"Sending result to {tag} from offset {human_bytes(offset)}")
        send_file_range(conn, str(job.output_path), offset, size)


# =========================================================================== CLI
def daemonize(pidfile: str | None) -> None:
    if os.name != "posix":
        raise SystemExit("--daemon is only supported on Linux/macOS. On Windows use scripts/run_server.bat "
                         "or Task Scheduler (see README).")
    if os.fork() > 0:
        os._exit(0)
    os.setsid()
    if os.fork() > 0:
        os._exit(0)
    os.chdir("/")
    devnull = os.open(os.devnull, os.O_RDWR)
    for fd in (0, 1, 2):
        os.dup2(devnull, fd)
    if pidfile:
        Path(pidfile).write_text(str(os.getpid()))


def setup_logging(workdir: Path, verbose: bool, foreground: bool) -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    fh = logging.handlers.RotatingFileHandler(workdir / "worker.log", maxBytes=5_000_000, backupCount=3)
    fh.setFormatter(fmt)
    root.addHandler(fh)
    if foreground:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(sh)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="GPU worker daemon for distributed task offloading")
    p.add_argument("--host", default="0.0.0.0", help="bind address (default 0.0.0.0)")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--token", default=os.environ.get("OFFLOAD_TOKEN", DEFAULT_TOKEN),
                   help="shared secret (or env OFFLOAD_TOKEN)")
    p.add_argument("--workdir", default=str(ROOT / "server" / "workdir"))
    p.add_argument("--engine", choices=["auto", "gpu", "cpu"], default="auto",
                   help="auto: NVENC if available else CPU; gpu: require NVENC; cpu: force software")
    p.add_argument("--no-hwdecode", action="store_true", help="do not use -hwaccel cuda for decoding")
    p.add_argument("--workers", type=int, default=1, help="concurrent GPU jobs (default 1)")
    p.add_argument("--max-gb", type=float, default=20.0, help="max accepted upload size")
    p.add_argument("--retention-hours", type=float, default=24.0)
    p.add_argument("--daemon", action="store_true", help="detach into the background (Linux/macOS)")
    p.add_argument("--pidfile", default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    setup_logging(workdir, args.verbose, foreground=not args.daemon)
    pidfile = str(Path(args.pidfile).resolve()) if args.pidfile else None
    if args.daemon:
        daemonize(pidfile)

    try:
        engine = GpuEngine(mode=args.engine, hwdecode=not args.no_hwdecode)
    except EngineError as exc:
        log.error("%s", exc)
        return 2
    server = WorkerServer(args.host, args.port, args.token, workdir, engine,
                          workers=args.workers, max_gb=args.max_gb, retention_h=args.retention_hours)

    def _sig(*_):
        log.info("shutting down ...")
        server._stop.set()

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    d = engine.describe()
    log.info("=" * 60)
    log.info("GPU worker daemon listening on %s:%d", args.host, server.port if server._sock else args.port)
    log.info("Reachable at: %s", ", ".join(local_ips()) or "(no external IPv4 found)")
    log.info("GPU(s): %s", "; ".join(f"{g['name']} {g['memory_mb']}MB" for g in d["gpus"]) or "none detected")
    log.info("NVENC: %s | %s | torch: %s", "available" if d["nvenc"] else "NOT available (CPU fallback)",
             d["ffmpeg"][:40], d["torch"])
    if args.token == DEFAULT_TOKEN:
        log.warning("Using the default token - set --token or OFFLOAD_TOKEN for real use!")
    log.info("Work directory: %s", workdir)
    log.info("=" * 60)
    try:
        server.serve_forever()
    except OSError as exc:
        log.error("cannot start server: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
