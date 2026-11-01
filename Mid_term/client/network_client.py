"""
================================================================================
CSC-334: Parallel and Distributed Computing
Lab 4: Custom Distributed Task Offloading & Remote GPU Rendering System
Module: client/network_client.py
Description: Robust Network Client handling handshake, ping, streaming & checksums.
================================================================================
"""

import os
import sys
import time
import socket
import hashlib
import uuid
from typing import Dict, Any, Tuple, Optional, Callable, List

from protocol import (
    MsgType, send_packet, send_json, read_packet, read_json,
    CHUNK_SIZE, SOCKET_TIMEOUT, PROTOCOL_VERSION
)
from utils import compute_sha256, format_bytes


class OffloadingClient:
    """
    Client-side communication manager:
    - Task 1: Performs handshake and ping latency check with remote worker.
    - Task 4: Chunked file streaming with SHA-256 verification and progress streaming.
    """
    def __init__(self, host: str = "127.0.0.1", port: int = 5000):
        self.host = host
        self.port = port
        self.client_id = f"client_{uuid.uuid4().hex[:8]}"
        self.sock: Optional[socket.socket] = None
        self.worker_info: Dict[str, Any] = {}
        self.is_connected = False

    def connect(self, timeout: float = 5.0) -> bool:
        """Establishes TCP connection to the remote worker node."""
        self.disconnect()
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(timeout)
            self.sock.connect((self.host, self.port))
            self.is_connected = True
            return True
        except Exception as e:
            self.disconnect()
            raise ConnectionError(f"Could not connect to Remote Worker at {self.host}:{self.port} -> {e}")

    def disconnect(self) -> None:
        """Closes the socket connection safely."""
        self.is_connected = False
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None

    def perform_handshake(self) -> Dict[str, Any]:
        """
        TASK 1: Handshake Protocol
        Sends client info, verifies worker protocol version, and retrieves worker capabilities.
        """
        if not self.is_connected or not self.sock:
            self.connect()

        handshake_payload = {
            "client_id": self.client_id,
            "protocol_version": PROTOCOL_VERSION,
            "timestamp": time.time()
        }
        send_json(self.sock, MsgType.HANDSHAKE_SYN, handshake_payload)

        msg_type, ack_data = read_json(self.sock)
        if msg_type != MsgType.HANDSHAKE_ACK:
            raise ValueError(f"Handshake failed. Expected HANDSHAKE_ACK, got {msg_type!r}")

        self.worker_info = ack_data
        return ack_data

    def measure_ping_latency(self, num_samples: int = 3) -> float:
        """
        TASK 1: Latency / Ping Check
        Calculates average round-trip time (RTT) in milliseconds.
        """
        if not self.is_connected or not self.sock:
            self.connect()

        latencies = []
        for _ in range(num_samples):
            t_start = time.time()
            send_json(self.sock, MsgType.PING_REQ, {"client_timestamp": t_start})
            msg_type, pong = read_json(self.sock)
            if msg_type == MsgType.PONG_RESP:
                rtt = (time.time() - t_start) * 1000.0  # ms
                latencies.append(rtt)
            time.sleep(0.05)

        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        return round(avg_latency, 2)

    def offload_video_transcode(
        self,
        input_file: str,
        output_dir: str,
        render_config: Dict[str, Any],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None
    ) -> Dict[str, Any]:
        """
        TASK 2 & 4: Submits video transcoding task to remote GPU worker.
        Transfers asset with SHA-256 verification, receives progress, downloads output.
        """
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")

        file_size = os.path.getsize(input_file)
        task_id = f"task_{uuid.uuid4().hex[:8]}"

        # Calculate SHA-256 for transmission integrity verification
        if progress_callback:
            progress_callback({"status": "Computing SHA-256 input checksum...", "percent": 0.0})
        input_sha256 = compute_sha256(input_file)

        if not self.is_connected or not self.sock:
            self.connect()
            self.perform_handshake()

        # 1. Submit Job Metadata
        if progress_callback:
            progress_callback({"status": "Submitting job configuration to remote worker...", "percent": 0.0})

        job_meta = {
            "task_id": task_id,
            "task_type": "video_transcode",
            "file_name": os.path.basename(input_file),
            "file_size": file_size,
            "sha256": input_sha256,
            "config": render_config
        }
        try:
            send_json(self.sock, MsgType.JOB_SUBMIT, job_meta)
        except (OSError, ConnectionError):
            self.connect()
            self.perform_handshake()
            send_json(self.sock, MsgType.JOB_SUBMIT, job_meta)

        # Wait for worker acceptance
        msg_type, ack = read_json(self.sock)
        if msg_type != MsgType.JOB_ACCEPTED:
            raise ConnectionError(f"Worker did not accept job: {ack}")

        # 2. Stream File Asset to Worker
        upload_start = time.time()
        sent_bytes = 0
        with open(input_file, "rb") as f:
            while chunk := f.read(CHUNK_SIZE):
                if cancel_check and cancel_check():
                    send_json(self.sock, MsgType.CANCEL_REQ, {"task_id": task_id})
                    raise InterruptedError("Job cancelled by user.")
                
                send_packet(self.sock, MsgType.FILE_CHUNK, chunk)
                sent_bytes += len(chunk)
                upload_pct = (sent_bytes / file_size) * 100.0
                if progress_callback and sent_bytes % (CHUNK_SIZE * 4) == 0:
                    speed = (sent_bytes / (time.time() - upload_start + 0.001)) / (1024 * 1024)
                    progress_callback({
                        "status": f"Uploading asset ({format_bytes(sent_bytes)}/{format_bytes(file_size)}) @ {speed:.2f} MB/s",
                        "percent": upload_pct,
                        "upload_progress": upload_pct
                    })

        send_packet(self.sock, MsgType.FILE_EOF, input_sha256.encode("utf-8"))
        upload_duration = time.time() - upload_start

        # 3. Stream Real-Time Progress from Worker
        remote_compute_start = time.time()
        result_payload = None

        while True:
            if cancel_check and cancel_check():
                send_json(self.sock, MsgType.CANCEL_REQ, {"task_id": task_id})
                raise InterruptedError("Job cancelled by user.")

            msg_type, payload = read_packet(self.sock)

            if msg_type == MsgType.PROGRESS:
                import json
                prog_data = json.loads(payload.decode("utf-8"))
                if progress_callback:
                    progress_callback(prog_data)

            elif msg_type == MsgType.JOB_RESULT:
                import json
                result_payload = json.loads(payload.decode("utf-8"))
                break

            elif msg_type == MsgType.ERROR:
                import json
                err_data = json.loads(payload.decode("utf-8"))
                raise RuntimeError(f"Remote Worker Execution Error: {err_data.get('error')}")

            else:
                pass

        remote_compute_duration = time.time() - remote_compute_start

        # 4. Download Rendered Output File with Checksum Validation
        out_filename = result_payload.get("output_file_name", f"output_{task_id}.mp4")
        expected_out_size = result_payload.get("output_size_bytes", 0)
        expected_out_sha256 = result_payload.get("output_sha256", "")
        download_start = time.time()

        os.makedirs(output_dir, exist_ok=True)
        local_output_path = os.path.join(output_dir, out_filename)

        if progress_callback:
            progress_callback({"status": "Downloading rendered asset from worker node...", "percent": 100.0})

        send_json(self.sock, MsgType.DOWNLOAD_REQ, {"task_id": task_id, "file_name": out_filename})

        # Read download acceptance
        msg_type, dl_ack = read_json(self.sock)
        recv_bytes = 0
        hasher = hashlib.sha256()

        with open(local_output_path, "wb") as f:
            while True:
                c_type, c_data = read_packet(self.sock)
                if c_type == MsgType.FILE_CHUNK:
                    f.write(c_data)
                    hasher.update(c_data)
                    recv_bytes += len(c_data)
                elif c_type == MsgType.FILE_EOF:
                    break

        download_duration = time.time() - download_start
        computed_out_sha256 = hasher.hexdigest()

        # Validate download integrity
        checksum_verified = (computed_out_sha256.lower() == expected_out_sha256.lower()) if expected_out_sha256 else True

        # 5. Download Server Execution Log File
        log_filename = result_payload.get("log_file_name", "")
        server_logs_text = result_payload.get("logs_text", "")
        local_log_path = ""

        if log_filename:
            local_log_path = os.path.join(output_dir, log_filename)
            try:
                if progress_callback:
                    progress_callback({"status": "Downloading server execution logs...", "percent": 100.0})
                send_json(self.sock, MsgType.DOWNLOAD_REQ, {"task_id": task_id, "file_name": log_filename})
                msg_type, log_ack = read_json(self.sock)
                with open(local_log_path, "wb") as f:
                    while True:
                        c_type, c_data = read_packet(self.sock)
                        if c_type == MsgType.FILE_CHUNK:
                            f.write(c_data)
                        elif c_type == MsgType.FILE_EOF:
                            break
                if not server_logs_text and os.path.exists(local_log_path):
                    with open(local_log_path, "r", encoding="utf-8", errors="ignore") as f:
                        server_logs_text = f.read()
            except Exception:
                # If downloading log file had any network issue, write text directly if available
                if server_logs_text:
                    try:
                        with open(local_log_path, "w", encoding="utf-8") as f:
                            f.write(server_logs_text)
                    except Exception:
                        pass
        elif server_logs_text:
            # Generate local log file if server provided text
            local_log_path = os.path.join(output_dir, f"out_{task_id}_render_log.txt")
            try:
                with open(local_log_path, "w", encoding="utf-8") as f:
                    f.write(server_logs_text)
            except Exception:
                pass

        total_turnaround_time = upload_duration + remote_compute_duration + download_duration
        network_overhead_time = upload_duration + download_duration

        return {
            "success": True,
            "task_id": task_id,
            "engine": result_payload.get("engine", "Remote Worker"),
            "local_output_path": local_output_path,
            "local_log_path": local_log_path,
            "server_logs_text": server_logs_text,
            "input_file_size": file_size,
            "output_file_size": os.path.getsize(local_output_path),
            "upload_duration_sec": round(upload_duration, 3),
            "remote_compute_duration_sec": round(remote_compute_duration, 3),
            "download_duration_sec": round(download_duration, 3),
            "network_overhead_sec": round(network_overhead_time, 3),
            "total_turnaround_sec": round(total_turnaround_time, 3),
            "input_sha256": input_sha256,
            "output_sha256": computed_out_sha256,
            "checksum_verified": checksum_verified
        }

    def offload_video_merge(
        self,
        input_files: List[str],
        output_dir: str,
        render_config: Dict[str, Any],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None
    ) -> Dict[str, Any]:
        """
        Submits a multi-clip merge + render task to the remote worker node.
        Protocol:
          1. Send JOB_SUBMIT with task_type='video_merge' and files list metadata.
          2. Wait for JOB_ACCEPTED.
          3. Stream each clip file (FILE_CHUNK* + FILE_EOF) in order.
          4. Wait for PROGRESS stream, then JOB_RESULT.
          5. Download rendered output + log.
        """
        if not input_files:
            raise ValueError("No input files provided for merge.")
        for fp in input_files:
            if not os.path.exists(fp):
                raise FileNotFoundError(f"Clip not found: {fp}")

        task_id = f"merge_{uuid.uuid4().hex[:8]}"

        # --- Build per-file metadata ---
        if progress_callback:
            progress_callback({"status": "Computing SHA-256 checksums for all clips...", "percent": 0.0})

        files_meta = []
        total_bytes = 0
        for i, fpath in enumerate(input_files):
            fsize = os.path.getsize(fpath)
            fsha = compute_sha256(fpath)
            files_meta.append({
                "index": i,
                "file_name": os.path.basename(fpath),
                "file_size": fsize,
                "sha256": fsha
            })
            total_bytes += fsize

        if not self.is_connected or not self.sock:
            self.connect()
            self.perform_handshake()

        # --- 1. Submit job metadata ---
        if progress_callback:
            progress_callback({"status": f"Submitting {len(input_files)}-clip merge job to remote worker...", "percent": 1.0})

        job_meta = {
            "task_id": task_id,
            "task_type": "video_merge",
            "files": files_meta,
            "config": render_config
        }
        try:
            send_json(self.sock, MsgType.JOB_SUBMIT, job_meta)
        except (OSError, ConnectionError):
            self.connect()
            self.perform_handshake()
            send_json(self.sock, MsgType.JOB_SUBMIT, job_meta)

        msg_type, ack = read_json(self.sock)
        if msg_type != MsgType.JOB_ACCEPTED:
            raise ConnectionError(f"Worker did not accept merge job: {ack}")

        # --- 2. Upload each clip sequentially ---
        upload_start = time.time()
        sent_total = 0
        for i, fpath in enumerate(input_files):
            fmeta = files_meta[i]
            clip_bytes = 0
            with open(fpath, "rb") as f:
                while chunk := f.read(CHUNK_SIZE):
                    if cancel_check and cancel_check():
                        send_json(self.sock, MsgType.CANCEL_REQ, {"task_id": task_id})
                        raise InterruptedError("Merge job cancelled by user.")
                    send_packet(self.sock, MsgType.FILE_CHUNK, chunk)
                    clip_bytes += len(chunk)
                    sent_total += len(chunk)
                    if progress_callback:
                        overall_pct = min(49.0, (sent_total / max(total_bytes, 1)) * 50.0)
                        speed = (sent_total / (time.time() - upload_start + 0.001)) / (1024 * 1024)
                        progress_callback({
                            "status": f"Uploading clip {i+1}/{len(input_files)}: {fmeta['file_name']} @ {speed:.1f} MB/s",
                            "percent": overall_pct,
                            "upload_progress": overall_pct
                        })
            send_packet(self.sock, MsgType.FILE_EOF, fmeta["sha256"].encode("utf-8"))

        upload_duration = time.time() - upload_start
        if progress_callback:
            progress_callback({"status": f"All {len(input_files)} clips uploaded. Waiting for server merge & render...", "percent": 50.0})

        # --- 3. Stream real-time progress from worker ---
        remote_compute_start = time.time()
        result_payload = None

        while True:
            if cancel_check and cancel_check():
                send_json(self.sock, MsgType.CANCEL_REQ, {"task_id": task_id})
                raise InterruptedError("Merge job cancelled by user.")

            msg_type, payload = read_packet(self.sock)

            if msg_type == MsgType.PROGRESS:
                import json as _json
                prog_data = _json.loads(payload.decode("utf-8"))
                if progress_callback:
                    # Remap progress 0-100 → 50-95 so upload + render together span 0-95%
                    raw_pct = prog_data.get("percent", 0.0)
                    prog_data["percent"] = 50.0 + (raw_pct * 0.45)
                    progress_callback(prog_data)
            elif msg_type == MsgType.JOB_RESULT:
                import json as _json
                result_payload = _json.loads(payload.decode("utf-8"))
                break
            elif msg_type == MsgType.ERROR:
                import json as _json
                err = _json.loads(payload.decode("utf-8"))
                raise RuntimeError(f"Remote Merge Error: {err.get('error')}")

        remote_compute_duration = time.time() - remote_compute_start

        # --- 4. Download merged+rendered output ---
        out_filename = result_payload.get("output_file_name", f"merged_{task_id}.mp4")
        expected_out_sha256 = result_payload.get("output_sha256", "")
        download_start = time.time()

        os.makedirs(output_dir, exist_ok=True)
        local_output_path = os.path.join(output_dir, out_filename)

        if progress_callback:
            progress_callback({"status": "Downloading merged & rendered video from server...", "percent": 96.0})

        send_json(self.sock, MsgType.DOWNLOAD_REQ, {"task_id": task_id, "file_name": out_filename})
        msg_type, _dl_ack = read_json(self.sock)

        recv_bytes = 0
        hasher = hashlib.sha256()
        with open(local_output_path, "wb") as f:
            while True:
                c_type, c_data = read_packet(self.sock)
                if c_type == MsgType.FILE_CHUNK:
                    f.write(c_data)
                    hasher.update(c_data)
                    recv_bytes += len(c_data)
                elif c_type == MsgType.FILE_EOF:
                    break

        download_duration = time.time() - download_start
        computed_out_sha256 = hasher.hexdigest()
        checksum_verified = (computed_out_sha256.lower() == expected_out_sha256.lower()) if expected_out_sha256 else True

        # --- 5. Download server log (if available) ---
        log_filename = result_payload.get("log_file_name", "")
        server_logs_text = result_payload.get("logs_text", "")
        local_log_path = ""

        if log_filename:
            local_log_path = os.path.join(output_dir, log_filename)
            try:
                send_json(self.sock, MsgType.DOWNLOAD_REQ, {"task_id": task_id, "file_name": log_filename})
                msg_type, _log_ack = read_json(self.sock)
                with open(local_log_path, "wb") as f:
                    while True:
                        c_type, c_data = read_packet(self.sock)
                        if c_type == MsgType.FILE_CHUNK:
                            f.write(c_data)
                        elif c_type == MsgType.FILE_EOF:
                            break
            except Exception:
                if server_logs_text:
                    with open(local_log_path, "w", encoding="utf-8") as f:
                        f.write(server_logs_text)

        if progress_callback:
            tag = "SHA-256: VERIFIED ✔" if checksum_verified else "SHA-256: MISMATCH ✗"
            progress_callback({"status": tag, "percent": 100.0})

        total_turnaround = upload_duration + remote_compute_duration + download_duration
        return {
            "success": True,
            "task_id": task_id,
            "clips_merged": len(input_files),
            "engine": result_payload.get("engine", "Remote Worker"),
            "local_output_path": local_output_path,
            "local_log_path": local_log_path,
            "server_logs_text": server_logs_text,
            "total_input_size": total_bytes,
            "output_file_size": os.path.getsize(local_output_path),
            "upload_duration_sec": round(upload_duration, 3),
            "remote_compute_duration_sec": round(remote_compute_duration, 3),
            "download_duration_sec": round(download_duration, 3),
            "total_turnaround_sec": round(total_turnaround, 3),
            "output_sha256": computed_out_sha256,
            "checksum_verified": checksum_verified
        }

    def offload_cuda_compute(

        self,
        matrix_size: int,
        iterations: int,
        output_dir: str,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """Submits a parallel matrix tensor compute workload to remote worker."""
        task_id = f"cuda_{uuid.uuid4().hex[:8]}"
        job_meta = {
            "task_id": task_id,
            "task_type": "cuda_compute",
            "file_name": "",
            "file_size": 0,
            "sha256": "",
            "config": {"matrix_size": matrix_size, "iterations": iterations}
        }
        send_json(self.sock, MsgType.JOB_SUBMIT, job_meta)

        msg_type, ack = read_json(self.sock)
        if msg_type != MsgType.JOB_ACCEPTED:
            raise ConnectionError(f"Job rejected: {ack}")

        start_time = time.time()
        result_payload = None

        while True:
            msg_type, payload = read_packet(self.sock)
            if msg_type == MsgType.PROGRESS:
                import json
                prog_data = json.loads(payload.decode("utf-8"))
                if progress_callback:
                    progress_callback(prog_data)
            elif msg_type == MsgType.JOB_RESULT:
                import json
                result_payload = json.loads(payload.decode("utf-8"))
                break
            elif msg_type == MsgType.ERROR:
                import json
                err = json.loads(payload.decode("utf-8"))
                raise RuntimeError(err.get("error"))

        total_time = time.time() - start_time
        return {
            "success": True,
            "task_id": task_id,
            "total_turnaround_sec": round(total_time, 3),
            "result_payload": result_payload
        }

    def offload_custom_script(
        self,
        script_file: str,
        output_dir: str,
        config: Dict[str, Any],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None
    ) -> Dict[str, Any]:
        """
        Submits a custom Python project script or heavy computation to remote worker node.
        Uploads script, streams live execution stdout in real-time, downloads results.
        """
        if not os.path.exists(script_file):
            raise FileNotFoundError(f"Project script not found: {script_file}")

        file_size = os.path.getsize(script_file)
        task_id = f"proj_{uuid.uuid4().hex[:8]}"

        if progress_callback:
            progress_callback({"status": "Computing script checksum...", "percent": 0.0})
        script_sha256 = compute_sha256(script_file)

        if not self.is_connected or not self.sock:
            self.connect()
            self.perform_handshake()

        # 1. Submit Job Metadata
        job_meta = {
            "task_id": task_id,
            "task_type": "custom_script",
            "file_name": os.path.basename(script_file),
            "file_size": file_size,
            "sha256": script_sha256,
            "config": config
        }
        send_json(self.sock, MsgType.JOB_SUBMIT, job_meta)

        # 2. Wait for Acceptance
        msg_type, ack = read_json(self.sock)
        if msg_type != MsgType.JOB_ACCEPTED:
            raise ConnectionError(f"Worker rejected script job: {ack}")

        # 3. Stream Script to Worker
        upload_start = time.time()
        with open(script_file, "rb") as f:
            while chunk := f.read(CHUNK_SIZE):
                if cancel_check and cancel_check():
                    send_json(self.sock, MsgType.CANCEL_REQ, {"task_id": task_id})
                    raise InterruptedError("Job cancelled by user.")
                send_packet(self.sock, MsgType.FILE_CHUNK, chunk)

        send_packet(self.sock, MsgType.FILE_EOF, script_sha256.encode("utf-8"))
        upload_duration = time.time() - upload_start

        # 4. Stream Live Execution Output from Worker Node
        remote_compute_start = time.time()
        result_payload = None

        while True:
            if cancel_check and cancel_check():
                send_json(self.sock, MsgType.CANCEL_REQ, {"task_id": task_id})
                raise InterruptedError("Job cancelled by user.")

            msg_type, payload = read_packet(self.sock)

            if msg_type == MsgType.PROGRESS:
                import json
                prog_data = json.loads(payload.decode("utf-8"))
                if progress_callback:
                    progress_callback(prog_data)

            elif msg_type == MsgType.JOB_RESULT:
                import json
                result_payload = json.loads(payload.decode("utf-8"))
                break

            elif msg_type == MsgType.ERROR:
                import json
                err_data = json.loads(payload.decode("utf-8"))
                raise RuntimeError(f"Remote Worker Error: {err_data.get('error')}")

        remote_compute_duration = time.time() - remote_compute_start

        # 5. Download Output Summary from Worker
        out_filename = result_payload.get("output_file_name", f"out_{task_id}.json")
        expected_sha256 = result_payload.get("output_sha256", "")
        download_start = time.time()

        os.makedirs(output_dir, exist_ok=True)
        local_output_path = os.path.join(output_dir, out_filename)

        send_json(self.sock, MsgType.DOWNLOAD_REQ, {"task_id": task_id, "file_name": out_filename})
        msg_type, dl_ack = read_json(self.sock)

        hasher = hashlib.sha256()
        with open(local_output_path, "wb") as f:
            while True:
                c_type, c_data = read_packet(self.sock)
                if c_type == MsgType.FILE_CHUNK:
                    f.write(c_data)
                    hasher.update(c_data)
                elif c_type == MsgType.FILE_EOF:
                    break

        download_duration = time.time() - download_start
        computed_sha256 = hasher.hexdigest()
        checksum_verified = (computed_sha256.lower() == expected_sha256.lower()) if expected_sha256 else True
        total_time = upload_duration + remote_compute_duration + download_duration

        return {
            "success": True,
            "task_id": task_id,
            "engine": result_payload.get("engine", "Remote Worker Node"),
            "local_output_path": local_output_path,
            "upload_duration_sec": round(upload_duration, 3),
            "remote_compute_duration_sec": round(remote_compute_duration, 3),
            "download_duration_sec": round(download_duration, 3),
            "network_overhead_sec": round(upload_duration + download_duration, 3),
            "total_turnaround_sec": round(total_time, 3),
            "checksum_verified": checksum_verified,
            "result_payload": result_payload
        }

    def list_server_files(self) -> list:
        """Requests list of available files stored in Server's shared storage."""
        if not self.is_connected or not self.sock:
            self.connect()
            self.perform_handshake()

        send_json(self.sock, MsgType.FILE_LIST_REQ, {})
        msg_type, data = read_json(self.sock)
        if msg_type == MsgType.FILE_LIST_RESP:
            return data.get("files", [])
        return []

    def upload_shared_file(self, file_path: str, progress_callback: Optional[Callable[[str], None]] = None) -> bool:
        """Uploads a file directly to the Server's shared storage hub over LAN."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if not self.is_connected or not self.sock:
            self.connect()
            self.perform_handshake()

        file_size = os.path.getsize(file_path)
        file_name = os.path.basename(file_path)
        sha256 = compute_sha256(file_path)

        send_json(self.sock, MsgType.FILE_SHARE_UP, {
            "file_name": file_name,
            "file_size": file_size,
            "sha256": sha256
        })

        msg_type, ack = read_json(self.sock)
        if msg_type != MsgType.FILE_SHARE_ACK:
            raise ConnectionError(f"Server rejected file upload: {ack}")

        sent_bytes = 0
        with open(file_path, "rb") as f:
            while chunk := f.read(CHUNK_SIZE):
                send_packet(self.sock, MsgType.FILE_CHUNK, chunk)
                sent_bytes += len(chunk)
                if progress_callback:
                    pct = (sent_bytes / file_size) * 100.0
                    progress_callback(f"Uploaded {format_bytes(sent_bytes)}/{format_bytes(file_size)} ({pct:.1f}%)")

        send_packet(self.sock, MsgType.FILE_EOF, sha256.encode("utf-8"))
        msg_type, final_ack = read_json(self.sock)
        return msg_type == MsgType.FILE_SHARE_ACK

    def download_shared_file(self, file_name: str, output_dir: str, progress_callback: Optional[Callable[[str], None]] = None) -> str:
        """Downloads a file from Server's shared storage hub to local directory."""
        if not self.is_connected or not self.sock:
            self.connect()
            self.perform_handshake()

        os.makedirs(output_dir, exist_ok=True)
        dest_path = os.path.join(output_dir, file_name)

        send_json(self.sock, MsgType.DOWNLOAD_REQ, {"file_name": file_name})
        msg_type, meta = read_json(self.sock)

        if msg_type == MsgType.ERROR:
            raise RuntimeError(meta.get("error", "Error downloading file from server"))

        total_size = meta.get("file_size", 0)
        expected_sha = meta.get("sha256", "")

        recv_bytes = 0
        hasher = hashlib.sha256()

        with open(dest_path, "wb") as f:
            while True:
                c_type, c_data = read_packet(self.sock)
                if c_type == MsgType.FILE_CHUNK:
                    f.write(c_data)
                    hasher.update(c_data)
                    recv_bytes += len(c_data)
                    if progress_callback and total_size > 0:
                        pct = (recv_bytes / total_size) * 100.0
                        progress_callback(f"Downloaded {format_bytes(recv_bytes)}/{format_bytes(total_size)} ({pct:.1f}%)")
                elif c_type == MsgType.FILE_EOF:
                    break

        return dest_path
