"""
================================================================================
CSC-334: Parallel and Distributed Computing
Lab 4: Custom Distributed Task Offloading & Remote GPU Rendering System
Module: server/network_server.py
Description: Multi-threaded Network Listener, Protocol Handler, Task Execution & File Sharing.
================================================================================
"""

import os
import time
import socket
import select
import hashlib
import threading
from typing import Dict, Any, Optional, List, Callable

from protocol import (
    MsgType, send_packet, send_json, read_packet, read_json,
    CHUNK_SIZE, SOCKET_TIMEOUT, PROTOCOL_VERSION
)
from utils import compute_sha256, format_bytes, detect_system_capabilities
from task_queue import TaskQueueManager, TaskItem, TaskStatus
from execution_engine import RemoteExecutionEngine


class RemoteWorkerServer:
    """
    Core Server Daemon that listens for client connections, processes handshakes,
    streams files with checksum verification, executes tasks on remote GPU/CPU,
    and supports LAN file sharing and GUI monitoring.
    """
    def __init__(self, host: str = "0.0.0.0", port: int = 5000, storage_dir: str = "server_storage"):
        self.host = host
        self.port = port
        self.storage_dir = os.path.abspath(storage_dir)
        os.makedirs(self.storage_dir, exist_ok=True)

        self.queue_manager = TaskQueueManager()
        self.engine = RemoteExecutionEngine()
        self.sys_info = detect_system_capabilities()

        self._server_sock: Optional[socket.socket] = None
        self._shutdown_event = threading.Event()
        self._lock = threading.Lock()
        self.active_clients = 0
        self.connected_clients: Dict[str, Dict[str, Any]] = {}
        self.total_tasks_completed = 0
        self.start_time: float = 0.0

        # Optional GUI / Monitoring callbacks
        self.on_client_connected: Optional[Callable[[Dict[str, Any]], None]] = None
        self.on_client_disconnected: Optional[Callable[[Dict[str, Any]], None]] = None
        self.on_task_update: Optional[Callable[[Dict[str, Any]], None]] = None
        self.on_log: Optional[Callable[[str], None]] = None

    def log(self, msg: str) -> None:
        """Centralized logger that prints to stdout safely and notifies GUI listeners."""
        try:
            print(msg)
        except UnicodeEncodeError:
            try:
                print(msg.encode("ascii", errors="replace").decode("ascii"))
            except Exception:
                pass
        if self.on_log:
            try:
                self.on_log(msg)
            except Exception:
                pass

    def start(self) -> None:
        """Starts the server listener and worker thread pool."""
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind((self.host, self.port))
        self._server_sock.listen(10)
        self._server_sock.settimeout(1.0)  # Non-blocking accept check for clean shutdown
        self.start_time = time.time()

        self.log("=" * 70)
        self.log("  REMOTE GPU & DISTRIBUTED COMPUTE SERVER DAEMON (CSC-334)")
        self.log("=" * 70)
        self.log(f"[*] Bound to Address       : {self.host}:{self.port}")
        self.log(f"[*] Storage Directory      : {self.storage_dir}")
        self.log(f"[*] Node Hostname          : {self.sys_info['hostname']}")
        self.log(f"[*] Platform OS            : {self.sys_info['os']}")
        self.log(f"[*] CPU Cores              : {self.sys_info['cpu_count']}")
        self.log(f"[*] GPU/NVENC Acceleration : {self.sys_info['nvenc_status']}")
        self.log(f"[*] Protocol Version       : {PROTOCOL_VERSION}")
        self.log("=" * 70)
        self.log("[*] Worker daemon listening for incoming client offload tasks...")

        # Start background task queue execution thread
        worker_thread = threading.Thread(target=self._queue_worker_loop, daemon=True)
        worker_thread.start()

        try:
            while not self._shutdown_event.is_set():
                try:
                    client_sock, client_addr = self._server_sock.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break

                with self._lock:
                    self.active_clients += 1
                
                self.log(f"\n[+] [CLIENT CONNECTED] Incoming connection from: {client_addr[0]}:{client_addr[1]}")
                handler_thread = threading.Thread(
                    target=self._handle_client_connection,
                    args=(client_sock, client_addr),
                    daemon=True
                )
                handler_thread.start()
        finally:
            self.stop()

    def stop(self) -> None:
        """Gracefully shuts down the server daemon."""
        self._shutdown_event.set()
        if self._server_sock:
            try:
                self._server_sock.close()
            except Exception:
                pass
        self.log("[-] Remote Worker Server Daemon stopped gracefully.")

    def _handle_client_connection(self, sock: socket.socket, addr: tuple) -> None:
        """Handles an interactive client session."""
        sock.settimeout(None)  # Keep persistent connection alive for interactive GUI
        addr_key = f"{addr[0]}:{addr[1]}"
        client_record = {
            "addr": addr_key,
            "ip": addr[0],
            "port": addr[1],
            "client_id": "Initializing...",
            "connected_at": time.time(),
            "status": "CONNECTED",
            "tasks_count": 0
        }

        with self._lock:
            self.connected_clients[addr_key] = client_record

        try:
            while not self._shutdown_event.is_set():
                try:
                    msg_type, payload = read_packet(sock)
                except (socket.timeout, ConnectionResetError, EOFError):
                    break

                # 1. TASK 1: Robust Handshake Protocol
                if msg_type == MsgType.HANDSHAKE_SYN:
                    import json
                    client_meta = json.loads(payload.decode("utf-8")) if payload else {}
                    client_id = client_meta.get("client_id", "unknown_client")
                    
                    with self._lock:
                        if addr_key in self.connected_clients:
                            self.connected_clients[addr_key]["client_id"] = client_id
                            self.connected_clients[addr_key]["status"] = "AUTHENTICATED"

                    self.log(f"[+] [HANDSHAKE ESTABLISHED] Client Computer '{client_id}' ({addr[0]}) verified protocol v{PROTOCOL_VERSION}!")
                    self.log(f"   Node ready to receive video render tasks & file shares from {addr[0]}")
                    
                    if self.on_client_connected:
                        try:
                            self.on_client_connected(self.connected_clients[addr_key])
                        except Exception:
                            pass

                    ack_data = {
                        "status": "READY",
                        "server_time": time.time(),
                        "hostname": self.sys_info["hostname"],
                        "os": self.sys_info["os"],
                        "cpu_count": self.sys_info["cpu_count"],
                        "nvenc_available": self.sys_info["nvenc_available"],
                        "nvenc_status": self.sys_info["nvenc_status"],
                        "cuda_available": self.sys_info["cuda_available"],
                        "queue_size": self.queue_manager.get_queue_size(),
                        "protocol_version": PROTOCOL_VERSION
                    }
                    send_json(sock, MsgType.HANDSHAKE_ACK, ack_data)

                # 2. TASK 1: Initial Latency / Ping Check
                elif msg_type == MsgType.PING_REQ:
                    import json
                    ping_data = json.loads(payload.decode("utf-8")) if payload else {}
                    pong_data = {
                        "client_send_time": ping_data.get("client_timestamp", time.time()),
                        "server_receive_time": time.time(),
                        "active_workers": self.queue_manager.get_active_count()
                    }
                    send_json(sock, MsgType.PONG_RESP, pong_data)

                # 3. TASK 2 & 4: Job Submission & Video Render Offloading
                elif msg_type == MsgType.JOB_SUBMIT:
                    with self._lock:
                        if addr_key in self.connected_clients:
                            self.connected_clients[addr_key]["status"] = "EXECUTING_TASK"
                            self.connected_clients[addr_key]["tasks_count"] += 1

                    self._process_job_submission(sock, payload, addr)

                    with self._lock:
                        if addr_key in self.connected_clients:
                            self.connected_clients[addr_key]["status"] = "IDLE"

                # 4. TASK 4: Download Processed Output or Shared File
                elif msg_type == MsgType.DOWNLOAD_REQ:
                    self._process_download_request(sock, payload)

                # 5. Direct File Sharing: List Shared Files on Server
                elif msg_type == MsgType.FILE_LIST_REQ:
                    files = self.get_shared_files()
                    send_json(sock, MsgType.FILE_LIST_RESP, {"files": files})

                # 6. Direct File Sharing: Upload File to Shared Storage
                elif msg_type == MsgType.FILE_SHARE_UP:
                    self._process_file_share_upload(sock, payload, addr)

                # 7. Cancel Request
                elif msg_type == MsgType.CANCEL_REQ:
                    import json
                    req = json.loads(payload.decode("utf-8")) if payload else {}
                    task_id = req.get("task_id", "")
                    cancelled = self.queue_manager.cancel_task(task_id)
                    send_json(sock, MsgType.JOB_RESULT, {"task_id": task_id, "status": "CANCELLED" if cancelled else "NOT_FOUND"})

                else:
                    self.log(f"[!] Unknown message type received: {msg_type!r}")

        except Exception as e:
            self.log(f"[!] Error handling client {addr}: {e}")
        finally:
            with self._lock:
                self.active_clients = max(0, self.active_clients - 1)
                disc_info = self.connected_clients.pop(addr_key, None)

            try:
                sock.close()
            except Exception:
                pass

            self.log(f"[-] [CLIENT DISCONNECTED] Client {addr[0]}:{addr[1]} disconnected.")
            if self.on_client_disconnected and disc_info:
                try:
                    self.on_client_disconnected(disc_info)
                except Exception:
                    pass

    def _process_job_submission(self, sock: socket.socket, payload: bytes, addr: tuple) -> None:
        """Receives asset file(s), validates SHA-256 checksum(s), and schedules task."""
        import json
        meta = json.loads(payload.decode("utf-8"))
        task_id = meta["task_id"]
        task_type = meta["task_type"]
        config = meta.get("config", {})

        self.log(f"\n[*] [TASK RECEIVED] Task ID: {task_id} ({task_type}) from {addr[0]}")

        # ----------------------------------------------------------------
        # VIDEO MERGE: Receive multiple clip files sequentially
        # ----------------------------------------------------------------
        if task_type == "video_merge":
            files_meta = meta.get("files", [])
            self.log(f"   Multi-Clip Merge Job: {len(files_meta)} clips to merge & render")

            # Acknowledge submission before receiving files
            send_json(sock, MsgType.JOB_ACCEPTED, {"task_id": task_id, "status": "READY_FOR_UPLOAD"})

            input_paths = []
            for file_info in files_meta:
                idx = file_info["index"]
                fname = file_info["file_name"]
                fsize = file_info["file_size"]
                fsha256 = file_info.get("sha256", "")
                clip_path = os.path.join(
                    self.storage_dir,
                    f"in_{task_id}_{idx:02d}_{os.path.basename(fname)}"
                )
                self.log(f"   [MERGE] Receiving clip {idx+1}/{len(files_meta)}: {fname} ({format_bytes(fsize)})")

                hasher = hashlib.sha256()
                with open(clip_path, "wb") as f:
                    while True:
                        ct, cd = read_packet(sock)
                        if ct == MsgType.FILE_CHUNK:
                            f.write(cd)
                            hasher.update(cd)
                        elif ct == MsgType.FILE_EOF:
                            break
                        else:
                            raise ValueError(f"Unexpected chunk type during merge upload: {ct!r}")

                computed = hasher.hexdigest()
                if fsha256 and computed.lower() != fsha256.lower():
                    self.log(f"[!] CHECKSUM MISMATCH for clip {fname}")
                    send_json(sock, MsgType.ERROR, {"error": f"Checksum mismatch for clip {fname}"})
                    return
                self.log(f"   [INTEGRITY] Clip {idx+1} SHA-256 verified.")
                input_paths.append(clip_path)

            # Store paths in config so the execution engine can access them
            config["input_paths"] = input_paths
            output_path = os.path.join(self.storage_dir, f"out_{task_id}_merged.mp4")
            input_path = ""  # Not used for merge tasks

        # ----------------------------------------------------------------
        # SINGLE FILE TASKS: video_transcode, custom_script, etc.
        # ----------------------------------------------------------------
        else:
            file_name = meta.get("file_name", f"{task_id}.dat")
            expected_size = meta.get("file_size", 0)
            expected_sha256 = meta.get("sha256", "")

            if expected_size > 0:
                self.log(f"   Incoming Asset: {file_name} ({format_bytes(expected_size)})")

            input_path = os.path.join(self.storage_dir, f"in_{task_id}_{os.path.basename(file_name)}") if file_name else ""

            # Acknowledge job submission
            send_json(sock, MsgType.JOB_ACCEPTED, {"task_id": task_id, "status": "READY_FOR_UPLOAD"})

            if expected_size > 0:
                received_bytes = 0
                hasher = hashlib.sha256()
                with open(input_path, "wb") as f:
                    while True:
                        chunk_type, chunk_data = read_packet(sock)
                        if chunk_type == MsgType.FILE_CHUNK:
                            f.write(chunk_data)
                            hasher.update(chunk_data)
                            received_bytes += len(chunk_data)
                        elif chunk_type == MsgType.FILE_EOF:
                            break
                        else:
                            raise ValueError(f"Unexpected chunk type during upload: {chunk_type!r}")

                computed_sha256 = hasher.hexdigest()
                self.log(f"   Upload Complete. SHA-256: {computed_sha256[:12]}...")

                if expected_sha256 and computed_sha256.lower() != expected_sha256.lower():
                    self.log(f"[!] CHECKSUM MISMATCH for {file_name}!")
                    send_json(sock, MsgType.ERROR, {"error": "Checksum verification failed. Corrupted transfer."})
                    if os.path.exists(input_path):
                        os.remove(input_path)
                    return
                else:
                    self.log(f"   [INTEGRITY] SHA-256 checksum verified successfully!")
            else:
                input_path = ""

            # Prepare output path based on task type
            if task_type == "video_transcode":
                out_ext = ".mp4"
            elif task_type in ["custom_script", "python_task"]:
                out_ext = ".json"
            else:
                out_ext = ".json"
            output_path = os.path.join(self.storage_dir, f"out_{task_id}{out_ext}")

        task_item = TaskItem(task_id, task_type, config, input_path, output_path)
        task_item.client_socket = sock

        # Add to Task Queue
        self.queue_manager.submit_task(task_item)

        # Directly monitor execution on this socket channel
        self._stream_task_execution(task_item, sock)

    def _stream_task_execution(self, task: TaskItem, sock: socket.socket) -> None:
        """Executes task and streams real-time progress directly to client socket."""
        task.status = TaskStatus.RUNNING
        task.start_time = time.time()

        def progress_callback(percent: float, fps: float, speed: str, eta: float, status_text: str):
            task.progress_percent = percent
            task.current_fps = fps
            task.speed_multiplier = speed
            progress_packet = {
                "task_id": task.task_id,
                "percent": round(percent, 1),
                "fps": round(fps, 1),
                "speed": speed,
                "eta_seconds": round(eta, 1),
                "status": status_text
            }
            try:
                send_json(sock, MsgType.PROGRESS, progress_packet)
            except Exception:
                pass

            if self.on_task_update:
                try:
                    self.on_task_update({
                        "task_id": task.task_id,
                        "task_type": task.task_type,
                        "percent": percent,
                        "status": status_text
                    })
                except Exception:
                    pass

        try:
            # 1. Video Transcoding Workload
            if task.task_type == "video_transcode":
                res = self.engine.execute_video_transcode(
                    task.input_path,
                    task.output_path,
                    task.config,
                    progress_callback=progress_callback,
                    check_cancelled=lambda: task.is_cancelled
                )
            # 1b. Multi-Clip Merge & Render Workload
            elif task.task_type == "video_merge":
                input_paths = task.config.get("input_paths", [])
                self.log(f"   [MERGE ENGINE] Starting merge of {len(input_paths)} clips...")
                res = self.engine.execute_video_merge(
                    input_paths,
                    task.output_path,
                    task.config,
                    progress_callback=progress_callback,
                    check_cancelled=lambda: task.is_cancelled
                )
                if res.get("success"):
                    self.log(f"   [MERGE COMPLETE] {res.get('clips_merged',0)} clips merged & encoded in {res.get('duration',0):.2f}s")
            # 2. CUDA / Tensor / NumPy Matrix Workload
            elif task.task_type == "cuda_compute":
                matrix_size = task.config.get("matrix_size", 1024)
                iterations = task.config.get("iterations", 10)
                res = self.engine.execute_cuda_compute(
                    matrix_size,
                    iterations,
                    task.output_path,
                    progress_callback=progress_callback,
                    check_cancelled=lambda: task.is_cancelled
                )
            # 3. Complex Project Python Script Workload
            elif task.task_type in ["custom_script", "python_task"]:
                res = self.engine.execute_custom_script(
                    task.input_path,
                    task.output_path,
                    task.config,
                    progress_callback=progress_callback,
                    check_cancelled=lambda: task.is_cancelled
                )
            else:
                res = {"success": False, "error": f"Unknown task type: {task.task_type}"}

            task.end_time = time.time()
            task.execution_duration = task.end_time - task.start_time

            if res.get("success"):
                task.status = TaskStatus.COMPLETED
                self.total_tasks_completed += 1
                output_size = os.path.getsize(task.output_path) if os.path.exists(task.output_path) else 0
                output_sha256 = compute_sha256(task.output_path) if output_size > 0 else ""

                log_file_name = res.get("log_file_name", "")
                log_path = res.get("log_path", "")
                log_size = os.path.getsize(log_path) if log_path and os.path.exists(log_path) else 0
                log_sha256 = compute_sha256(log_path) if log_size > 0 else ""

                result_payload = {
                    "task_id": task.task_id,
                    "status": "COMPLETED",
                    "engine": res.get("engine", "Default"),
                    "duration_sec": round(task.execution_duration, 3),
                    "output_file_name": os.path.basename(task.output_path),
                    "output_size_bytes": output_size,
                    "output_sha256": output_sha256,
                    "log_file_name": log_file_name,
                    "log_size_bytes": log_size,
                    "log_sha256": log_sha256,
                    "logs_text": res.get("logs_text", ""),
                    "metrics": res
                }
                self.log(f"[+] [TASK COMPLETE] Task {task.task_id} completed in {task.execution_duration:.2f}s using {res.get('engine')}")
                if log_file_name:
                    self.log(f"   [RENDER LOG] Execution log generated: {log_file_name} ({format_bytes(log_size)})")
                send_json(sock, MsgType.JOB_RESULT, result_payload)
            else:
                task.status = TaskStatus.FAILED
                task.error_message = res.get("error", "Unknown execution error")
                self.log(f"[-] [TASK FAILED] Task {task.task_id}: {task.error_message}")
                send_json(sock, MsgType.ERROR, {"task_id": task.task_id, "error": task.error_message})

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            self.log(f"[TASK EXCEPTION] Task {task.task_id}: {e}")
            send_json(sock, MsgType.ERROR, {"task_id": task.task_id, "error": str(e)})

    def _process_file_share_upload(self, sock: socket.socket, payload: bytes, addr: tuple) -> None:
        """Receives a file directly shared from client to server storage without job execution."""
        import json
        meta = json.loads(payload.decode("utf-8"))
        file_name = meta.get("file_name", "shared_file.dat")
        file_size = meta.get("file_size", 0)
        expected_sha256 = meta.get("sha256", "")

        dest_path = os.path.join(self.storage_dir, os.path.basename(file_name))
        self.log(f"[*] [FILE SHARE] Incoming file from {addr[0]}: {file_name} ({format_bytes(file_size)})")

        send_json(sock, MsgType.FILE_SHARE_ACK, {"status": "READY_TO_RECEIVE"})

        hasher = hashlib.sha256()
        with open(dest_path, "wb") as f:
            while True:
                c_type, c_data = read_packet(sock)
                if c_type == MsgType.FILE_CHUNK:
                    f.write(c_data)
                    hasher.update(c_data)
                elif c_type == MsgType.FILE_EOF:
                    break

        computed_sha = hasher.hexdigest()
        verified = (computed_sha.lower() == expected_sha256.lower()) if expected_sha256 else True

        if verified:
            self.log(f"[+] [FILE SAVED] {file_name} saved to shared storage. SHA-256 verified.")
            send_json(sock, MsgType.FILE_SHARE_ACK, {"status": "SUCCESS", "file_name": file_name, "sha256": computed_sha})
        else:
            self.log(f"[-] [FILE ERROR] Checksum mismatch for {file_name}")
            send_json(sock, MsgType.ERROR, {"error": "Checksum mismatch during file share."})

    def _process_download_request(self, sock: socket.socket, payload: bytes) -> None:
        """Streams rendered output file or shared file back to client in chunks with SHA-256 validation."""
        import json
        req = json.loads(payload.decode("utf-8"))
        task_id = req.get("task_id", "")
        file_name = req.get("file_name", "")

        file_path = os.path.join(self.storage_dir, file_name)
        if not os.path.exists(file_path):
            send_json(sock, MsgType.ERROR, {"error": f"File '{file_name}' not found on server"})
            return

        file_size = os.path.getsize(file_path)
        sha256 = compute_sha256(file_path)
        self.log(f"[*] [STREAMING FILE] Streaming {file_name} ({format_bytes(file_size)}) to client...")

        # Send download metadata header
        send_json(sock, MsgType.JOB_ACCEPTED, {"file_name": file_name, "file_size": file_size, "sha256": sha256})

        # Stream chunks
        with open(file_path, "rb") as f:
            while chunk := f.read(CHUNK_SIZE):
                send_packet(sock, MsgType.FILE_CHUNK, chunk)

        # Send EOF
        send_packet(sock, MsgType.FILE_EOF, sha256.encode("utf-8"))
        self.log(f"[+] [DOWNLOAD COMPLETE] Successfully streamed {file_name} to client.")

    def get_shared_files(self) -> List[Dict[str, Any]]:
        """Returns details of all files stored in server_storage for file sharing."""
        results = []
        if not os.path.exists(self.storage_dir):
            return results

        for fname in os.listdir(self.storage_dir):
            fpath = os.path.join(self.storage_dir, fname)
            if os.path.isfile(fpath):
                stat = os.stat(fpath)
                results.append({
                    "name": fname,
                    "size_bytes": stat.st_size,
                    "size_str": format_bytes(stat.st_size),
                    "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
                })
        results.sort(key=lambda x: x["name"])
        return results

    def get_active_clients_list(self) -> List[Dict[str, Any]]:
        """Returns snapshot of connected clients for GUI display."""
        with self._lock:
            return list(self.connected_clients.values())

    def _queue_worker_loop(self) -> None:
        """Background worker thread loop for task queue execution."""
        while not self._shutdown_event.is_set():
            task = self.queue_manager.get_task(timeout=1.0)
            if task:
                self.queue_manager.task_done()
