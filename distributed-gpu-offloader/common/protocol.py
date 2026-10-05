"""Wire protocol shared by the client and the GPU worker daemon.

Every message on the TCP stream is a *frame*:

    +--------+-----------------+----------------------+
    | 1 byte | 4 bytes (BE)    | <length> bytes       |
    | type   | payload length  | payload              |
    +--------+-----------------+----------------------+

    type 'J' -> payload is a UTF-8 JSON object (control message)
    type 'D' -> payload is raw binary data (file chunk)

Handshake (Task 1)
------------------
    worker -> client : WELCOME {version, nonce, server}
    client -> worker : HELLO   {version, client, auth=HMAC-SHA256(token, nonce)}
    worker -> client : OK {server info}        or  ERROR {code:'auth'|'version'}

After the handshake the client may send PING / BWTEST / INFO / SUBMIT / WATCH /
FETCH / CANCEL / CLEANUP / BYE requests (see server/daemon.py).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import socket
import struct
from typing import Callable, Optional

PROTOCOL_VERSION = 1
DEFAULT_PORT = 5050
DEFAULT_TOKEN = "offload-secret"
CHUNK_SIZE = 1024 * 1024  # 1 MiB data frames
MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_DATA_BYTES = 16 * 1024 * 1024

FRAME_JSON = b"J"
FRAME_DATA = b"D"
_HEADER = struct.Struct("!cI")


# --------------------------------------------------------------------------- errors
class ProtocolError(Exception):
    """Malformed or unexpected data on the wire."""


class ConnectionClosed(ProtocolError):
    """The peer closed the TCP connection."""


class BusyError(ProtocolError):
    """The worker is temporarily busy with this job (retryable)."""


class ChecksumError(ProtocolError):
    """SHA-256 mismatch after a transfer (retryable, restarts the transfer)."""


class AuthError(ProtocolError):
    """Token / protocol version rejected by the worker."""


class RemoteError(ProtocolError):
    """The worker answered with an ERROR message."""

    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


class Cancelled(Exception):
    """The user cancelled the operation."""


# --------------------------------------------------------------------------- sockets
def tune_socket(sock: socket.socket) -> None:
    """Low-latency + large buffers; best effort."""
    for level, opt, val in (
        (socket.IPPROTO_TCP, socket.TCP_NODELAY, 1),
        (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
        (socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024),
        (socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024),
    ):
        try:
            sock.setsockopt(level, opt, val)
        except OSError:
            pass


def recv_exact(sock: socket.socket, n: int) -> bytearray:
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        r = sock.recv_into(view[got:], n - got)
        if r == 0:
            raise ConnectionClosed("peer closed the connection")
        got += r
    return buf


def send_json(sock: socket.socket, obj: dict) -> None:
    payload = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    sock.sendall(_HEADER.pack(FRAME_JSON, len(payload)) + payload)


def send_data(sock: socket.socket, data: bytes) -> None:
    sock.sendall(_HEADER.pack(FRAME_DATA, len(data)))
    sock.sendall(data)


def recv_frame(sock: socket.socket) -> tuple[bytes, bytearray]:
    kind, length = _HEADER.unpack(recv_exact(sock, _HEADER.size))
    if kind == FRAME_JSON and length > MAX_JSON_BYTES:
        raise ProtocolError(f"JSON frame too large ({length} bytes)")
    if kind == FRAME_DATA and length > MAX_DATA_BYTES:
        raise ProtocolError(f"data frame too large ({length} bytes)")
    if kind not in (FRAME_JSON, FRAME_DATA):
        raise ProtocolError(f"unknown frame type {kind!r}")
    return kind, recv_exact(sock, length) if length else bytearray()


def recv_json(sock: socket.socket) -> dict:
    kind, payload = recv_frame(sock)
    if kind != FRAME_JSON:
        raise ProtocolError("expected a control message, got a data frame")
    try:
        msg = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"invalid JSON frame: {exc}") from exc
    if not isinstance(msg, dict):
        raise ProtocolError("control message must be a JSON object")
    return msg


def expect(sock: socket.socket, *types: str) -> dict:
    """Receive one control message; raise on ERROR or an unexpected type."""
    msg = recv_json(sock)
    mtype = msg.get("type")
    if mtype == "ERROR":
        code = msg.get("code", "error")
        if code == "busy":
            raise BusyError(msg.get("message", "worker busy"))
        if code in ("auth", "version"):
            raise AuthError(msg.get("message", code))
        raise RemoteError(code, msg.get("message", ""))
    if types and mtype not in types:
        raise ProtocolError(f"expected {'/'.join(types)}, got {mtype!r}")
    return msg


def send_error(sock: socket.socket, code: str, message: str) -> None:
    try:
        send_json(sock, {"type": "ERROR", "code": code, "message": message})
    except OSError:
        pass


# --------------------------------------------------------------------------- file transfer
def send_file_range(
    sock: socket.socket,
    path: str,
    offset: int,
    total: int,
    progress_cb: Optional[Callable[[int], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> int:
    """Send bytes [offset, total) of *path* as data frames. Returns bytes sent."""
    sent = offset
    with open(path, "rb") as fh:
        fh.seek(offset)
        while sent < total:
            if cancel_check and cancel_check():
                raise Cancelled()
            chunk = fh.read(min(CHUNK_SIZE, total - sent))
            if not chunk:
                raise ProtocolError("file is shorter than its declared size")
            send_data(sock, chunk)
            sent += len(chunk)
            if progress_cb:
                progress_cb(sent)
    return sent - offset


def recv_file_range(
    sock: socket.socket,
    sink,
    expected: int,
    progress_cb: Optional[Callable[[int], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> int:
    """Receive exactly *expected* bytes of data frames and write them to *sink*."""
    got = 0
    while got < expected:
        if cancel_check and cancel_check():
            raise Cancelled()
        kind, payload = recv_frame(sock)
        if kind == FRAME_JSON:
            msg = json.loads(payload.decode("utf-8"))
            if msg.get("type") == "ERROR":
                raise RemoteError(msg.get("code", "error"), msg.get("message", ""))
            raise ProtocolError(f"unexpected control message during transfer: {msg.get('type')}")
        got += len(payload)
        if got > expected:
            raise ProtocolError("peer sent more data than announced")
        sink.write(payload)
        if progress_cb:
            progress_cb(got)
    return got


class NullSink:
    """File-like object that discards data (bandwidth tests)."""

    def write(self, data) -> int:  # noqa: D401
        return len(data)


# --------------------------------------------------------------------------- integrity / auth
def sha256_file(path: str, progress_cb: Optional[Callable[[int], None]] = None) -> str:
    h = hashlib.sha256()
    done = 0
    with open(path, "rb") as fh:
        while True:
            block = fh.read(4 * 1024 * 1024)
            if not block:
                break
            h.update(block)
            done += len(block)
            if progress_cb:
                progress_cb(done)
    return h.hexdigest()


def make_nonce() -> str:
    return os.urandom(16).hex()


def auth_response(token: str, nonce: str) -> str:
    return hmac.new(token.encode("utf-8"), nonce.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_auth(token: str, nonce: str, response: str) -> bool:
    return hmac.compare_digest(auth_response(token, nonce), str(response))
