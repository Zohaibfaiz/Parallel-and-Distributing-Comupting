"""
================================================================================
CSC-334: Parallel and Distributed Computing
Lab 4: Custom Distributed Task Offloading & Remote GPU Rendering System
Module: server/protocol.py
Description: Robust Binary + JSON Socket Framing Protocol for Task Offloading.
================================================================================
"""

import json
import struct
import socket
import time
from typing import Tuple, Dict, Any, Optional

# Protocol Header Specification:
# [ 2 bytes MAGIC: 'DC' ] [ 2 bytes MSG_TYPE ] [ 4 bytes PAYLOAD_LENGTH (uint32, big-endian) ]
MAGIC_BYTES = b"DC"
HEADER_FORMAT = ">2s2sI"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # 8 bytes

# Message Types (2 ASCII characters)
class MsgType:
    HANDSHAKE_SYN = b"HS"  # Client -> Server: Handshake initialization
    HANDSHAKE_ACK = b"HA"  # Server -> Client: Handshake acknowledgement & system capabilities
    PING_REQ      = b"PI"  # Client -> Server: Latency measurement ping
    PONG_RESP     = b"PO"  # Server -> Client: Latency measurement pong
    JOB_SUBMIT    = b"JS"  # Client -> Server: Job config & incoming file metadata
    JOB_ACCEPTED  = b"JA"  # Server -> Client: Job queued / accepted confirmation
    FILE_CHUNK    = b"FC"  # Raw Binary: Transferred chunk of asset/output
    FILE_EOF      = b"FE"  # End-Of-File marker with checksum
    PROGRESS      = b"PR"  # Server -> Client: Real-time progress percentage & stats
    JOB_RESULT    = b"JR"  # Server -> Client: Final job completion status & metrics
    DOWNLOAD_REQ  = b"DR"  # Client -> Server: Request to stream rendered output back
    CANCEL_REQ    = b"CR"  # Client -> Server: Cancel current job
    FILE_LIST_REQ = b"FL"  # Client -> Server: Request list of shared files
    FILE_LIST_RESP= b"FR"  # Server -> Client: Response with list of shared files
    FILE_SHARE_UP = b"FU"  # Client -> Server: Upload file to shared storage
    FILE_SHARE_ACK= b"FA"  # Server -> Client: Acknowledge shared file upload
    ERROR         = b"ER"  # Bi-directional: Error code & reason

# Default Network Configuration Constants
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5000
CHUNK_SIZE = 64 * 1024  # 64 KB chunk size for network transfers
SOCKET_TIMEOUT = 15.0   # 15 seconds socket read timeout
PROTOCOL_VERSION = "2.0.0"


def send_packet(sock: socket.socket, msg_type: bytes, payload: bytes = b"") -> None:
    """
    Packs and sends a complete framed packet over a TCP socket.
    Format: [ MAGIC (2B) | MSG_TYPE (2B) | LENGTH (4B) ] [ PAYLOAD (NB) ]
    """
    if len(msg_type) != 2:
        raise ValueError("msg_type must be exactly 2 bytes")
    
    length = len(payload)
    header = struct.pack(HEADER_FORMAT, MAGIC_BYTES, msg_type, length)
    sock.sendall(header + payload)


def send_json(sock: socket.socket, msg_type: bytes, data: Dict[str, Any]) -> None:
    """Serializes a Python dict into JSON bytes and sends it."""
    payload = json.dumps(data).encode("utf-8")
    send_packet(sock, msg_type, payload)


def recvall(sock: socket.socket, num_bytes: int) -> bytes:
    """
    Reliably receives exactly `num_bytes` from the socket stream.
    Addresses TCP stream fragmentation and Head-of-Line chunking.
    Raises ConnectionError if connection closes prematurely.
    """
    buffer = bytearray(num_bytes)
    view = memoryview(buffer)
    received = 0
    while received < num_bytes:
        chunk = sock.recv_into(view[received:], num_bytes - received)
        if chunk == 0:
            raise ConnectionResetError("Connection closed unexpectedly by remote peer.")
        received += chunk
    return bytes(buffer)


def read_packet(sock: socket.socket) -> Tuple[bytes, bytes]:
    """
    Reads the 8-byte header, validates magic bytes, and reads the full payload.
    Returns (msg_type, payload_bytes).
    """
    header_data = recvall(sock, HEADER_SIZE)
    magic, msg_type, length = struct.unpack(HEADER_FORMAT, header_data)

    if magic != MAGIC_BYTES:
        raise ValueError(f"Invalid protocol magic bytes: {magic!r}. Expected {MAGIC_BYTES!r}")

    payload = recvall(sock, length) if length > 0 else b""
    return msg_type, payload


def read_json(sock: socket.socket) -> Tuple[bytes, Dict[str, Any]]:
    """Reads a packet and deserializes its payload as JSON."""
    msg_type, payload = read_packet(sock)
    data = json.loads(payload.decode("utf-8")) if payload else {}
    return msg_type, data
