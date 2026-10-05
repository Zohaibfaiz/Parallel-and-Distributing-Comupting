"""End-to-end tests on loopback (no NVIDIA GPU needed: the worker falls back to CPU).

Run:  python -m unittest discover -s tests -v
"""
from __future__ import annotations

import os
import shutil
import socket
import sys
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from client.net_client import OffloadClient, TransferError  # noqa: E402
from common.ffmpeg_utils import gen_test_video, normalize_settings, probe  # noqa: E402
from common.protocol import (  # noqa: E402
    PROTOCOL_VERSION, auth_response, expect, send_file_range, send_json, sha256_file, tune_socket,
    recv_json,
)
from server.daemon import WorkerServer  # noqa: E402
from server.gpu_engine import GpuEngine  # noqa: E402

TOKEN = "unit-test-token"
HAS_FFMPEG = shutil.which("ffmpeg") and shutil.which("ffprobe")


@unittest.skipUnless(HAS_FFMPEG, "ffmpeg/ffprobe required")
class SystemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="offload-test-"))
        cls.video = str(cls.tmp / "clip.mp4")
        gen_test_video(cls.video, 640, 360, 3, fps=24)
        cls.engine = GpuEngine(mode="cpu")  # deterministic on any machine
        cls.server = WorkerServer("127.0.0.1", 0, TOKEN, cls.tmp / "work", cls.engine).start()
        cls.port = cls.server.port

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def client(self, **kw) -> OffloadClient:
        kw.setdefault("log", lambda m: None)
        kw.setdefault("timeout", 10)
        return OffloadClient("127.0.0.1", self.port, kw.pop("token", TOKEN), **kw)

    # ---------------------------------------------------------------- Task 1
    def test_handshake_and_ping(self):
        r = self.client().check_worker(pings=5)
        self.assertTrue(r["reachable"], r)
        self.assertLess(r["rtt_avg_ms"], 100)
        self.assertEqual(r["server"]["version"], PROTOCOL_VERSION)

    def test_wrong_token_rejected(self):
        r = self.client(token="wrong").check_worker()
        self.assertFalse(r["reachable"])
        self.assertIn("authentication", r["error"])

    def test_unreachable_worker_reports_cleanly(self):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()  # nothing listens here any more
        r = OffloadClient("127.0.0.1", port, TOKEN, timeout=2).check_worker()
        self.assertFalse(r["reachable"])
        self.assertIn("refused", r["error"])

    def test_bandwidth(self):
        r = self.client().bandwidth_test(4)
        self.assertGreater(r["upload_mbps"], 1)
        self.assertGreater(r["download_mbps"], 1)

    # ---------------------------------------------------------------- Task 2 + 4
    def test_full_offload_roundtrip(self):
        out = str(self.tmp / "out.mp4")
        events = []
        c = self.client(on_progress=events.append)
        res = c.run_transcode(self.video, {"resolution": "240", "bitrate": "500k", "preset": "p1"}, out)
        info = probe(out)
        self.assertEqual(info["height"], 240)
        self.assertEqual(res["output_size"], os.path.getsize(out))
        self.assertGreater(res["render_s"], 0)
        stages = {e["stage"] for e in events}
        self.assertTrue({"hashing", "uploading", "rendering", "downloading", "done"} <= stages, stages)
        self.assertEqual(max(e["overall"] for e in events), 100)

    def test_bad_settings_rejected(self):
        with self.assertRaises(ValueError):
            normalize_settings({"bitrate": "5M; rm -rf /"})
        with self.assertRaises(ValueError):
            normalize_settings({"preset": "../../etc"})

    def test_corrupted_upload_is_detected(self):
        """Send the right size but a wrong SHA-256: the worker must refuse the file."""
        sock = socket.create_connection(("127.0.0.1", self.port), timeout=10)
        tune_socket(sock)
        w = expect(sock, "WELCOME")
        send_json(sock, {"type": "HELLO", "version": PROTOCOL_VERSION, "client": "t",
                         "auth": auth_response(TOKEN, w["nonce"])})
        expect(sock, "OK")
        size = os.path.getsize(self.video)
        send_json(sock, {"type": "SUBMIT", "job_id": "ab" * 16, "kind": "transcode", "filename": "x.mp4",
                         "size": size, "sha256": "0" * 64, "settings": {}})
        acc = expect(sock, "ACCEPT")
        self.assertTrue(acc["upload_required"])
        send_file_range(sock, self.video, 0, size)
        msg = expect(sock, "UPLOAD_OK", "UPLOAD_BAD")
        self.assertEqual(msg["type"], "UPLOAD_BAD")
        sock.close()

    def test_upload_resume_after_disconnect(self):
        """Kill the connection mid-upload; the client must resume from the received offset."""
        big = str(self.tmp / "big.mp4")
        gen_test_video(big, 1920, 1080, 3, fps=24)
        state = {"n": 0, "fired": False}

        def hook(point):
            if point == "upload":
                state["n"] += 1
                if state["n"] == 2 and not state["fired"]:
                    state["fired"] = True
                    raise ConnectionResetError("simulated cable pull")

        logs = []
        c = self.client(log=logs.append)
        c.fault_hook = hook
        out = str(self.tmp / "resumed.mp4")
        res = c.run_transcode(big, {"resolution": "360", "bitrate": "400k", "preset": "p1"}, out)
        self.assertTrue(state["fired"])
        self.assertGreaterEqual(res["retries"], 1)
        self.assertTrue(any("resuming upload" in m for m in logs), logs)
        self.assertEqual(probe(out)["height"], 360)

    def test_progress_stream_survives_disconnect(self):
        state = {"n": 0}

        def hook(point):
            if point == "watch":
                state["n"] += 1
                if state["n"] == 4:
                    raise ConnectionResetError("simulated drop during progress stream")

        c = self.client()
        c.fault_hook = hook
        out = str(self.tmp / "watch.mp4")
        res = c.run_transcode(self.video, {"resolution": "240", "bitrate": "400k", "preset": "p1"}, out)
        self.assertTrue(os.path.exists(out))
        self.assertGreaterEqual(res["retries"], 1)

    def test_download_resume_and_corruption(self):
        state = {"n": 0}

        def hook(point):
            if point == "download":
                state["n"] += 1
                if state["n"] == 1:
                    raise ConnectionResetError("simulated drop during download")

        c = self.client()
        c.fault_hook = hook
        out = str(self.tmp / "dl.mp4")
        c.run_transcode(self.video, {"resolution": "240", "bitrate": "3M", "preset": "p1"}, out)
        self.assertTrue(os.path.exists(out))
        self.assertFalse(os.path.exists(out + ".part"))

    def test_gives_up_after_max_retries(self):
        c = self.client(max_retries=1)

        def hook(point):
            if point == "watch":
                raise ConnectionResetError("always failing")

        c.fault_hook = hook
        import client.net_client as nc
        orig = nc.OffloadClient._sleep
        nc.OffloadClient._sleep = lambda self, s: None  # do not wait in tests
        try:
            with self.assertRaises(TransferError):
                c.run_transcode(self.video, {"resolution": "240", "bitrate": "400k"}, str(self.tmp / "x.mp4"))
        finally:
            nc.OffloadClient._sleep = orig

    def test_cancel(self):
        from common.protocol import Cancelled
        ev = threading.Event()
        c = self.client(cancel_event=ev)
        big = str(self.tmp / "cancel.mp4")
        gen_test_video(big, 1920, 1080, 6, fps=30)

        def hook(point):
            if point == "watch":
                ev.set()

        c.fault_hook = hook
        with self.assertRaises(Cancelled):
            c.run_transcode(big, {"resolution": "1080", "bitrate": "8M", "preset": "p7"},
                            str(self.tmp / "cancelled.mp4"))

    def test_unknown_job_and_bad_id(self):
        sock = socket.create_connection(("127.0.0.1", self.port), timeout=10)
        w = expect(sock, "WELCOME")
        send_json(sock, {"type": "HELLO", "version": PROTOCOL_VERSION, "client": "t",
                         "auth": auth_response(TOKEN, w["nonce"])})
        expect(sock, "OK")
        send_json(sock, {"type": "WATCH", "job_id": "f" * 32})
        self.assertEqual(recv_json(sock)["code"], "unknown_job")
        send_json(sock, {"type": "SUBMIT", "job_id": "../../etc/passwd", "size": 1, "sha256": "0" * 64})
        self.assertEqual(recv_json(sock)["code"], "bad_job_id")
        sock.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
