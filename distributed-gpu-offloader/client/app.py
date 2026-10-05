#!/usr/bin/env python3
"""Desktop GUI client (Task 3) - CustomTkinter.

    python client/app.py

Left column : worker connection, input file, render settings, output folder.
Right column: stage + progress bar, live stats, action buttons, live log terminal.
All network / render work runs in background threads and talks to the UI through a
queue (Tk widgets are only touched from the main thread).
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from client.local_render import local_nvenc_available, render_local  # noqa: E402
from client.net_client import JobFailed, OffloadClient, TransferError  # noqa: E402
from common.ffmpeg_utils import (  # noqa: E402
    CODECS, PRESETS, RESOLUTIONS, EncodeCancelled, EncodeError, find_binary, probe,
)
from common.protocol import DEFAULT_PORT, DEFAULT_TOKEN, AuthError, Cancelled, RemoteError  # noqa: E402
from common.util import human_bytes, human_secs  # noqa: E402

CONFIG_PATH = Path.home() / ".gpu_offload_client.json"
PRESET_HELP = {
    "p1": "p1 - fastest, lowest quality", "p2": "p2 - faster", "p3": "p3 - fast",
    "p4": "p4 - balanced (default)", "p5": "p5 - slow, good quality", "p6": "p6 - slower",
    "p7": "p7 - slowest, best quality",
}
BITRATES = ["1M", "2M", "4M", "5M", "8M", "12M", "20M", "40M"]
STAGE_LABEL = {"hashing": "Computing checksum", "uploading": "Uploading to worker", "queued": "Waiting in queue",
               "rendering": "Rendering on remote GPU", "downloading": "Downloading result",
               "verifying": "Verifying integrity", "done": "Finished", "local": "Rendering locally"}


class App(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.title("Distributed GPU Offloader - Client")
        self.geometry("1180x760")
        self.minsize(1020, 680)

        self.q: "queue.Queue[tuple]" = queue.Queue()
        self.cancel_event = threading.Event()
        self.busy = False
        self.last_local_s: dict[tuple, float] = {}
        self.last_remote_s: dict[tuple, float] = {}
        self.last_output: str | None = None
        self.t_start = 0.0

        self._build()
        self._load_config()
        self._log("Client ready. Enter the worker IP, press 'Test connection', pick a video and press "
                  "'Start offload'.", "info")
        if not find_binary("ffmpeg"):
            self._log("⚠ ffmpeg not found locally - 'Render locally' will not work (offloading still does).",
                      "warn")
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._drain)

    # ------------------------------------------------------------------ UI construction
    def _build(self) -> None:
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        side = ctk.CTkScrollableFrame(self, width=330, corner_radius=0, label_text="Job configuration",
                                      label_font=ctk.CTkFont(size=15, weight="bold"))
        side.grid(row=0, column=0, sticky="nsew")

        # --- connection
        self._section(side, "1  Remote worker")
        row = ctk.CTkFrame(side, fg_color="transparent"); row.pack(fill="x", padx=10, pady=2)
        self.ip_var = ctk.StringVar(value="192.168.1.1")
        self.port_var = ctk.StringVar(value=str(DEFAULT_PORT))
        ctk.CTkLabel(row, text="Server IP", width=70, anchor="w").pack(side="left")
        ctk.CTkEntry(row, textvariable=self.ip_var, width=130).pack(side="left", padx=(0, 6))
        ctk.CTkLabel(row, text="Port", width=34, anchor="w").pack(side="left")
        ctk.CTkEntry(row, textvariable=self.port_var, width=60).pack(side="left")
        row = ctk.CTkFrame(side, fg_color="transparent"); row.pack(fill="x", padx=10, pady=2)
        self.token_var = ctk.StringVar(value=DEFAULT_TOKEN)
        ctk.CTkLabel(row, text="Token", width=70, anchor="w").pack(side="left")
        ctk.CTkEntry(row, textvariable=self.token_var, show="•", width=200).pack(side="left")
        self.btn_test = ctk.CTkButton(side, text="Test connection (ping)", command=self._test_connection)
        self.btn_test.pack(fill="x", padx=10, pady=(6, 2))
        self.status_lbl = ctk.CTkLabel(side, text="●  not tested", text_color="gray70", anchor="w",
                                       justify="left", wraplength=290)
        self.status_lbl.pack(fill="x", padx=12, pady=(0, 4))

        # --- input
        self._section(side, "2  Input video")
        row = ctk.CTkFrame(side, fg_color="transparent"); row.pack(fill="x", padx=10, pady=2)
        self.file_var = ctk.StringVar()
        ctk.CTkEntry(row, textvariable=self.file_var, placeholder_text="No file selected", width=210
                     ).pack(side="left", padx=(0, 6))
        ctk.CTkButton(row, text="Browse…", width=70, command=self._pick_file).pack(side="left")
        self.file_info = ctk.CTkLabel(side, text="", text_color="gray70", anchor="w", justify="left",
                                      wraplength=290)
        self.file_info.pack(fill="x", padx=12)

        # --- settings
        self._section(side, "3  Render quality")
        self.res_var = ctk.StringVar(value="original")
        self.codec_var = ctk.StringVar(value="h264")
        self.bitrate_var = ctk.StringVar(value="5M")
        self.preset_var = ctk.StringVar(value=PRESET_HELP["p4"])
        self._option(side, "Resolution", self.res_var,
                     [r if r == "original" else f"{r}" for r in RESOLUTIONS])
        ctk.CTkLabel(side, text="(height in pixels; 'original' keeps the source size)", text_color="gray60",
                     font=ctk.CTkFont(size=11), anchor="w").pack(fill="x", padx=12)
        self._option(side, "Codec", self.codec_var, list(CODECS))
        row = ctk.CTkFrame(side, fg_color="transparent"); row.pack(fill="x", padx=10, pady=2)
        ctk.CTkLabel(row, text="Bitrate", width=70, anchor="w").pack(side="left")
        ctk.CTkComboBox(row, variable=self.bitrate_var, values=BITRATES, width=120).pack(side="left")
        ctk.CTkLabel(row, text="e.g. 8M or 800k", text_color="gray60",
                     font=ctk.CTkFont(size=11)).pack(side="left", padx=6)
        self._option(side, "Preset", self.preset_var, [PRESET_HELP[p] for p in PRESETS])

        # --- output
        self._section(side, "4  Output folder")
        row = ctk.CTkFrame(side, fg_color="transparent"); row.pack(fill="x", padx=10, pady=(2, 10))
        self.out_var = ctk.StringVar(value=str(Path.home() / "Videos" if (Path.home() / "Videos").exists()
                                               else Path.home()))
        ctk.CTkEntry(row, textvariable=self.out_var, width=210).pack(side="left", padx=(0, 6))
        ctk.CTkButton(row, text="Browse…", width=70, command=self._pick_outdir).pack(side="left")

        # ------------------------------------------------------------ right column
        main = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        main.grid(row=0, column=1, sticky="nsew", padx=14, pady=12)
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(4, weight=1)

        ctk.CTkLabel(main, text="Remote GPU Rendering", font=ctk.CTkFont(size=24, weight="bold"),
                     anchor="w").grid(row=0, column=0, sticky="ew")
        self.stage_lbl = ctk.CTkLabel(main, text="Idle", anchor="w", font=ctk.CTkFont(size=14))
        self.stage_lbl.grid(row=1, column=0, sticky="ew", pady=(8, 2))
        prow = ctk.CTkFrame(main, fg_color="transparent"); prow.grid(row=2, column=0, sticky="ew")
        prow.grid_columnconfigure(0, weight=1)
        self.progress = ctk.CTkProgressBar(prow, height=18)
        self.progress.set(0)
        self.progress.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.pct_lbl = ctk.CTkLabel(prow, text="0.0 %", width=70, font=ctk.CTkFont(size=14, weight="bold"))
        self.pct_lbl.grid(row=0, column=1)
        self.detail_lbl = ctk.CTkLabel(main, text="", anchor="w", text_color="gray70")
        self.detail_lbl.grid(row=3, column=0, sticky="ew", pady=(4, 6))

        # buttons
        brow = ctk.CTkFrame(main, fg_color="transparent")
        brow.grid(row=5, column=0, sticky="ew", pady=(8, 0))
        self.btn_start = ctk.CTkButton(brow, text="▶  Start offload", height=38, command=self._start_offload)
        self.btn_start.pack(side="left", padx=(0, 8))
        self.btn_cancel = ctk.CTkButton(brow, text="■  Cancel", height=38, fg_color="#a33", hover_color="#822",
                                        state="disabled", command=self._cancel)
        self.btn_cancel.pack(side="left", padx=(0, 8))
        self.btn_local = ctk.CTkButton(brow, text="Render locally (compare)", height=38, fg_color="gray30",
                                       hover_color="gray25", command=self._start_local)
        self.btn_local.pack(side="left", padx=(0, 8))
        self.btn_open = ctk.CTkButton(brow, text="Open output folder", height=38, fg_color="gray30",
                                      hover_color="gray25", command=self._open_output)
        self.btn_open.pack(side="left")

        # log terminal
        log_frame = ctk.CTkFrame(main)
        log_frame.grid(row=4, column=0, sticky="nsew")
        log_frame.grid_columnconfigure(0, weight=1)
        log_frame.grid_rowconfigure(1, weight=1)
        hdr = ctk.CTkFrame(log_frame, fg_color="transparent"); hdr.grid(row=0, column=0, sticky="ew", padx=8, pady=4)
        ctk.CTkLabel(hdr, text="Live log terminal", font=ctk.CTkFont(weight="bold")).pack(side="left")
        ctk.CTkButton(hdr, text="Clear", width=60, height=24, fg_color="gray30",
                      command=self._clear_log).pack(side="right")
        self.log_box = ctk.CTkTextbox(log_frame, font=ctk.CTkFont(family="Consolas", size=12), wrap="word",
                                      state="disabled")
        self.log_box.grid(row=1, column=0, sticky="nsew", padx=6, pady=(0, 6))
        for tag, color in (("info", "#9ecbff"), ("ok", "#7ee787"), ("warn", "#f2cc60"), ("error", "#ff7b72"),
                           ("worker", "#c9d1d9"), ("plain", "#8b949e")):
            self.log_box.tag_config(tag, foreground=color)

    def _section(self, parent, text: str) -> None:
        ctk.CTkLabel(parent, text=text, font=ctk.CTkFont(size=13, weight="bold"), anchor="w"
                     ).pack(fill="x", padx=10, pady=(12, 2))

    def _option(self, parent, label: str, var: ctk.StringVar, values: list[str]) -> None:
        row = ctk.CTkFrame(parent, fg_color="transparent"); row.pack(fill="x", padx=10, pady=2)
        ctk.CTkLabel(row, text=label, width=70, anchor="w").pack(side="left")
        ctk.CTkOptionMenu(row, variable=var, values=values, width=200).pack(side="left")

    # ------------------------------------------------------------------ config persistence
    def _load_config(self) -> None:
        try:
            cfg = json.loads(CONFIG_PATH.read_text())
        except (OSError, ValueError):
            return
        self.ip_var.set(cfg.get("ip", self.ip_var.get()))
        self.port_var.set(str(cfg.get("port", self.port_var.get())))
        self.token_var.set(cfg.get("token", self.token_var.get()))
        self.res_var.set(cfg.get("resolution", "original"))
        self.codec_var.set(cfg.get("codec", "h264"))
        self.bitrate_var.set(cfg.get("bitrate", "5M"))
        self.preset_var.set(PRESET_HELP.get(cfg.get("preset", "p4"), PRESET_HELP["p4"]))
        if cfg.get("out_dir") and Path(cfg["out_dir"]).exists():
            self.out_var.set(cfg["out_dir"])

    def _save_config(self) -> None:
        cfg = {"ip": self.ip_var.get().strip(), "port": self.port_var.get().strip(), "token": self.token_var.get(),
               "resolution": self.res_var.get(), "codec": self.codec_var.get(),
               "bitrate": self.bitrate_var.get().strip(), "preset": self._preset(), "out_dir": self.out_var.get()}
        try:
            CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
        except OSError:
            pass

    # ------------------------------------------------------------------ helpers
    def _preset(self) -> str:
        return self.preset_var.get().split(" ")[0]

    def _settings(self) -> dict:
        return {"resolution": self.res_var.get(), "bitrate": self.bitrate_var.get().strip(),
                "preset": self._preset(), "codec": self.codec_var.get(), "engine": "auto"}

    def _log(self, text: str, tag: str | None = None) -> None:
        if tag is None:
            if text.startswith("[worker]"):
                tag = "worker"
            elif text.startswith("✔"):
                tag = "ok"
            elif text.startswith("⚠"):
                tag = "warn"
            elif text.startswith("✖"):
                tag = "error"
            else:
                tag = "plain"
        stamp = time.strftime("%H:%M:%S")
        self.log_box.configure(state="normal")
        self.log_box.insert("end", f"{stamp}  {text}\n", tag)
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")

    def _set_busy(self, busy: bool) -> None:
        self.busy = busy
        self.btn_start.configure(state="disabled" if busy else "normal")
        self.btn_local.configure(state="disabled" if busy else "normal")
        self.btn_test.configure(state="disabled" if busy else "normal")
        self.btn_cancel.configure(state="normal" if busy else "disabled")

    def _client(self) -> OffloadClient:
        try:
            port = int(self.port_var.get())
        except ValueError:
            raise ValueError("Port must be a number")
        host = self.ip_var.get().strip()
        if not host:
            raise ValueError("Enter the worker's IP address")
        return OffloadClient(host, port, self.token_var.get(), log=lambda m: self.q.put(("log", m)),
                             on_progress=lambda p: self.q.put(("progress", p)), cancel_event=self.cancel_event)

    def _validate_input(self) -> str | None:
        path = self.file_var.get().strip()
        if not path or not os.path.isfile(path):
            messagebox.showwarning("No input", "Please choose an existing video file first.")
            return None
        if not Path(self.out_var.get()).is_dir():
            messagebox.showwarning("Output folder", "The output folder does not exist.")
            return None
        return path

    def _output_path(self, src: str, suffix: str) -> str:
        s = self._settings()
        name = f"{Path(src).stem}_{suffix}_{s['resolution']}_{s['preset']}_{s['codec']}.mp4"
        return str(Path(self.out_var.get()) / name)

    def _key(self, src: str) -> tuple:
        s = self._settings()
        return (src, s["resolution"], s["bitrate"], s["preset"], s["codec"])

    # ------------------------------------------------------------------ file pickers
    def _pick_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Select input video",
            filetypes=[("Video files", "*.mp4 *.mkv *.mov *.avi *.webm *.m4v *.flv *.wmv *.ts"), ("All files", "*.*")])
        if not path:
            return
        self.file_var.set(path)
        text = human_bytes(os.path.getsize(path))
        if find_binary("ffprobe"):
            try:
                i = probe(path)
                text = (f"{i['width']}x{i['height']}  {i['vcodec']}  {i['fps']:.0f} fps  "
                        f"{human_secs(i['duration'])}  {text}")
            except EncodeError:
                pass
        self.file_info.configure(text=text)
        self._log(f"Selected {path} ({text})", "info")

    def _pick_outdir(self) -> None:
        d = filedialog.askdirectory(title="Select output folder", initialdir=self.out_var.get())
        if d:
            self.out_var.set(d)

    def _open_output(self) -> None:
        target = Path(self.last_output).parent if self.last_output else Path(self.out_var.get())
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(target))  # noqa: S606
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(target)])
            else:
                subprocess.Popen(["xdg-open", str(target)])
        except OSError as exc:
            messagebox.showerror("Open folder", str(exc))

    # ------------------------------------------------------------------ actions
    def _test_connection(self) -> None:
        try:
            client = self._client()
        except ValueError as exc:
            messagebox.showwarning("Connection", str(exc))
            return
        self._save_config()
        self.status_lbl.configure(text="●  testing…", text_color="#f2cc60")
        self._log(f"Testing connection to {client.host}:{client.port} …", "info")
        self.btn_test.configure(state="disabled")

        def work():
            r = client.check_worker(pings=10)
            if r["reachable"]:
                try:
                    r["bandwidth"] = client.bandwidth_test(16)
                except Exception as exc:  # noqa: BLE001
                    r["bandwidth_error"] = str(exc)
            self.q.put(("ping", r))

        threading.Thread(target=work, daemon=True).start()

    def _start_offload(self) -> None:
        src = self._validate_input()
        if not src:
            return
        try:
            client = self._client()
        except ValueError as exc:
            messagebox.showwarning("Connection", str(exc))
            return
        self._save_config()
        out = self._output_path(src, "gpu")
        settings = self._settings()
        self.cancel_event.clear()
        self._set_busy(True)
        self.t_start = time.perf_counter()
        self.progress.set(0)
        self.last_output = out
        self._log(f"Settings: {settings}", "info")

        def work():
            try:
                chk = client.check_worker(pings=3)
                if not chk["reachable"]:
                    raise TransferError(f"worker not reachable: {chk['error']}")
                self.q.put(("log", f"✔ Worker online - RTT {chk['rtt_avg_ms']} ms"
                                   f"{' | NVENC ready' if chk['server'].get('nvenc') else ' | NO NVENC (CPU fallback)'}"))
                res = client.run_transcode(src, settings, out)
                self.q.put(("done", res, src))
            except Cancelled:
                self.q.put(("cancelled",))
            except (AuthError, RemoteError, TransferError, JobFailed) as exc:
                self.q.put(("error", str(exc)))
            except Exception as exc:  # noqa: BLE001
                self.q.put(("error", f"unexpected error: {exc!r}"))

        threading.Thread(target=work, daemon=True).start()

    def _start_local(self) -> None:
        src = self._validate_input()
        if not src:
            return
        if not find_binary("ffmpeg"):
            messagebox.showerror("ffmpeg missing", "Install FFmpeg on this machine to render locally.")
            return
        self._save_config()
        out = self._output_path(src, "local")
        settings = self._settings()
        self.cancel_event.clear()
        self._set_busy(True)
        self.progress.set(0)
        eng = "NVENC (local GPU)" if local_nvenc_available() else "CPU (libx264/libx265)"
        self._log(f"Local render on this laptop using {eng} …", "info")

        def work():
            try:
                res = render_local(src, out, settings, "auto",
                                   lambda p: self.q.put(("local_progress", p)), self.cancel_event)
                self.q.put(("local_done", res, src, out))
            except EncodeCancelled:
                self.q.put(("cancelled",))
            except Exception as exc:  # noqa: BLE001
                self.q.put(("error", f"local render failed: {exc}"))

        threading.Thread(target=work, daemon=True).start()

    def _cancel(self) -> None:
        self.cancel_event.set()
        self.btn_cancel.configure(state="disabled")
        self._log("Cancelling …", "warn")

    # ------------------------------------------------------------------ queue pump (main thread)
    def _drain(self) -> None:
        try:
            while True:
                msg = self.q.get_nowait()
                getattr(self, f"_on_{msg[0]}")(*msg[1:])
        except queue.Empty:
            pass
        self.after(100, self._drain)

    def _on_log(self, text: str) -> None:
        self._log(text)

    def _on_progress(self, p: dict) -> None:
        self.progress.set(p["overall"] / 100)
        self.pct_lbl.configure(text=f"{p['overall']:.1f} %")
        self.stage_lbl.configure(text=f"{STAGE_LABEL.get(p['stage'], p['stage'])}  ({p['stage_percent']:.0f}%)")
        self.detail_lbl.configure(text=f"{p.get('text', '')}   |   elapsed {human_secs(time.perf_counter() - self.t_start)}")

    def _on_local_progress(self, p: dict) -> None:
        self.progress.set(p["percent"] / 100)
        self.pct_lbl.configure(text=f"{p['percent']:.1f} %")
        self.stage_lbl.configure(text=STAGE_LABEL["local"])
        eta = f"  ETA {human_secs(p['eta_s'])}" if p.get("eta_s") is not None else ""
        self.detail_lbl.configure(text=f"{p['fps']:.0f} fps  {p['speed']:.2f}x{eta}")

    def _on_ping(self, r: dict) -> None:
        self.btn_test.configure(state="normal")
        if not r["reachable"]:
            self.status_lbl.configure(text=f"●  OFFLINE - {r['error']}", text_color="#ff7b72")
            self._log(f"✖ Worker not reachable: {r['error']}" + (f" ({r['hint']})" if r["hint"] else ""))
            return
        s = r["server"]
        gpus = ", ".join(g["name"] for g in s.get("gpus", [])) or "no GPU reported"
        bw = r.get("bandwidth")
        bw_txt = f" | ↑{bw['upload_mbps']:.0f} ↓{bw['download_mbps']:.0f} Mbit/s" if bw else ""
        self.status_lbl.configure(
            text=f"●  ONLINE  {s['server']}\nRTT {r['rtt_avg_ms']:.2f} ms (jitter {r['jitter_ms']:.2f}){bw_txt}\n"
                 f"{gpus} | NVENC {'✔' if s.get('nvenc') else '✖'}",
            text_color="#7ee787" if s.get("nvenc") else "#f2cc60")
        self._log(f"✔ Worker {s['server']} online: RTT avg {r['rtt_avg_ms']} ms, min {r['rtt_min_ms']}, "
                  f"max {r['rtt_max_ms']}, jitter {r['jitter_ms']}")
        if bw:
            self._log(f"✔ Link throughput: upload {bw['upload_mbps']} Mbit/s, download {bw['download_mbps']} Mbit/s")
        if r["hint"]:
            self._log(f"⚠ {r['hint']}")

    def _compare(self, key: tuple) -> None:
        if key in self.last_local_s and key in self.last_remote_s:
            l, r = self.last_local_s[key], self.last_remote_s[key]
            self._log(f"★ Local {l:.1f}s vs remote {r:.1f}s end-to-end  →  speedup ×{l / r:.2f}", "ok")

    def _on_done(self, res: dict, src: str) -> None:
        self._set_busy(False)
        self.progress.set(1)
        self.pct_lbl.configure(text="100 %")
        self.stage_lbl.configure(text="Finished")
        self.detail_lbl.configure(text=f"Saved to {res['output_path']}")
        sr = res.get("server_resource") or {}
        self._log(f"Timing: hash {res['hash_s']:.1f}s | upload {res['upload_s']:.1f}s | queue "
                  f"{res['queue_wait_s']:.1f}s | render {res['render_s']:.1f}s ({res['engine']}) | download "
                  f"{res['download_s']:.1f}s | verify {res['verify_s']:.1f}s | TOTAL {res['total_s']:.1f}s", "info")
        if sr.get("gpu_util_avg") is not None:
            self._log(f"Worker GPU avg {sr['gpu_util_avg']}% (max {sr['gpu_util_max']}%), VRAM max "
                      f"{sr['gpu_mem_max_mb']} MB", "info")
        self.last_remote_s[self._key(src)] = res["total_s"]
        self._compare(self._key(src))

    def _on_local_done(self, res: dict, src: str, out: str) -> None:
        self._set_busy(False)
        self.progress.set(1)
        self.pct_lbl.configure(text="100 %")
        self.stage_lbl.configure(text="Finished (local)")
        self.detail_lbl.configure(text=f"Saved to {out}")
        self.last_output = out
        self._log(f"✔ Local render ({res['engine']}) finished in {res['render_s']:.1f}s", "ok")
        self.last_local_s[self._key(src)] = res["render_s"]
        self._compare(self._key(src))

    def _on_error(self, text: str) -> None:
        self._set_busy(False)
        self.stage_lbl.configure(text="Failed")
        self._log(f"✖ {text}")
        messagebox.showerror("Error", text)

    def _on_cancelled(self) -> None:
        self._set_busy(False)
        self.stage_lbl.configure(text="Cancelled")
        self.progress.set(0)
        self.pct_lbl.configure(text="0.0 %")
        self._log("✖ Cancelled by user")

    def _on_close(self) -> None:
        if self.busy and not messagebox.askyesno("Quit", "A job is running. Cancel it and quit?"):
            return
        self.cancel_event.set()
        self._save_config()
        self.destroy()


def main() -> None:
    App().mainloop()


if __name__ == "__main__":
    main()
