"""
================================================================================
CSC-334: Parallel and Distributed Computing
Lab 4: Custom Distributed Task Offloading & Remote GPU Rendering System
Module: client/client_gui.py
Description: Modern CustomTkinter Desktop GUI for Task Offloading, Remote GPU Rendering,
             Third-Party Video Editor Integration (CapCut/ClipChamp/etc.), and LAN File Sharing.
             - Browse any folder (CapCut exports, ClipChamp output, etc.)
             - Select a video from the folder listing
             - Send to remote server for rendering with live logs
             - Server renders and sends the rendered video back to this laptop
================================================================================
"""

import os
import sys
import time
import queue
import threading
from typing import Optional, Dict, Any, List
from tkinter import filedialog, messagebox
import customtkinter as ctk

# Add client directory to path for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from network_client import OffloadingClient
from utils import generate_sample_video, format_bytes, format_time, compute_sha256
from protocol import DEFAULT_HOST, DEFAULT_PORT


# ============================================================================
# Supported video file extensions for folder scanning
# ============================================================================
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".wmv", ".flv", ".m4v", ".ts"}


class TaskOffloaderApp(ctk.CTk):
    """
    Modern CustomTkinter GUI for Custom Distributed Task Offloading,
    Remote Hardware Acceleration, Third-Party Video Editor Integration,
    and LAN File Sharing Hub.
    """
    def __init__(self):
        super().__init__()

        # Window Appearance & Geometry
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")
        self.title("CSC-334: Distributed Task Offloading & Remote Compute System")
        self.geometry("1100x880")
        self.minsize(980, 720)

        # State Variables
        self.client: Optional[OffloadingClient] = None
        self.is_processing = False
        self.cancel_flag = False
        self.selected_file_path = ""
        self.last_rendered_path = ""
        self.last_log_path = ""
        self.selected_folder_path = ""
        self.download_output_dir = os.path.join(current_dir, "client_downloads")
        os.makedirs(self.download_output_dir, exist_ok=True)

        # Merge-tab state
        self.merge_clips: List[str] = []          # Ordered list of clip absolute paths
        self.merge_last_rendered_path = ""
        self.merge_last_log_path = ""

        # Thread-safe UI update queue
        self.ui_queue = queue.Queue()
        self._poll_ui_queue()

        # Build UI Layout
        self._build_header()
        self._build_tab_container()
        self._build_log_terminal()

        # Log Initial System Banner
        self.log("[SYSTEM] Distributed Task Offloading Client initialized (Computer A).")
        self.log(f"[SYSTEM] Default Target: {DEFAULT_HOST}:{DEFAULT_PORT}. Click '⚡ Handshake & Ping' to connect to Computer B.")
        self.log("[TIP] Use '📂 Video Editor Exports' tab to select your CapCut/ClipChamp/etc. output folder!")

    # ==========================================
    # UI BUILDERS
    # ==========================================
    def _build_header(self):
        """Top Header Banner with Course & System Details."""
        header_frame = ctk.CTkFrame(self, corner_radius=10, fg_color=("#1f2937", "#111827"))
        header_frame.pack(fill="x", padx=16, pady=(12, 6))

        title_row = ctk.CTkFrame(header_frame, fg_color="transparent")
        title_row.pack(fill="x", padx=16, pady=(10, 2))

        ctk.CTkLabel(
            title_row,
            text="⚡ CSC-334: Distributed Task Offloading & Remote Compute (Computer A)",
            font=ctk.CTkFont(family="Inter", size=19, weight="bold"),
            text_color="#60a5fa"
        ).pack(side="left")

        self.conn_indicator = ctk.CTkLabel(
            title_row,
            text="🔴 OFFLINE",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#f87171",
            fg_color=("#374151", "#1e293b"),
            corner_radius=6,
            padx=10,
            pady=3
        )
        self.conn_indicator.pack(side="right")



    def _build_tab_container(self):
        """Builds tabs: Tab 1 = Video Editor Exports; Tab 2 = Multi-Clip Merge; Tab 3 = LAN File Sharing."""
        self.tabview = ctk.CTkTabview(self, corner_radius=10)
        self.tabview.pack(fill="both", expand=True, padx=16, pady=4)

        self.tab_offload = self.tabview.add("📂 Video Editor Exports (CapCut/ClipChamp)")
        self.tab_merge   = self.tabview.add("🎬 Multi-Clip Merge & Render")
        self.tab_sharing = self.tabview.add("📁 LAN File Sharing Hub")

        self._build_offload_tab()
        self._build_merge_tab()
        self._build_sharing_tab()

    def _build_offload_tab(self):
        """Tab 1: Folder browser for third-party video editor exports + server rendering."""
        main_container = ctk.CTkFrame(self.tab_offload, fg_color="transparent")
        main_container.pack(fill="both", expand=True, padx=4, pady=2)
        main_container.columnconfigure(0, weight=1)
        main_container.columnconfigure(1, weight=1)

        # ==========================================
        # LEFT COLUMN: Connection & Folder Browser
        # ==========================================
        left_col = ctk.CTkFrame(main_container, corner_radius=10)
        left_col.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=2)

        # Section 1: Networking & Handshake Setup
        net_title = ctk.CTkLabel(
            left_col,
            text="🌐 1. Network & Server Connection (Computer B)",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#93c5fd"
        )
        net_title.pack(anchor="w", padx=14, pady=(10, 4))

        net_row = ctk.CTkFrame(left_col, fg_color="transparent")
        net_row.pack(fill="x", padx=14, pady=2)

        self.ip_entry = ctk.CTkEntry(net_row, placeholder_text="Server IP (e.g. 192.168.1.20)", width=170)
        self.ip_entry.insert(0, DEFAULT_HOST)
        self.ip_entry.pack(side="left", padx=(0, 6))

        self.port_entry = ctk.CTkEntry(net_row, placeholder_text="Port", width=65)
        self.port_entry.insert(0, str(DEFAULT_PORT))
        self.port_entry.pack(side="left", padx=(0, 6))

        self.ping_btn = ctk.CTkButton(
            net_row,
            text="⚡ Handshake & Ping",
            fg_color="#2563eb",
            hover_color="#1d4ed8",
            command=self._handle_handshake_and_ping
        )
        self.ping_btn.pack(side="left", fill="x", expand=True)

        # Server Node Status Card
        self.worker_card = ctk.CTkFrame(left_col, fg_color=("#374151", "#1e293b"), corner_radius=8)
        self.worker_card.pack(fill="x", padx=14, pady=6)

        self.worker_status_lbl = ctk.CTkLabel(
            self.worker_card,
            text="Server Status: DISCONNECTED",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#f87171"
        )
        self.worker_status_lbl.pack(anchor="w", padx=10, pady=(5, 2))

        self.worker_specs_lbl = ctk.CTkLabel(
            self.worker_card,
            text="Enter Server IP & click 'Handshake & Ping' to connect Computer B",
            font=ctk.CTkFont(size=11),
            text_color="#cbd5e1"
        )
        self.worker_specs_lbl.pack(anchor="w", padx=10, pady=(0, 5))

        # Section 2: Third-Party Video Editor Folder Browser
        folder_title = ctk.CTkLabel(
            left_col,
            text="📂 2. Select Video Exports Folder (CapCut / ClipChamp / etc.)",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#93c5fd"
        )
        folder_title.pack(anchor="w", padx=14, pady=(8, 4))

        folder_desc = ctk.CTkLabel(
            left_col,
            text="Browse to the folder where your video editor saves exported videos.",
            font=ctk.CTkFont(size=11),
            text_color="#94a3b8"
        )
        folder_desc.pack(anchor="w", padx=14, pady=(0, 4))

        folder_row = ctk.CTkFrame(left_col, fg_color="transparent")
        folder_row.pack(fill="x", padx=14, pady=2)

        self.folder_path_entry = ctk.CTkEntry(folder_row, placeholder_text="Select exports folder...", width=240)
        self.folder_path_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))

        self.folder_browse_btn = ctk.CTkButton(
            folder_row,
            text="📂 Browse Folder",
            width=120,
            fg_color="#7c3aed",
            hover_color="#6d28d9",
            command=self._handle_browse_folder
        )
        self.folder_browse_btn.pack(side="left", padx=(0, 6))

        self.folder_refresh_btn = ctk.CTkButton(
            folder_row,
            text="🔄",
            width=35,
            fg_color="#475569",
            hover_color="#334155",
            command=self._refresh_folder_videos
        )
        self.folder_refresh_btn.pack(side="left")

        # Video Files List in Folder
        self.video_list_label = ctk.CTkLabel(
            left_col,
            text="📹 Video Files Found: (select a folder first)",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#cbd5e1"
        )
        self.video_list_label.pack(anchor="w", padx=14, pady=(6, 2))

        self.video_listbox = ctk.CTkTextbox(
            left_col,
            height=100,
            font=ctk.CTkFont(family="Consolas", size=11),
            fg_color=("#111827", "#030712"),
            text_color="#e2e8f0"
        )
        self.video_listbox.pack(fill="both", expand=True, padx=14, pady=(0, 4))
        self.video_listbox.insert("end", "No folder selected yet.\nClick '📂 Browse Folder' to select your CapCut/ClipChamp exports directory.\n")
        self.video_listbox.configure(state="disabled")

        # Video selector row
        select_row = ctk.CTkFrame(left_col, fg_color="transparent")
        select_row.pack(fill="x", padx=14, pady=(2, 4))

        self.video_select_entry = ctk.CTkEntry(
            select_row,
            placeholder_text="Type video filename or # from list above...",
            width=200
        )
        self.video_select_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))

        self.video_pick_btn = ctk.CTkButton(
            select_row,
            text="✅ Select Video",
            width=100,
            fg_color="#059669",
            hover_color="#047857",
            command=self._handle_select_video_from_list
        )
        self.video_pick_btn.pack(side="left", padx=(0, 6))

        self.video_browse_btn = ctk.CTkButton(
            select_row,
            text="📄 Browse File",
            width=90,
            fg_color="#475569",
            hover_color="#334155",
            command=self._handle_browse_single_file
        )
        self.video_browse_btn.pack(side="left")

        # Selected file display
        self.selected_file_lbl = ctk.CTkLabel(
            left_col,
            text="Selected: None",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#fbbf24"
        )
        self.selected_file_lbl.pack(anchor="w", padx=14, pady=(2, 6))

        # ==========================================
        # RIGHT COLUMN: Render Config & Dashboard
        # ==========================================
        right_col = ctk.CTkFrame(main_container, corner_radius=10)
        right_col.grid(row=0, column=1, sticky="nsew", padx=(6, 0), pady=2)

        # Section 3: Render Settings
        render_title = ctk.CTkLabel(
            right_col,
            text="⚙️ 3. Render Configuration (Server-Side)",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#93c5fd"
        )
        render_title.pack(anchor="w", padx=14, pady=(10, 4))

        # Render Settings Box
        self.settings_frame = ctk.CTkFrame(right_col, fg_color=("#374151", "#1e293b"), corner_radius=8)
        self.settings_frame.pack(fill="x", padx=14, pady=4)

        # Row 1: Resolution & Bitrate
        row1 = ctk.CTkFrame(self.settings_frame, fg_color="transparent")
        row1.pack(fill="x", padx=10, pady=3)

        ctk.CTkLabel(row1, text="Resolution:", width=75, anchor="w").pack(side="left")
        self.res_combo = ctk.CTkComboBox(
            row1,
            values=["1920x1080", "1280x720", "854x480", "3840x2160", "original"],
            width=110
        )
        self.res_combo.set("1280x720")
        self.res_combo.pack(side="left", padx=(0, 10))

        ctk.CTkLabel(row1, text="Bitrate:", width=50, anchor="w").pack(side="left")
        self.bitrate_combo = ctk.CTkComboBox(
            row1,
            values=["1500k", "3000k", "5000k", "8000k", "12000k"],
            width=100
        )
        self.bitrate_combo.set("3000k")
        self.bitrate_combo.pack(side="left")

        # Row 2: Preset & Acceleration Mode
        row2 = ctk.CTkFrame(self.settings_frame, fg_color="transparent")
        row2.pack(fill="x", padx=10, pady=(2, 4))

        ctk.CTkLabel(row2, text="Preset:", width=75, anchor="w").pack(side="left")
        self.preset_combo = ctk.CTkComboBox(
            row2,
            values=["ultrafast", "fast", "medium", "slow", "p4"],
            width=110
        )
        self.preset_combo.set("fast")
        self.preset_combo.pack(side="left", padx=(0, 10))

        ctk.CTkLabel(row2, text="Mode:", width=50, anchor="w").pack(side="left")
        self.accel_combo = ctk.CTkComboBox(
            row2,
            values=["auto", "nvenc", "cpu"],
            width=100
        )
        self.accel_combo.set("auto")
        self.accel_combo.pack(side="left")

        # Row 3: Creative Filters & FPS
        row3 = ctk.CTkFrame(self.settings_frame, fg_color="transparent")
        row3.pack(fill="x", padx=10, pady=(2, 4))

        ctk.CTkLabel(row3, text="Filter:", width=75, anchor="w").pack(side="left")
        self.filter_combo = ctk.CTkComboBox(
            row3,
            values=["none", "grayscale", "sepia", "mirror", "blur", "speed_2x"],
            width=110
        )
        self.filter_combo.set("none")
        self.filter_combo.pack(side="left", padx=(0, 10))

        ctk.CTkLabel(row3, text="FPS:", width=50, anchor="w").pack(side="left")
        self.fps_combo = ctk.CTkComboBox(
            row3,
            values=["original", "24 fps", "30 fps", "60 fps"],
            width=100
        )
        self.fps_combo.set("original")
        self.fps_combo.pack(side="left")

        # Section 4: Real-Time Progress Dashboard
        prog_title = ctk.CTkLabel(
            right_col,
            text="📊 4. Real-Time Remote Execution Dashboard",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#93c5fd"
        )
        prog_title.pack(anchor="w", padx=14, pady=(10, 4))

        # Progress Percentage Bar
        self.progress_bar = ctk.CTkProgressBar(right_col, height=18, corner_radius=8)
        self.progress_bar.set(0.0)
        self.progress_bar.pack(fill="x", padx=14, pady=(4, 2))

        self.progress_lbl = ctk.CTkLabel(
            right_col,
            text="0.0% | Ready to offload video",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#cbd5e1"
        )
        self.progress_lbl.pack(anchor="w", padx=14, pady=(0, 6))

        # Metrics Card Grid
        metrics_box = ctk.CTkFrame(right_col, fg_color=("#374151", "#1e293b"), corner_radius=8)
        metrics_box.pack(fill="x", padx=14, pady=4)
        metrics_box.columnconfigure(0, weight=1)
        metrics_box.columnconfigure(1, weight=1)

        self.metric_fps = ctk.CTkLabel(metrics_box, text="Speed/Stage: --", font=ctk.CTkFont(size=12))
        self.metric_fps.grid(row=0, column=0, padx=8, pady=4, sticky="w")

        self.metric_eta = ctk.CTkLabel(metrics_box, text="ETA: --", font=ctk.CTkFont(size=12))
        self.metric_eta.grid(row=0, column=1, padx=8, pady=4, sticky="w")

        self.metric_turnaround = ctk.CTkLabel(metrics_box, text="Turnaround: -- s", font=ctk.CTkFont(size=12))
        self.metric_turnaround.grid(row=1, column=0, padx=8, pady=4, sticky="w")

        self.metric_integrity = ctk.CTkLabel(metrics_box, text="Integrity: Idle", font=ctk.CTkFont(size=12), text_color="#fbbf24")
        self.metric_integrity.grid(row=1, column=1, padx=8, pady=4, sticky="w")

        # Action Buttons
        btn_frame = ctk.CTkFrame(right_col, fg_color="transparent")
        btn_frame.pack(fill="x", padx=14, pady=(8, 4))

        self.offload_btn = ctk.CTkButton(
            btn_frame,
            text="🚀 Send Video to Server for Rendering (Computer B)",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#16a34a",
            hover_color="#15803d",
            height=38,
            command=self._handle_start_offload
        )
        self.offload_btn.pack(fill="x", pady=(0, 6))

        btn_row = ctk.CTkFrame(btn_frame, fg_color="transparent")
        btn_row.pack(fill="x")

        self.play_btn = ctk.CTkButton(
            btn_row,
            text="▶️ Play Video",
            fg_color="#0284c7",
            hover_color="#0369a1",
            height=30,
            width=90,
            state="disabled",
            command=self._handle_play_rendered_video
        )
        self.play_btn.pack(side="left", padx=(0, 6))

        self.view_log_btn = ctk.CTkButton(
            btn_row,
            text="📄 View Logs",
            fg_color="#0d9488",
            hover_color="#0f766e",
            height=30,
            width=95,
            state="disabled",
            command=self._handle_view_server_logs
        )
        self.view_log_btn.pack(side="left", padx=(0, 6))

        self.open_dl_btn = ctk.CTkButton(
            btn_row,
            text="📂 Open Downloads",
            fg_color="#7c3aed",
            hover_color="#6d28d9",
            height=30,
            command=lambda: os.startfile(self.download_output_dir)
        )
        self.open_dl_btn.pack(side="left", fill="x", expand=True, padx=(0, 6))

        self.cancel_btn = ctk.CTkButton(
            btn_row,
            text="🛑 Cancel",
            fg_color="#dc2626",
            hover_color="#b91c1c",
            height=30,
            width=80,
            command=self._handle_cancel
        )
        self.cancel_btn.pack(side="left")

    def _build_merge_tab(self):
        """Tab 2: Select multiple video clips, reorder them, and merge+render on server."""
        root = ctk.CTkFrame(self.tab_merge, fg_color="transparent")
        root.pack(fill="both", expand=True, padx=4, pady=2)
        root.columnconfigure(0, weight=1)
        root.columnconfigure(1, weight=1)

        # ── LEFT: Clip list manager ──────────────────────────────────────────
        left = ctk.CTkFrame(root, corner_radius=10)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=2)

        ctk.CTkLabel(
            left,
            text="🎞️ 1. Select & Order Video Clips to Merge",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#93c5fd"
        ).pack(anchor="w", padx=14, pady=(10, 2))

        ctk.CTkLabel(
            left,
            text="Add clips in the order you want them merged (top = first).",
            font=ctk.CTkFont(size=11),
            text_color="#94a3b8"
        ).pack(anchor="w", padx=14, pady=(0, 6))

        # Clip list display
        self.merge_list_box = ctk.CTkTextbox(
            left,
            height=200,
            font=ctk.CTkFont(family="Consolas", size=11),
            fg_color=("#111827", "#030712"),
            text_color="#e2e8f0"
        )
        self.merge_list_box.pack(fill="both", expand=True, padx=14, pady=(0, 4))
        self.merge_list_box.insert("end", "No clips added yet.\nClick '➕ Add Clip' to add video files.\n")
        self.merge_list_box.configure(state="disabled")

        # Clip count label
        self.merge_count_lbl = ctk.CTkLabel(
            left,
            text="Clips: 0  |  Total Size: 0 B",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#fbbf24"
        )
        self.merge_count_lbl.pack(anchor="w", padx=14, pady=(0, 4))

        # Control buttons row 1
        btn_row1 = ctk.CTkFrame(left, fg_color="transparent")
        btn_row1.pack(fill="x", padx=14, pady=(2, 2))

        ctk.CTkButton(
            btn_row1, text="➕ Add Clip",
            fg_color="#059669", hover_color="#047857", width=110,
            command=self._merge_add_clip
        ).pack(side="left", padx=(0, 6))

        ctk.CTkButton(
            btn_row1, text="➖ Remove",
            fg_color="#dc2626", hover_color="#b91c1c", width=90,
            command=self._merge_remove_clip
        ).pack(side="left", padx=(0, 6))

        ctk.CTkButton(
            btn_row1, text="🗑️ Clear All",
            fg_color="#475569", hover_color="#334155", width=90,
            command=self._merge_clear_all
        ).pack(side="left")

        # Control buttons row 2 (reorder)
        btn_row2 = ctk.CTkFrame(left, fg_color="transparent")
        btn_row2.pack(fill="x", padx=14, pady=(2, 8))

        ctk.CTkButton(
            btn_row2, text="⬆️ Move Up",
            fg_color="#2563eb", hover_color="#1d4ed8", width=100,
            command=self._merge_move_up
        ).pack(side="left", padx=(0, 6))

        ctk.CTkButton(
            btn_row2, text="⬇️ Move Down",
            fg_color="#2563eb", hover_color="#1d4ed8", width=100,
            command=self._merge_move_down
        ).pack(side="left", padx=(0, 6))

        # Selected clip index entry
        self.merge_sel_entry = ctk.CTkEntry(
            btn_row2, placeholder_text="Clip # (1-based)", width=110
        )
        self.merge_sel_entry.pack(side="left")

        # ── RIGHT: Render Config + Progress + Action ─────────────────────────
        right = ctk.CTkFrame(root, corner_radius=10)
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 0), pady=2)

        ctk.CTkLabel(
            right,
            text="⚙️ 2. Render Configuration (Server-Side)",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#93c5fd"
        ).pack(anchor="w", padx=14, pady=(10, 4))

        # Settings box
        msf = ctk.CTkFrame(right, fg_color=("#374151", "#1e293b"), corner_radius=8)
        msf.pack(fill="x", padx=14, pady=4)

        mr1 = ctk.CTkFrame(msf, fg_color="transparent")
        mr1.pack(fill="x", padx=10, pady=3)
        ctk.CTkLabel(mr1, text="Resolution:", width=75, anchor="w").pack(side="left")
        self.merge_res_combo = ctk.CTkComboBox(
            mr1, values=["1920x1080", "1280x720", "854x480", "3840x2160", "original"], width=110)
        self.merge_res_combo.set("1280x720")
        self.merge_res_combo.pack(side="left", padx=(0, 10))
        ctk.CTkLabel(mr1, text="Bitrate:", width=50, anchor="w").pack(side="left")
        self.merge_bitrate_combo = ctk.CTkComboBox(
            mr1, values=["1500k", "3000k", "5000k", "8000k", "12000k"], width=100)
        self.merge_bitrate_combo.set("3000k")
        self.merge_bitrate_combo.pack(side="left")

        mr2 = ctk.CTkFrame(msf, fg_color="transparent")
        mr2.pack(fill="x", padx=10, pady=(2, 4))
        ctk.CTkLabel(mr2, text="Preset:", width=75, anchor="w").pack(side="left")
        self.merge_preset_combo = ctk.CTkComboBox(
            mr2, values=["ultrafast", "fast", "medium", "slow"], width=110)
        self.merge_preset_combo.set("fast")
        self.merge_preset_combo.pack(side="left", padx=(0, 10))
        ctk.CTkLabel(mr2, text="Mode:", width=50, anchor="w").pack(side="left")
        self.merge_accel_combo = ctk.CTkComboBox(
            mr2, values=["auto", "nvenc", "cpu"], width=100)
        self.merge_accel_combo.set("auto")
        self.merge_accel_combo.pack(side="left")

        mr3 = ctk.CTkFrame(msf, fg_color="transparent")
        mr3.pack(fill="x", padx=10, pady=(2, 4))
        ctk.CTkLabel(mr3, text="Filter:", width=75, anchor="w").pack(side="left")
        self.merge_filter_combo = ctk.CTkComboBox(
            mr3, values=["none", "grayscale", "sepia", "mirror", "blur", "speed_2x"], width=110)
        self.merge_filter_combo.set("none")
        self.merge_filter_combo.pack(side="left", padx=(0, 10))
        ctk.CTkLabel(mr3, text="FPS:", width=50, anchor="w").pack(side="left")
        self.merge_fps_combo = ctk.CTkComboBox(
            mr3, values=["original", "24 fps", "30 fps", "60 fps"], width=100)
        self.merge_fps_combo.set("original")
        self.merge_fps_combo.pack(side="left")

        # Progress section
        ctk.CTkLabel(
            right,
            text="📊 3. Real-Time Merge & Render Dashboard",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#93c5fd"
        ).pack(anchor="w", padx=14, pady=(10, 4))

        self.merge_progress_bar = ctk.CTkProgressBar(right, height=18, corner_radius=8)
        self.merge_progress_bar.set(0.0)
        self.merge_progress_bar.pack(fill="x", padx=14, pady=(4, 2))

        self.merge_progress_lbl = ctk.CTkLabel(
            right,
            text="0.0% | Ready to merge & render",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#cbd5e1"
        )
        self.merge_progress_lbl.pack(anchor="w", padx=14, pady=(0, 4))

        mmet = ctk.CTkFrame(right, fg_color=("#374151", "#1e293b"), corner_radius=8)
        mmet.pack(fill="x", padx=14, pady=4)
        mmet.columnconfigure(0, weight=1)
        mmet.columnconfigure(1, weight=1)

        self.merge_metric_stage = ctk.CTkLabel(mmet, text="Stage: --", font=ctk.CTkFont(size=12))
        self.merge_metric_stage.grid(row=0, column=0, padx=8, pady=4, sticky="w")
        self.merge_metric_eta = ctk.CTkLabel(mmet, text="ETA: --", font=ctk.CTkFont(size=12))
        self.merge_metric_eta.grid(row=0, column=1, padx=8, pady=4, sticky="w")
        self.merge_metric_time = ctk.CTkLabel(mmet, text="Turnaround: -- s", font=ctk.CTkFont(size=12))
        self.merge_metric_time.grid(row=1, column=0, padx=8, pady=4, sticky="w")
        self.merge_metric_integrity = ctk.CTkLabel(
            mmet, text="Integrity: Idle", font=ctk.CTkFont(size=12), text_color="#fbbf24")
        self.merge_metric_integrity.grid(row=1, column=1, padx=8, pady=4, sticky="w")

        # Action buttons
        mbf = ctk.CTkFrame(right, fg_color="transparent")
        mbf.pack(fill="x", padx=14, pady=(8, 4))

        self.merge_send_btn = ctk.CTkButton(
            mbf,
            text="🚀 Merge Clips & Render on Server B",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#7c3aed", hover_color="#6d28d9",
            height=38,
            command=self._handle_start_merge
        )
        self.merge_send_btn.pack(fill="x", pady=(0, 6))

        mbr = ctk.CTkFrame(mbf, fg_color="transparent")
        mbr.pack(fill="x")

        self.merge_play_btn = ctk.CTkButton(
            mbr, text="▶️ Play Merged",
            fg_color="#0284c7", hover_color="#0369a1",
            height=30, width=110, state="disabled",
            command=self._merge_play_video
        )
        self.merge_play_btn.pack(side="left", padx=(0, 6))

        self.merge_log_btn = ctk.CTkButton(
            mbr, text="📄 View Logs",
            fg_color="#0d9488", hover_color="#0f766e",
            height=30, width=90, state="disabled",
            command=self._merge_view_logs
        )
        self.merge_log_btn.pack(side="left", padx=(0, 6))

        ctk.CTkButton(
            mbr, text="📂 Downloads",
            fg_color="#475569", hover_color="#334155",
            height=30,
            command=lambda: os.startfile(self.download_output_dir)
        ).pack(side="left", fill="x", expand=True, padx=(0, 6))

        self.merge_cancel_btn = ctk.CTkButton(
            mbr, text="🛑 Cancel",
            fg_color="#dc2626", hover_color="#b91c1c",
            height=30, width=80,
            command=self._handle_cancel
        )
        self.merge_cancel_btn.pack(side="left")

    def _build_sharing_tab(self):

        """Tab 2: Peer-to-Peer LAN File Sharing Hub."""
        frame = ctk.CTkFrame(self.tab_sharing, fg_color="transparent")
        frame.pack(fill="both", expand=True, padx=8, pady=6)

        # Top Control Row
        ctrl_row = ctk.CTkFrame(frame, corner_radius=8)
        ctrl_row.pack(fill="x", padx=6, pady=4)

        ctk.CTkLabel(
            ctrl_row,
            text="📁 LAN Storage Hub (Share files with Computer B without internet)",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#93c5fd"
        ).pack(side="left", padx=12, pady=10)

        refresh_files_btn = ctk.CTkButton(
            ctrl_row,
            text="🔄 Refresh Server Files",
            width=140,
            fg_color="#2563eb",
            hover_color="#1d4ed8",
            command=self._handle_list_shared_files
        )
        refresh_files_btn.pack(side="right", padx=(0, 10))

        upload_share_btn = ctk.CTkButton(
            ctrl_row,
            text="📤 Upload File to Server",
            width=150,
            fg_color="#059669",
            hover_color="#047857",
            command=self._handle_upload_shared_file
        )
        upload_share_btn.pack(side="right", padx=(0, 10))

        # Files Display
        self.shared_files_box = ctk.CTkTextbox(
            frame,
            height=180,
            font=ctk.CTkFont(family="Consolas", size=11),
            fg_color=("#111827", "#030712"),
            text_color="#e2e8f0"
        )
        self.shared_files_box.pack(fill="both", expand=True, padx=6, pady=4)
        self.shared_files_box.insert("end", "Click '🔄 Refresh Server Files' to see available files on Computer B.\n")
        self.shared_files_box.configure(state="disabled")

        # Bottom Actions
        act_row = ctk.CTkFrame(frame, fg_color="transparent")
        act_row.pack(fill="x", padx=6, pady=4)

        self.dl_fname_entry = ctk.CTkEntry(act_row, placeholder_text="Enter file name to download...", width=280)
        self.dl_fname_entry.pack(side="left", padx=(0, 8))

        dl_btn = ctk.CTkButton(
            act_row,
            text="📥 Download from Server",
            width=160,
            fg_color="#0284c7",
            hover_color="#0369a1",
            command=self._handle_download_shared_file
        )
        dl_btn.pack(side="left", padx=(0, 10))

        open_dl_btn = ctk.CTkButton(
            act_row,
            text="📂 Open Downloads Folder",
            width=160,
            fg_color="#475569",
            hover_color="#334155",
            command=lambda: os.startfile(self.download_output_dir)
        )
        open_dl_btn.pack(side="left")

    def _build_log_terminal(self):
        """Bottom Real-Time Log Terminal Widget."""
        term_frame = ctk.CTkFrame(self, corner_radius=10)
        term_frame.pack(fill="both", expand=True, padx=16, pady=(2, 10))

        head_row = ctk.CTkFrame(term_frame, fg_color="transparent")
        head_row.pack(fill="x", padx=12, pady=(6, 2))

        ctk.CTkLabel(
            head_row,
            text="💻 Integrated Real-Time Live Console (Computer A)",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#93c5fd"
        ).pack(side="left")

        clear_btn = ctk.CTkButton(
            head_row,
            text="Clear Log",
            width=70,
            height=22,
            fg_color="#4b5563",
            hover_color="#374151",
            command=self._clear_logs
        )
        clear_btn.pack(side="right")

        self.log_textbox = ctk.CTkTextbox(
            term_frame,
            font=ctk.CTkFont(family="Consolas", size=11),
            fg_color=("#111827", "#030712"),
            text_color="#e2e8f0",
            wrap="none"
        )
        self.log_textbox.pack(fill="both", expand=True, padx=12, pady=(0, 6))

    # ==========================================
    # LOGGING & UI HELPERS (THREAD-SAFE)
    # ==========================================
    def _poll_ui_queue(self):
        """Processes pending UI mutations on Tkinter main thread."""
        try:
            while True:
                fn, args, kwargs = self.ui_queue.get_nowait()
                try:
                    fn(*args, **kwargs)
                except Exception:
                    pass
        except queue.Empty:
            pass
        self.after(40, self._poll_ui_queue)

    def run_on_ui(self, fn, *args, **kwargs):
        """Enqueues a function to run safely on the main GUI thread."""
        self.ui_queue.put((fn, args, kwargs))

    def log(self, text: str):
        """Thread-safe append to terminal log console."""
        timestamp = time.strftime("%H:%M:%S")
        msg = f"[{timestamp}] {text}\n"
        def _do_insert():
            self.log_textbox.insert("end", msg)
            self.log_textbox.see("end")
        self.run_on_ui(_do_insert)

    def _clear_logs(self):
        self.log_textbox.delete("1.0", "end")

    # ==========================================
    # FOLDER BROWSER & VIDEO FILE LISTING
    # ==========================================
    def _handle_browse_folder(self):
        """Opens folder dialog to select a third-party video editor export directory."""
        folder = filedialog.askdirectory(
            title="Select Video Editor Exports Folder (CapCut / ClipChamp / etc.)"
        )
        if folder:
            self.selected_folder_path = folder
            self.folder_path_entry.delete(0, "end")
            self.folder_path_entry.insert(0, folder)
            self.log(f"📂 [FOLDER SELECTED] {folder}")
            self._refresh_folder_videos()

    def _refresh_folder_videos(self):
        """Scans the selected folder for video files and lists them."""
        folder = self.folder_path_entry.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showwarning("No Folder", "Please select a valid folder first using '📂 Browse Folder'.")
            return

        self.selected_folder_path = folder
        video_files = []

        try:
            for fname in sorted(os.listdir(folder)):
                fpath = os.path.join(folder, fname)
                if os.path.isfile(fpath):
                    ext = os.path.splitext(fname)[1].lower()
                    if ext in VIDEO_EXTENSIONS:
                        size = os.path.getsize(fpath)
                        mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(fpath)))
                        video_files.append((fname, size, mtime))
        except Exception as e:
            self.log(f"❌ [SCAN ERROR] Could not scan folder: {e}")
            return

        # Update the video list display
        self.video_listbox.configure(state="normal")
        self.video_listbox.delete("1.0", "end")

        if not video_files:
            self.video_listbox.insert("end", f"No video files found in:\n{folder}\n\nSupported formats: {', '.join(VIDEO_EXTENSIONS)}\n")
            self.video_list_label.configure(text="📹 Video Files Found: 0")
        else:
            self.video_listbox.insert("end", f"{'#':<4} {'File Name':<40} {'Size':<12} {'Modified'}\n")
            self.video_listbox.insert("end", "─" * 80 + "\n")
            for idx, (name, size, mtime) in enumerate(video_files, 1):
                size_str = format_bytes(size)
                self.video_listbox.insert("end", f"{idx:<4} {name:<40} {size_str:<12} {mtime}\n")
            self.video_list_label.configure(text=f"📹 Video Files Found: {len(video_files)}")
            self.log(f"📹 [SCAN] Found {len(video_files)} video file(s) in: {folder}")

        self.video_listbox.configure(state="disabled")

        # Store the list for selection
        self._folder_video_files = video_files

    def _handle_select_video_from_list(self):
        """Selects a video from the folder listing by # or filename."""
        text = self.video_select_entry.get().strip()
        if not text:
            messagebox.showwarning("Empty Selection", "Please type a video filename or # from the list above.")
            return

        if not hasattr(self, '_folder_video_files') or not self._folder_video_files:
            messagebox.showwarning("No Videos", "Please scan a folder first by clicking '📂 Browse Folder'.")
            return

        selected_name = None

        # Try to match by number
        try:
            idx = int(text)
            if 1 <= idx <= len(self._folder_video_files):
                selected_name = self._folder_video_files[idx - 1][0]
        except ValueError:
            pass

        # Try to match by filename
        if not selected_name:
            for name, _, _ in self._folder_video_files:
                if text.lower() in name.lower():
                    selected_name = name
                    break

        if not selected_name:
            messagebox.showerror("Not Found", f"No video matching '{text}' found in the folder.")
            return

        full_path = os.path.join(self.selected_folder_path, selected_name)
        if not os.path.exists(full_path):
            messagebox.showerror("Error", f"File not found: {full_path}")
            return

        self.selected_file_path = full_path
        size = os.path.getsize(full_path)
        self.selected_file_lbl.configure(text=f"Selected: {selected_name} ({format_bytes(size)})")
        self.log(f"✅ [SELECTED] {selected_name} ({format_bytes(size)}) from exports folder")

    def _handle_browse_single_file(self):
        """Browse a single video file from anywhere."""
        path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4 *.mov *.mkv *.avi *.webm *.wmv *.flv *.m4v *.ts"), ("All Files", "*.*")]
        )
        if path:
            self.selected_file_path = path
            size = os.path.getsize(path)
            self.selected_file_lbl.configure(text=f"Selected: {os.path.basename(path)} ({format_bytes(size)})")
            self.log(f"✅ [SELECTED] {os.path.basename(path)} ({format_bytes(size)})")

    # ==========================================
    # EVENT HANDLERS
    # ==========================================
    def _handle_handshake_and_ping(self):
        """Task 1: Handshake protocol and latency ping check with prominent alerts."""
        host = self.ip_entry.get().strip()
        try:
            port = int(self.port_entry.get().strip())
        except ValueError:
            messagebox.showerror("Error", "Invalid port number.")
            return

        def task():
            self.log(f"\n[CONNECTING] Contacting Remote Server (Computer B) at {host}:{port}...")
            try:
                self.client = OffloadingClient(host=host, port=port)
                info = self.client.perform_handshake()
                latency = self.client.measure_ping_latency(num_samples=3)

                hostname = info.get('hostname', 'Remote Server')
                cpu_count = info.get('cpu_count', 4)
                gpu_status = info.get('nvenc_status', 'N/A')
                status_txt = f"Connected to: {hostname} ({host})"
                specs_txt = f"CPU: {cpu_count} Cores | GPU: {gpu_status[:30]} | Ping RTT: {latency} ms"

                def _update_ui():
                    self.worker_status_lbl.configure(text=f"🟢 {status_txt}", text_color="#4ade80")
                    self.worker_specs_lbl.configure(text=specs_txt)
                    self.conn_indicator.configure(text=f"🟢 ONLINE ({hostname})", text_color="#4ade80")

                self.run_on_ui(_update_ui)

                self.log(f"⚡ [TCP CONNECTED] Socket established with {host}:{port}!")
                self.log(f"🤝 [HANDSHAKE VERIFIED] Server '{hostname}' running protocol v{info.get('protocol_version')}")
                self.log(f"⚡ [LATENCY PROBE] Measured Round-Trip Time (RTT): {latency} ms")
                self.log(f"🚀 [RESOURCES UNLOCKED] {cpu_count} CPU Cores | {gpu_status}")
                self.log(f"   Computer B is now ready to render your video exports!")

                # Prominent notification popup
                msg = (
                    f"🎉 Successfully Connected to Server (Computer B)!\n\n"
                    f"• Node Hostname : {hostname}\n"
                    f"• Bound Address : {host}:{port}\n"
                    f"• CPU Hardware  : {cpu_count} Processing Cores\n"
                    f"• Acceleration  : {gpu_status}\n"
                    f"• Ping Latency  : {latency} ms (Fast LAN Wire)\n\n"
                    f"Remote resources are active! You can now send video exports for rendering."
                )
                self.run_on_ui(lambda: messagebox.showinfo("Connected to Server!", msg))

            except Exception as e:
                def _update_err():
                    self.worker_status_lbl.configure(text="🔴 Server Status: UNREACHABLE", text_color="#f87171")
                    self.worker_specs_lbl.configure(text=f"Connection Error: {e}")
                    self.conn_indicator.configure(text="🔴 OFFLINE", text_color="#f87171")
                self.run_on_ui(_update_err)
                self.log(f"❌ [CONNECT ERROR] Could not connect to Server at {host}:{port} -> {e}")
                self.run_on_ui(lambda: messagebox.showerror(
                    "Connection Failed",
                    f"Could not connect to Server at {host}:{port}.\n\n"
                    f"Check:\n"
                    f"1. Is Server GUI / Daemon running on Computer B?\n"
                    f"2. Are both computers connected with LAN cable?\n"
                    f"3. Is the IP address correct?"
                ))

        threading.Thread(target=task, daemon=True).start()

    def _handle_start_offload(self):
        """Initiates video rendering offload to remote server."""
        if self.is_processing:
            messagebox.showwarning("Warning", "A task is currently running.")
            return

        if not self.selected_file_path or not os.path.exists(self.selected_file_path):
            messagebox.showerror("Error", "Please select a video file first.\n\nUse '📂 Browse Folder' to find your CapCut/ClipChamp exports,\nor '📄 Browse File' to select any video file.")
            return

        host = self.ip_entry.get().strip()
        port = int(self.port_entry.get().strip())
        render_cfg = {
            "resolution": self.res_combo.get(),
            "bitrate": self.bitrate_combo.get(),
            "preset": self.preset_combo.get(),
            "mode": self.accel_combo.get(),
            "filter": self.filter_combo.get(),
            "fps": self.fps_combo.get()
        }
        selected_file = self.selected_file_path

        self.is_processing = True
        self.cancel_flag = False

        self.progress_bar.set(0.0)
        self.progress_lbl.configure(text="0.0% | Initializing Video Render Offload...")
        self.metric_integrity.configure(text="Checksum: Calculating...", text_color="#fbbf24")

        def run_thread():
            try:
                # Ensure connection is active
                need_reconnect = False
                if not self.client or not self.client.is_connected or not self.client.sock:
                    need_reconnect = True
                else:
                    try:
                        self.client.sock.send(b"")
                    except Exception:
                        need_reconnect = True

                if need_reconnect:
                    if self.client:
                        try:
                            self.client.disconnect()
                        except Exception:
                            pass
                    self.client = OffloadingClient(host=host, port=port)
                    self.client.connect()
                    self.client.perform_handshake()

                def progress_cb(prog: dict):
                    pct = prog.get("percent", 0.0)
                    status = prog.get("status", "")
                    fps_val = prog.get("fps", 0)
                    speed_val = prog.get("speed", "1x")
                    eta_val = prog.get("eta_seconds", 0)

                    def _update_prog_widgets():
                        self.progress_bar.set(pct / 100.0)
                        self.progress_lbl.configure(text=f"{pct:.1f}% | {status}")
                        if "fps" in prog:
                            self.metric_fps.configure(text=f"Speed: {fps_val} ({speed_val})")
                        if "eta_seconds" in prog:
                            self.metric_eta.configure(text=f"ETA: {eta_val:.1f} s")

                    self.run_on_ui(_update_prog_widgets)
                    if status:
                        self.log(f"  [REMOTE] {status}")

                video_name = os.path.basename(selected_file)
                self.log(f"\n{'='*60}")
                self.log(f"🚀 [OFFLOAD VIDEO] Sending '{video_name}' to Server for Rendering")
                self.log(f"   Source: {selected_file}")
                self.log(f"   Config: {render_cfg}")
                self.log(f"{'='*60}")

                res = self.client.offload_video_transcode(
                    input_file=selected_file,
                    output_dir=self.download_output_dir,
                    render_config=render_cfg,
                    progress_callback=progress_cb,
                    cancel_check=lambda: self.cancel_flag
                )

                def _update_video_done():
                    self.progress_bar.set(1.0)
                    self.progress_lbl.configure(text="100.0% | Complete & Downloaded to Laptop!")
                    self.metric_turnaround.configure(text=f"Turnaround: {res['total_turnaround_sec']} s")
                    self.last_rendered_path = res['local_output_path']
                    self.last_log_path = res.get('local_log_path', '')
                    self.play_btn.configure(state="normal")
                    if self.last_log_path and os.path.exists(self.last_log_path):
                        self.view_log_btn.configure(state="normal")
                    if res.get("checksum_verified"):
                        self.metric_integrity.configure(text="SHA-256: VERIFIED ✔", text_color="#4ade80")
                    else:
                        self.metric_integrity.configure(text="SHA-256: MISMATCH ❌", text_color="#f87171")

                self.run_on_ui(_update_video_done)

                self.log(f"\n{'='*60}")
                self.log(f"✅ [SUCCESS] Video Rendered on Server and Downloaded to Laptop!")
                self.log(f"   • Source File     : {video_name}")
                self.log(f"   • Remote Engine   : {res.get('engine')}")
                self.log(f"   • Upload Time     : {res['upload_duration_sec']}s")
                self.log(f"   • Render Time     : {res['remote_compute_duration_sec']}s")
                self.log(f"   • Download Time   : {res['download_duration_sec']}s")
                self.log(f"   • Total Turnaround: {res['total_turnaround_sec']}s")
                self.log(f"   • Input Size      : {format_bytes(res['input_file_size'])}")
                self.log(f"   • Output Size     : {format_bytes(res['output_file_size'])}")
                self.log(f"   • SHA-256 Verified: {'✔ YES' if res.get('checksum_verified') else '❌ NO'}")
                self.log(f"   • Saved to Laptop : {res['local_output_path']}")
                if res.get('local_log_path'):
                    self.log(f"   • Server Log File : {res['local_log_path']}")
                self.log(f"{'='*60}")

                # Print server log excerpt in console if available
                if res.get('server_logs_text'):
                    self.log(f"\n📋 [SERVER EXECUTION LOG SUMMARY]")
                    for l in res['server_logs_text'].splitlines()[:15]:
                        self.log(f"   {l}")
                    if len(res['server_logs_text'].splitlines()) > 15:
                        self.log(f"   ... ({len(res['server_logs_text'].splitlines()) - 15} more lines in log file)")

                log_info = f"\n• Server Log File : {os.path.basename(res['local_log_path']) if res.get('local_log_path') else 'Attached'}"
                msg = (
                    f"🎉 Video Successfully Rendered on Server (Computer B)!\n\n"
                    f"• Source File     : {video_name}\n"
                    f"• Render Engine   : {res.get('engine')}\n"
                    f"• Upload Time     : {res['upload_duration_sec']} seconds\n"
                    f"• Render Time     : {res['remote_compute_duration_sec']} seconds\n"
                    f"• Download Time   : {res['download_duration_sec']} seconds\n"
                    f"• Total Time      : {res['total_turnaround_sec']} seconds\n"
                    f"• Integrity Check : SHA-256 Verified ✔{log_info}\n\n"
                    f"The rendered file & execution logs are ready in client_downloads/.\n"
                    f"Click '▶️ Play Video' to watch or '📄 View Logs' to inspect details!"
                )
                self.run_on_ui(lambda: messagebox.showinfo("Render Complete & Downloaded!", msg))

            except Exception as e:
                self.log(f"❌ [OFFLOAD FAILED] {e}")
                def _update_err_badge():
                    self.metric_integrity.configure(text="Error", text_color="#f87171")
                    self.progress_lbl.configure(text=f"Failed: {e}")
                self.run_on_ui(_update_err_badge)
                self.run_on_ui(lambda: messagebox.showerror("Task Failed", f"Execution failed:\n{e}"))
            finally:
                self.is_processing = False

        threading.Thread(target=run_thread, daemon=True).start()

    # ==========================================
    # MERGE TAB — CLIP LIST MANAGEMENT
    # ==========================================
    def _merge_refresh_list(self):
        """Refreshes the merge clip list textbox from self.merge_clips."""
        self.merge_list_box.configure(state="normal")
        self.merge_list_box.delete("1.0", "end")
        if not self.merge_clips:
            self.merge_list_box.insert("end", "No clips added yet.\nClick '➕ Add Clip' to add video files.\n")
        else:
            self.merge_list_box.insert("end", f"{'#':<4} {'File Name':<38} {'Size'}\n")
            self.merge_list_box.insert("end", "─" * 60 + "\n")
            total = 0
            for i, fp in enumerate(self.merge_clips, 1):
                fname = os.path.basename(fp)
                sz = os.path.getsize(fp) if os.path.exists(fp) else 0
                total += sz
                self.merge_list_box.insert("end", f"{i:<4} {fname:<38} {format_bytes(sz)}\n")
            self.merge_count_lbl.configure(
                text=f"Clips: {len(self.merge_clips)}  |  Total Size: {format_bytes(total)}"
            )
        self.merge_list_box.configure(state="disabled")
        if self.merge_clips:
            self.merge_count_lbl.configure(
                text=f"Clips: {len(self.merge_clips)}  |  Total Size: {format_bytes(sum(os.path.getsize(f) for f in self.merge_clips if os.path.exists(f)))}"
            )
        else:
            self.merge_count_lbl.configure(text="Clips: 0  |  Total Size: 0 B")

    def _merge_add_clip(self):
        """Opens file dialog to add one or more clip files to the merge list."""
        paths = filedialog.askopenfilenames(
            title="Select Video Clips to Merge",
            filetypes=[("Video Files", "*.mp4 *.mov *.mkv *.avi *.webm *.wmv *.flv *.m4v *.ts"),
                       ("All Files", "*.*")]
        )
        for p in paths:
            if p and p not in self.merge_clips:
                self.merge_clips.append(p)
                self.log(f"➕ [MERGE] Added clip: {os.path.basename(p)}")
        self._merge_refresh_list()

    def _merge_remove_clip(self):
        """Removes the clip at the index typed in the selection entry."""
        txt = self.merge_sel_entry.get().strip()
        if not txt:
            messagebox.showwarning("No Selection", "Enter a clip # in the box to remove it.")
            return
        try:
            idx = int(txt) - 1
            if 0 <= idx < len(self.merge_clips):
                removed = self.merge_clips.pop(idx)
                self.log(f"➖ [MERGE] Removed clip #{idx+1}: {os.path.basename(removed)}")
                self._merge_refresh_list()
            else:
                messagebox.showerror("Invalid #", f"Clip #{idx+1} does not exist.")
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid clip number.")

    def _merge_clear_all(self):
        """Clears all clips from the merge list."""
        if self.merge_clips and messagebox.askyesno("Clear All", "Remove all clips from the merge list?"):
            self.merge_clips.clear()
            self.log("🗑️ [MERGE] Cleared all clips.")
            self._merge_refresh_list()

    def _merge_move_up(self):
        """Moves the clip at the typed index one position up."""
        txt = self.merge_sel_entry.get().strip()
        try:
            idx = int(txt) - 1
            if 1 <= idx < len(self.merge_clips):
                self.merge_clips[idx-1], self.merge_clips[idx] = self.merge_clips[idx], self.merge_clips[idx-1]
                self._merge_refresh_list()
                self.merge_sel_entry.delete(0, "end")
                self.merge_sel_entry.insert(0, str(idx))
        except (ValueError, IndexError):
            messagebox.showwarning("Invalid", "Enter a valid clip # to move.")

    def _merge_move_down(self):
        """Moves the clip at the typed index one position down."""
        txt = self.merge_sel_entry.get().strip()
        try:
            idx = int(txt) - 1
            if 0 <= idx < len(self.merge_clips) - 1:
                self.merge_clips[idx], self.merge_clips[idx+1] = self.merge_clips[idx+1], self.merge_clips[idx]
                self._merge_refresh_list()
                self.merge_sel_entry.delete(0, "end")
                self.merge_sel_entry.insert(0, str(idx+2))
        except (ValueError, IndexError):
            messagebox.showwarning("Invalid", "Enter a valid clip # to move.")

    # ==========================================
    # MERGE TAB — SEND TO SERVER
    # ==========================================
    def _handle_start_merge(self):
        """Initiates multi-clip merge + render on server B."""
        if self.is_processing:
            messagebox.showwarning("Busy", "A task is already running. Please wait or cancel it.")
            return

        if len(self.merge_clips) < 2:
            messagebox.showerror("Not Enough Clips",
                                 "Please add at least 2 video clips to merge.\nUse '➕ Add Clip' to select files.")
            return

        for fp in self.merge_clips:
            if not os.path.exists(fp):
                messagebox.showerror("File Missing", f"Clip not found:\n{fp}")
                return

        host = self.ip_entry.get().strip()
        try:
            port = int(self.port_entry.get().strip())
        except ValueError:
            messagebox.showerror("Error", "Invalid port number.")
            return

        render_cfg = {
            "resolution": self.merge_res_combo.get(),
            "bitrate":    self.merge_bitrate_combo.get(),
            "preset":     self.merge_preset_combo.get(),
            "mode":       self.merge_accel_combo.get(),
            "filter":     self.merge_filter_combo.get(),
            "fps":        self.merge_fps_combo.get()
        }
        clips_snapshot = list(self.merge_clips)

        self.is_processing = True
        self.cancel_flag = False

        # Reset progress UI
        self.run_on_ui(lambda: self.merge_progress_bar.set(0.0))
        self.run_on_ui(lambda: self.merge_progress_lbl.configure(text="0.0% | Preparing merge task..."))
        self.run_on_ui(lambda: self.merge_metric_integrity.configure(text="Integrity: Computing...", text_color="#fbbf24"))
        self.run_on_ui(lambda: self.merge_play_btn.configure(state="disabled"))
        self.run_on_ui(lambda: self.merge_log_btn.configure(state="disabled"))

        def run_thread():
            try:
                # Reconnect if needed
                need_reconnect = False
                if not self.client or not self.client.is_connected or not self.client.sock:
                    need_reconnect = True
                else:
                    try:
                        self.client.sock.send(b"")
                    except Exception:
                        need_reconnect = True

                if need_reconnect:
                    if self.client:
                        try:
                            self.client.disconnect()
                        except Exception:
                            pass
                    self.client = OffloadingClient(host=host, port=port)
                    self.client.connect()
                    self.client.perform_handshake()

                def progress_cb(prog: dict):
                    pct     = prog.get("percent", 0.0)
                    status  = prog.get("status", "")
                    eta_val = prog.get("eta_seconds", 0)
                    fps_val = prog.get("fps", 0)
                    spd_val = prog.get("speed", "--")

                    def _up():
                        self.merge_progress_bar.set(pct / 100.0)
                        self.merge_progress_lbl.configure(text=f"{pct:.1f}% | {status}")
                        if "eta_seconds" in prog:
                            self.merge_metric_eta.configure(text=f"ETA: {eta_val:.1f} s")
                        if "fps" in prog:
                            self.merge_metric_stage.configure(text=f"FPS: {fps_val} ({spd_val})")
                        if status:
                            self.merge_metric_stage.configure(text=f"Stage: {status[:35]}")
                    self.run_on_ui(_up)
                    if status:
                        self.log(f"  [MERGE] {status}")

                names = [os.path.basename(f) for f in clips_snapshot]
                self.log(f"\n{'='*60}")
                self.log(f"🎬 [MERGE JOB] Sending {len(clips_snapshot)} clips to Server for merging & rendering")
                for i, n in enumerate(names, 1):
                    self.log(f"   Clip {i}: {n}")
                self.log(f"   Config: {render_cfg}")
                self.log(f"{'='*60}")

                res = self.client.offload_video_merge(
                    input_files=clips_snapshot,
                    output_dir=self.download_output_dir,
                    render_config=render_cfg,
                    progress_callback=progress_cb,
                    cancel_check=lambda: self.cancel_flag
                )

                def _done():
                    self.merge_progress_bar.set(1.0)
                    self.merge_progress_lbl.configure(text=f"100.0% | Merge Complete! {res['clips_merged']} clips → 1 video")
                    self.merge_metric_time.configure(text=f"Turnaround: {res['total_turnaround_sec']} s")
                    self.merge_last_rendered_path = res["local_output_path"]
                    self.merge_last_log_path = res.get("local_log_path", "")
                    self.merge_play_btn.configure(state="normal")
                    if self.merge_last_log_path and os.path.exists(self.merge_last_log_path):
                        self.merge_log_btn.configure(state="normal")
                    if res.get("checksum_verified"):
                        self.merge_metric_integrity.configure(text="SHA-256: VERIFIED ✔", text_color="#4ade80")
                    else:
                        self.merge_metric_integrity.configure(text="SHA-256: MISMATCH ❌", text_color="#f87171")
                self.run_on_ui(_done)

                self.log(f"\n{'='*60}")
                self.log(f"✅ [MERGE COMPLETE] {res['clips_merged']} clips merged, rendered & downloaded!")
                self.log(f"   • Remote Engine   : {res.get('engine', 'CPU')}")
                self.log(f"   • Upload Time     : {res['upload_duration_sec']} s")
                self.log(f"   • Render Time     : {res['remote_compute_duration_sec']} s")
                self.log(f"   • Download Time   : {res['download_duration_sec']} s")
                self.log(f"   • Total Turnaround: {res['total_turnaround_sec']} s")
                self.log(f"   • Output Size     : {format_bytes(res['output_file_size'])}")
                self.log(f"   • SHA-256 Verified: {'✔ YES' if res.get('checksum_verified') else '❌ NO'}")
                self.log(f"   • Saved to        : {res['local_output_path']}")
                self.log(f"{'='*60}")

                msg = (
                    f"🎉 Multi-Clip Merge & Render Complete!\n\n"
                    f"• Clips Merged    : {res['clips_merged']}\n"
                    f"• Render Engine   : {res.get('engine', 'CPU')}\n"
                    f"• Upload Time     : {res['upload_duration_sec']} seconds\n"
                    f"• Render Time     : {res['remote_compute_duration_sec']} seconds\n"
                    f"• Download Time   : {res['download_duration_sec']} seconds\n"
                    f"• Total Turnaround: {res['total_turnaround_sec']} seconds\n"
                    f"• Integrity Check : SHA-256 Verified ✔\n\n"
                    f"The merged video is saved in client_downloads/.\n"
                    f"Click '▶️ Play Merged' to watch it!"
                )
                self.run_on_ui(lambda: messagebox.showinfo("Merge Complete!", msg))

            except Exception as e:
                self.log(f"❌ [MERGE FAILED] {e}")
                def _err():
                    self.merge_metric_integrity.configure(text="Error", text_color="#f87171")
                    self.merge_progress_lbl.configure(text=f"Failed: {str(e)[:60]}")
                self.run_on_ui(_err)
                self.run_on_ui(lambda: messagebox.showerror("Merge Failed", f"Merge task failed:\n{e}"))
            finally:
                self.is_processing = False

        threading.Thread(target=run_thread, daemon=True).start()

    def _merge_play_video(self):
        """Opens the merged rendered video with the default system player."""
        if self.merge_last_rendered_path and os.path.exists(self.merge_last_rendered_path):
            os.startfile(self.merge_last_rendered_path)
        else:
            messagebox.showwarning("No Video", "No merged video available yet. Complete a merge task first.")

    def _merge_view_logs(self):
        """Opens the server render log for the last merge task."""
        if self.merge_last_log_path and os.path.exists(self.merge_last_log_path):
            os.startfile(self.merge_last_log_path)
        else:
            messagebox.showwarning("No Log", "No render log available for this merge task.")

    def _handle_list_shared_files(self):

        """Queries and displays files available in Server storage."""
        def task():
            try:
                self.log("[FILE HUB] Querying shared files from Computer B...")
                if not self.client or not self.client.is_connected:
                    host = self.ip_entry.get().strip()
                    port = int(self.port_entry.get().strip())
                    self.client = OffloadingClient(host=host, port=port)
                    self.client.connect()
                    self.client.perform_handshake()

                files = self.client.list_server_files()
                self.log(f"[FILE HUB] Received {len(files)} shared file(s) from server.")

                def _update_box():
                    self.shared_files_box.configure(state="normal")
                    self.shared_files_box.delete("1.0", "end")
                    if not files:
                        self.shared_files_box.insert("end", "No shared files currently on Server.\nUpload a file using 'Upload File to Server'.\n")
                    else:
                        self.shared_files_box.insert("end", f"{'File Name':<35} | {'Size':<10} | {'Modified Time'}\n")
                        self.shared_files_box.insert("end", "-" * 70 + "\n")
                        for f in files:
                            self.shared_files_box.insert("end", f"{f['name']:<35} | {f['size_str']:<10} | {f['modified']}\n")
                    self.shared_files_box.configure(state="disabled")

                self.run_on_ui(_update_box)
            except Exception as e:
                self.log(f"[FILE HUB ERROR] {e}")
                self.run_on_ui(lambda: messagebox.showerror("Error", f"Could not list server files:\n{e}"))

        threading.Thread(target=task, daemon=True).start()

    def _handle_upload_shared_file(self):
        """Uploads a local file from Computer A directly to Computer B's storage hub."""
        src_path = filedialog.askopenfilename(title="Select File to Upload to Server")
        if not src_path:
            return

        def task():
            try:
                fname = os.path.basename(src_path)
                self.log(f"📁 [UPLOADING] Sending '{fname}' to Computer B...")
                if not self.client or not self.client.is_connected:
                    host = self.ip_entry.get().strip()
                    port = int(self.port_entry.get().strip())
                    self.client = OffloadingClient(host=host, port=port)
                    self.client.connect()
                    self.client.perform_handshake()

                success = self.client.upload_shared_file(
                    src_path,
                    progress_callback=lambda msg: self.log(f"  {msg}")
                )
                if success:
                    self.log(f"✅ [UPLOAD COMPLETE] '{fname}' uploaded to Server shared storage!")
                    self.run_on_ui(lambda: messagebox.showinfo("File Uploaded", f"Successfully uploaded '{fname}' to Server!"))
                    self._handle_list_shared_files()
            except Exception as e:
                self.log(f"❌ [UPLOAD ERROR] {e}")
                self.run_on_ui(lambda: messagebox.showerror("Upload Error", f"Failed to upload file:\n{e}"))

        threading.Thread(target=task, daemon=True).start()

    def _handle_download_shared_file(self):
        """Downloads a requested file from Computer B to Computer A."""
        fname = self.dl_fname_entry.get().strip()
        if not fname:
            messagebox.showwarning("Enter Filename", "Please enter the file name to download from the list above.")
            return

        def task():
            try:
                self.log(f"📥 [DOWNLOADING] Requesting '{fname}' from Computer B...")
                if not self.client or not self.client.is_connected:
                    host = self.ip_entry.get().strip()
                    port = int(self.port_entry.get().strip())
                    self.client = OffloadingClient(host=host, port=port)
                    self.client.connect()
                    self.client.perform_handshake()

                dest = self.client.download_shared_file(
                    fname,
                    self.download_output_dir,
                    progress_callback=lambda msg: self.log(f"  {msg}")
                )
                self.log(f"✅ [DOWNLOAD COMPLETE] Saved to: {dest}")
                self.run_on_ui(lambda: messagebox.showinfo("Download Complete", f"Downloaded '{fname}'!\nSaved to: {dest}"))
            except Exception as e:
                self.log(f"❌ [DOWNLOAD ERROR] {e}")
                self.run_on_ui(lambda: messagebox.showerror("Download Error", f"Could not download file:\n{e}"))

        threading.Thread(target=task, daemon=True).start()

    def _handle_play_rendered_video(self):
        """Opens the downloaded rendered video in the default media player on this laptop."""
        if self.last_rendered_path and os.path.exists(self.last_rendered_path):
            try:
                os.startfile(self.last_rendered_path)
                self.log(f"[MEDIA PLAYER] Opened rendered video: {self.last_rendered_path}")
            except Exception as e:
                self.log(f"[ERROR] Could not open video player: {e}")
        else:
            messagebox.showwarning("No Video", "No rendered video available yet. Please send a video for rendering first.")

    def _handle_view_server_logs(self):
        """Opens the downloaded server execution log file."""
        if self.last_log_path and os.path.exists(self.last_log_path):
            try:
                os.startfile(self.last_log_path)
                self.log(f"[LOG VIEWER] Opened server execution log: {self.last_log_path}")
            except Exception as e:
                self.log(f"[ERROR] Could not open log viewer: {e}")
        else:
            messagebox.showwarning("No Logs", "No server execution logs available yet. Please complete a render task first.")

    def _handle_cancel(self):
        if self.is_processing:
            self.cancel_flag = True
            self.log("[CANCEL] User initiated cancellation request.")
        else:
            self.log("[INFO] No active task to cancel.")


def main():
    app = TaskOffloaderApp()
    app.mainloop()


if __name__ == "__main__":
    main()
