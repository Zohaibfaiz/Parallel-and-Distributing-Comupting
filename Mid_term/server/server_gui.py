"""
================================================================================
CSC-334: Parallel and Distributed Computing
Lab 4: Custom Distributed Task Offloading & Remote GPU Rendering System
Module: server/server_gui.py
Description: Modern CustomTkinter Desktop GUI for the Server Worker Node.
             Displays real-time hardware status, connected clients, live logs,
             task progress, LAN IP addresses, and file sharing hub.
================================================================================
"""

import os
import sys
import time
import queue
import socket
import threading
from typing import Optional, Dict, Any, List
from tkinter import filedialog, messagebox
import customtkinter as ctk

# Add server directory to path for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from network_server import RemoteWorkerServer
from utils import get_all_local_ips, format_bytes, detect_system_capabilities
from protocol import DEFAULT_PORT, PROTOCOL_VERSION


class ServerDashboardApp(ctk.CTk):
    """
    Modern CustomTkinter GUI for the Remote Worker Server (Computer B).
    """
    def __init__(self):
        super().__init__()

        # Window Appearance & Geometry
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")
        self.title("CSC-334: Remote Worker Server Node — Hardware Acceleration & Compute Hub")
        self.geometry("1120x820")
        self.minsize(980, 700)

        # State Variables
        self.server: Optional[RemoteWorkerServer] = None
        self.server_thread: Optional[threading.Thread] = None
        self.is_running = False
        self.storage_dir = os.path.join(current_dir, "server_storage")
        os.makedirs(self.storage_dir, exist_ok=True)

        self.sys_info = detect_system_capabilities()
        self.local_ips = get_all_local_ips()

        # Thread-safe UI update queue
        self.ui_queue = queue.Queue()
        self._poll_ui_queue()

        # Build UI Layout
        self._build_header()
        self._build_network_bar()
        self._build_metrics_cards()
        self._build_middle_workspace()
        self._build_log_console()

        # Auto-refresh timer for clients & files
        self._schedule_periodic_refresh()

        self.log("[SYSTEM] Server Dashboard Initialized.")
        self.log(f"[HARDWARE] Detected: {self.sys_info['hostname']} | OS: {self.sys_info['os']} | CPU: {self.sys_info['cpu_count']} Cores")
        self.log(f"[ACCELERATION] {self.sys_info['nvenc_status']}")
        self.log(f"[NETWORK] Available LAN IPs: {', '.join(self.local_ips)}")
        self.log("[READY] Click '🚀 Start Server' to begin listening for incoming client connections.")

    # ==========================================
    # UI BUILDERS
    # ==========================================
    def _build_header(self):
        """Top Header Banner."""
        header_frame = ctk.CTkFrame(self, corner_radius=10, fg_color=("#1e293b", "#0f172a"))
        header_frame.pack(fill="x", padx=16, pady=(12, 6))

        title_row = ctk.CTkFrame(header_frame, fg_color="transparent")
        title_row.pack(fill="x", padx=16, pady=(10, 2))

        ctk.CTkLabel(
            title_row,
            text="⚡ CSC-334: Remote Worker Server Node (Computer B)",
            font=ctk.CTkFont(family="Inter", size=20, weight="bold"),
            text_color="#38bdf8"
        ).pack(side="left")

        self.status_badge = ctk.CTkLabel(
            title_row,
            text="🔴 SERVER OFFLINE",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#f87171",
            fg_color=("#374151", "#1e293b"),
            corner_radius=6,
            padx=10,
            pady=3
        )
        self.status_badge.pack(side="right")



    def _build_network_bar(self):
        """Network control, IP selector, and Start/Stop button."""
        net_frame = ctk.CTkFrame(self, corner_radius=10)
        net_frame.pack(fill="x", padx=16, pady=4)

        row = ctk.CTkFrame(net_frame, fg_color="transparent")
        row.pack(fill="x", padx=14, pady=10)

        # Bind IP
        ctk.CTkLabel(row, text="Bind Address:", font=ctk.CTkFont(size=12, weight="bold")).pack(side="left", padx=(0, 6))
        self.bind_entry = ctk.CTkEntry(row, width=120)
        self.bind_entry.insert(0, "0.0.0.0")
        self.bind_entry.pack(side="left", padx=(0, 12))

        # Port
        ctk.CTkLabel(row, text="Port:", font=ctk.CTkFont(size=12, weight="bold")).pack(side="left", padx=(0, 6))
        self.port_entry = ctk.CTkEntry(row, width=70)
        self.port_entry.insert(0, str(DEFAULT_PORT))
        self.port_entry.pack(side="left", padx=(0, 16))

        # Start / Stop Toggle
        self.start_btn = ctk.CTkButton(
            row,
            text="🚀 Start Server",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#16a34a",
            hover_color="#15803d",
            width=140,
            command=self._toggle_server
        )
        self.start_btn.pack(side="left", padx=(0, 16))

        # LAN IP Helper Display
        lan_ip_display = self.local_ips[0] if self.local_ips else "127.0.0.1"
        self.lan_info_lbl = ctk.CTkLabel(
            row,
            text=f"📌 Computer A Connect IP:  {lan_ip_display}",
            font=ctk.CTkFont(family="Consolas", size=12, weight="bold"),
            text_color="#fbbf24",
            fg_color=("#1f2937", "#111827"),
            corner_radius=6,
            padx=10,
            pady=4
        )
        self.lan_info_lbl.pack(side="left", padx=(0, 8))

        copy_btn = ctk.CTkButton(
            row,
            text="📋 Copy IP",
            width=80,
            height=28,
            fg_color="#475569",
            hover_color="#334155",
            command=lambda: self._copy_ip_to_clipboard(lan_ip_display)
        )
        copy_btn.pack(side="left")

    def _build_metrics_cards(self):
        """Summary status cards."""
        cards_frame = ctk.CTkFrame(self, fg_color="transparent")
        cards_frame.pack(fill="x", padx=16, pady=4)
        cards_frame.columnconfigure((0, 1, 2, 3), weight=1)

        # Card 1: Clients Connected
        self.card_clients = self._create_metric_card(cards_frame, 0, "Connected Clients", "0 Active", "#38bdf8")

        # Card 2: CPU Cores
        self.card_cpu = self._create_metric_card(cards_frame, 1, "CPU Cores", f"{self.sys_info['cpu_count']} Cores", "#a78bfa")

        # Card 3: Acceleration Engine
        gpu_label = "NVENC GPU" if self.sys_info["nvenc_available"] else "CPU Multi-Thread"
        self.card_gpu = self._create_metric_card(cards_frame, 2, "Hardware Engine", gpu_label, "#4ade80")

        # Card 4: Tasks Completed
        self.card_tasks = self._create_metric_card(cards_frame, 3, "Tasks Executed", "0 Finished", "#fb923c")

    def _create_metric_card(self, parent, col, title, value, val_color):
        card = ctk.CTkFrame(parent, corner_radius=10, fg_color=("#1e293b", "#0f172a"))
        card.grid(row=0, column=col, padx=4, sticky="ew")

        ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=11), text_color="#94a3b8").pack(anchor="w", padx=12, pady=(8, 0))
        val_lbl = ctk.CTkLabel(card, text=value, font=ctk.CTkFont(size=16, weight="bold"), text_color=val_color)
        val_lbl.pack(anchor="w", padx=12, pady=(0, 8))
        return val_lbl

    def _build_middle_workspace(self):
        """Two columns: Left = Connected Clients & Active Task; Right = File Sharing Hub."""
        mid_container = ctk.CTkFrame(self, fg_color="transparent")
        mid_container.pack(fill="both", expand=False, padx=16, pady=4)
        mid_container.columnconfigure(0, weight=5)
        mid_container.columnconfigure(1, weight=5)

        # LEFT: Connected Clients & Active Task
        left_col = ctk.CTkFrame(mid_container, corner_radius=10)
        left_col.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=4)

        ctk.CTkLabel(
            left_col,
            text="💻 Active Connected Clients (Computer A)",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#93c5fd"
        ).pack(anchor="w", padx=12, pady=(8, 4))

        self.client_listbox = ctk.CTkTextbox(
            left_col,
            height=120,
            font=ctk.CTkFont(family="Consolas", size=11),
            fg_color=("#111827", "#030712"),
            text_color="#e2e8f0"
        )
        self.client_listbox.pack(fill="x", padx=12, pady=(0, 8))
        self.client_listbox.insert("end", "No clients currently connected.\nConnect Computer A by clicking 'Handshake & Ping' in Client GUI.\n")
        self.client_listbox.configure(state="disabled")

        # Active Task Progress Section
        ctk.CTkLabel(
            left_col,
            text="⚙️ Current Workload Execution",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#cbd5e1"
        ).pack(anchor="w", padx=12, pady=(2, 2))

        self.task_prog_bar = ctk.CTkProgressBar(left_col, height=14, corner_radius=6)
        self.task_prog_bar.set(0.0)
        self.task_prog_bar.pack(fill="x", padx=12, pady=(4, 2))

        self.task_prog_lbl = ctk.CTkLabel(
            left_col,
            text="Idle — Waiting for offloaded tasks from Computer A",
            font=ctk.CTkFont(size=11),
            text_color="#94a3b8"
        )
        self.task_prog_lbl.pack(anchor="w", padx=12, pady=(0, 8))

        # RIGHT: LAN File Sharing Hub
        right_col = ctk.CTkFrame(mid_container, corner_radius=10)
        right_col.grid(row=0, column=1, sticky="nsew", padx=(6, 0), pady=4)

        fhead = ctk.CTkFrame(right_col, fg_color="transparent")
        fhead.pack(fill="x", padx=12, pady=(8, 4))

        ctk.CTkLabel(
            fhead,
            text="📁 Server Storage & LAN File Sharing Hub",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#93c5fd"
        ).pack(side="left")

        refresh_btn = ctk.CTkButton(
            fhead,
            text="🔄 Refresh",
            width=70,
            height=24,
            fg_color="#334155",
            hover_color="#1e293b",
            command=self._refresh_file_list
        )
        refresh_btn.pack(side="right")

        self.files_textbox = ctk.CTkTextbox(
            right_col,
            height=120,
            font=ctk.CTkFont(family="Consolas", size=11),
            fg_color=("#111827", "#030712"),
            text_color="#e2e8f0"
        )
        self.files_textbox.pack(fill="x", padx=12, pady=(0, 6))

        # File actions row
        factions = ctk.CTkFrame(right_col, fg_color="transparent")
        factions.pack(fill="x", padx=12, pady=(0, 8))

        open_folder_btn = ctk.CTkButton(
            factions,
            text="📂 Open Storage Folder",
            height=28,
            fg_color="#0284c7",
            hover_color="#0369a1",
            command=self._open_storage_folder
        )
        open_folder_btn.pack(side="left", padx=(0, 8))

        share_file_btn = ctk.CTkButton(
            factions,
            text="📤 Share Local File to Hub",
            height=28,
            fg_color="#059669",
            hover_color="#047857",
            command=self._share_file_from_server
        )
        share_file_btn.pack(side="left")

    def _build_log_console(self):
        """Bottom Real-Time Log Console."""
        term_frame = ctk.CTkFrame(self, corner_radius=10)
        term_frame.pack(fill="both", expand=True, padx=16, pady=(4, 12))

        head_row = ctk.CTkFrame(term_frame, fg_color="transparent")
        head_row.pack(fill="x", padx=12, pady=(6, 2))

        ctk.CTkLabel(
            head_row,
            text="💻 Integrated Real-Time Live Server Console",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#93c5fd"
        ).pack(side="left")

        clear_btn = ctk.CTkButton(
            head_row,
            text="Clear Log",
            width=70,
            height=22,
            fg_color="#475569",
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
        self.log_textbox.pack(fill="both", expand=True, padx=12, pady=(0, 8))

    # ==========================================
    # LOGGING & THREAD SAFETY
    # ==========================================
    def _poll_ui_queue(self):
        """Processes pending UI updates on the main thread."""
        try:
            while True:
                fn, args, kwargs = self.ui_queue.get_nowait()
                try:
                    fn(*args, **kwargs)
                except Exception:
                    pass
        except queue.Empty:
            pass
        self.after(50, self._poll_ui_queue)

    def run_on_ui(self, fn, *args, **kwargs):
        """Enqueues a callable to execute safely on the GUI thread."""
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

    def _copy_ip_to_clipboard(self, text: str):
        self.clipboard_clear()
        self.clipboard_append(text)
        self.log(f"[CLIPBOARD] Copied Server IP '{text}' to clipboard!")
        messagebox.showinfo("Copied", f"Server IP '{text}' copied!\nEnter this IP in Client GUI on Computer A.")

    # ==========================================
    # SERVER CONTROLS & EVENT LISTENERS
    # ==========================================
    def _toggle_server(self):
        if not self.is_running:
            self._start_server()
        else:
            self._stop_server()

    def _start_server(self):
        host = self.bind_entry.get().strip() or "0.0.0.0"
        try:
            port = int(self.port_entry.get().strip())
        except ValueError:
            messagebox.showerror("Invalid Port", "Please enter a valid integer port number (e.g. 5000).")
            return

        try:
            self.server = RemoteWorkerServer(host=host, port=port, storage_dir=self.storage_dir)
            self.server.on_log = lambda msg: self.log(msg)
            self.server.on_client_connected = self._on_client_connected
            self.server.on_client_disconnected = self._on_client_disconnected
            self.server.on_task_update = self._on_task_update

            self.server_thread = threading.Thread(target=self.server.start, daemon=True)
            self.server_thread.start()

            self.is_running = True
            self.start_btn.configure(text="🛑 Stop Server", fg_color="#dc2626", hover_color="#b91c1c")
            self.status_badge.configure(text="🟢 SERVER LISTENING", text_color="#4ade80")
            self.bind_entry.configure(state="disabled")
            self.port_entry.configure(state="disabled")

            self.log(f"[SERVER STARTED] Listening on {host}:{port}. Ready for Computer A to connect!")
        except Exception as e:
            self.log(f"[START ERROR] Failed to start server: {e}")
            messagebox.showerror("Error", f"Failed to start server:\n{e}")

    def _stop_server(self):
        if self.server:
            self.server.stop()
            self.server = None

        self.is_running = False
        self.start_btn.configure(text="🚀 Start Server", fg_color="#16a34a", hover_color="#15803d")
        self.status_badge.configure(text="🔴 SERVER OFFLINE", text_color="#f87171")
        self.bind_entry.configure(state="normal")
        self.port_entry.configure(state="normal")
        self.log("[SERVER STOPPED] Server daemon halted.")

    def _on_client_connected(self, client_info: Dict[str, Any]):
        """Called when Computer A performs a handshake."""
        client_id = client_info.get("client_id", "Client")
        ip = client_info.get("ip", "Unknown")
        self.log(f"🎉 [CONNECTED] Computer A ('{client_id}' at {ip}) successfully connected!")

        # Update client card
        def _update():
            count = len(self.server.connected_clients) if self.server else 0
            self.card_clients.configure(text=f"{count} Active")
            self._refresh_client_list()
        self.run_on_ui(_update)

    def _on_client_disconnected(self, client_info: Dict[str, Any]):
        """Called when a client disconnects."""
        ip = client_info.get("ip", "Unknown")
        self.log(f"🔴 [DISCONNECTED] Client at {ip} disconnected.")
        def _update():
            count = len(self.server.connected_clients) if self.server else 0
            self.card_clients.configure(text=f"{count} Active")
            self._refresh_client_list()
        self.run_on_ui(_update)

    def _on_task_update(self, task_info: Dict[str, Any]):
        """Called on real-time task progress."""
        pct = task_info.get("percent", 0.0)
        status = task_info.get("status", "")
        task_id = task_info.get("task_id", "")
        task_type = task_info.get("task_type", "")

        def _update():
            self.task_prog_bar.set(pct / 100.0)
            self.task_prog_lbl.configure(text=f"Task {task_id} ({task_type}) - {pct:.1f}% | {status}")
            if self.server:
                self.card_tasks.configure(text=f"{self.server.total_tasks_completed} Finished")
        self.run_on_ui(_update)

    def _schedule_periodic_refresh(self):
        """Periodically refreshes client list and file list."""
        if self.is_running and self.server:
            self._refresh_client_list()
            self._refresh_file_list()
        self.after(3000, self._schedule_periodic_refresh)

    def _refresh_client_list(self):
        """Updates the connected clients textbox."""
        if not self.server:
            return

        clients = self.server.get_active_clients_list()
        self.client_listbox.configure(state="normal")
        self.client_listbox.delete("1.0", "end")

        if not clients:
            self.client_listbox.insert("end", "No clients connected.\nWaiting for Computer A on LAN...")
        else:
            for idx, c in enumerate(clients, 1):
                uptime = int(time.time() - c["connected_at"])
                self.client_listbox.insert(
                    "end",
                    f"[{idx}] Client ID: {c['client_id']}\n"
                    f"    Address  : {c['addr']}\n"
                    f"    Status   : {c['status']} | Tasks Run: {c['tasks_count']} | Uptime: {uptime}s\n"
                    f"    --------------------------------------------------\n"
                )
        self.client_listbox.configure(state="disabled")

    def _refresh_file_list(self):
        """Updates the shared files listbox."""
        if not self.server:
            return

        files = self.server.get_shared_files()
        self.files_textbox.configure(state="normal")
        self.files_textbox.delete("1.0", "end")

        if not files:
            self.files_textbox.insert("end", "Shared storage is empty.\nShare files from Computer B or upload from Computer A.\n")
        else:
            for f in files:
                self.files_textbox.insert(
                    "end",
                    f"• {f['name']:<30} | {f['size_str']:<10} | {f['modified']}\n"
                )
        self.files_textbox.configure(state="disabled")

    def _open_storage_folder(self):
        """Opens server_storage folder in Windows Explorer."""
        try:
            os.startfile(self.storage_dir)
            self.log(f"[EXPLORER] Opened server storage folder: {self.storage_dir}")
        except Exception as e:
            self.log(f"[ERROR] Could not open storage folder: {e}")

    def _share_file_from_server(self):
        """Copies any file from Computer B into server_storage so Computer A can download it."""
        src_path = filedialog.askopenfilename(title="Select File to Share with Computer A")
        if src_path:
            import shutil
            fname = os.path.basename(src_path)
            dest = os.path.join(self.storage_dir, fname)
            try:
                shutil.copy2(src_path, dest)
                self.log(f"📁 [FILE SHARED] Added '{fname}' ({format_bytes(os.path.getsize(dest))}) to shared storage.")
                self._refresh_file_list()
                messagebox.showinfo("File Shared", f"File '{fname}' added to Shared Storage!\nComputer A can now download it over the LAN.")
            except Exception as e:
                self.log(f"[ERROR] Could not share file: {e}")
                messagebox.showerror("Error", f"Could not share file: {e}")


def main():
    app = ServerDashboardApp()
    app.mainloop()


if __name__ == "__main__":
    main()
