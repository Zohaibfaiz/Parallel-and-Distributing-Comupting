# Screenshots & demo media

Add your own captures here (taken on your real client + worker) before pushing - the main README links to them:

| File | What to capture |
|---|---|
| `01_network_ping.png` | Terminal: `ping 192.168.1.1` + `python client/cli.py check --host 192.168.1.1` (shows RTT / NVENC) |
| `02_worker_daemon.png` | Worker terminal after starting `server/daemon.py` (GPU name, NVENC available) |
| `03_gui_idle.png` | GUI before starting: IP, file, quality settings filled in, status "ONLINE" |
| `04_gui_progress.png` | GUI while rendering: progress bar, stage label, live log lines from the worker |
| `05_gui_done.png` | GUI after finishing: timing line in the log + local-vs-remote speedup line |
| `06_worker_gpu.png` | `nvidia-smi` (or Task Manager > GPU > Video Encode) on the worker while rendering |
| `07_benchmark_report.png` | The generated `REPORT.md` charts (or the terminal output of `benchmark.py`) |
| `../demo.gif` | 15-30 s screen recording of an offload from click to finished file (ScreenToGif / peek / OBS) |
