# Distributed Task Offloading & Remote GPU Rendering System

**CSC-334 Parallel and Distributed Computing - Assignment**
Author: Muhammad Abdul Rehman (FA23-BSE-068) - COMSATS University Islamabad, Vehari Campus

A client/server system that offloads heavy video transcoding from a weak laptop to a remote worker with an NVIDIA GPU
(FFmpeg + **NVENC**, optional **PyTorch CUDA** jobs) over a direct Ethernet/Wi-Fi link, with live progress streaming,
integrity checking and automatic recovery from network failures.

```text
 CLIENT LAPTOP                                              REMOTE WORKER (NVIDIA GPU)
┌──────────────────────────────┐   TCP :5050 (custom     ┌─────────────────────────────────┐
│ CustomTkinter GUI / CLI      │   framed protocol)      │ Listener (thread / connection)  │
│  file picker, settings, log  │ ──────────────────────► │  HMAC handshake + ping          │
│ net_client:                  │   1 handshake+ping      │  Job manager + FIFO queue       │
│  SHA-256, upload (resume)    │   2 upload  (SHA-256)   │  GPU engine:                    │
│  progress watcher (reconnect)│   3 progress stream ◄── │   ffmpeg -c:v h264_nvenc        │
│  download (resume, verify)   │   4 download (SHA-256)  │   (-hwaccel cuda) / torch CUDA  │
└──────────────────────────────┘                         └─────────────────────────────────┘
```

## Features

| Assignment task | What is implemented |
|---|---|
| **1. Networking & handshake** (20) | Static-IP guide + scripts (`192.168.1.1` / `.2`); framed TCP protocol; challenge-response handshake (HMAC-SHA256 of a server nonce with a shared token, protocol-version check); **latency ping test** (min/avg/max/jitter) and TCP throughput test before any job; friendly diagnosis of refused / timed-out / wrong-token / no-NVENC situations. |
| **2. Remote GPU daemon** (25) | `server/daemon.py`: multi-client threaded daemon, FIFO job queue with worker threads, strict whitelist validation of all render parameters (no shell injection), rotating log file, optional `--daemon` mode and systemd unit. `server/gpu_engine.py`: **FFmpeg `h264_nvenc` / `hevc_nvenc`** with optional CUDA hardware decoding (automatic software-decode retry), **PyTorch CUDA matmul** job, and a clearly-logged CPU fallback (libx264/x265) so the code can be tested without a GPU. |
| **3. Client GUI** (25) | `client/app.py` (CustomTkinter, dark theme): server IP/port/token fields + *Test connection*, file picker with ffprobe info, **Resolution / Bitrate / Preset / Codec** controls, output folder, stage label + progress bar, stats line, **live log terminal** (colour coded, worker log lines streamed in), *Cancel*, *Render locally (compare)* with automatic speedup message, settings saved between sessions. A headless `client/cli.py` offers the same functions. |
| **4. Progress & robustness** (15) | Worker pushes `PROGRESS`/`EVENT` messages (percent, fps, speed, ETA) on a dedicated watch connection; per-phase weighting gives one overall bar. **SHA-256 verified in both directions**; **resumable upload and download** (byte offsets); automatic reconnect with exponential back-off (jobs keep running on the worker when the socket dies and the client re-attaches and replays missed events); socket timeouts, heartbeat, cancellation, idle-connection reaping, clean error messages. |
| **5. Benchmark & report** (15) | `benchmark/benchmark.py` generates test clips (or uses yours), runs local vs remote N times, and records hash / upload / queue / render / download / verify times, CPU of both machines, GPU utilisation / VRAM / power of the worker. `benchmark/report.py` turns the data into charts + a formal `REPORT.md` with **speedup**, **compute-only speedup**, **network overhead**, **break-even bandwidth** and written analysis. |

## Repository layout

```text
client/      app.py (GUI)  cli.py  net_client.py (protocol client)  local_render.py
server/      daemon.py     gpu_engine.py   gpu-worker.service (systemd)
common/      protocol.py (wire format, auth, checksums)  ffmpeg_utils.py  util.py
benchmark/   benchmark.py  report.py
scripts/     run_server.{sh,bat}  run_client.{sh,bat}  setup_static_ip_{linux.sh,windows.ps1}
tests/       test_system.py   (13 end-to-end tests on loopback)
docs/        network_setup.md  screenshots/  sample_output/
```

## 1. Setup

### Prerequisites (both machines)
* Python **3.10+**
* **FFmpeg + ffprobe** in `PATH` (`ffmpeg -version`).
  * Windows: download a *full* build from <https://www.gyan.dev/ffmpeg/builds/> or BtbN, add `bin/` to `PATH`.
  * Ubuntu: `sudo apt install ffmpeg` (check NVENC: `ffmpeg -encoders | grep nvenc`). macOS: `brew install ffmpeg` (client only).

### Worker node (server, NVIDIA GPU)
1. Install the **NVIDIA driver** (`nvidia-smi` must work) and an FFmpeg build that contains NVENC.
2. Copy / clone this repository, then:
   ```bash
   pip install -r server/requirements.txt
   # optional, only for the PyTorch CUDA job type (pick the CUDA build for your driver):
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```

### Client laptop
```bash
pip install -r client/requirements.txt          # customtkinter + psutil
pip install -r benchmark/requirements.txt       # only needed to run the benchmark/report (matplotlib)
```
On Linux the GUI also needs Tk: `sudo apt install python3-tk`.

## 2. Network configuration (static IPs over Ethernet / Wi-Fi)

Full guide with troubleshooting: **[docs/network_setup.md](docs/network_setup.md)**. Short version:

| | IP | Mask |
|---|---|---|
| Worker (server) | `192.168.1.1` | `255.255.255.0` |
| Client laptop | `192.168.1.2` | `255.255.255.0` |

```powershell
# Windows, Administrator PowerShell (use -Role client on the laptop)
.\scripts\setup_static_ip_windows.ps1 -Role server -Interface "Ethernet"
```
```bash
# Linux
sudo ./scripts/setup_static_ip_linux.sh server enp3s0
```
Then check the cable: `ping 192.168.1.1` from the laptop.

## 3. Running

### Start the daemon on the worker
```bash
# Linux/macOS
OFFLOAD_TOKEN=mysecret python3 server/daemon.py --host 0.0.0.0 --port 5050
# Windows
set OFFLOAD_TOKEN=mysecret
python server\daemon.py --host 0.0.0.0 --port 5050
```
(or `scripts/run_server.sh` / `scripts\run_server.bat`). The start-up banner shows the GPU, whether NVENC works and
the IP addresses the worker is reachable on. Useful flags:

| Flag | Meaning |
|---|---|
| `--token` / `OFFLOAD_TOKEN` | shared secret - **must be identical on the client** |
| `--engine auto\|gpu\|cpu` | `auto` = NVENC if available else CPU; `gpu` = refuse to start without NVENC; `cpu` = force software |
| `--no-hwdecode` | do not use `-hwaccel cuda` for decoding |
| `--workers N` | concurrent GPU jobs (default 1; others wait in the queue) |
| `--max-gb`, `--retention-hours` | upload size limit, automatic clean-up of old jobs |
| `--daemon --pidfile F` | detach into the background (Linux/macOS); on Linux you can also use `server/gpu-worker.service` |
| `--workdir DIR` | where uploads / results / `worker.log` are stored |

### Launch the GUI on the laptop
```bash
python client/app.py            # or scripts/run_client.sh | scripts\run_client.bat
```
1. Enter **Server IP** `192.168.1.1`, port `5050`, the same **token** → *Test connection* (shows RTT, jitter, link speed, GPU, NVENC).
2. *Browse…* an input video; choose **Resolution, Codec, Bitrate, Preset** (`p1` fastest … `p7` best quality) and the output folder.
3. **Start offload** - watch the bar go through *checksum → upload → queue → rendering on remote GPU → download → verify*, with the worker's own log lines in the terminal panel.
4. Optional: **Render locally (compare)** with the same settings; the log then prints the speedup factor.

### Command line (same engine, no GUI)
```bash
python client/cli.py check     --host 192.168.1.1 --token mysecret
python client/cli.py bandwidth --host 192.168.1.1 --token mysecret
python client/cli.py transcode --host 192.168.1.1 --token mysecret input.mp4 --resolution 1080 --bitrate 8M --preset p5
python client/cli.py local     input.mp4 --resolution 1080 --bitrate 8M --preset p5
python client/cli.py torch     --host 192.168.1.1 --token mysecret --n 4096 --iters 100     # PyTorch CUDA TFLOPS
```

## 4. Benchmark & technical report (Task 5)

```bash
python benchmark/benchmark.py --host 192.168.1.1 --token mysecret \
    --matrix full --runs 3 --res 1080 --bitrate 8M --preset p5 \
    --client-desc "Laptop: <CPU>, <GPU 500MB>, <RAM>" --server-desc "PC: <CPU>, <GPU 4GB>, <RAM>"
# optionally add your own clips:  --inputs C:\videos\a.mp4 C:\videos\b.mkv
```
`--matrix quick` (3 clips) or `full` (6 clips, 360p→2160p, 20-90 s). Output goes to `benchmark/results/<timestamp>/`:
`results.json`, `results.csv`, four PNG charts and **`REPORT.md`** (setup tables, results, analysis, conclusion).
Regenerate the report from old data with `python benchmark/report.py <results.json>`; export `REPORT.md` to PDF with
any Markdown viewer / `pandoc REPORT.md -o REPORT.pdf` for submission.

Definitions used in the report:

| Metric | Formula |
|---|---|
| Speedup | `S = T_local / T_remote_total` |
| Compute-only speedup | `S_c = T_local / T_render_remote` |
| Network overhead | `(T_upload + T_download) / T_remote_total` |
| Integrity overhead | `(T_hash + T_verify) / T_remote_total` |
| Break-even bandwidth | `B* = 8·(Size_in + Size_out) / (T_local − T_render_remote)` |

> **Honest note:** the report states which encoder the worker really used. If NVENC is missing it prints a warning and the
> numbers are CPU-vs-CPU. `docs/sample_output/` contains a loopback/CPU sample that only demonstrates the pipeline - replace it
> with your real measurements before submitting.

## Protocol summary

Every message is a frame `type(1 byte) | length(4 bytes, big-endian) | payload`; `J` = JSON control message, `D` = raw data chunk (1 MiB).

```text
worker → WELCOME {version, nonce}        client → HELLO {version, HMAC-SHA256(token, nonce)}      worker → OK {GPU/NVENC info}
client → PING/PONG (latency)             client → BWTEST (throughput)
client → SUBMIT {job_id, size, sha256, settings}  worker → ACCEPT {offset}   (offset > 0 = resume)
client → D D D …  worker → UPLOAD_OK | UPLOAD_BAD (checksum)  worker → QUEUED
client → WATCH {job_id, since}           worker → EVENT / PROGRESS / HEARTBEAT … WATCH_END       (reconnect-safe)
client → FETCH {job_id, offset}          worker → RESULT {size, sha256} then D D D …             (resume-safe)
client → CANCEL | CLEANUP | BYE
```

## Testing

```bash
python -m unittest discover -s tests -v
```
13 end-to-end tests start a real worker on loopback (CPU engine) and verify: handshake & latency, wrong token, unreachable
host message, bandwidth test, full round trip with progress stages, corrupted-upload detection, **upload resume after a
simulated cable pull**, **progress-stream reconnect**, **download resume**, retry exhaustion, cancellation, parameter-injection
rejection and bad job ids.

## Security notes
* Shared-token HMAC challenge-response (the token is never sent); constant-time comparison; failed logins are delayed.
* Render parameters are validated against a whitelist and FFmpeg is always started with an argument list (no shell).
* Job ids are 32 hex characters (no path traversal); upload size and free disk space are checked.
* The traffic is **not encrypted** - this is intended for a private point-to-point link. Do not expose port 5050 to the Internet.

## Screenshots & demo

<img width="1600" height="900" alt="WhatsApp Image 2026-10-04 at 8 44 34 AM" src="https://github.com/user-attachments/assets/e3a03bfd-1370-476f-b2f0-736b392933d4" />
<img width="945" height="601" alt="WhatsApp Image 2026-10-04 at 8 44 35 AM" src="https://github.com/user-attachments/assets/f6f1e3d1-b207-42c4-bc50-980ef2db02d8" />
<img width="1600" height="916" alt="WhatsApp Image 2026-10-04 at 8 44 34 AM (2)" src="https://github.com/user-attachments/assets/19172467-1ed8-43dc-b186-71f5f9ce3558" />
<img width="1600" height="902" alt="WhatsApp Image 2026-10-04 at 8 44 34 AM (1)" src="https://github.com/user-attachments/assets/95ba1cb4-45b1-4934-b6bc-12427ea33977" />

## License
MIT - see [LICENSE](LICENSE).
