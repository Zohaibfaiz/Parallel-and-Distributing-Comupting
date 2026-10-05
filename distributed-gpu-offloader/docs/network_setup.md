# Network configuration guide

Goal: a private point-to-point link between the **worker PC** (NVIDIA GPU, 4 GB+) and the **client laptop**.

| Machine | Role | Static IP | Subnet mask | Gateway / DNS |
|---|---|---|---|---|
| Worker PC | server | `192.168.1.1` | `255.255.255.0` (/24) | leave empty |
| Client laptop | client | `192.168.1.2` | `255.255.255.0` (/24) | leave empty |

> A gateway is not needed: both machines are on the same /24 subnet. Leaving it empty also stops Windows
> from routing your Internet traffic into the cable.

## Option A - Direct CAT6 Ethernet cable (recommended)

Modern NICs auto-detect crossover (Auto-MDI/MDIX), so a normal CAT6 patch cable is enough.
Expect 1 Gbit/s (about 110 MB/s) on gigabit NICs; latency is usually below 1 ms.

### Windows (worker and/or client)
GUI: *Settings -> Network & Internet -> Ethernet -> Edit IP assignment -> Manual -> IPv4 on* ->
IP address `192.168.1.1` (worker) or `192.168.1.2` (client), subnet prefix length `24`, leave gateway/DNS empty.

Or one command in an **Administrator PowerShell** (list adapters with `Get-NetAdapter`):

```powershell
.\scripts\setup_static_ip_windows.ps1 -Role server -Interface "Ethernet"   # on the worker
.\scripts\setup_static_ip_windows.ps1 -Role client -Interface "Ethernet"   # on the laptop
.\scripts\setup_static_ip_windows.ps1 -Role reset  -Interface "Ethernet"   # undo (back to DHCP)
```
The worker variant also creates the Windows Firewall rules for **TCP 5050** and ping (Private profile).

### Linux
```bash
ip -br link                                              # find the interface name, e.g. enp3s0
sudo ./scripts/setup_static_ip_linux.sh server enp3s0    # worker -> 192.168.1.1/24 (+ ufw rule)
sudo ./scripts/setup_static_ip_linux.sh client enp3s0    # laptop -> 192.168.1.2/24
```

### macOS (client)
*System Settings -> Network -> Ethernet/USB adapter -> Details -> TCP/IP -> Configure IPv4: Manually*,
address `192.168.1.2`, subnet mask `255.255.255.0`, router empty.

## Option B - Dedicated Wi-Fi subnet
Use a router/hotspot that is dedicated to the two machines (no other clients), give them the same static
addresses (or DHCP reservations), keep both on 5 GHz, and place them close together. Expect 200-600 Mbit/s and
2-10 ms latency. For the best numbers use the cable.

## Verify the link (do this before starting the app)

```text
# from the laptop
ping 192.168.1.1                      # replies < 1 ms on a direct cable
# after the daemon is running on the worker:
python client/cli.py check --host 192.168.1.1 --token <token>
python client/cli.py bandwidth --host 192.168.1.1 --token <token>   # should be ~900 Mbit/s on gigabit
```
Optional second opinion with iperf3: `iperf3 -s` on the worker, `iperf3 -c 192.168.1.1` on the laptop.

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `ping` times out | Cable not in the right port, wrong adapter edited, a different subnet, or Windows firewall blocking ICMP (run the PowerShell script on the worker). |
| Ping works but the app says *connection refused* | The daemon is not running, or runs on another port (`--port`). |
| Ping works but the app says *timed out* | Firewall on the worker blocks TCP 5050. Windows: allow the port for the **Private** profile (the Ethernet link must be classified as *Private*, not *Public*). Linux: `sudo ufw allow 5050/tcp`. |
| *authentication failed* | `--token` on the daemon differs from the Token field in the GUI. |
| Only ~90 Mbit/s | One side negotiated 100 Mbit/s (bad cable or port). Check *Adapter -> Speed* (`Get-NetAdapter` / `ethtool enp3s0`). |
| Internet disappeared on the laptop | You set a gateway/DNS on the cable adapter. Clear them; use Wi-Fi for Internet and the cable only for the worker. |
| Worker shows *NVENC NOT available* | Update the NVIDIA driver; use an FFmpeg build with NVENC (Windows: gyan.dev/BtbN "full/gpl" builds; Linux: `ffmpeg -encoders | grep nvenc`). |
