#!/usr/bin/env python3
"""Turn benchmark results.json into charts + a formal technical report (REPORT.md).

    python benchmark/report.py benchmark/results/<run>/results.json [--out DIR]

Metrics (all per test clip, averaged over the repetitions)
  Speedup (end-to-end)   S      = T_local / T_remote_total
  Compute-only speedup   S_c    = T_local / T_render_remote
  Network overhead       O_net  = (T_upload + T_download) / T_remote_total
  Integrity overhead     O_int  = (T_hash + T_verify)     / T_remote_total
  Break-even bandwidth   B*     = 8 * (Size_in + Size_out) / (T_local - T_render_remote)   [Mbit/s]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import OrderedDict
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # charts are optional
    plt = None


def _mean(v):
    v = [x for x in v if x is not None]
    return statistics.mean(v) if v else None


def _std(v):
    v = [x for x in v if x is not None]
    return statistics.pstdev(v) if len(v) > 1 else 0.0


def aggregate(rows: list[dict]) -> list[dict]:
    groups: "OrderedDict[str, list[dict]]" = OrderedDict()
    for r in rows:
        groups.setdefault(r["case"], []).append(r)
    out = []
    for case, rs in groups.items():
        g = lambda k: _mean([r.get(k) for r in rs])  # noqa: E731
        a = {"case": case, "runs": len(rs), "width": rs[0]["width"], "height": rs[0]["height"],
             "duration_s": rs[0]["duration_s"], "input_mb": g("input_mb"), "output_mb": g("output_mb"),
             "local_s": g("local_s"), "local_std": _std([r.get("local_s") for r in rs]),
             "remote_total_s": g("remote_total_s"), "remote_std": _std([r.get("remote_total_s") for r in rs]),
             "hash_s": g("hash_s"), "upload_s": g("upload_s"), "queue_s": g("queue_s"), "render_s": g("render_s"),
             "download_s": g("download_s"), "verify_s": g("verify_s"), "upload_mbps": g("upload_mbps"),
             "download_mbps": g("download_mbps"), "avg_fps": g("avg_fps"), "retries": sum(r.get("retries", 0) for r in rs),
             "local_cpu": g("local_cpu_avg"), "local_gpu": g("local_gpu_avg"), "client_cpu_off": g("client_cpu_avg_offload"),
             "server_gpu": g("server_gpu_avg"), "server_gpu_max": g("server_gpu_max"), "server_vram": g("server_vram_mb"),
             "server_cpu": g("server_cpu_avg"), "server_power": g("server_gpu_power_w"),
             "local_engine": rs[0].get("local_engine"), "remote_engine": rs[0].get("remote_engine")}
        T = a["remote_total_s"]
        a["net_s"] = a["upload_s"] + a["download_s"]
        a["integrity_s"] = a["hash_s"] + a["verify_s"]
        a["net_pct"] = a["net_s"] / T * 100 if T else None
        a["int_pct"] = a["integrity_s"] / T * 100 if T else None
        a["render_pct"] = a["render_s"] / T * 100 if T else None
        if a["local_s"]:
            a["speedup"] = a["local_s"] / T
            a["speedup_compute"] = a["local_s"] / a["render_s"] if a["render_s"] else None
            gain = a["local_s"] - a["render_s"]
            a["breakeven_mbps"] = (8 * (a["input_mb"] + a["output_mb"]) / gain) if gain > 0 else None
        else:
            a["speedup"] = a["speedup_compute"] = a["breakeven_mbps"] = None
        out.append(a)
    return out


# --------------------------------------------------------------------------- charts
def make_charts(agg: list[dict], out_dir: Path) -> dict[str, str]:
    if plt is None:
        return {}
    charts = {}
    names = [a["case"] for a in agg]
    x = range(len(agg))
    have_local = all(a["local_s"] for a in agg)

    fig, ax = plt.subplots(figsize=(9, 4.6))
    w = 0.38
    if have_local:
        ax.bar([i - w / 2 for i in x], [a["local_s"] for a in agg], w, yerr=[a["local_std"] for a in agg],
               label="Local (client)", color="#d9534f", capsize=3)
    ax.bar([i + w / 2 for i in x], [a["remote_total_s"] for a in agg], w, yerr=[a["remote_std"] for a in agg],
           label="Remote GPU (end-to-end)", color="#2f7ed8", capsize=3)
    ax.set_xticks(list(x)); ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Time (s)"); ax.set_title("Local rendering vs. remote GPU offload"); ax.legend(); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(out_dir / "chart_time_comparison.png", dpi=140); plt.close(fig)
    charts["time"] = "chart_time_comparison.png"

    fig, ax = plt.subplots(figsize=(9, 4.6))
    bottom = [0.0] * len(agg)
    for key, label, color in (("hash_s", "SHA-256 hash", "#8e8e93"), ("upload_s", "Upload", "#f0ad4e"),
                              ("queue_s", "Queue wait", "#b39ddb"), ("render_s", "GPU render", "#2f7ed8"),
                              ("download_s", "Download", "#5cb85c"), ("verify_s", "Verify", "#bdbdbd")):
        vals = [a[key] or 0 for a in agg]
        ax.bar(names, vals, bottom=bottom, label=label, color=color)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_ylabel("Time (s)"); ax.set_title("Where the remote time goes"); ax.legend(ncol=3, fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right"); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(out_dir / "chart_remote_breakdown.png", dpi=140); plt.close(fig)
    charts["breakdown"] = "chart_remote_breakdown.png"

    if have_local:
        fig, ax = plt.subplots(figsize=(9, 4.4))
        order = sorted(agg, key=lambda a: a["input_mb"])
        ax.plot([a["input_mb"] for a in order], [a["speedup"] for a in order], "o-", color="#2f7ed8", label="End-to-end speedup")
        ax.plot([a["input_mb"] for a in order], [a["speedup_compute"] or 0 for a in order], "s--", color="#5cb85c",
                label="Compute-only speedup")
        ax.axhline(1, color="#d9534f", ls=":", label="break-even (x1)")
        for a in order:
            ax.annotate(a["case"], (a["input_mb"], a["speedup"]), textcoords="offset points", xytext=(4, 6), fontsize=7)
        ax.set_xlabel("Input file size (MB)"); ax.set_ylabel("Speedup (x)")
        ax.set_title("Speedup vs. input size"); ax.legend(); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(out_dir / "chart_speedup.png", dpi=140); plt.close(fig)
        charts["speedup"] = "chart_speedup.png"

    fig, ax = plt.subplots(figsize=(9, 4.4))
    series = (("Client CPU - local render", "local_cpu", "#d9534f"),
              ("Client CPU - during offload", "client_cpu_off", "#f0ad4e"),
              ("Worker GPU util (avg)", "server_gpu", "#2f7ed8"), ("Worker CPU (avg)", "server_cpu", "#8e8e93"))
    bw = 0.2
    for j, (label, key, color) in enumerate(series):
        ax.bar([i + (j - 1.5) * bw for i in x], [a[key] or 0 for a in agg], bw, label=label, color=color)
    ax.set_xticks(list(x)); ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Utilisation (%)"); ax.set_ylim(0, 105); ax.set_title("Resource utilisation")
    ax.legend(fontsize=8, ncol=2); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(out_dir / "chart_resources.png", dpi=140); plt.close(fig)
    charts["resources"] = "chart_resources.png"
    return charts


# --------------------------------------------------------------------------- markdown
def _f(v, nd=1, suffix=""):
    return "n/a" if v is None else f"{v:.{nd}f}{suffix}"


def _gpu_txt(gpus):
    return "; ".join(f"{g['name']} ({g['memory_mb']} MB)" for g in gpus) or "none detected"


def generate(results_json: Path, out_dir: Path | None = None) -> Path:
    results_json = Path(results_json)
    out_dir = Path(out_dir or results_json.parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(results_json.read_text())
    meta, rows = data["meta"], data["rows"]
    agg = aggregate(rows)
    charts = make_charts(agg, out_dir)
    c, s, lk, st = meta["client"], meta["server"], meta["link"], meta["settings"]
    have_local = all(a["local_s"] for a in agg)
    L = []
    add = L.append

    add("# Performance Evaluation Report - Distributed Task Offloading & Remote GPU Rendering\n")
    add(f"*Course: CSC-334 Parallel and Distributed Computing - generated {meta['timestamp']}*\n")
    add("## 1. Objective\n")
    add("This report evaluates whether offloading video transcoding from a resource-constrained client laptop to a "
        "remote worker with a dedicated NVIDIA GPU (FFmpeg + NVENC) reduces the end-to-end completion time. "
        "It quantifies the **speedup**, the **network transfer overhead** and the **resource utilisation** of both machines.\n")

    add("## 2. Experimental setup\n")
    add("| | Client (local baseline) | Worker (remote GPU) |\n|---|---|---|")
    add(f"| Description | {c['description']} | {s['description']} |")
    add(f"| Host / OS | {c['platform']} | {s.get('hostname') or 'n/a'} |")
    add(f"| CPU | {c['cpu']} ({c['cpu_threads']} threads) | see description |")
    add(f"| RAM | {_f(c['ram_gb'], 1, ' GB')} | see description |")
    add(f"| GPU(s) | {_gpu_txt(c['gpus'])} | {_gpu_txt(s['gpus'])} |")
    add(f"| Encoder used | {agg[0]['local_engine'] or 'n/a'} | {agg[0]['remote_engine'] or 'n/a'} (NVENC available: {s['nvenc']}) |")
    add(f"| FFmpeg | local install | {s.get('ffmpeg') or 'n/a'} |\n")
    if not s.get("nvenc") or "nvenc" not in str(agg[0]["remote_engine"] or "").lower():
        add("> **Warning:** the worker did not use NVENC for these runs (encoder: "
            f"`{agg[0]['remote_engine']}`). The numbers below therefore describe CPU-to-CPU offloading and must not be "
            "presented as GPU results - re-run the benchmark on a worker with a working NVIDIA GPU.\n")
    add(f"**Network link:** {lk['description']} - target `{lk['host']}`; ping RTT avg **{_f(lk['rtt_avg_ms'], 2)} ms** "
        f"(min {_f(lk['rtt_min_ms'], 2)}, max {_f(lk['rtt_max_ms'], 2)}, jitter {_f(lk['jitter_ms'], 2)} ms); "
        f"raw TCP throughput measured by the system: **{_f(lk['iperf_like_up_mbps'], 0)} Mbit/s up / "
        f"{_f(lk['iperf_like_down_mbps'], 0)} Mbit/s down**.\n")
    add(f"**Render settings (identical on both sides):** resolution height `{st['resolution']}`, bitrate `{st['bitrate']}`, "
        f"preset `{st['preset']}`, codec `{st['codec']}`. Each clip was processed **{meta['runs']}x**; values are means "
        "(± population std-dev for totals).\n")

    add("## 3. Methodology\n")
    add("* Test clips are synthetic (FFmpeg `testsrc2` + noise, H.264, realistic source bitrate, AAC audio) or user supplied, "
        "covering several resolutions and durations (= file sizes).\n"
        "* **Local run:** the client transcodes the clip with FFmpeg on its own hardware (NVENC if the laptop has one, otherwise libx264/libx265).\n"
        "* **Remote run:** the client offloads the clip through the custom protocol: SHA-256 hash → upload → worker queue → "
        "GPU render → download → SHA-256 verification. All phases are timed (`time.perf_counter`).\n"
        "* CPU utilisation is sampled once per second with `psutil`; GPU utilisation / VRAM / power on the worker through `nvidia-smi`.\n"
        "* Definitions: " + "  \n".join([
            "Speedup *S* = T_local / T_remote_total",
            "Compute-only speedup *S_c* = T_local / T_render_remote",
            "Network overhead *O_net* = (T_upload + T_download) / T_remote_total",
            "Integrity overhead *O_int* = (T_hash + T_verify) / T_remote_total",
            "Break-even bandwidth *B\\** = 8·(Size_in + Size_out) / (T_local − T_render_remote)"]) + "\n")

    add("## 4. Results\n")
    add("### 4.1 Execution time\n")
    add("| Clip | Resolution | Dur. | In (MB) | Out (MB) | Local (s) | Remote total (s) | Remote render (s) | Speedup S | Compute speedup S_c |\n"
        "|---|---|---|---|---|---|---|---|---|---|")
    for a in agg:
        add(f"| {a['case']} | {a['width']}x{a['height']} | {a['duration_s']:.0f}s | {a['input_mb']:.1f} | {a['output_mb']:.1f} | "
            f"{_f(a['local_s'])} ± {a['local_std']:.1f} | {a['remote_total_s']:.1f} ± {a['remote_std']:.1f} | {a['render_s']:.1f} | "
            f"**{_f(a['speedup'], 2, 'x')}** | {_f(a['speedup_compute'], 2, 'x')} |")
    add("")
    if "time" in charts:
        add(f"![Local vs remote]({charts['time']})\n")
    if "speedup" in charts:
        add(f"![Speedup vs size]({charts['speedup']})\n")

    add("### 4.2 Network transfer overhead\n")
    add("| Clip | Hash (s) | Upload (s) | Upload Mbit/s | Download (s) | Download Mbit/s | Verify (s) | Network share O_net | Integrity share O_int | Render share |\n"
        "|---|---|---|---|---|---|---|---|---|---|")
    for a in agg:
        add(f"| {a['case']} | {a['hash_s']:.2f} | {a['upload_s']:.2f} | {_f(a['upload_mbps'], 0)} | {a['download_s']:.2f} | "
            f"{_f(a['download_mbps'], 0)} | {a['verify_s']:.2f} | {_f(a['net_pct'], 1, '%')} | {_f(a['int_pct'], 1, '%')} | "
            f"{_f(a['render_pct'], 1, '%')} |")
    add("")
    if "breakdown" in charts:
        add(f"![Remote time breakdown]({charts['breakdown']})\n")

    add("### 4.3 Resource utilisation\n")
    add("| Clip | Client CPU - local (%) | Client CPU - offload (%) | Worker GPU avg / max (%) | Worker VRAM (MB) | Worker GPU power (W) | Worker CPU (%) | Render fps (worker) |\n"
        "|---|---|---|---|---|---|---|---|")
    for a in agg:
        add(f"| {a['case']} | {_f(a['local_cpu'], 0)} | {_f(a['client_cpu_off'], 0)} | {_f(a['server_gpu'], 0)} / "
            f"{_f(a['server_gpu_max'], 0)} | {_f(a['server_vram'], 0)} | {_f(a['server_power'], 0)} | {_f(a['server_cpu'], 0)} | "
            f"{_f(a['avg_fps'], 0)} |")
    add("")
    if "resources" in charts:
        add(f"![Resource utilisation]({charts['resources']})\n")

    # ---------------- automatic analysis
    add("## 5. Analysis\n")
    if have_local:
        best = max(agg, key=lambda a: a["speedup"])
        worst = min(agg, key=lambda a: a["speedup"])
        mean_s = _mean([a["speedup"] for a in agg])
        mean_net = _mean([a["net_pct"] for a in agg])
        mean_int = _mean([a["int_pct"] for a in agg])
        mean_render = _mean([a["render_pct"] for a in agg])
        add(f"* **Speedup.** Offloading changed the end-to-end time by a mean factor of **{mean_s:.2f}x** "
            f"(best: **{best['speedup']:.2f}x** on `{best['case']}`, worst: **{worst['speedup']:.2f}x** on `{worst['case']}`). "
            + ("Offloading is therefore beneficial for every tested clip. " if worst["speedup"] > 1 else
               "For at least one clip offloading was *slower* than rendering locally - see the break-even analysis below. "))
        mean_sc = _mean([a["speedup_compute"] for a in agg if a["speedup_compute"]])
        if mean_sc:
            add(f"* **Compute-only speedup.** Ignoring all transfers, the worker rendered the clips {mean_sc:.2f}x faster than the client "
                "on average. This is the upper bound of what offloading can achieve; the gap to the end-to-end figure is the cost of moving data.")
        add(f"* **Network & integrity overhead.** On average **{mean_net:.1f}%** of the remote time was spent transferring data "
            f"(upload + download) and **{mean_int:.1f}%** on SHA-256 integrity checks, while **{mean_render:.1f}%** was the actual remote rendering. "
            + ("The transfer cost is small compared to the compute time, so offloading scales well." if mean_net < 25 else
               "Transfer time is a significant part of the total, so the link bandwidth is the limiting factor."))
        ordered = sorted(agg, key=lambda a: a["input_mb"])
        if len(ordered) >= 2:
            first, last = ordered[0], ordered[-1]
            trend = "increases" if last["speedup"] > first["speedup"] else "decreases"
            add(f"* **Effect of file size / resolution.** From `{first['case']}` ({first['input_mb']:.1f} MB) to `{last['case']}` "
                f"({last['input_mb']:.1f} MB) the end-to-end speedup {trend} from {first['speedup']:.2f}x to {last['speedup']:.2f}x. "
                "Small jobs pay the fixed costs (connection set-up, hashing, queueing, process start-up) with little compute to amortise them; "
                "heavy jobs (higher resolution, longer duration) benefit most because compute grows faster than the bytes that must be moved.")
        be = [a for a in agg if a["breakeven_mbps"]]
        if be:
            hi = max(be, key=lambda a: a["breakeven_mbps"])
            link = min(x for x in (lk["iperf_like_up_mbps"], lk["iperf_like_down_mbps"]) if x) if (lk["iperf_like_up_mbps"] or lk["iperf_like_down_mbps"]) else None
            add(f"* **Break-even bandwidth.** Offloading pays off as long as the link is faster than B* = 8·(S_in+S_out)/(T_local−T_render). "
                f"The most demanding case (`{hi['case']}`) needs at least **{hi['breakeven_mbps']:.1f} Mbit/s**"
                + (f"; the measured link ({link:.0f} Mbit/s) is {link / hi['breakeven_mbps']:.1f}x above that." if link else ".")
                + " This is why a direct CAT6 link is preferable to congested Wi-Fi.")
        else:
            add("* **Break-even bandwidth.** For the tested settings the worker was not faster than the client at pure compute, "
                "so no link speed makes offloading worthwhile (B* is undefined).")
        mean_lc, mean_oc = _mean([a["local_cpu"] for a in agg]), _mean([a["client_cpu_off"] for a in agg])
        if mean_lc is not None and mean_oc is not None:
            if mean_oc < mean_lc - 10:
                tail = ", leaving the laptop responsive (less heat and no OOM risk on the client)."
            else:
                tail = ("; the client CPU is still busy during offloading (hashing, socket I/O and, in this run, "
                        "possibly the worker sharing the same machine), so no large relief is visible.")
            add(f"* **Client resources.** During local rendering the client CPU averaged **{mean_lc:.0f}%**; while offloading it was "
                f"**{mean_oc:.0f}%**{tail}")
        mg = _mean([a["server_gpu"] for a in agg])
        if mg is not None:
            add(f"* **Worker resources.** Average GPU utilisation during rendering was **{mg:.0f}%** "
                f"(peak {_f(max(a['server_gpu_max'] or 0 for a in agg), 0)}%), VRAM peak "
                f"{_f(max(a['server_vram'] or 0 for a in agg), 0)} MB. NVENC uses a dedicated fixed-function encoder block, "
                "so the 3D/CUDA utilisation reported by `nvidia-smi` can look low even when the encoder is saturated.")
    else:
        add("* Local baseline was skipped (`--skip-local`); only remote timings are reported.")
    total_retries = sum(a["retries"] for a in agg)
    add(f"* **Robustness.** {total_retries} network retries were needed during the benchmark; every transfer was verified by SHA-256 "
        "and all results were accepted.\n")

    add("## 6. Discussion\n")
    add("**Why offloading can win.** Hardware encoders (NVENC) are ASICs built into the GPU; they encode H.264/HEVC far faster and with "
        "much less power than software encoders on a laptop CPU. Offloading converts the problem into `T_remote = T_hash + T_up + T_queue + T_render + T_down + T_verify`. "
        "As long as `T_up + T_down + overheads < T_local − T_render`, the user gains time, and the laptop is free for other work.\n")
    add("**Limits (Amdahl-style).** The transfer and integrity phases are serial and independent of GPU speed, so even an infinitely fast GPU "
        "cannot beat `S_max = T_local / (T_hash + T_up + T_down + T_verify)`. Output size depends on the chosen bitrate, so a high target bitrate "
        "increases download time.\n")
    add("**Possible improvements.** (1) Pipeline: start rendering while the upload is still in progress (chunked/streaming transcode); "
        "(2) compress or use a lossless-friendly container for upload; (3) use jumbo frames on the direct link; "
        "(4) run several jobs concurrently on GPUs that allow multiple NVENC sessions; (5) overlap hashing with upload.\n")

    add("## 7. Conclusion\n")
    if have_local:
        ms = _mean([a["speedup"] for a in agg])
        verdict = ("offloading was faster than local rendering" if ms > 1 else
                   "offloading was NOT faster than local rendering under these conditions")
        add(f"For the tested workloads {verdict}: mean end-to-end speedup **{ms:.2f}x**, with a mean network share of "
            f"**{_mean([a['net_pct'] for a in agg]):.1f}%** of the total time. "
            "The custom protocol provided latency checking, authenticated sessions, integrity validation, resumable transfers and live progress streaming.")
    else:
        add("The custom protocol provided latency checking, authenticated sessions, integrity validation, resumable transfers and live progress streaming.")
    add("\n---\n*Raw data: `results.json`, `results.csv`. Regenerate this report with `python benchmark/report.py results.json`.*")

    md_path = out_dir / "REPORT.md"
    md_path.write_text("\n".join(L), encoding="utf-8")
    return md_path


def main() -> int:
    p = argparse.ArgumentParser(description="Generate the technical report from benchmark results")
    p.add_argument("results", help="path to results.json")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    print(generate(Path(a.results), Path(a.out) if a.out else None))
    return 0


if __name__ == "__main__":
    sys.exit(main())
