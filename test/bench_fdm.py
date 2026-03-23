#!/usr/bin/env python3
"""
Benchmark script for pochoir FDM solvers.

Measures wall-clock solve time and peak GPU memory for the numpy and
torch engines on representative 2D and 3D Laplace problems, then
queries nvidia-smi for live GPU utilisation.

Usage:
    python bench_fdm.py [--epochs N] [--steps N] [--prec F]

Output is printed to stdout as a plain table.
"""

import argparse
import json
import os
import subprocess
import time

import numpy


# ---------------------------------------------------------------------------
# Problem factories
# ---------------------------------------------------------------------------

def make_2d(shape=(64, 80)):
    iarr = numpy.zeros(shape)
    iarr[0, :] = 1.0
    iarr[-1, :] = -1.0
    barr = numpy.zeros(shape, dtype=bool)
    barr[0, :] = True
    barr[-1, :] = True
    periodic = (False, True)
    return iarr, barr, periodic


def make_3d(shape=(32, 40, 48)):
    iarr = numpy.zeros(shape)
    iarr[:, :, 0] = 1.0
    iarr[:, :, -1] = -1.0
    barr = numpy.zeros(shape, dtype=bool)
    barr[:, :, 0] = True
    barr[:, :, -1] = True
    periodic = (True, True, False)
    return iarr, barr, periodic


# ---------------------------------------------------------------------------
# GPU utilities
# ---------------------------------------------------------------------------

def gpu_info():
    """Return dict with GPU name, total/free memory (MiB) and utilisation (%)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).decode().strip()
        rows = []
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 4:
                rows.append({
                    "name": parts[0],
                    "mem_total_mib": int(parts[1]),
                    "mem_free_mib": int(parts[2]),
                    "util_pct": int(parts[3]),
                })
        return rows
    except Exception:
        return []


def peak_gpu_memory_mib():
    """Return peak GPU memory allocated by torch in MiB (0 if no CUDA)."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 ** 2)
    except Exception:
        pass
    return 0.0


# ---------------------------------------------------------------------------
# Benchmarking helpers
# ---------------------------------------------------------------------------

def bench_numpy(iarr, barr, periodic, prec, epoch, nepochs):
    from pochoir.fdm_numpy import solve
    t0 = time.perf_counter()
    solve(iarr, barr, periodic, prec, epoch, nepochs)
    return time.perf_counter() - t0


def bench_torch(iarr, barr, periodic, prec, epoch, nepochs, dtype_str="float64"):
    import torch
    from pochoir.fdm_torch import solve

    dtype = torch.float64 if dtype_str == "float64" else torch.float32

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    solve(
        iarr, barr, periodic, prec, epoch, nepochs,
        info_msg=lambda msg: None,
        _dtype=dtype,
        phi0=None,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak_mib = peak_gpu_memory_mib()
    return elapsed, peak_mib


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(epochs, steps, prec):
    problems = [
        ("2D 64×80",    make_2d((64, 80))),
        ("3D 32×40×48", make_3d((32, 40, 48))),
    ]

    col_w = 20
    header = (
        f"{'Problem':<{col_w}} {'Engine':<14} {'dtype':<8}"
        f" {'Time (s)':>10} {'Peak GPU (MiB)':>15}"
    )
    sep = "-" * len(header)

    print(sep)
    print(header)
    print(sep)

    for label, (iarr, barr, periodic) in problems:
        # numpy
        t_np = bench_numpy(iarr, barr, periodic, prec, steps, epochs)
        print(f"{label:<{col_w}} {'numpy':<14} {'float64':<8} {t_np:>10.3f} {'n/a':>15}")

        # torch float32
        t32, m32 = bench_torch(iarr, barr, periodic, prec, steps, epochs, "float32")
        mem32 = f"{m32:.1f}" if m32 > 0 else "n/a"
        print(f"{label:<{col_w}} {'torch':<14} {'float32':<8} {t32:>10.3f} {mem32:>15}")

        # torch float64
        t64, m64 = bench_torch(iarr, barr, periodic, prec, steps, epochs, "float64")
        mem64 = f"{m64:.1f}" if m64 > 0 else "n/a"
        print(f"{label:<{col_w}} {'torch':<14} {'float64':<8} {t64:>10.3f} {mem64:>15}")

        print()

    print(sep)

    # GPU device info from nvidia-smi
    gpus = gpu_info()
    if gpus:
        print("\nnvidia-smi snapshot (taken after benchmarks):")
        for i, g in enumerate(gpus):
            print(
                f"  GPU {i}: {g['name']}  "
                f"mem {g['mem_total_mib'] - g['mem_free_mib']}/{g['mem_total_mib']} MiB used  "
                f"util {g['util_pct']}%"
            )
    else:
        print("\n(nvidia-smi not available or no GPUs detected)")


# ---------------------------------------------------------------------------
# Trace analysis & plotting
# ---------------------------------------------------------------------------

def plot_trace(trace_path="pochoir_fdm_trace.json", out_dir=None, top_n=10):
    """Parse a PyTorch profiler trace (pochoir_fdm_trace.json) and save plots.

    Produces four figures:
      1. Top-N GPU kernels by total execution time (horizontal bar chart).
      2. Per-iteration latency over time (line plot of each CompiledFxGraph call).
      3. Distribution of durations for the top-N kernels (box plot).
      4. Compile-phase overhead vs. steady-state run time (stacked bar chart).

    Parameters
    ----------
    trace_path : str
        Path to the JSON trace file.
    out_dir : str or None
        Directory where PNG files are saved.  Defaults to the directory that
        contains the trace file.
    top_n : int
        Number of kernels to highlight in plots 1 and 3.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        raise ImportError("matplotlib is required for plot_trace(); install it with 'pip install matplotlib'.")

    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(trace_path))

    print(f"Loading trace: {trace_path} …", flush=True)
    with open(trace_path) as f:
        data = json.load(f)

    events = data.get("traceEvents", [])
    print(f"  {len(events):,} events loaded.", flush=True)

    # ------------------------------------------------------------------
    # Separate event categories
    # ------------------------------------------------------------------
    kernel_events   = [e for e in events if e.get("cat") == "kernel"]
    cpu_fx_calls    = sorted(
        [e for e in events
         if e.get("cat") == "user_annotation" and "CompiledFxGraph" in e.get("name", "")],
        key=lambda x: x["ts"],
    )
    compile_events  = [e for e in events
                       if e.get("cat") == "user_annotation" and "dynamo_timed" in e.get("name", "")]

    # ------------------------------------------------------------------
    # Figure 1 – Top-N kernels by total GPU time
    # ------------------------------------------------------------------
    kernel_totals: dict[str, float] = {}
    for k in kernel_events:
        kernel_totals[k["name"]] = kernel_totals.get(k["name"], 0.0) + k["dur"]

    sorted_kernels = sorted(kernel_totals.items(), key=lambda x: -x[1])
    top_kernels    = sorted_kernels[:top_n]

    def _short(name, maxlen=55):
        return name if len(name) <= maxlen else "…" + name[-(maxlen - 1):]

    labels = [_short(n) for n, _ in top_kernels]
    totals_ms = [t / 1_000 for _, t in top_kernels]  # µs → ms

    fig1, ax1 = plt.subplots(figsize=(10, 0.55 * top_n + 1.5))
    bars = ax1.barh(range(top_n), totals_ms, color="steelblue", edgecolor="white")
    ax1.set_yticks(range(top_n))
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlabel("Total GPU time (ms)")
    ax1.set_title(f"Top-{top_n} GPU kernels by total execution time")
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    for bar, val in zip(bars, totals_ms):
        ax1.text(bar.get_width() * 1.005, bar.get_y() + bar.get_height() / 2,
                 f"{val:,.1f} ms", va="center", fontsize=7)
    fig1.tight_layout()
    out1 = os.path.join(out_dir, "trace_top_kernels.png")
    fig1.savefig(out1, dpi=150)
    plt.close(fig1)
    print(f"  Saved: {out1}")

    # ------------------------------------------------------------------
    # Figure 2 – Per-iteration latency over time
    # ------------------------------------------------------------------
    if cpu_fx_calls:
        iter_dur_ms = numpy.array([e["dur"] / 1_000 for e in cpu_fx_calls])
        iter_idx    = numpy.arange(len(iter_dur_ms))

        # Smooth with a simple rolling window for readability
        window = max(1, len(iter_dur_ms) // 200)
        if window > 1:
            kernel_w     = numpy.ones(window) / window
            iter_smoothed = numpy.convolve(iter_dur_ms, kernel_w, mode="valid")
            idx_smoothed  = iter_idx[window // 2: window // 2 + len(iter_smoothed)]
        else:
            iter_smoothed, idx_smoothed = iter_dur_ms, iter_idx

        fig2, ax2 = plt.subplots(figsize=(11, 4))
        ax2.plot(iter_idx, iter_dur_ms, color="lightsteelblue", linewidth=0.4,
                 alpha=0.7, label="raw")
        ax2.plot(idx_smoothed, iter_smoothed, color="steelblue", linewidth=1.5,
                 label=f"rolling mean (w={window})")
        ax2.set_xlabel("Iteration index")
        ax2.set_ylabel("Duration (ms)")
        ax2.set_title("Per-iteration latency (CompiledFxGraph calls)")
        ax2.legend(fontsize=8)
        ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.2f}"))
        fig2.tight_layout()
        out2 = os.path.join(out_dir, "trace_iteration_latency.png")
        fig2.savefig(out2, dpi=150)
        plt.close(fig2)
        print(f"  Saved: {out2}")

    # ------------------------------------------------------------------
    # Figure 3 – Duration distribution of top-N kernels (box plot)
    # ------------------------------------------------------------------
    # Group durations per kernel name for the top-N kernels
    top_names = {n for n, _ in top_kernels}
    kernel_groups: dict[str, list[float]] = {n: [] for n in top_names}
    for k in kernel_events:
        if k["name"] in kernel_groups:
            kernel_groups[k["name"]].append(k["dur"] / 1_000)  # µs → ms

    # Order matches Figure 1 (descending total time)
    ordered_names = [n for n, _ in top_kernels]
    box_data  = [kernel_groups[n] for n in ordered_names]
    box_labels = [_short(n, 45) for n in ordered_names]

    fig3, ax3 = plt.subplots(figsize=(10, 0.55 * top_n + 1.5))
    ax3.boxplot(box_data, vert=False, patch_artist=True,
                boxprops=dict(facecolor="steelblue", alpha=0.6),
                medianprops=dict(color="navy", linewidth=1.5),
                flierprops=dict(marker=".", markersize=2, alpha=0.3))
    ax3.set_yticks(range(1, top_n + 1))
    ax3.set_yticklabels(box_labels, fontsize=8)
    ax3.set_xlabel("Kernel duration (ms)")
    ax3.set_title(f"Duration distribution for top-{top_n} kernels")
    fig3.tight_layout()
    out3 = os.path.join(out_dir, "trace_kernel_distributions.png")
    fig3.savefig(out3, dpi=150)
    plt.close(fig3)
    print(f"  Saved: {out3}")

    # ------------------------------------------------------------------
    # Figure 4 – Compile-phase overhead vs. steady-state runtime
    # ------------------------------------------------------------------
    # Compile overhead: sum of top-level dynamo_timed annotations
    # (use _compile.compile_inner as the canonical compile cost)
    compile_total_ms = sum(
        e["dur"] / 1_000
        for e in compile_events
        if "_compile.compile_inner" in e["name"]
    )
    # Steady-state runtime: total duration of all CompiledFxGraph CPU calls
    # minus the very first call which may include warm-up
    steady_total_ms = sum(e["dur"] / 1_000 for e in cpu_fx_calls[1:])

    categories   = ["Compile (dynamo)", "Steady-state iterations"]
    values_ms    = [compile_total_ms, steady_total_ms]
    colors       = ["#e07b54", "steelblue"]

    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(10, 4))

    # Left: absolute bar
    ax4a.bar(categories, values_ms, color=colors, edgecolor="white", width=0.5)
    ax4a.set_ylabel("Time (ms)")
    ax4a.set_title("Compile overhead vs. steady-state runtime")
    ax4a.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    for i, v in enumerate(values_ms):
        ax4a.text(i, v * 1.01, f"{v:,.1f} ms", ha="center", fontsize=9)

    # Right: pie
    total = sum(values_ms)
    ax4b.pie(values_ms, labels=categories, colors=colors, autopct="%1.1f%%",
             startangle=90, wedgeprops=dict(edgecolor="white"))
    ax4b.set_title(f"Time breakdown  (total {total:,.1f} ms)")

    fig4.tight_layout()
    out4 = os.path.join(out_dir, "trace_compile_vs_runtime.png")
    fig4.savefig(out4, dpi=150)
    plt.close(fig4)
    print(f"  Saved: {out4}")

    print("Done.")
    return dict(out_top_kernels=out1, out_iter_latency=out2,
                out_distributions=out3, out_compile_vs_runtime=out4)


def main():
    parser = argparse.ArgumentParser(description="Benchmark pochoir FDM solvers")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of epochs (default: 20)")
    parser.add_argument("--steps", type=int, default=500,
                        help="Steps per epoch (default: 500)")
    parser.add_argument("--prec", type=float, default=1e-6,
                        help="Convergence precision (default: 1e-6)")
    parser.add_argument("--plot-trace", metavar="JSON",
                        help="Parse a pochoir_fdm_trace.json and save plots, then exit.")
    parser.add_argument("--plot-out", metavar="DIR", default=None,
                        help="Output directory for trace plots (default: same as JSON).")
    parser.add_argument("--top-n", type=int, default=10,
                        help="Number of top kernels to show in trace plots (default: 10).")
    args = parser.parse_args()

    if args.plot_trace:
        plot_trace(args.plot_trace, out_dir=args.plot_out, top_n=args.top_n)
        return

    print(f"pochoir FDM benchmark  epochs={args.epochs}  steps={args.steps}  prec={args.prec}\n")
    run(args.epochs, args.steps, args.prec)


if __name__ == "__main__":
    main()
