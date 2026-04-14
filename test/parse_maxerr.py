#!/usr/bin/env python3
"""Parse maxerr values for torch.float64 entries from pochoir.log."""

import re
import sys

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

LOG_FILE = "pochoir.log"


def parse_maxerr(log_file, dtype):
    """Parse iteration, maxerr, and epoch for a given dtype.

    Returns a dict mapping "{dtype} (run N)" -> list of (iteration, maxerr, epoch) tuples.
    Each run is delimited by a 'bi_core shape' log line, so two runs that share the same
    dtype are stored under separate keys.
    """
    shape_pattern = re.compile(r"bi_core shape = ")
    epoch_pattern = re.compile(r"={3,}\s*epoch:\s*(\d+)/")
    iter_pattern = re.compile(
        rf"iteration\s*:\s*(\d+),\s*maxerr\s*=\s*([\d.eE+\-]+),.*dtype\s*=\s*{re.escape(dtype)}"
    )
    current_epoch = None
    section = 0
    results = {}
    with open(log_file) as f:
        for line in f:
            if shape_pattern.search(line):
                section += 1
                continue
            em = epoch_pattern.search(line)
            if em:
                current_epoch = int(em.group(1))
                continue
            m = iter_pattern.search(line)
            if m and section > 0:
                key = f"{dtype} (run {section})"
                results.setdefault(key, []).append(
                    (int(m.group(1)), float(m.group(2)), current_epoch)
                )
    return results


def parse_maxerr_float32(log_file):
    return parse_maxerr(log_file, "torch.float32")

def parse_maxerr_float16(log_file):
    return parse_maxerr(log_file, "torch.float16")

def parse_maxerr_float64(log_file):
    return parse_maxerr(log_file, "torch.float64")


def plot_maxerr(series, output_file=None):
    """Plot maxerr vs iteration for one or more dtype series.

    Each dtype gets its own subplot; epochs are shown as separate lines with legends.

    Args:
        series: dict mapping label (str) to list of (iteration, maxerr, epoch) tuples.
        output_file: path to save the figure, or None to display interactively.
    """
    n = len(series)
    fig, axes = plt.subplots(n, 1, figsize=(10, 5 * n), squeeze=False)
    # set font size for all axes to 15
    plt.rcParams.update({'font.size': 15})
    for ax, (label, entries) in zip(axes[:, 0], series.items()):
        # Group entries by epoch
        epochs = {}
        for iteration, maxerr, epoch in entries:
            epochs.setdefault(epoch, ([], []))[0].append(iteration)
            epochs.setdefault(epoch, ([], []))[1].append(maxerr)
        for epoch, (iterations, maxerr_values) in sorted(epochs.items()):
            ax.plot(iterations, maxerr_values, linewidth=1, label=f"epoch {epoch}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Max Error")
        ax.set_title(f"Max Error vs Iteration ({label})")
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        # ax.set_ylim(bottom=1e-10)  # adjust as needed for visibility
        ax.legend()
    fig.tight_layout()

    if output_file:
        fig.savefig(output_file, dpi=150)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


def parse_runtime(log_file):
    """Parse per-epoch runtimes grouped by dtype and run.

    Returns a dict mapping "{dtype} (run N)" -> list of (epoch, runtime_seconds).
    Each run is delimited by a 'bi_core shape' log line, so two runs that share the same
    dtype are stored under separate keys.
    """
    shape_pattern = re.compile(r"bi_core shape = ")
    dtype_pattern = re.compile(r"bi_core dtype = (torch\.float\d+)")
    epoch_time_pattern = re.compile(r"^.*epoch (\d+) time:\s*([\d.]+)\s*seconds")
    current_dtype = None
    section = 0
    results = {}
    seen_epochs = {}  # key -> set of epoch numbers already recorded
    with open(log_file) as f:
        for line in f:
            if shape_pattern.search(line):
                section += 1
                current_dtype = None  # reset until dtype line is seen for this run
                continue
            dm = dtype_pattern.search(line)
            if dm:
                current_dtype = dm.group(1)
                continue
            tm = epoch_time_pattern.match(line)
            if tm and current_dtype and section > 0:
                epoch = int(tm.group(1))
                runtime = float(tm.group(2))
                key = f"{current_dtype} (run {section})"
                if epoch not in seen_epochs.setdefault(key, set()):
                    seen_epochs[key].add(epoch)
                    results.setdefault(key, []).append((epoch, runtime))
    return results


def save_pdf_summary(series, runtime_data, path_to_output):
    """Save a PDF with maxerr and runtime pages grouped by dtype.

    For each dtype:
      - One page: maxerr vs iteration (epochs as separate lines).
      - One page: runtime vs epoch plot (title shows total runtime),
        or a text page if only one epoch exists.

    Args:
        series: dict mapping dtype -> list of (iteration, maxerr, epoch).
        runtime_data: dict returned by parse_runtime().
        path_to_output: path to the output PDF file.
    """
    plt.rcParams.update({"font.size": 14})
    all_dtypes = list(dict.fromkeys(list(series.keys()) + list(runtime_data.keys())))
    with PdfPages(path_to_output) as pdf:
        for dtype in all_dtypes:
            # --- maxerr page ---
            if dtype in series:
                entries = series[dtype]
                fig, ax = plt.subplots(figsize=(10, 5))
                epoch_groups = {}
                for iteration, maxerr, epoch in entries:
                    epoch_groups.setdefault(epoch, ([], []))[0].append(iteration)
                    epoch_groups.setdefault(epoch, ([], []))[1].append(maxerr)
                for epoch, (iterations, maxerr_values) in sorted(epoch_groups.items()):
                    ax.plot(iterations, maxerr_values, linewidth=1, label=f"epoch {epoch}")
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Max Error")
                ax.set_title(f"Max Error vs Iteration ({dtype})")
                ax.set_yscale("log")
                ax.grid(True, which="both", linestyle="--", alpha=0.5)
                ax.legend()
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

            # --- runtime page ---
            if dtype in runtime_data:
                epoch_runtimes = runtime_data[dtype]
                epochs = [e for e, _ in epoch_runtimes]
                runtimes = [r for _, r in epoch_runtimes]
                total_runtime = sum(runtimes)
                fig, ax = plt.subplots(figsize=(8, 5))
                if len(epoch_runtimes) > 1:
                    ax.plot(epochs, runtimes, marker="o", linewidth=2)
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("Runtime (seconds)")
                    ax.grid(True, linestyle="--", alpha=0.5)
                    ax.set_title(
                        f"Runtime vs Epoch ({dtype})\n"
                        f"Total runtime: {total_runtime:.2f} seconds"
                    )
                else:
                    ax.axis("off")
                    ax.set_title(f"Runtime Summary ({dtype})")
                    text = (
                        f"Epoch {epochs[0]}\n"
                        f"Runtime: {runtimes[0]:.2f} seconds"
                    )
                    ax.text(
                        0.5, 0.5, text,
                        transform=ax.transAxes,
                        fontsize=18, ha="center", va="center",
                        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
                    )
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
    print(f"Summary PDF saved to {path_to_output}")


def print_runtime_summary(runtime_data):
    """Print per-epoch runtimes to stdout."""
    for dtype, epoch_runtimes in runtime_data.items():
        total = sum(r for _, r in epoch_runtimes)
        print(f"\n--- Runtime: {dtype} ---")
        print(f"{'epoch':>6}  {'runtime (s)':>14}")
        print("-" * 26)
        for epoch, runtime in epoch_runtimes:
            print(f"{epoch:>6}  {runtime:>14.2f}")
        print("-" * 26)
        print(f"Total runtime : {total:.2f} seconds")


def print_summary(label, entries):
    maxerr_values = [v for _, v, _ in entries]
    print(f"\n--- {label} ---")
    print(f"{'epoch':>6}  {'iteration':>12}  {'maxerr':>20}")
    print("-" * 44)
    for iteration, maxerr, epoch in entries:
        print(f"{str(epoch):>6}  {iteration:>12}  {maxerr:>20.10g}")
    print("-" * 44)
    print(f"Total entries : {len(entries)}")
    print(f"First maxerr  : {maxerr_values[0]:.10g}")
    print(f"Last maxerr   : {maxerr_values[-1]:.10g}")
    print(f"Min maxerr    : {min(maxerr_values):.10g}")
    print(f"Max maxerr    : {max(maxerr_values):.10g}")


if __name__ == "__main__":
    ## How to run:
    """
      python parse_maxerr.py pochoir.log [maxerr_plot.png] [runtime_summary.pdf]
    """
    log_file = sys.argv[1] if len(sys.argv) > 1 else LOG_FILE
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    pdf_output = sys.argv[3] if len(sys.argv) > 3 else None

    sections_f32 = parse_maxerr_float32(log_file)
    sections_f64 = parse_maxerr_float64(log_file)
    sections_f16 = parse_maxerr_float16(log_file)

    if not sections_f32 and not sections_f64 and not sections_f16:
        print("No maxerr entries found.")
        sys.exit(0)

    series = {}
    for label, entries in {**sections_f32, **sections_f64, **sections_f16}.items():
        print_summary(label, entries)
        series[label] = entries
    plot_maxerr(series, output_file)

    # Parse and report runtimes
    runtime_data = parse_runtime(log_file)
    if runtime_data:
        print_runtime_summary(runtime_data)
    else:
        print("No runtime entries found.")

    if pdf_output:
        save_pdf_summary(series, runtime_data, pdf_output)
