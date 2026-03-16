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


def main():
    parser = argparse.ArgumentParser(description="Benchmark pochoir FDM solvers")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of epochs (default: 20)")
    parser.add_argument("--steps", type=int, default=500,
                        help="Steps per epoch (default: 500)")
    parser.add_argument("--prec", type=float, default=1e-6,
                        help="Convergence precision (default: 1e-6)")
    args = parser.parse_args()

    print(f"pochoir FDM benchmark  epochs={args.epochs}  steps={args.steps}  prec={args.prec}\n")
    run(args.epochs, args.steps, args.prec)


if __name__ == "__main__":
    main()
