#!/usr/bin/env python3
"""Compare a pochoir field-response .npy against the tred reference npz.

Implements the convention from the reference notebook
(FieldResponse/pochoir/test/for_pixel/compare_fr_tred.ipynb):

  * tred response units  : eplus / us
  * pochoir raw units    : eplus / ns   ->  multiply by 1000 to get eplus/us
  * spatial layout       : pochoir (Npaths, Nt) reshaped C-order to (45, 45, Nt),
                           compared at identical (i, j) -- NO spatial shift.
  * time                 : both at 0.05 us / tick.

tred (response_44_v2a_full.npz) is an INDEPENDENT calculation (Jiangmei's tred),
so an exact 1e-6 match is not physically expected.  This tool reports the real
agreement: integrated charge, peak amplitude, waveform correlation, and the
per-pixel/global diff after optional peak time-alignment.

Usage:
    compare_fr_tred.py POCHOIR_NPY [REFERENCE_NPZ] [--side 45] [--align peak|none]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

DEFAULT_REF = "/home/rrazakami/work/ND-LAr/data_tred/responses/response_44_v2a_full.npz"
NS_PER_US = 1000.0  # pochoir eplus/ns -> eplus/us


def load_ref(path):
    z = np.load(path)
    return (z["response"], float(z["time_tick"]),
            float(z["drift_length"]), float(z["bin_size"]))


def peak_align(p, r):
    """Return (p_block, r_block, shift) aligned on the global |sum_xy| peak tick."""
    pp = int(np.argmax(np.abs(p).sum(axis=(0, 1))))
    rp = int(np.argmax(np.abs(r).sum(axis=(0, 1))))
    shift = rp - pp
    nt_p, nt_r = p.shape[2], r.shape[2]
    r_lo, r_hi = max(0, shift), min(nt_r, nt_p + shift)
    p_lo, p_hi = r_lo - shift, r_hi - shift
    return p[:, :, p_lo:p_hi], r[:, :, r_lo:r_hi], shift


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("pochoir_npy")
    ap.add_argument("reference_npz", nargs="?", default=DEFAULT_REF)
    ap.add_argument("--side", type=int, default=45)
    ap.add_argument("--align", choices=["peak", "none"], default="peak")
    ap.add_argument("--tick-us", type=float, default=0.05)
    args = ap.parse_args(argv)

    resp, tred_tick, tred_drift, tred_bin = load_ref(args.reference_npz)
    raw = np.load(args.pochoir_npy)

    print(f"# pochoir : {args.pochoir_npy}  raw shape={raw.shape} "
          f"range[{raw.min():.4e},{raw.max():.4e}]")
    print(f"# tred    : {args.reference_npz}  shape={resp.shape} "
          f"tick={tred_tick}us drift={tred_drift} bin={tred_bin}")

    # units + layout
    poch = raw * NS_PER_US                      # eplus/us
    s = args.side
    if poch.ndim == 2:
        if poch.shape[0] != s * s:
            print(f"!! Npaths {poch.shape[0]} != {s}x{s}; cannot reshape")
            return 2
        poch = poch.reshape(s, s, -1)           # (y, x, t), C-order
    print(f"# pochoir grid (eplus/us): {poch.shape}  "
          f"range[{poch.min():.4e},{poch.max():.4e}]")

    out = {"tred_meta": {"tick_us": tred_tick, "drift": tred_drift, "bin": tred_bin}}

    # ---- integrated charge map (eplus) ----
    poch_int = poch.sum(axis=-1) * args.tick_us
    tred_int = resp.sum(axis=-1) * tred_tick
    out["integral_eplus"] = {
        "pochoir_max": float(poch_int.max()),
        "tred_max": float(tred_int.max()),
        "pochoir_center(0,0)": float(poch_int[0, 0]),
        "tred_center(0,0)": float(tred_int[0, 0]),
    }

    # ---- peak amplitude map (eplus/us) ----
    out["peak_amp_eplus_per_us"] = {
        "pochoir_max": float(poch.max()),
        "tred_max": float(resp.max()),
        "ratio_poch_over_tred": float(poch.max() / resp.max()),
    }

    # ---- waveform agreement, peak-aligned ----
    if args.align == "peak":
        pb, rb, shift = peak_align(poch, resp)
    else:
        n = min(poch.shape[2], resp.shape[2])
        pb, rb, shift = poch[:, :, :n], resp[:, :, :n], 0
    diff = pb - rb
    rms = float(np.sqrt(np.mean(diff ** 2)))
    ref_rms = float(np.sqrt(np.mean(rb ** 2)))
    # central-pixel waveform correlation
    cy = cx = 0  # pixel center is (0,0) in this quadrant convention
    a, b = pb[cy, cx], rb[cy, cx]
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    corr = float(np.dot(a, b) / denom)
    out["aligned"] = {
        "shift_ticks": int(shift),
        "overlap_shape": list(diff.shape),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "rms_diff": rms,
        "tred_rms": ref_rms,
        "rel_rms": rms / (ref_rms + 1e-30),
        "center_waveform_cosine_sim": corr,
    }
    out["pass_1e-6"] = out["aligned"]["max_abs_diff"] < 1e-6

    print(json.dumps(out, indent=2))
    rpt = os.path.splitext(args.pochoir_npy)[0] + ".tredcompare.json"
    with open(rpt, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"# wrote {rpt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
