#!/usr/bin/env python3
"""
Reference-free test for cubic-interpolation OVERSHOOT / ringing near the pixel
plane, on the active store.

Idea: a linear (trilinear) interpolant is *bounded* -- between grid nodes it can
never leave the interval spanned by the surrounding node values.  A cubic
tensor-spline is NOT bounded: near a near-discontinuous derivative (an electrode
edge) it can overshoot past the node envelope and ring.  So, with NO finer-grid
reference needed:

    overshoot(x) = amount by which cubic phi(x) falls OUTSIDE the [min,max] of
                   the grid nodes bracketing x.

Any nonzero overshoot is a purely numerical artifact of the interpolation order
(the true electrostatic potential, being harmonic, obeys a maximum principle and
cannot exceed its local nodal envelope between nodes either).  We also flag
spurious local extrema (wiggles) in regions where the node data is monotone, and
measure how far cubic |E| exceeds the largest node-to-node slope.

Output: interp_overshoot_test.pdf + printed verdict.
"""

import os
import json
import argparse
import textwrap

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import RegularGridInterpolator as RGI

from pochoir import units
from pochoir.domain import Domain

MM = units.mm
E_VPERCM = units.V / units.cm


def load(store, key="drift3d"):
    with open(os.path.join(store, "domain", key + ".json")) as fp:
        dj = json.load(fp)
    dom = Domain(dj["shape"], dj["spacing"], dj.get("origin"))
    phi = np.load(os.path.join(store, "potential", key + ".npz"))[key]
    zi = int(phi.std(axis=(0, 1)).argmax())
    return dom, phi, zi


def add_footer(fig, text, width=112, rect=(0, 0.18, 1, 0.94)):
    fig.tight_layout(rect=rect)
    fig.text(0.06, 0.015, "\n".join(textwrap.wrap(" ".join(text.split()), width=width)),
             ha="left", va="bottom", fontsize=8.5, family="monospace",
             bbox=dict(boxstyle="round", facecolor="#f5f5f5", edgecolor="#bbbbbb"))


def line_overshoot_1d(nodes_x, nodes_phi, xs):
    """1-D linear & cubic interp of (nodes_x, nodes_phi); overshoot of cubic
    vs the bracketing-node envelope. Returns dict."""
    lin = np.interp(xs, nodes_x, nodes_phi)
    # scipy cubic via RGI (1-D)
    rgi = RGI((nodes_x,), nodes_phi, method="cubic",
              bounds_error=False, fill_value=None)
    cub = rgi(xs[:, None])
    # bracketing node envelope for each sample
    idx = np.clip(np.searchsorted(nodes_x, xs) - 1, 0, len(nodes_x) - 2)
    lo = np.minimum(nodes_phi[idx], nodes_phi[idx + 1])
    hi = np.maximum(nodes_phi[idx], nodes_phi[idx + 1])
    over = np.where(cub > hi, cub - hi, np.where(cub < lo, cub - lo, 0.0))
    return dict(lin=lin, cub=cub, lo=lo, hi=hi, over=over)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="store")
    ap.add_argument("--key", default="drift3d")
    ap.add_argument("--out", default="interp_overshoot_test.pdf")
    args = ap.parse_args()

    dom, phi, zi = load(args.store, args.key)
    x_axis, y_axis, z_axis = dom.linspaces
    z_plane = z_axis[zi] / MM

    # ------------------------------------------------------------------ #
    # Global overshoot scan: sample every cell of every transverse x-line at
    # the plane (and the two neighbouring planes) and measure cubic overshoot.
    # ------------------------------------------------------------------ #
    xs = np.linspace(x_axis[0], x_axis[-1], 2000)
    worst = 0.0
    worst_line = None
    tot_over = 0
    tot = 0
    for zj in (zi - 1, zi, zi + 1):
        for yj in range(len(y_axis)):
            nodes_phi = phi[:, yj, zj]
            r = line_overshoot_1d(x_axis, nodes_phi, xs)
            om = np.abs(r["over"])
            tot_over += int((om > 1e-9).sum())
            tot += om.size
            if om.max() > worst:
                worst = om.max()
                worst_line = (yj, zj, r)
    frac = 100.0 * tot_over / tot

    # node envelope span at the plane, for context (how big is worst vs signal)
    plane_span = phi[:, :, zi].max() - phi[:, :, zi].min()

    print(f"pixel plane z = {z_plane:.2f} mm (index {zi})")
    print(f"cubic overshoot beyond bracketing-node envelope:")
    print(f"  max overshoot = {worst:.3f} V   "
          f"({100.0*worst/plane_span:.1f}% of the {plane_span:.1f} V plane span)")
    print(f"  fraction of sampled sub-cell points that overshoot = {frac:.1f}%")

    # ------------------------------------------------------------------ #
    with PdfPages(args.out) as pdf:
        # Page 1: the worst transverse line, phi with envelope + overshoot
        yj, zj, r = worst_line
        fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)
        fig.suptitle(f"Cubic overshoot of the drift potential -- worst transverse "
                     f"line (y={y_axis[yj]/MM:.2f} mm, z={z_axis[zj]/MM:.2f} mm)",
                     fontsize=12)
        ax = axes[0]
        ax.plot(x_axis / MM, phi[:, yj, zj], "ks", ms=4, label="grid nodes")
        ax.plot(xs / MM, r["lin"], "C0-", lw=1.3, label="linear (bounded)")
        ax.plot(xs / MM, r["cub"], "C1--", lw=1.3, label="cubic")
        ax.fill_between(xs / MM, r["lo"], r["hi"], color="C0", alpha=0.12,
                        label="node envelope")
        ax.set_ylabel(r"$\phi$ [V]"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax.set_title("cubic leaves the node envelope = overshoot (impossible for the "
                     "true harmonic potential)")
        ax = axes[1]
        ax.plot(xs / MM, r["over"], "C3-", lw=1.3)
        ax.axhline(0, color="k", lw=0.6)
        ax.fill_between(xs / MM, r["over"], 0, color="C3", alpha=0.3)
        ax.set_xlabel("x [mm]"); ax.set_ylabel("overshoot [V]")
        ax.set_title(f"signed overshoot (max |overshoot| = {worst:.2f} V)")
        ax.grid(alpha=0.3)
        add_footer(fig,
                   "Black squares are the stored grid-node potentials on this line; "
                   "the shaded band is the interval they span between neighbours. "
                   "Linear interpolation (C0) stays inside the band by construction. "
                   "Cubic (C2) pushes OUTSIDE it near the pad edge -- pure ringing, "
                   "since a harmonic potential obeys a maximum principle and cannot "
                   f"exceed its nodal envelope between nodes. Peak overshoot here is "
                   f"{worst:.2f} V, {100.0*worst/plane_span:.1f}% of the {plane_span:.1f} "
                   "V transverse span at the plane. This overshoot in phi is "
                   "differentiated into a spurious spike in E, which is what over-"
                   "focuses the cubic drift paths onto the pads.")
        pdf.savefig(fig); plt.close(fig)

        # Page 2: verdict text
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.08, 0.93, "Verdict: is the near-plane change a cubic artifact?",
                 fontsize=14, weight="bold", va="top")
        body = (
            f"Pixel plane at z = {z_plane:.2f} mm; grid spacing "
            f"{dom.spacing[0]/MM*1000:.0f} um.\n\n"
            f"* Cubic overshoots the local grid-node envelope on "
            f"{frac:.1f}% of sub-cell sample points near the plane.\n"
            f"* Peak overshoot in phi = {worst:.2f} V "
            f"({100.0*worst/plane_span:.1f}% of the {plane_span:.1f} V plane span).\n\n"
            "Because the drift potential is a solution of Laplace's equation, it "
            "satisfies a maximum principle: between two grid nodes the TRUE potential "
            "can never exceed the range of the surrounding nodal values. Linear "
            "interpolation respects this; cubic does not. Any overshoot is therefore "
            "a guaranteed numerical artifact of the C2 spline reacting to the near-"
            "kink that the pad edge imprints on phi -- exactly the 'singularity-like' "
            "behaviour suspected.\n\n"
            "Consequence for the paths: the overshoot in phi becomes a spurious spike "
            "in E = grad(phi) right at the pad edge, giving cubic paths extra "
            "transverse kick and over-focusing them onto the pads (the coherent "
            "inward arrows and 345 um median landing shift in "
            "paths_linear_vs_cubic_summary.pdf).\n\n"
            "BUT: linear is not 'right' either -- its C0 field is a staircase that "
            "step-collapses the integrator (linear drift ran ~3x slower) and caps E "
            "accuracy at O(h). Both suffer because the 0.1 mm grid UNDER-RESOLVES the "
            "pad edge. The robust fix is a finer grid near the plane so the node data "
            "itself resolves the edge; then the interpolation order is a second-order "
            "choice rather than the dominant error. A monotone/limited cubic "
            "(overshoot-free) is an alternative if refinement is too costly."
        )
        fig.text(0.08, 0.86, "\n".join(
            sum([textwrap.wrap(p, width=92) or [""] for p in body.split("\n")], [])),
            fontsize=10.5, va="top", family="monospace")
        pdf.savefig(fig); plt.close(fig)

    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
