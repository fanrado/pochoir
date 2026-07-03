#!/usr/bin/env python3
"""
Compare drift paths computed with LINEAR vs CUBIC interpolation of the scalar
drift potential, on the *same* active store.

Both path sets are produced by re-running ``pochoir drift`` on the same starts
and the same stored potential, changing only ``--interp-order``.  This isolates
the effect of the interpolation order on the trajectories -- in particular near
the pixel plane, where the geometry is strong and where the field reconstruction
differs most between the two orders.

The question this addresses: near the pixel plane the two orders give very
different fields; does that translate into a large, physically meaningful change
in the paths, or into a spline-overshoot artifact in the landing pattern?

Output: a multi-page PDF, one figure per page, each with a description footer.

Usage:
    python paths_linear_vs_cubic_summary.py \
        --store store \
        --linear paths/drift3d_tight_linear \
        --cubic  paths/drift3d_tight_cubic \
        --out    paths_linear_vs_cubic_summary.pdf
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

from pochoir import units

MM = units.mm


# --------------------------------------------------------------------------- #
def load(store, key):
    base = key.split("/")[-1]
    return np.load(os.path.join(store, key + ".npz"))[base]


def load_domain(store, key="drift3d"):
    from pochoir.domain import Domain
    with open(os.path.join(store, "domain", key + ".json")) as fp:
        dj = json.load(fp)
    return Domain(dj["shape"], dj["spacing"], dj.get("origin"))


def load_potential_plane(store, key="drift3d"):
    """Return (phi array, z-index of pixel plane, domain)."""
    dom = load_domain(store, key)
    phi = np.load(os.path.join(store, "potential", key + ".npz"))[key]
    zi = int(phi.std(axis=(0, 1)).argmax())
    return phi, zi, dom


def add_footer(fig, text, width=112, rect=(0, 0.16, 1, 0.94)):
    fig.tight_layout(rect=rect)
    wrapped = "\n".join(textwrap.wrap(" ".join(text.split()), width=width))
    fig.text(0.06, 0.015, wrapped, ha="left", va="bottom", fontsize=8.5,
             family="monospace",
             bbox=dict(boxstyle="round", facecolor="#f5f5f5", edgecolor="#bbbbbb"))


def page_text(pdf, title, body):
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.08, 0.92, title, fontsize=15, weight="bold", va="top")
    fig.text(0.08, 0.85, "\n".join(textwrap.wrap(" ".join(body.split()), width=95)),
             fontsize=11, va="top", family="monospace")
    pdf.savefig(fig)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def endpoints(paths):
    """Final (x,y,z) of every path, in mm."""
    return paths[:, -1, :] / MM


def page_landing(pdf, lin, cub, phi, zi, dom):
    """Where paths land on the pixel plane: linear vs cubic + displacement."""
    eL = endpoints(lin)
    eC = endpoints(cub)
    d = np.linalg.norm(eC - eL, axis=1)          # 3D endpoint displacement, mm
    dxy = np.linalg.norm(eC[:, :2] - eL[:, :2], axis=1)

    x_axis, y_axis, _ = dom.linspaces
    fig, axes = plt.subplots(1, 2, figsize=(12, 7))
    fig.suptitle("Where drift paths land on the pixel plane: linear vs cubic",
                 fontsize=13)

    ax = axes[0]
    # backdrop: potential at the plane (the pad footprint)
    im = ax.pcolormesh(x_axis / MM, y_axis / MM, phi[:, :, zi].T,
                       shading="auto", cmap="Greys_r", alpha=0.8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$\phi$ at plane [V]")
    ax.scatter(eL[:, 0], eL[:, 1], s=14, c="C0", label="linear", zorder=3)
    ax.scatter(eC[:, 0], eC[:, 1], s=14, c="C1", marker="x", label="cubic", zorder=3)
    for a, b in zip(eL, eC):
        ax.annotate("", xy=(b[0], b[1]), xytext=(a[0], a[1]),
                    arrowprops=dict(arrowstyle="->", color="C3", lw=0.6, alpha=0.7))
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    ax.set_title("landing (x,y); arrows linear->cubic"); ax.legend(loc="upper right")
    ax.set_aspect("equal")

    ax = axes[1]
    ax.hist(dxy * 1000.0, bins=30, color="C4", alpha=0.85)
    ax.axvline(dom.spacing[0] / MM * 1000.0, color="k", ls="--",
               label=f"1 cell = {dom.spacing[0]/MM*1000:.0f} um")
    ax.set_xlabel(r"transverse landing shift |cubic - linear|  [$\mu$m]")
    ax.set_ylabel("number of paths")
    ax.set_title("distribution of landing shift"); ax.legend()

    add_footer(fig,
               f"Each of the {len(lin)} start points is drifted to the pixel plane "
               "twice. Points show the landing (x,y) for linear (dots) and cubic "
               "(crosses); red arrows connect the same start. The histogram is the "
               "transverse landing shift. median shift = "
               f"{np.median(dxy)*1000:.1f} um, 90th pct = "
               f"{np.percentile(dxy,90)*1000:.1f} um, max = {dxy.max()*1000:.1f} um "
               f"(grid cell = {dom.spacing[0]/MM*1000:.0f} um). Shifts that are a "
               "large fraction of a cell -- or larger -- near the pad edges are the "
               "signature of the interpolation order actually steering the paths "
               "there. A cubic-overshoot artifact would show up as landing shifts "
               "clustered at pad EDGES pointing in a consistent (spurious) direction.")
    pdf.savefig(fig)
    plt.close(fig)
    return d, dxy, eL, eC


def page_overlays(pdf, lin, cub, idxs, phi, zi, dom, labeltxt):
    """Overlay representative path pairs in x-z and y-z projections."""
    z_axis = dom.linspaces[2]
    z_plane = z_axis[zi] / MM
    fig, axes = plt.subplots(1, 2, figsize=(12, 7.5))
    fig.suptitle("Representative path pairs (linear solid, cubic dashed)", fontsize=13)
    cmap = plt.get_cmap("tab10")
    for j, i in enumerate(idxs):
        c = cmap(j % 10)
        L = lin[i] / MM
        C = cub[i] / MM
        axes[0].plot(L[:, 0], L[:, 2], color=c, lw=1.3)
        axes[0].plot(C[:, 0], C[:, 2], color=c, lw=1.3, ls="--")
        axes[1].plot(L[:, 1], L[:, 2], color=c, lw=1.3,
                     label=f"start ({L[0,0]:.2f},{L[0,1]:.2f})")
        axes[1].plot(C[:, 1], C[:, 2], color=c, lw=1.3, ls="--")
    for ax, lab in ((axes[0], "x [mm]"), (axes[1], "y [mm]")):
        ax.axhline(z_plane, color="k", ls=":", lw=0.8)
        ax.set_xlabel(lab); ax.set_ylabel("z [mm]")
        ax.set_ylim(z_plane - 0.5, z_plane + 4.0)     # zoom near the plane
    axes[0].set_title("x-z projection (zoom near plane)")
    axes[1].set_title("y-z projection (zoom near plane)")
    axes[1].legend(fontsize=7, loc="upper right")
    add_footer(fig, labeltxt)
    pdf.savefig(fig)
    plt.close(fig)


def page_divergence(pdf, lin, cub, dom, zi):
    """How the linear/cubic separation grows as paths approach the plane."""
    z_plane = dom.linspaces[2][zi] / MM
    # separation between the two paths at each tick (they share t_eval), mm
    sep = np.linalg.norm(lin - cub, axis=2) / MM              # (npaths, nticks)
    # use the linear path's z as the common vertical coordinate
    zL = lin[:, :, 2] / MM

    fig, axes = plt.subplots(1, 2, figsize=(12, 7))
    fig.suptitle("Where the two solutions separate", fontsize=13)

    ax = axes[0]
    for i in range(0, lin.shape[0], max(1, lin.shape[0] // 40)):
        ax.plot(zL[i], sep[i] * 1000.0, color="C0", lw=0.5, alpha=0.4)
    ax.axvline(z_plane, color="k", ls="--", lw=1.0, label=f"plane z={z_plane:.1f}")
    ax.set_xlabel("z [mm]"); ax.set_ylabel(r"|linear - cubic|  [$\mu$m]")
    ax.set_title("path separation vs z (sample of paths)")
    ax.set_xlim(z_plane - 0.5, zL.max())
    ax.legend()

    ax = axes[1]
    finalsep = sep[:, -1] * 1000.0
    ax.hist(finalsep, bins=30, color="C2", alpha=0.85)
    ax.set_xlabel(r"final 3D separation |linear - cubic|  [$\mu$m]")
    ax.set_ylabel("number of paths")
    ax.set_title("final separation distribution")

    add_footer(fig,
               "Left: 3D distance between the linear and cubic solution of the same "
               "start, plotted against height z (a sample of paths). The two agree "
               "far from the plane (right side) and separate as they descend into the "
               "strong-geometry region near the pad -- confirming the divergence is "
               "driven by the near-plane field reconstruction, not by accumulated ODE "
               "error in the bulk. Right: distribution of the final separation. If the "
               "separation grew smoothly and monotonically toward the plane the change "
               "is a genuine field effect; a sudden jump or oscillation right at the "
               "plane would instead point to cubic overshoot.")
    pdf.savefig(fig)
    plt.close(fig)
    return sep, zL


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default="store")
    ap.add_argument("--linear", default="paths/drift3d_tight_linear")
    ap.add_argument("--cubic", default="paths/drift3d_tight_cubic")
    ap.add_argument("--key", default="drift3d", help="potential/domain key")
    ap.add_argument("--out", default="paths_linear_vs_cubic_summary.pdf")
    args = ap.parse_args()

    lin = load(args.store, args.linear)
    cub = load(args.store, args.cubic)
    assert lin.shape == cub.shape, f"shape mismatch {lin.shape} vs {cub.shape}"
    phi, zi, dom = load_potential_plane(args.store, args.key)
    z_plane = dom.linspaces[2][zi] / MM

    eL = endpoints(lin); eC = endpoints(cub)
    dxy = np.linalg.norm(eC[:, :2] - eL[:, :2], axis=1)

    # choose representative paths: the largest-shift ones (near the pad edges)
    order = np.argsort(dxy)[::-1]
    big = order[:6]
    small = order[-3:]
    reps = list(big) + list(small)

    with PdfPages(args.out) as pdf:
        page_text(pdf,
                  "Drift paths: linear vs. cubic interpolation of the drift potential",
                  "Both path sets were produced by re-running `pochoir drift` on the "
                  "SAME active store, same starts and same stored scalar potential, "
                  "changing only --interp-order (linear vs cubic). Every difference "
                  "below is therefore purely the effect of the interpolation order on "
                  "the trajectory. The focus is the pixel-plane region, where the "
                  "geometry is strong and where the reconstructed field differs most "
                  f"between the two orders. Pixel plane at z = {z_plane:.2f} mm; "
                  f"{lin.shape[0]} paths, {lin.shape[1]} time steps each.")

        page_landing(pdf, lin, cub, phi, zi, dom)
        page_divergence(pdf, lin, cub, dom, zi)
        page_overlays(pdf, lin, cub, reps, phi, zi, dom,
                      "Six largest-shift path pairs (near pad edges) plus three "
                      "smallest-shift (near-uniform region). Linear solid, cubic "
                      "dashed, same color = same start. In the bulk (top) the pairs "
                      "overlie; near the plane (dotted line) the large-shift pairs "
                      "peel apart, bending toward/away from the pad. Whether cubic "
                      "bends them toward a smoother, more physical landing or into a "
                      "spurious overshoot is the crux of the linear-vs-cubic choice.")

        page_text(pdf, "Summary (numbers)",
                  f"Transverse landing shift |cubic-linear| over {lin.shape[0]} paths: "
                  f"median {np.median(dxy)*1000:.1f} um, mean {dxy.mean()*1000:.1f} um, "
                  f"90th pct {np.percentile(dxy,90)*1000:.1f} um, "
                  f"max {dxy.max()*1000:.1f} um. Grid cell = "
                  f"{dom.spacing[0]/MM*1000:.0f} um. "
                  "Interpretation: shifts small compared to a cell mean the "
                  "interpolation order barely moves the landing (paths robust); shifts "
                  "comparable to or larger than a cell, concentrated at the pad edges, "
                  "mean the order materially steers the paths there -- which is exactly "
                  "the region where we must decide whether cubic is reconstructing the "
                  "true field or overshooting. The companion report "
                  "interp_linear_vs_cubic_summary.pdf shows the field-level view of "
                  "the same comparison.")

    print(f"wrote {args.out}")
    print(f"landing shift (um): median {np.median(dxy)*1000:.2f} "
          f"mean {dxy.mean()*1000:.2f} max {dxy.max()*1000:.2f} "
          f"cell {dom.spacing[0]/MM*1000:.0f}")


if __name__ == "__main__":
    main()
