#!/usr/bin/env python3
"""
Summary-plot generator for the central-pixel-border drift/induced-current study.

This is a script conversion of test/Paths_at_centralpixBorder.ipynb. It loads one
pochoir ``store_*`` dataset and writes a multi-page PDF, one analysis figure per
page, each annotated with a short description of what it shows and why it matters.

Default dataset:
    test/store_0.1mmSpacing_0.05usTimeStep_0.1mmPixPlaneWidth

Usage:
    python paths_at_centralpix_summary.py \
        --data store_0.1mmSpacing_0.05usTimeStep_0.1mmPixPlaneWidth \
        --out  paths_at_centralpix_summary.pdf

    # optional second dataset, only used for the delta-phi overlay page:
    python paths_at_centralpix_summary.py --compare store_0.05mmSpacing_0.05usTimeStep_0.05mmPixPlaneWidth
"""

import os
import argparse
import textwrap

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.ticker import FuncFormatter
from matplotlib.backends.backend_pdf import PdfPages
from scipy.integrate import simpson, trapezoid, romb

# One time tick = 0.05 us for this dataset (us per tick); also used as the
# integration step for the induced-current -> charge integral.
DX_US = 0.05

# Geometry constant used by the notebook fr reshape (10 paths/pixel x 5x5 block
# folded into a 25x25 grid of drift-path start points).
NGRID = 25


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_dataset(path):
    """Load every array the summary needs from a store_* directory."""
    def npz(rel, key):
        return np.load(os.path.join(path, rel))[key]

    fr = np.load(os.path.join(path, "fr_4p4pitch_3.8pix_nogrid_10pathsperpixel.npy"))
    ntick = fr.size // (NGRID * NGRID)
    fr = fr.reshape(NGRID, NGRID, ntick) * 1000.0          # -> e-/us

    shifted_paths = np.load(os.path.join(path, "tmp", "shifted_paths.npy"))
    shifted_paths = shifted_paths.reshape(NGRID, NGRID, -1, 3)

    return dict(
        path=path,
        fr=fr,
        shifted_paths=shifted_paths,
        velocity=npz("velocity/drift3d.npz", "drift3d"),     # (3, X, Y, Z)
        drift=npz("potential/drift3d.npz", "drift3d"),       # (X, Y, Z)
        drift_domain=npz("boundary/drift3d.npz", "drift3d"), # (X, Y, Z)
        weight3d=npz("potential/weight3d.npz", "weight3d"),  # (X, Y, Z)
    )


# --------------------------------------------------------------------------- #
# PDF page helper: draw a figure, stamp a wrapped description footer on it.
# --------------------------------------------------------------------------- #
def add_footer(fig, description, width=110):
    """Reserve a bottom band on the figure and write a wrapped description."""
    fig.subplots_adjust(bottom=0.22)
    wrapped = "\n".join(textwrap.wrap(" ".join(description.split()), width=width))
    fig.text(0.06, 0.015, wrapped, ha="left", va="bottom", fontsize=9,
             family="monospace",
             bbox=dict(boxstyle="round", facecolor="#f5f5f5", edgecolor="#bbbbbb"))


def text_page(pdf, title, body):
    """Emit a full-page text page (for tables / numeric results)."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.5, 0.94, title, ha="center", va="top", fontsize=16, weight="bold")
    fig.text(0.06, 0.86, body, ha="left", va="top", fontsize=10, family="monospace")
    pdf.savefig(fig)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Individual plot pages. Each returns nothing; each saves one page to `pdf`.
# --------------------------------------------------------------------------- #
def page_title(pdf, data, compare):
    fig = plt.figure(figsize=(11, 8.5))
    lines = [
        "Central-pixel-border drift & induced-current summary",
        "",
        f"dataset : {os.path.basename(os.path.normpath(data['path']))}",
        f"fr shape (folded)        : {data['fr'].shape}",
        f"shifted_paths shape      : {data['shifted_paths'].shape}",
        f"drift potential shape    : {data['drift'].shape}",
        f"drift velocity shape     : {data['velocity'].shape}",
        f"weighting potential shape: {data['weight3d'].shape}",
        f"time tick                : {DX_US} us",
    ]
    if compare is not None:
        lines.append(f"compare dataset          : "
                     f"{os.path.basename(os.path.normpath(compare['path']))}")
    fig.text(0.5, 0.72, "\n".join(lines[:1]), ha="center", va="center",
             fontsize=18, weight="bold")
    fig.text(0.12, 0.55, "\n".join(lines[2:]), ha="left", va="top",
             fontsize=12, family="monospace")
    fig.text(0.12, 0.12,
             "Generated from test/Paths_at_centralpixBorder.ipynb. Each following "
             "page carries one figure and a short description of what it shows.",
             ha="left", va="bottom", fontsize=10, style="italic", wrap=True)
    pdf.savefig(fig)
    plt.close(fig)


def page_paths_xz(pdf, data):
    sp = data["shifted_paths"]
    fig, ax = plt.subplots(figsize=(11, 7))
    for i in range(min(5, sp.shape[0])):
        ax.scatter(sp[i, 4, :, 0], sp[i, 4, :, 2], s=10, color="black")
    # pixel pad footprint (x extent x z surface slab), in mm
    for (x_left, x_right) in [(11, 12.7)]:
        ax.add_patch(patches.Rectangle((x_left, 10.0), x_right - x_left, 0.1,
                     linewidth=2, edgecolor="red", facecolor="white", alpha=0.4))
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z (mm)")
    ax.set_ylim(9.5, 12)
    ax.set_title("Drift paths approaching the central-pixel border")
    add_footer(fig,
        "Electron drift paths (x vs z) for the first few start points near the "
        "central pixel. The red slab marks the pixel-pad footprint at the anode "
        "(z = 10.0-10.1 mm). Shows how trajectories converge onto the pad edge "
        "as charge is collected.")
    pdf.savefig(fig)
    plt.close(fig)


def page_velocity(pdf, data):
    v = data["velocity"]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(v[2, 38, 38, :] * 1000.0)
    ax.set_xlabel("z (0.1 mm / tick)")
    ax.set_ylabel(r"$v_z$ (mm/$\mu$s)")
    ax.set_title("Drift velocity at the corner of the pixel")
    add_footer(fig,
        "z-component of the drift velocity along depth at the pixel-corner column "
        "(index 38,38). The rise near the anode reflects the field intensifying "
        "as the charge nears the collecting pad.")
    pdf.savefig(fig)
    plt.close(fig)


def page_drift_diff(pdf, data, compare):
    drift = data["drift"]
    drift_diff = drift[4, 4, 1:] - drift[4, 4, :-1]
    fig, ax = plt.subplots(figsize=(11, 6))
    if compare is not None:
        c = compare["drift"]
        cdiff = c[4, 4, 1:] - c[4, 4, :-1]
        ax.plot(cdiff, label="compare dataset")
    ax.plot(drift_diff, label="this dataset")
    ax.set_xlabel("z (0.1 mm / tick)")
    ax.set_ylabel(r"$\delta\phi$ (V)")
    ax.set_xlim(150, 250)
    ax.legend()
    ax.set_title("Per-step drift-potential difference along z")
    add_footer(fig,
        "Finite difference of the drift potential along z at column (4,4). The "
        "spike near the anode is the steep potential drop at the pad; comparing "
        "spacings/datasets here checks grid-resolution effects on the field.")
    pdf.savefig(fig)
    plt.close(fig)


def page_charge_maps(pdf, data):
    fr = data["fr"]
    fr_integral = simpson(fr, dx=DX_US, axis=2)
    fr_clip = fr_integral.copy()
    fr_clip[:5, :5] = np.min(fr_integral[10:, 10:])

    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (img, ttl) in zip(axs, [
            (fr_integral, "full charge map"),
            (fr_integral[:5, :5], "central 5x5 block"),
            (fr_clip, "central block clipped to bulk min")]):
        im = ax.imshow(img, cmap="viridis", origin="lower")
        ax.set_xlabel("x index")
        ax.set_ylabel("y index")
        ax.set_title(ttl)
        fig.colorbar(im, ax=ax, label="Charge (e-)")
    fig.suptitle("Collected charge per drift-start point (Simpson integral of fr)")
    add_footer(fig,
        "Time-integrated induced current = collected charge for each start point. "
        "Left: full 25x25 map. Middle: central pixel's 5x5 block (the collecting "
        "pixel). Right: full map with the central block clipped to the bulk "
        "minimum to expose neighbour-pixel structure.")
    pdf.savefig(fig)
    plt.close(fig)


def page_integration_methods(pdf, data):
    """Stats table + telescoping identity + downsampling, all as a text page."""
    fr = data["fr"]
    maps = {
        "left_riemann": fr.sum(axis=2) * DX_US,        # == Q_end - Q_start
        "trapezoid":    trapezoid(fr, dx=DX_US, axis=2),
        "simpson":      simpson(fr, dx=DX_US, axis=2),
    }
    n = fr.shape[2]
    k = int(np.floor(np.log2(n - 1)))
    m = 2 ** k + 1
    maps["romb_trunc"] = romb(fr[:, :, :m], dx=DX_US, axis=2)

    lines = []
    lines.append(f"romb uses first {m} of {n} samples (tail ~0).")
    lines.append("")
    mask = np.abs(maps["simpson"]) > 0.5
    lines.append(f"central-pixel block: {mask.sum()} cells")
    lines.append("")
    lines.append(f"{'method':14s}{'mean':>12s}{'std':>12s}{'min':>12s}{'max':>12s}")
    for name, mp in maps.items():
        v = mp[mask]
        lines.append(f"{name:14s}{v.mean():12.6f}{v.std():12.2e}"
                     f"{v.min():12.6f}{v.max():12.6f}")
    lines.append("")
    lines.append("left_riemann/trapezoid std ~ 5e-5 (uniform);")
    lines.append("simpson std ~ 2e-3 (~45x larger) -> spatial artifact.")
    lines.append("")

    # Telescoping identity on the brightest block cell.
    y, x = np.unravel_index(np.argmax(maps["simpson"]), maps["simpson"].shape)
    w = fr[y, x]
    tele = w.sum() * DX_US
    lines.append(f"Telescoping identity at brightest cell ({y},{x}):")
    lines.append(f"  left-Riemann sum*dx = {tele:.8f}  (exact Q_end - Q_start)")
    lines.append(f"  trapezoid           = {trapezoid(w, dx=DX_US):.8f}  "
                 f"(diff {trapezoid(w, dx=DX_US) - tele:+.2e})")
    s = simpson(w, dx=DX_US)
    lines.append(f"  simpson             = {s:.8f}  "
                 f"(diff {s - tele:+.2e}, {(s - tele) / tele * 100:+.3f}%)")
    lines.append("")

    # Downsampling stability.
    lines.append(f"Downsampling stability at the same cell:")
    lines.append(f"{'step':>5s}{'nsamp':>8s}{'simpson':>14s}{'trapezoid':>14s}")
    for step in (1, 2, 4, 8, 16):
        ws = w[::step]
        lines.append(f"{step:5d}{ws.size:8d}"
                     f"{simpson(ws, dx=DX_US * step):14.6f}"
                     f"{trapezoid(ws, dx=DX_US * step):14.6f}")
    lines.append("")
    lines.append("Conclusion: integrate this finite-difference current with")
    lines.append("trapezoid (== fr.sum*dx == Q_end-Q_start), NOT simpson.")
    lines.append("Simpson injects a spatially-varying 0.1-0.6% wobble by fitting")
    lines.append("parabolas across the already-differenced current.")

    text_page(pdf, "Induced-current integration: method comparison",
              "\n".join(lines))


def page_integration_artifact(pdf, data):
    fr = data["fr"]
    simp = simpson(fr, dx=DX_US, axis=2)
    trap = trapezoid(fr, dx=DX_US, axis=2)
    diff = simp - trap
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    for a, (img, ttl) in zip(ax[:2], [(simp, "simpson"), (trap, "trapezoid")]):
        im = a.imshow(img, cmap="viridis", origin="lower")
        a.set_title(ttl)
        a.set_xlabel("x index")
        a.set_ylabel("y index")
        fig.colorbar(im, ax=a, label="Charge (e-)")
    vlim = np.abs(diff).max()
    im = ax[2].imshow(diff, cmap="RdBu_r", origin="lower", vmin=-vlim, vmax=vlim)
    ax[2].set_title("simpson - trapezoid (method artifact)")
    ax[2].set_xlabel("x index")
    ax[2].set_ylabel("y index")
    fig.colorbar(im, ax=ax[2], label="Charge diff (e-)")
    fig.suptitle("Spatial structure of the Simpson integration artifact")
    add_footer(fig,
        "Charge maps from Simpson vs trapezoid integration and their difference. "
        "Trapezoid (= exact telescoping sum) is uniform across the central block; "
        "Simpson's parabola fit over the spiky current produces the spatially "
        "varying residual on the right.")
    pdf.savefig(fig)
    plt.close(fig)


def page_waveforms(pdf, data):
    fr = data["fr"]
    x_range = np.arange(fr.shape[2])
    shift = np.argmax(fr[0, 0]) - np.argmax(fr[4, 4])
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(x_range, fr[0, 0], label="at [0,0]")
    ax.plot(x_range, fr[4, 4], label="at [4,4]")
    ax.set_xlabel("Time (one tick = 0.05 us)")
    ax.set_ylabel("Induced current")
    ax.set_xlim(150, 280)
    ax.legend()
    ax.set_title(f"Induced-current waveforms (peak shift = {shift} ticks)")
    add_footer(fig,
        "Induced-current waveform at two start points within the central pixel "
        "([0,0] vs [4,4]). The peak-time shift reflects the different drift path "
        "lengths from pixel centre vs corner before charge is collected.")
    pdf.savefig(fig)
    plt.close(fig)


def page_spacetime(pdf, data):
    fr = data["fr"]
    pos_max = np.argmax(fr[0, 0, :])
    nt = min(2000, fr.shape[2])
    sts = np.zeros((5, nt))
    sts[0, :] = fr[20, 0, :nt]   # pixel -2
    sts[1, :] = fr[10, 0, :nt]   # pixel -1
    sts[2, :] = fr[0, 0, :nt]    # pixel  0
    sts[3, :] = fr[10, 0, :nt]   # pixel +1
    sts[4, :] = fr[20, 0, :nt]   # pixel +2
    pixel_offsets = np.array([-2, -1, 0, 1, 2])
    T, P = np.meshgrid(np.arange(nt), pixel_offsets)

    fig, ax = plt.subplots(2, 1, figsize=(9, 13))
    fig.subplots_adjust(hspace=0.32)

    vmax = np.max(np.abs(sts))
    linthresh = 0.01
    norm = mcolors.SymLogNorm(linthresh=linthresh, linscale=1.0,
                              vmin=-vmax, vmax=vmax, base=10)
    c = ax[1].pcolormesh(T, P, sts, cmap="RdBu_r", norm=norm, shading="nearest")
    cbar = fig.colorbar(c, ax=ax[1], pad=0.02)
    cbar.set_label(r"Induced current ($\mathrm{e^{-}/\mu s}$)")
    cbar.locator = mticker.SymmetricalLogLocator(base=10, linthresh=linthresh)
    cbar.formatter = mticker.LogFormatterMathtext(base=10)
    cbar.update_normal(c)
    ax[1].set_xlim(0, 250)
    ax[1].set_ylim(-2.5, 2.5)
    ax[1].set_yticks(pixel_offsets)
    ax[1].set_yticklabels(["-2", "-1", "0", "+1", "+2"])
    ax[1].yaxis.set_minor_locator(mticker.NullLocator())
    ax[1].set_xlabel(r"Time, tick = $0.05\ \mu$s", fontsize=14)
    ax[1].set_ylabel("Pixel index\n(from central pixel)", fontsize=14)
    ax[1].set_title("Induced current per pixel vs drift time\n"
                    "(at the center of each pixel)", fontsize=14)

    def pixel_to_region(val, pos):
        if val <= 4:
            return "0"
        elif 5 <= val <= 14:
            return "+1"
        return "+2"
    formatter = FuncFormatter(pixel_to_region)

    im = ax[0].imshow(fr[:, :, pos_max], cmap="RdBu_r", origin="lower")
    fig.colorbar(im, ax=ax[0], label=r"Induced current ($e^-/\mu s$)")
    for v in (5, 15, 24):
        ax[0].axvline(v, linestyle="--")
        ax[0].axhline(v, linestyle="--")
    ax[0].xaxis.set_major_formatter(formatter)
    ax[0].yaxis.set_major_formatter(formatter)
    ax[0].set_xticks([2, 9, 19])
    ax[0].set_yticks([2, 9, 19])
    ax[0].set_xlabel("x Pixel index", fontsize=14)
    ax[0].set_ylabel("y Pixel index", fontsize=14)
    ax[0].set_title("Induced current at a fixed drift time", fontsize=14)

    add_footer(fig,
        "Top: snapshot of induced current across the pixel array at the peak time "
        "of the central pixel, with dashed lines separating pixel regions. Bottom: "
        "symmetric-log space-time map of the current seen by neighbouring pixels, "
        "showing the bipolar signal induced on adjacent pixels.")
    pdf.savefig(fig)
    plt.close(fig)


def page_path_xy(pdf, data):
    sp = data["shifted_paths"]
    seg = sp[4, 4, 202:255, :]
    fig, ax = plt.subplots(figsize=(11, 7))
    sc = ax.scatter(seg[:, 0], seg[:, 1], c=seg[:, 2], cmap="viridis", s=30)
    fig.colorbar(sc, ax=ax, label="z (mm)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Drift path in x-y over the collection time window")
    add_footer(fig,
        "Lateral (x-y) motion of one drift path (start point 4,4) over the time "
        "window where it is being collected, coloured by depth z. Shows the "
        "transverse displacement of the trajectory as it approaches the pad.")
    pdf.savefig(fig)
    plt.close(fig)


def page_chamfer(pdf, data):
    arr = np.ones((44, 44, 1), dtype=int)
    arr[:, 19:25, :] = 0
    arr[19:25, :, :] = 0

    def trimCorner(arr, x, y, z1, z2, corner):
        if corner == 0:
            arr[x-3:x+1, y, z1:z2] = 0; arr[x, y-3:y+1, z1:z2] = 0
            arr[x, y, z1:z2] = 0; arr[x-1, y-1, z1:z2] = 0
        if corner == 1:
            arr[x-3:x+1, y, z1:z2] = 0; arr[x, y:y+4, z1:z2] = 0
            arr[x, y, z1:z2] = 0; arr[x-1, y+1, z1:z2] = 0
        if corner == 2:
            arr[x:x+4, y, z1:z2] = 0; arr[x, y:y+4, z1:z2] = 0
            arr[x, y, z1:z2] = 0; arr[x+1, y+1, z1:z2] = 0
        if corner == 3:
            arr[x:x+4, y, z1:z2] = 0; arr[x, y-3:y+1, z1:z2] = 0
            arr[x, y, z1:z2] = 0; arr[x+1, y-1, z1:z2] = 0

    z1, z2 = 0, 1
    trimCorner(arr, 18, 18, z1, z2, 0)
    trimCorner(arr, 25, 18, z1, z2, 3)
    trimCorner(arr, 18, 25, z1, z2, 1)
    trimCorner(arr, 25, 25, z1, z2, 2)
    grid = arr[:, :, 0]

    fig, ax = plt.subplots(figsize=(9, 9))
    for row in range(44):
        for col in range(44):
            if grid[row, col] == 1:
                ax.add_patch(patches.Rectangle((col, row), 1, 1, linewidth=1.0,
                             edgecolor="#000000", facecolor="#f5c800"))
    ax.set_xlim(0, 44)
    ax.set_ylim(0, 44)
    ax.set_aspect("equal")
    ax.set_title("Pixel plane (chamfered-corner grid layout)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    add_footer(fig,
        "Reconstruction of the conducting pixel-plane mask used in the geometry: "
        "a pixel pad bounded by grounded grid lines, with chamfered (trimmed) "
        "corners. Filled cells are conductor; the cross-shaped gaps are the grid.")
    pdf.savefig(fig)
    plt.close(fig)


def page_weight_slice(pdf, data):
    wp = data["weight3d"]
    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(wp[110, :, :], origin="lower", extent=[0, wp.shape[2],
                   0, wp.shape[1]])
    fig.colorbar(im, ax=ax, label="weighting potential")
    ax.set_title("Weighting potential at x = 110 tick = 11 mm")
    add_footer(fig,
        "Weighting potential on a y-z plane at x index 110 (~11 mm). The "
        "weighting potential governs the induced current (Shockley-Ramo); it "
        "peaks at the pad and falls off into the bulk.")
    pdf.savefig(fig)
    plt.close(fig)


def page_weight_contour(pdf, data):
    wp = data["weight3d"]
    pixel_x, pixel_y, half_width = 110, 110, 20
    x0 = max(pixel_x - half_width, 0)
    x1 = min(pixel_x + half_width, wp.shape[0])
    block = wp[x0:x1, pixel_y, :]
    coord = np.arange(x0, x1)
    z_coord = np.arange(wp.shape[2])
    Z, C = np.meshgrid(z_coord, coord)

    fig, ax = plt.subplots(figsize=(8, 7))
    cf = ax.contourf(C, Z, block, levels=5, cmap="viridis")
    fig.colorbar(cf, ax=ax, label="Weighting potential")
    ax.axvline(pixel_x, color="white", linestyle="--", linewidth=1, alpha=0.8)
    ax.set_xlabel("x (pixel index)")
    ax.set_ylabel("z (depth)")
    ax.set_title(f"Weighting potential, XZ slice through pixel "
                 f"({pixel_x}, {pixel_y})")
    ax.invert_yaxis()
    add_footer(fig,
        "Contour of the weighting potential on a vertical x-z slice centred on "
        "the pixel (depth increases downward). Tight contours near the pad mean "
        "a charge only induces appreciable current once it is close to the pad.")
    pdf.savefig(fig)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="store_0.1mmSpacing_0.05usTimeStep_0.1mmPixPlaneWidth",
                    help="path to the store_* dataset directory")
    ap.add_argument("--out", default="paths_at_centralpix_summary.pdf",
                    help="output PDF path")
    ap.add_argument("--compare", default=None,
                    help="optional second dataset for the delta-phi overlay page")
    args = ap.parse_args()

    data = load_dataset(args.data)
    compare = load_dataset(args.compare) if args.compare else None

    with PdfPages(args.out) as pdf:
        page_title(pdf, data, compare)
        page_paths_xz(pdf, data)
        page_velocity(pdf, data)
        page_drift_diff(pdf, data, compare)
        page_charge_maps(pdf, data)
        page_integration_methods(pdf, data)
        page_integration_artifact(pdf, data)
        page_waveforms(pdf, data)
        page_spacetime(pdf, data)
        page_path_xy(pdf, data)
        page_chamfer(pdf, data)
        page_weight_slice(pdf, data)
        page_weight_contour(pdf, data)

    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
