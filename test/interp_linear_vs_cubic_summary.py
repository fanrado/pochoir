#!/usr/bin/env python3
"""
Compare linear vs. cubic interpolation of the *scalar drift potential* and its
downstream effect on E = grad(phi) and the drift velocity v = mu(|E|,T)*E.

The comparison is done with the exact code path the drift solver uses:
``pochoir.drift_numpy.PotentialField`` is instantiated twice on the same stored
potential array, once with ``method='linear'`` and once with ``method='cubic'``.
Everything downstream (the half-spacing central-difference gradient, the LAr
mobility model, the units) is therefore identical between the two curves; only
the interpolation order of phi differs.

Two regions are contrasted:

  * BULK region  -- deep in the drift gap where the field is nearly uniform.
    Here phi is almost linear in z, so linear and cubic *values* agree closely,
    but the linear gradient is only C^-1 (piecewise, staircased across cell
    faces) while the cubic gradient is smooth.  This isolates the "kink at every
    cell face" pathology that makes the adaptive ODE integrator step-collapse.

  * NEAR-PIXEL-PLANE region -- where the pixel-pad geometry drives strong,
    rapidly varying transverse and longitudinal fields.  Here the two orders
    disagree substantially in value as well as smoothness, and cubic overshoot
    near the electrode edges (if any) shows up.

Output: a multi-page PDF, one figure per page, each with a description footer.

Usage:
    python interp_linear_vs_cubic_summary.py \
        --store store \
        --out   interp_linear_vs_cubic_summary.pdf
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
from pochoir.domain import Domain
from pochoir.drift_numpy import PotentialField


# --------------------------------------------------------------------------- #
# Physical-unit conversion helpers (internal system-of-units -> readable units)
# --------------------------------------------------------------------------- #
E_VPERCM = units.V / units.cm          # divide an internal E by this -> V/cm
V_MMPERUS = units.mm / units.us         # divide an internal v by this -> mm/us
MM = units.mm


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load(store, key="drift3d"):
    """Load the domain + scalar potential from a pochoir store directory."""
    with open(os.path.join(store, "domain", key + ".json")) as fp:
        dj = json.load(fp)
    domain = Domain(dj["shape"], dj["spacing"], dj.get("origin"))
    phi = np.load(os.path.join(store, "potential", key + ".npz"))[key]
    paths = None
    ppath = os.path.join(store, "paths", "drift3d_tight.npz")
    if os.path.exists(ppath):
        paths = np.load(ppath)["drift3d_tight"]          # (npaths, nticks, 3)
    return domain, phi, paths


def find_pixel_plane_z(domain, phi):
    """z (mm) of the pixel plane = where transverse std(phi) is maximal."""
    tv = phi.std(axis=(0, 1))
    zi = int(tv.argmax())
    z_axis = domain.linspaces[2]
    return z_axis[zi], zi, tv


# --------------------------------------------------------------------------- #
# Sampling the two PotentialField interpolants
# --------------------------------------------------------------------------- #
def sample_line(pf, p0, p1, n):
    """Sample phi, |E|, E-components and |v| along the segment p0 -> p1.

    Returns a dict of arrays already converted to readable units:
        s     : arc-length parameter (mm) from p0
        phi   : interpolated scalar potential (raw stored units, ~volts)
        Emag  : |E| in V/cm
        E     : (n,3) E components in V/cm
        vmag  : |v| in mm/us
        vz    : v_z in mm/us
    """
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    ts = np.linspace(0.0, 1.0, n)
    pts = p0[None, :] + ts[:, None] * (p1 - p0)[None, :]
    s = np.linalg.norm(pts - p0[None, :], axis=1) / MM

    phi = np.empty(n)
    E = np.empty((n, 3))
    v = np.empty((n, 3))
    for i, p in enumerate(pts):
        phi[i] = pf.potential_at(p)
        e = pf.efield(p)
        E[i] = e
        v[i] = pf(0.0, p)
    Emag = np.linalg.norm(E, axis=1) / E_VPERCM
    Ec = E / E_VPERCM
    vmag = np.linalg.norm(v, axis=1) / V_MMPERUS
    vz = v[:, 2] / V_MMPERUS
    return dict(s=s, pts=pts, phi=phi, Emag=Emag, E=Ec, vmag=vmag, vz=vz)


def sample_plane_Emag(pf, xs, zs, yfix):
    """|E| (V/cm) on an (x,z) grid at fixed y.  Returns array shaped (len(zs),len(xs))."""
    out = np.empty((len(zs), len(xs)))
    for iz, z in enumerate(zs):
        for ix, x in enumerate(xs):
            e = pf.efield(np.array([x, yfix, z], float))
            out[iz, ix] = np.linalg.norm(e) / E_VPERCM
    return out


# --------------------------------------------------------------------------- #
# PDF page helper
# --------------------------------------------------------------------------- #
def add_footer(fig, text, width=112, rect=(0, 0.16, 1, 0.94)):
    """Lay the axes out inside `rect`, then stamp a wrapped description below."""
    fig.tight_layout(rect=rect)
    wrapped = "\n".join(textwrap.wrap(" ".join(text.split()), width=width))
    fig.text(0.06, 0.015, wrapped, ha="left", va="bottom", fontsize=8.5,
             family="monospace",
             bbox=dict(boxstyle="round", facecolor="#f5f5f5", edgecolor="#bbbbbb"))


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def page_context(pdf, domain, phi, paths, z_plane, tv):
    """Orientation page: where the pixel plane is, where the bulk is."""
    z_axis = domain.linspaces[2]
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.plot(z_axis, tv, color="C3", lw=1.6)
    ax.axvline(z_plane, color="k", ls="--", lw=1.0)
    ax.annotate(f"pixel plane\nz = {z_plane:.2f} mm\n(strong geometry)",
                xy=(z_plane, tv.max()), xytext=(z_plane + 4, tv.max() * 0.8),
                arrowprops=dict(arrowstyle="->"), fontsize=10)
    # shade bulk region
    ax.axvspan(z_plane + 3, z_axis[-1] - 2, color="C0", alpha=0.12)
    ax.text((z_plane + 3 + z_axis[-1] - 2) / 2, tv.max() * 0.5,
            "BULK\n(near-uniform field)", ha="center", color="C0", fontsize=11)
    if paths is not None:
        zmin, zmax = paths[..., 2].min() / MM, paths[..., 2].max() / MM
        ax.axvspan(zmin, zmax, color="C2", alpha=0.06)
        ax.text((zmin + zmax) / 2, tv.max() * 0.97,
                f"drift-path z-range\n[{zmin:.1f}, {zmax:.1f}] mm",
                ha="center", color="C2", fontsize=9)
    ax.set_xlabel("z  [mm]")
    ax.set_ylabel(r"transverse std of $\phi$ across (x,y)  [V]")
    ax.set_title("Where the geometry matters: transverse variation of the drift "
                 "potential vs. z")
    ax.grid(alpha=0.3)
    add_footer(fig,
               "The stored scalar potential phi is solved on a "
               f"{tuple(int(s) for s in domain.shape)} grid at "
               f"{domain.spacing[0]:.2f} mm spacing. "
               "The pixel-plane/anode sits at the z where phi varies most strongly "
               "across the transverse (x,y) plane; the drift gap above it is nearly "
               "field-uniform (bulk). Electrons start near the cathode (high z) and "
               "drift down to the pixel plane. The following pages compare LINEAR vs "
               "CUBIC interpolation of phi -- and its derived E and v -- first in the "
               "smooth bulk, then across the pixel plane where geometry dominates.")
    pdf.savefig(fig)
    plt.close(fig)


def page_line(pdf, pf_lin, pf_cub, p0, p1, n, title, footer, zoom=None):
    """4-panel page: phi, |E|, v_z, and (cubic-linear) along a line."""
    L = sample_line(pf_lin, p0, p1, n)
    C = sample_line(pf_cub, p0, p1, n)
    # arc-length is along z for vertical lines; label accordingly
    dz = np.abs(np.asarray(p1, float) - np.asarray(p0, float))
    along_z = np.argmax(dz) == 2
    xlab = "z along line  [mm]" if along_z else "transverse position along line  [mm]"
    xco = (np.asarray(p0, float)[2] / MM) + L["s"] * (1 if p1[2] >= p0[2] else -1) \
        if along_z else L["s"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle(title, fontsize=13)

    ax = axes[0, 0]
    ax.plot(xco, L["phi"], color="C0", lw=1.4, label="linear")
    ax.plot(xco, C["phi"], color="C1", lw=1.4, ls="--", label="cubic")
    ax.set_ylabel(r"$\phi$  [V]"); ax.set_xlabel(xlab)
    ax.set_title(r"scalar potential $\phi$"); ax.grid(alpha=0.3); ax.legend()

    ax = axes[0, 1]
    ax.plot(xco, L["Emag"], color="C0", lw=1.4, label="linear")
    ax.plot(xco, C["Emag"], color="C1", lw=1.4, ls="--", label="cubic")
    ax.set_ylabel(r"$|E| = |\nabla\phi|$  [V/cm]"); ax.set_xlabel(xlab)
    ax.set_title(r"field magnitude $|E|$  (note linear staircase / kinks)")
    ax.grid(alpha=0.3); ax.legend()
    if zoom is not None:
        axins = ax.inset_axes([0.5, 0.5, 0.47, 0.47])
        m = (xco >= zoom[0]) & (xco <= zoom[1])
        axins.plot(xco[m], L["Emag"][m], color="C0", lw=1.4)
        axins.plot(xco[m], C["Emag"][m], color="C1", lw=1.4, ls="--")
        axins.tick_params(labelsize=7)
        ax.indicate_inset_zoom(axins, edgecolor="black")

    ax = axes[1, 0]
    ax.plot(xco, L["vz"], color="C0", lw=1.4, label="linear")
    ax.plot(xco, C["vz"], color="C1", lw=1.4, ls="--", label="cubic")
    ax.set_ylabel(r"$v_z$  [mm/$\mu$s]"); ax.set_xlabel(xlab)
    ax.set_title(r"drift velocity $v_z = \mu(|E|,T)\,E_z$")
    ax.grid(alpha=0.3); ax.legend()

    ax = axes[1, 1]
    ax.plot(xco, C["phi"] - L["phi"], color="C3", lw=1.2, label=r"$\Delta\phi$ [V]")
    ax.plot(xco, C["Emag"] - L["Emag"], color="C4", lw=1.2,
            label=r"$\Delta|E|$ [V/cm]")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("cubic - linear"); ax.set_xlabel(xlab)
    ax.set_title("difference (cubic - linear)"); ax.grid(alpha=0.3); ax.legend()

    add_footer(fig, footer)
    pdf.savefig(fig)
    plt.close(fig)


def page_plane(pdf, pf_lin, pf_cub, domain, z_plane, yfix, footer):
    """(x,z) maps of |E| for linear, cubic, and their difference near the plane."""
    x_axis = domain.linspaces[0]
    xs = np.linspace(x_axis[0], x_axis[-1], 88)
    zs = np.linspace((z_plane - 1.5) * MM, (z_plane + 2.5) * MM, 80)
    EL = sample_plane_Emag(pf_lin, xs, zs, yfix)
    EC = sample_plane_Emag(pf_cub, xs, zs, yfix)
    diff = EC - EL

    ext = [xs[0] / MM, xs[-1] / MM, zs[0] / MM, zs[-1] / MM]
    fig, axes = plt.subplots(1, 3, figsize=(13, 7.2))
    fig.suptitle(f"|E| in an (x,z) slice at y={yfix/MM:.2f} mm across the pixel plane "
                 f"(z={z_plane:.2f} mm)", fontsize=12)
    vmax = np.percentile(np.concatenate([EL.ravel(), EC.ravel()]), 99)
    for ax, dat, ttl in ((axes[0], EL, "linear"), (axes[1], EC, "cubic")):
        im = ax.imshow(dat, origin="lower", extent=ext, aspect="auto",
                       cmap="viridis", vmin=0, vmax=vmax)
        ax.set_title(f"|E|  ({ttl})"); ax.set_xlabel("x [mm]"); ax.set_ylabel("z [mm]")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="V/cm")
    dmax = np.percentile(np.abs(diff), 99)
    im = axes[2].imshow(diff, origin="lower", extent=ext, aspect="auto",
                        cmap="RdBu_r", vmin=-dmax, vmax=dmax)
    axes[2].set_title("|E| cubic - linear"); axes[2].set_xlabel("x [mm]")
    axes[2].set_ylabel("z [mm]")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04, label="V/cm")
    add_footer(fig, footer)
    pdf.savefig(fig)
    plt.close(fig)


def page_text(pdf, title, body):
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.08, 0.92, title, fontsize=15, weight="bold", va="top")
    wrapped = "\n".join(textwrap.wrap(" ".join(body.split()), width=95))
    fig.text(0.08, 0.85, wrapped, fontsize=11, va="top", family="monospace")
    pdf.savefig(fig)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default="store", help="pochoir store directory")
    ap.add_argument("--key", default="drift3d", help="drift potential key")
    ap.add_argument("--temperature", type=float, default=87.0,
                    help="LAr temperature (Kelvin, system-of-units)")
    ap.add_argument("--out", default="interp_linear_vs_cubic_summary.pdf")
    ap.add_argument("--n", type=int, default=600, help="samples per line")
    args = ap.parse_args()

    domain, phi, paths = load(args.store, args.key)
    z_plane, zi, tv = find_pixel_plane_z(domain, phi)
    x_axis, y_axis, z_axis = domain.linspaces

    pf_lin = PotentialField(domain, phi, args.temperature, method="linear")
    pf_cub = PotentialField(domain, phi, args.temperature, method="cubic")

    # ------- pick sampling geometry -------------------------------------- #
    # Transverse location with the strongest in-plane gradient at the plane,
    # used for the near-plane vertical line (passes through a pixel-pad edge).
    gx = np.abs(np.gradient(phi[:, 0, zi]))
    x_edge = x_axis[int(gx.argmax())]

    # Vertical bulk line: fixed (x,y) well above the plane, through near-uniform
    # field; spans several cells so the linear staircase in E is visible.
    xb = x_axis[len(x_axis) // 2]
    yb = y_axis[len(y_axis) // 2]
    bulk_z0 = min(z_plane + 5.0, z_axis[-1] - 6.0) * MM
    bulk_z1 = (z_axis[-1] - 2.0) * MM
    # a ~1 mm zoom window (in mm along z) for the staircase inset
    zw0 = bulk_z0 / MM + 1.0
    zoom_bulk = (zw0, zw0 + 1.0)

    # Vertical near-plane line through the pad edge.
    near_z0 = (z_plane - 1.2) * MM
    near_z1 = (z_plane + 3.0) * MM

    # Transverse line across the pad at the plane (x varying, y=0).
    tp_y = y_axis[0]

    print(f"pixel plane z = {z_plane:.3f} mm (index {zi})")
    print(f"bulk vertical line  @ (x={xb/MM:.2f}, y={yb/MM:.2f}) mm, "
          f"z {bulk_z0/MM:.2f}->{bulk_z1/MM:.2f} mm")
    print(f"near-plane vertical line @ (x={x_edge/MM:.2f}, y={tp_y/MM:.2f}) mm, "
          f"z {near_z0/MM:.2f}->{near_z1/MM:.2f} mm")

    with PdfPages(args.out) as pdf:
        page_text(pdf,
                  "Linear vs. cubic interpolation of the scalar drift potential",
                  "This report compares how the drift solver reconstructs the "
                  "electric field E = grad(phi) and the drift velocity "
                  "v = mu(|E|,T) * E when the scalar potential phi is interpolated "
                  "with LINEAR (trilinear, C0) vs CUBIC (C2) order. Both curves come "
                  "from pochoir.drift_numpy.PotentialField evaluated on the same "
                  "stored potential array; only the RegularGridInterpolator order "
                  "differs, so every difference seen here is purely the effect of "
                  "interpolation order. Physics recap: a C0 (linear) phi has a "
                  "piecewise-constant gradient, so E is discontinuous across every "
                  "grid-cell face. Those kinks (a) cap the between-node accuracy of E "
                  "at O(h) and (b) make the adaptive Radau integrator repeatedly "
                  "shrink its step at cell faces. A C2 (cubic) phi yields a smooth E "
                  "and v. Two regimes are examined: the near-uniform BULK (where the "
                  "value difference is tiny but the smoothness difference is stark) "
                  "and the NEAR-PIXEL-PLANE region (where geometry makes the two "
                  "orders differ in value as well, and where cubic overshoot near "
                  "electrode edges, if any, would appear).")

        page_context(pdf, domain, phi, paths, z_plane, tv)

        page_line(pdf, pf_lin, pf_cub,
                  [xb, yb, bulk_z0], [xb, yb, bulk_z1], args.n,
                  f"BULK region -- vertical line at (x={xb/MM:.2f}, y={yb/MM:.2f}) mm",
                  "Deep in the drift gap phi is nearly linear in z, so the linear and "
                  "cubic POTENTIAL curves overlap and the endpoint difference is "
                  "negligible (bottom-right). But the derived |E| exposes the "
                  "difference in smoothness: the linear field is a staircase with a "
                  "kink at every 0.1 mm cell face (see zoom), while the cubic field is "
                  "smooth. This staircase -- not the value error -- is what forces the "
                  "adaptive ODE integrator to collapse its step size at each face. The "
                  "drift velocity v_z inherits the same texture.",
                  zoom=zoom_bulk)

        page_line(pdf, pf_lin, pf_cub,
                  [x_edge, tp_y, near_z0], [x_edge, tp_y, near_z1], args.n,
                  f"NEAR PIXEL PLANE -- vertical line at (x={x_edge/MM:.2f}, "
                  f"y={tp_y/MM:.2f}) mm through a pad edge",
                  "Crossing the pixel plane the potential bends sharply. Here linear "
                  "and cubic disagree in VALUE as well as smoothness: cubic tracks the "
                  "curvature of phi (bottom-right delta is no longer negligible) and "
                  "produces a smoothly peaked |E| and v_z, whereas the linear "
                  "reconstruction is jagged. Watch for cubic overshoot/undershoot "
                  "right at the electrode edge -- the known trade-off of C2 "
                  "interpolation against a near-singular geometry.")

        page_line(pdf, pf_lin, pf_cub,
                  [x_axis[0], tp_y, z_plane * MM],
                  [x_axis[-1], tp_y, z_plane * MM], args.n,
                  f"NEAR PIXEL PLANE -- transverse line (x varying) at "
                  f"y={tp_y/MM:.2f} mm, z={z_plane:.2f} mm",
                  "A horizontal cut straight across the pixel pad at the plane. The "
                  "pad footprint appears as a well in phi; its edges are where the "
                  "transverse field E_x is largest. Linear interpolation renders the "
                  "well with flat facets and corners (staircased E_x), cubic renders a "
                  "rounded well with a continuous E_x. This transverse geometry is the "
                  "dominant driver of where drift paths bend as they approach the "
                  "anode.")

        page_plane(pdf, pf_lin, pf_cub, domain, z_plane, tp_y,
                   "|E| maps in an (x,z) slice spanning the pixel plane. The linear "
                   "map shows blocky cell-face banding; the cubic map is smooth. The "
                   "right panel (cubic - linear) localizes where the interpolation "
                   "order matters most: concentrated at the pad edges and along cell "
                   "faces near the plane, and near-zero out in the bulk. This is the "
                   "spatial picture behind the line plots.")

        page_text(pdf, "Summary",
                  "1) In the BULK the two orders give the same potential to within "
                  "rounding, so switching to cubic does NOT change drift-path "
                  "endpoints there -- its benefit is a smooth E/v that removes the "
                  "cell-face kinks and lets the adaptive integrator take long steps. "
                  "2) NEAR THE PIXEL PLANE the orders differ in value too, because "
                  "cubic captures the curvature of phi that linear cannot; this is "
                  "where cubic changes the paths, and where any cubic overshoot near "
                  "electrode edges must be watched. 3) Cubic improves the between-node "
                  "reconstruction of the SAME node data (C0->C2, O(h)->O(h^3) in E); "
                  "it does not improve the node values themselves -- that requires a "
                  "finer FDM grid. The two levers are complementary: finer grid fixes "
                  "node fidelity, cubic fixes between-node smoothness. Finer grid alone "
                  "does not remove the kinks; only a higher-order interpolant does.")

    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
