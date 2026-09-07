#!/usr/bin/env python3
"""
Quantify the near/far seam discontinuity in a hybrid (stitched) pochoir store.

The hybrid solve stitches a fine near-field solution to an upsampled coarse
far-field solution at ``--interface``.  The Laplacian residual metric spikes at
that plane but does NOT reveal which side is wrong; the E_z profile down the
pad-centre axis does.  This script reports that profile plus the value jump, the
gradient kink and the transverse corrugation amplitude, so a seam can be judged
without plotting anything.

READ-ONLY.  It imports only ``pochoir.main.Main`` to read the store with the
same npz conventions as the other test/ analysis scripts (see
test/sample_field_along_paths.py); it does not touch any solver code path and
writes nothing.

Usage:
    ./env/bin/python test/seam_profile.py --store <dir> \
        [--potential potential/drift3d] [--domain domain/drift3d] \
        [--interface 19.8] [--window 3.0]

Works for both fields:
    --potential potential/drift3d   (volts)
    --potential potential/weight3d  (dimensionless unit probe in [0,1])

The volt-valued design-field normalisation (56.0 V/mm) is applied only when the
potential actually looks volt-valued, or when --design-field is given.  The kink
is always reported as a raw number and as a percentage of the local |E_z|.
"""
import argparse
import sys

import numpy as np

from pochoir.main import Main


DEFAULT_DESIGN_FIELD = 56.0        # V/mm, the drift geometry's design field


def _circular_centroid_index(weights):
    """Index of the centroid of a 1-D nonnegative profile, wrap-aware.

    The drift domain is one transversely PERIODIC pixel tile, so the pad
    footprint can straddle the wrap; a plain centre-of-mass would then land in
    the gap.  Treat the axis as a circle and take the angular mean.
    """
    n = len(weights)
    tot = weights.sum()
    if tot <= 0:
        return n // 2
    ang = 2.0 * np.pi * np.arange(n) / n
    c = (weights * np.cos(ang)).sum() / tot
    s = (weights * np.sin(ang)).sum() / tot
    if abs(c) < 1e-12 and abs(s) < 1e-12:
        return n // 2
    a = np.arctan2(s, c) % (2.0 * np.pi)
    return int(round(a * n / (2.0 * np.pi))) % n


def _pad_centre_indices(pot, boundary, zaxis=2):
    """(ix, iy) of the pad-centre transverse axis.

    Uses the boundary (electrode Dirichlet) mask when the store has one: the
    plane with the most boundary cells is the pad plane, and the wrap-aware
    centroid of its per-axis metal counts is the pad centre.  Falls back to the
    geometric centre when no mask is available.
    """
    if boundary is None:
        return pot.shape[0] // 2, pot.shape[1] // 2
    b = boundary.astype(bool)
    per_plane = b.reshape(-1, b.shape[zaxis]).sum(axis=0) \
        if zaxis == 2 else None
    if per_plane is None or not per_plane.any():
        return pot.shape[0] // 2, pot.shape[1] // 2
    # The pad plane is the densest metal plane; the cathode is a FULL plane, so
    # prefer the densest plane that is not fully metal.
    ntrans = b.shape[0] * b.shape[1]
    cand = np.where(per_plane < ntrans)[0]
    if len(cand) == 0:
        return pot.shape[0] // 2, pot.shape[1] // 2
    iz = cand[np.argmax(per_plane[cand])]
    sl = b[:, :, iz]
    ix = _circular_centroid_index(sl.sum(axis=1).astype(float))
    iy = _circular_centroid_index(sl.sum(axis=0).astype(float))
    return ix, iy


def _ez(phi_z, dz):
    """E_z = -dphi/dz by central difference, one-sided at the two ends."""
    return -np.gradient(phi_z, dz)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Quantify the near/far seam in a hybrid pochoir store.")
    ap.add_argument('--store', required=True)
    ap.add_argument('--potential', default='potential/drift3d',
                    help="store key of the stitched full-volume potential")
    ap.add_argument('--domain', default=None,
                    help="domain key (default: derived from --potential)")
    ap.add_argument('--boundary', default=None,
                    help="boundary key (default: derived from --potential)")
    ap.add_argument('--interface', type=float, default=19.8,
                    help="near/far interface depth in mm")
    ap.add_argument('--window', type=float, default=3.0,
                    help="+/- mm around the interface to tabulate")
    ap.add_argument('--ix', type=int, default=None,
                    help="override the transverse x index of the profile axis")
    ap.add_argument('--iy', type=int, default=None,
                    help="override the transverse y index of the profile axis")
    ap.add_argument('--design-field', type=float, default=None,
                    help="reference field in units/mm for the kink percentage "
                         "(default: 56.0 for a volt-valued potential, none for "
                         "a dimensionless weighting probe)")
    ap.add_argument('--depths', default='10.5,15,19.8,30,40',
                    help="comma-separated mm depths for the corrugation table")
    a = ap.parse_args(argv)

    stem = a.potential.split('/')[-1]
    dom_key = a.domain or 'domain/%s' % stem
    bnd_key = a.boundary or 'boundary/%s' % stem

    m = Main(a.store)
    dom = m.get_domain(dom_key)
    pot = np.asarray(m.get(a.potential), dtype=float)
    try:
        boundary = np.asarray(m.get(bnd_key))
    except Exception:
        boundary = None

    if pot.ndim != 3:
        print("ERROR: %s is %dD, expected 3D" % (a.potential, pot.ndim),
              file=sys.stderr)
        return 1

    zs = np.asarray(dom.linspaces[2], dtype=float)
    dz = float(dom.spacing[2])
    nz = len(zs)

    # ---- field kind -------------------------------------------------------
    span = float(pot.max() - pot.min())
    volt_like = span > 1.5
    unit = 'V' if volt_like else '1'
    funit = 'V/mm' if volt_like else '1/mm'
    design = a.design_field
    if design is None and volt_like:
        design = DEFAULT_DESIGN_FIELD

    ix = a.ix if a.ix is not None else None
    iy = a.iy if a.iy is not None else None
    if ix is None or iy is None:
        cx, cy = _pad_centre_indices(pot, boundary)
        ix = cx if ix is None else ix
        iy = cy if iy is None else iy

    print("=" * 74)
    print("SEAM PROFILE  store=%s" % a.store)
    print("  potential %-24s shape %s" % (a.potential, tuple(pot.shape)))
    print("  domain    %-24s spacing %s mm" % (dom_key, tuple(dom.spacing)))
    print("  range     [%.6g, %.6g] %s  ->  treated as %s"
          % (pot.min(), pot.max(), unit,
             "VOLT-VALUED (drift)" if volt_like
             else "DIMENSIONLESS (weighting probe)"))
    print("  profile axis at transverse (ix,iy) = (%d,%d) = (%.4f,%.4f) mm%s"
          % (ix, iy, dom.linspaces[0][ix], dom.linspaces[1][iy],
             "" if boundary is not None else "  [geometric centre; no boundary mask]"))
    if design is not None:
        print("  reference field for percentages: %.6g %s" % (design, funit))
    print("=" * 74)

    # ---- 5. does the interface land on a node? ----------------------------
    k = (a.interface - zs[0]) / dz
    inode = int(round(k))
    on_node = abs(k - inode) < 1e-6 and 0 <= inode < nz
    print()
    print("[5] INTERFACE PLANE ON THE OUTPUT GRID")
    print("    z_iface           = %.6f mm" % a.interface)
    print("    (z_iface-z0)/dz   = %.9f" % k)
    if on_node:
        print("    ON A NODE, index %d of %d  (z = %.6f mm)"
              % (inode, nz, zs[inode]))
    else:
        print("    *** NOT A NODE of this grid *** nearest index %d (z = %.6f mm),"
              % (inode, zs[min(max(inode, 0), nz - 1)]))
        print("        off by %.6g mm -- check --interface against the store"
              % abs(a.interface - zs[min(max(inode, 0), nz - 1)]))
    if not (1 <= inode <= nz - 2):
        print("    interface too close to a z face to profile; stopping.")
        return 1

    phi = pot[ix, iy, :]
    ez = _ez(phi, dz)

    # ---- 1. phi(z) and E_z(z) across the seam ----------------------------
    lo = max(0, int(np.floor((a.interface - a.window - zs[0]) / dz)))
    hi = min(nz - 1, int(np.ceil((a.interface + a.window - zs[0]) / dz)))
    print()
    print("[1] phi(z) AND E_z(z) DOWN THE PAD-CENTRE AXIS, %.3f +/- %.3f mm"
          % (a.interface, a.window))
    print("    E_z by central difference on the %.4g mm output spacing." % dz)
    print()
    print("      %6s %12s %16s %16s" % ("index", "z [mm]", "phi [%s]" % unit,
                                        "E_z [%s]" % funit))
    for i in range(lo, hi + 1):
        mark = "  <== seam" if i == inode else ""
        print("      %6d %12.5f %16.9g %16.9g%s"
              % (i, zs[i], phi[i], ez[i], mark))

    # ---- 2. the VALUE jump at the seam plane -----------------------------
    # Compare phi at the seam against what each side's local linear trend
    # predicts there.  Fit each side on the nodes strictly below / above.
    nfit = max(3, int(round(1.0 / dz)))          # ~1mm of nodes per side
    blo = max(0, inode - nfit)
    bhi = min(nz - 1, inode + nfit)
    below = np.polyfit(zs[blo:inode], phi[blo:inode], 1)
    above = np.polyfit(zs[inode + 1:bhi + 1], phi[inode + 1:bhi + 1], 1)
    p_below = np.polyval(below, zs[inode])
    p_above = np.polyval(above, zs[inode])
    print()
    print("[2] VALUE JUMP AT THE SEAM PLANE (index %d, z = %.5f mm)"
          % (inode, zs[inode]))
    print("    phi(seam)                          = %16.9g %s" % (phi[inode], unit))
    print("    linear extrapolation from BELOW    = %16.9g %s" % (p_below, unit))
    print("    linear extrapolation from ABOVE    = %16.9g %s" % (p_above, unit))
    print("    departure from below-side trend    = %16.9g %s" % (phi[inode] - p_below, unit))
    print("    departure from above-side trend    = %16.9g %s" % (phi[inode] - p_above, unit))
    print("    one-sided trends disagree by       = %16.9g %s" % (p_above - p_below, unit))
    print("    |phi(z+dz) - phi(z-dz)|            = %16.9g %s"
          % (abs(phi[inode + 1] - phi[inode - 1]), unit))
    print("    (a C0 stitch makes phi(seam) itself continuous by construction;")
    print("     a nonzero trend disagreement is the gradient kink showing up in phi.)")

    # ---- 3. the GRADIENT kink -------------------------------------------
    ez_below = -(phi[inode] - phi[inode - 1]) / dz
    ez_above = -(phi[inode + 1] - phi[inode]) / dz
    kink = ez_above - ez_below
    local = 0.5 * (abs(ez_below) + abs(ez_above))
    print()
    print("[3] GRADIENT KINK AT THE SEAM  (the number that matters)")
    print("    E_z just BELOW the plane           = %16.9g %s" % (ez_below, funit))
    print("    E_z just ABOVE the plane           = %16.9g %s" % (ez_above, funit))
    print("    kink (above - below)               = %16.9g %s" % (kink, funit))
    if local > 0:
        print("    as %% of the local |E_z| (%-12.6g) = %13.6f %%"
              % (local, 100.0 * abs(kink) / local))
    else:
        print("    local |E_z| is zero here; percentage undefined")
    if design is not None:
        print("    as %% of the reference field %-8.6g = %13.6f %%"
              % (design, 100.0 * abs(kink) / abs(design)))
    else:
        print("    (no reference field for this dimensionless probe; use the")
        print("     local-|E_z| percentage above)")
    print("    slope of the one-sided fits: below %.9g, above %.9g %s"
          % (-below[0], -above[0], funit))

    # ---- 4. TRANSVERSE CORRUGATION --------------------------------------
    depths = []
    for tok in a.depths.split(','):
        tok = tok.strip()
        if tok:
            depths.append(float(tok))
    if a.interface not in depths:
        depths.append(a.interface)
    depths = sorted(set(depths))
    print()
    print("[4] TRANSVERSE CORRUGATION AMPLITUDE, (max-min) of phi ACROSS THE SLICE")
    print("    The pad-plane corrugation decays with a ~4.2mm length and is gone")
    print("    by z ~ 30-40mm (NOTES-weighting-farfield.md); the interface is at")
    print("    %.3f mm, so this says whether the seam sits inside that zone."
          % a.interface)
    print()
    print("      %10s %8s %16s %16s %16s" % ("z [mm]", "index", "min [%s]" % unit,
                                             "max [%s]" % unit,
                                             "max-min [%s]" % unit))
    for zd in depths:
        j = int(round((zd - zs[0]) / dz))
        if not (0 <= j < nz):
            print("      %10.4f %8s  (outside the grid)" % (zd, "-"))
            continue
        sl = pot[:, :, j]
        mark = "  <== seam" if j == inode else ""
        print("      %10.4f %8d %16.9g %16.9g %16.9g%s"
              % (zs[j], j, sl.min(), sl.max(), sl.max() - sl.min(), mark))

    print()
    print("=" * 74)
    return 0


if __name__ == '__main__':
    sys.exit(main())
