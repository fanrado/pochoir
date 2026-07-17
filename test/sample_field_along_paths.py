#!/usr/bin/env python3
"""
Sample the drift velocity and drift E-field at every point ALONG each drift path.

Motivation
----------
The saved paths give only the electron POSITIONS.  To argue about what happens
at t+dt (i.e. whether a charge that reaches the pixel plane in the gap is still
being driven, vs. sitting in equilibrium on the FR4 surface), we need the drift
velocity and E-field the electron actually experiences at each instant along its
trajectory.

This reconstructs exactly the field the drift integrator used: it rebuilds the
same ``PotentialField`` (interpolate the scalar drift potential -> central finite
difference for E = grad(phi) -> LAr mobility -> velocity), and evaluates it,
vectorised, at every path sample.  It does NOT touch any production code.

Outputs (into the same store):
    alongpath/velocity.npz  -> key 'velocity'  shape (Npaths, Nsteps, 3)
    alongpath/efield.npz    -> key 'efield'    shape (Npaths, Nsteps, 3)   [V/mm]
    alongpath/meta.npz      -> Emag_bulk (V/mm reference), z_surface, etc.

Usage: python sample_field_along_paths.py STORE_DIR \
           [--potential potential/drift3d] [--paths paths/drift3d_nodes] \
           [--insulator initial/drift_full_insulator] [--domain domain/drift_full] \
           [--temperature 87.0]
"""
import sys, argparse
import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI
from pochoir.main import Main
from pochoir import lar, units


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('store')
    ap.add_argument('--potential', default='potential/drift3d')
    ap.add_argument('--paths', default='paths/drift3d_nodes')
    ap.add_argument('--insulator', default='initial/drift_full_insulator')
    ap.add_argument('--domain', default='domain/drift_full')
    ap.add_argument('--temperature', type=float, default=87.0)
    ap.add_argument('--outprefix', default='alongpath')
    a = ap.parse_args()

    m = Main(a.store)
    dom = m.get_domain(a.domain)
    pot = np.asarray(m.get(a.potential))
    paths = np.asarray(m.get(a.paths))                     # (N, T, 3) system units
    try:
        ins = np.asarray(m.get(a.insulator)).astype(bool)
    except Exception:
        ins = None
    temp = a.temperature * units.K

    spacing = np.array(dom.spacing, dtype=float)
    origin = np.array(dom.origin, dtype=float)
    ishape = np.array(dom.shape, dtype=int)
    bb_lo = np.array(dom.bb[0], dtype=float)
    bb_hi = np.array(dom.bb[1], dtype=float)
    interp = RGI(dom.linspaces, pot, method='linear',
                 bounds_error=False, fill_value=None)     # same as PotentialField

    N, T, ndim = paths.shape
    flat = paths.reshape(-1, ndim)                         # (N*T, 3)

    def efield_batch(P):
        """Vectorised copy of the MASK-AWARE PotentialField.efield over an (M,3)
        array of points (pochoir-9keo / pochoir-0342).

        When an insulator mask is present the naive central difference would be
        taken ACROSS the frozen FR4 cells (phi=0), fabricating a spurious normal
        field at the surface.  Matching production drift_numpy.PotentialField.
        efield exactly: for each +/- half-cell sample, test the ADJACENT CELL in
        that direction (the same neighbour the solver's stencil_poisson_neumann
        drops its bond to); if exactly one neighbour cell is solid, mirror that
        sample about the point (ghost = active-side value) so the NORMAL
        component vanishes at the FR4 face while the TANGENTIAL components (whose
        samples stay in the LAr) survive; if both neighbours are solid there is
        no field.  Kept in lockstep with PotentialField.efield -- if that rule
        changes, update here too.
        """
        M = P.shape[0]
        E = np.zeros((M, ndim))
        if ins is not None:
            # nearest grid cell per point, clamped in range (== PotentialField._cell)
            base = np.clip(np.rint((P - origin) / spacing).astype(int),
                           0, ishape - 1)
        for d in range(ndim):
            h = 0.5 * spacing[d]
            pp = P.copy(); pm = P.copy()
            phigh = np.minimum(P[:, d] + h, bb_hi[d])
            plow = np.maximum(P[:, d] - h, bb_lo[d])
            pp[:, d] = phigh; pm[:, d] = plow
            denom = phigh - plow
            good = denom > 0
            phi_plus = np.zeros(M); phi_minus = np.zeros(M)
            phi_plus[good] = interp(pp[good])
            phi_minus[good] = interp(pm[good])
            sel = good.copy()
            if ins is not None:
                cp = base.copy(); cp[:, d] = np.minimum(cp[:, d] + 1, ishape[d] - 1)
                cm = base.copy(); cm[:, d] = np.maximum(cm[:, d] - 1, 0)
                plus_solid = ins[tuple(cp.T)]
                minus_solid = ins[tuple(cm.T)]
                mp = plus_solid & ~minus_solid       # i+1 solid -> mirror plus
                mm = minus_solid & ~plus_solid       # i-1 solid -> mirror minus
                both = plus_solid & minus_solid      # sandwiched -> no field
                phi_plus[mp] = phi_minus[mp]
                phi_minus[mm] = phi_plus[mm]
                sel = good & ~both
            val = np.zeros(M)
            val[sel] = (phi_plus[sel] - phi_minus[sel]) / denom[sel]
            E[:, d] = val
        return E * units.V

    # process in chunks to bound memory
    velo = np.zeros_like(flat)
    efld = np.zeros_like(flat)
    CH = 200000
    for s in range(0, flat.shape[0], CH):
        P = flat[s:s+CH]
        inside = np.all((P >= bb_lo) & (P <= bb_hi), axis=1)
        E = efield_batch(P)
        emag = np.sqrt((E**2).sum(axis=1))
        mu = lar.mobility(emag, temp)
        v = E * (mu / units.mm**2)[:, None]
        # Match the enforcement-free production drift (pochoir-h3y1): velocity is
        # zeroed ONLY outside the domain bbox (no field data there), NOT inside the
        # FR4 mask.  The in-insulator v=0 zeroing was removed from PotentialField
        # so the reconstructed velocity here must follow the Neumann-BC field too.
        v[~inside] = 0.0
        velo[s:s+CH] = v
        efld[s:s+CH] = E

    velo = velo.reshape(N, T, ndim)
    efld = efld.reshape(N, T, ndim)

    # bulk |E| reference (deep drift region, on-axis) for V/mm normalisation
    zc = ishape[2] // 2
    Ebulk = float(np.sqrt((efield_batch(np.array([[origin[0]+spacing[0]*ishape[0]//2,
                                                   origin[1]+spacing[1]*ishape[1]//2,
                                                   origin[2]+spacing[2]*zc]]))**2).sum()))
    vmm_per_unit = 50.0 / Ebulk if Ebulk > 0 else 1.0      # bulk = 50 V/mm by design
    efld_vmm = efld * vmm_per_unit                         # E in V/mm

    # FR4 top face = (max FR4 z-index + 0.5)*spacing, matching the termination
    # plane in drift_numpy.solve_potential (commit 09fb55c, pochoir-qqlw).
    max_fr4 = int(np.max(np.where(ins.any(axis=(0, 1)))[0])) if ins is not None else -1
    z_surface = float(origin[2] + (max_fr4 + 0.5) * spacing[2]) if max_fr4 >= 0 else float('nan')

    np.savez(f'{a.store}/{a.outprefix}/velocity.npz', velocity=velo)
    np.savez(f'{a.store}/{a.outprefix}/efield.npz', efield=efld_vmm)
    np.savez(f'{a.store}/{a.outprefix}/meta.npz',
             Emag_bulk_unit=Ebulk, vmm_per_unit=vmm_per_unit,
             z_surface_mm=z_surface / units.mm, temperature_K=a.temperature)
    print(f'sampled {N} paths x {T} steps ; bulk|E| unit={Ebulk:.4e} (=50 V/mm) ; '
          f'z_surface={z_surface/units.mm:.2f} mm')
    print(f'wrote {a.outprefix}/velocity.npz (system units), {a.outprefix}/efield.npz (V/mm)')


if __name__ == '__main__':
    import os
    os.makedirs(sys.argv[1] + '/alongpath', exist_ok=True)
    main()
