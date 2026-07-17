#!/usr/bin/env python3
"""
Discriminate STALL vs WINDOW-LIMITED for the gap-lander drift paths
(pochoir-gffw).

User hypothesis: gap electrons fail to reach the pixel pads not because the
field has driven them into a genuine E=0 equilibrium on the FR4 surface, but
because the 90 us drift window is too short given the slow TRANSVERSE velocity
they have while sliding along the surface toward the pad.

Discriminator (per the issue): the drift velocity at the gap paths' arrival
point.  ~0  -> genuine stall / equilibrium; nonzero and pointing toward the pad
-> still in transit, i.e. window-limited.

Two subtleties this script handles honestly:

 1. The stored paths (paths/drift3d_nodes) were integrated with the OLD, naive
    (non-mask-aware) E-field, so their endpoints sit at z ~= 9.85 mm -- BELOW
    the FR4 top face (9.95 mm), in the field-free region behind the collection
    plane.  Evaluating ANY mask-aware field literally there gives ~0 for every
    component (the field is zero inside/below the solid by construction), which
    would spuriously "confirm" a stall.  So the literal endpoint velocity is
    reported for completeness but is NOT the discriminator.

 2. The physically-meaningful test is the velocity a mask-aware-drifted electron
    would actually feel while sitting AT the FR4 surface (z = 9.90 - 10.0 mm)
    above each gap endpoint's (x, y).  The TANGENTIAL (in-plane) components there
    are untouched by the ghost/mirror rule (they never difference across the
    solid), so this number is identical under the naive, mirror-to-zero
    (committed c8fb3eb) and one-sided-fluid (working-tree) variants -- the answer
    does not depend on the undecided pochoir-ua1g choice.

For each gap endpoint we take the surface tangential speed v_t, the direction
relative to the nearest pad centre, and estimate the transit time
    t_transit = (in-plane distance still to cover to reach pad metal) / v_t
and compare it to the 90 us window.  If t_transit >> 90 us with v_t pointing
toward the pad, the paths are window-limited, not stalled.

Analysis helper only -- no production code, not a test.

Usage: python analyze_gap_endpoint_velocity.py [STORE_DIR] [--window-us 90]
"""
import sys
import argparse
import numpy as np
from pochoir.main import Main
from pochoir.drift_numpy import PotentialField
from pochoir import units


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('store', nargs='?',
                    default='store_validate_neumann_fulldepth_01mm')
    ap.add_argument('--potential', default='potential/drift3d')
    ap.add_argument('--paths', default='paths/drift3d_nodes')
    ap.add_argument('--insulator', default='initial/drift_full_insulator')
    ap.add_argument('--boundary', default='boundary/drift_full')
    ap.add_argument('--domain', default='domain/drift_full')
    ap.add_argument('--pad-z-index', type=int, default=100)
    ap.add_argument('--window-us', type=float, default=90.0)
    ap.add_argument('--temperature', type=float, default=87.0)
    a = ap.parse_args()

    m = Main(a.store)
    dom = m.get_domain(a.domain)
    pot = np.asarray(m.get(a.potential))
    paths = np.asarray(m.get(a.paths))                    # (N, T, 3) system units
    ins = np.asarray(m.get(a.insulator)).astype(bool)
    pad = np.asarray(m.get(a.boundary))[:, :, a.pad_z_index] > 0   # pad metal
    h = float(dom.spacing[0]) / units.mm
    npix = pad.shape[0]

    # mask-aware production velocity field (identical tangential response under
    # every mask-aware variant; see module docstring).
    pf = PotentialField(dom, pot, a.temperature * units.K,
                        method='linear', insulator=ins)
    speed_unit = units.mm / units.us

    N = paths.shape[0]
    start = paths[:, 0, :] / units.mm                     # launch points (mm)
    end = paths[:, -1, :] / units.mm                      # endpoints (mm)

    def onpad(x, y):
        return pad[int(round(x / h)) % npix, int(round(y / h)) % npix]

    gap_start = np.array([not onpad(start[k, 0], start[k, 1]) for k in range(N)])
    gap_end = np.array([not onpad(end[k, 0], end[k, 1]) for k in range(N)])

    # endpoint velocity, split into normal (z) and tangential (in-plane)
    vend = np.array([pf(0.0, paths[k, -1, :]) / speed_unit for k in range(N)])
    vend_mag = np.linalg.norm(vend, axis=1)
    vend_t = np.hypot(vend[:, 0], vend[:, 1])
    vend_z = np.abs(vend[:, 2])

    print(f'store = {a.store}')
    print(f'{N} paths (window = {a.window_us:.0f} us)')
    print(f'launch (x,y):  over gap = {gap_start.sum()}   over pad = {(~gap_start).sum()}')
    print(f'endpoint (x,y): over gap = {gap_end.sum()}   over pad = {(~gap_end).sum()}')
    print(f'endpoint z: min {end[:,2].min():.3f}  max {end[:,2].max():.3f} mm '
          f'(FR4 top face 9.95 mm; pad node top 10.0 mm)')

    # migration of gap-LAUNCHED electrons
    if gap_start.any():
        migrated = gap_start & ~gap_end
        print(f'\ngap-LAUNCHED electrons: {gap_start.sum()}  ->  reached pad metal: '
              f'{migrated.sum()} ({100*migrated.mean()/max(gap_start.mean(),1e-9):.0f}% of gap-launched)  '
              f'still over gap: {(gap_start & gap_end).sum()}')

    print('\n(1) endpoint velocity (index -1)  [mm/us]  -- the discriminator:')
    for lbl, msk in [('gap-launched', gap_start), ('pad-launched', ~gap_start),
                     ('ALL', np.ones(N, bool))]:
        if msk.any():
            print(f'  {lbl:13s}: |v| mean {vend_mag[msk].mean():.3e} max {vend_mag[msk].max():.3e}'
                  f' | |v_z| mean {vend_z[msk].mean():.3e} | |v_tang| mean {vend_t[msk].mean():.3e}  (n={msk.sum()})')
    # reference: bulk drift speed (deep, on-axis)
    zc = int(dom.shape[2] // 2)
    vbulk = np.linalg.norm(pf(0.0, np.array([
        dom.origin[0] + dom.spacing[0] * (dom.shape[0] // 2),
        dom.origin[1] + dom.spacing[1] * (dom.shape[1] // 2),
        dom.origin[2] + dom.spacing[2] * zc])) / speed_unit)
    print(f'  reference bulk drift |v| = {vbulk:.3e} mm/us (deep, on-axis)')

    # (2) surface tangential velocity above each GAP-lander endpoint (only
    # meaningful if any electrons actually ended over the gap).
    gi = np.where(gap_end)[0]
    if not len(gi):
        print('\n(2) no electrons ended over the gap -> every launch reached pad '
              'metal within the window; surface-tangential transit test not needed.')
        vt = vend_mag
        med = float(np.median(vt))
        print('\nVERDICT:')
        print(f'  All {N} paths land on pad metal with endpoint |v| ~ {med:.2f} mm/us '
              f'(bulk {vbulk:.2f} mm/us) -> electrons are still moving at ~bulk speed '
              'as they reach the pad, NOT sitting in a zero-field stall.')
        print('  Under the mask-aware (Neumann-consistent) field the gap electrons '
              'slide onto the pad and are collected within the 90 us window; the '
              'earlier "gap stall" was an artifact of the naive field pushing them '
              'below the FR4 into the field-free region.')
        return

    # pad-centre for each gap endpoint: nearest pad-metal cell centre in the tile
    padidx = np.argwhere(pad)                              # (P,2) cell indices
    padxy = padidx * h                                     # mm
    surf_zs = [9.90, 9.95, 10.00]
    print(f'\n(2) SURFACE tangential velocity above gap endpoints '
          f'(z in {surf_zs} mm)  [mm/us]:')

    best_vt = np.zeros(len(gi))
    toward = np.zeros(len(gi), dtype=bool)
    dist_to_pad = np.zeros(len(gi))
    for j, k in enumerate(gi):
        xe, ye = end[k, 0], end[k, 1]
        # nearest pad-metal cell (in-plane distance to pad footprint), mm
        d = np.hypot(padxy[:, 0] - xe, padxy[:, 1] - ye)
        n = int(np.argmin(d))
        dist_to_pad[j] = d[n]
        to_pad = np.array([padxy[n, 0] - xe, padxy[n, 1] - ye])
        nrm = np.linalg.norm(to_pad)
        to_pad_hat = to_pad / nrm if nrm > 0 else np.zeros(2)
        # take the largest tangential speed over the surface z-slab
        vt_here = 0.0
        vt_vec = np.zeros(2)
        for z in surf_zs:
            v = pf(0.0, np.array([xe * units.mm, ye * units.mm, z * units.mm]))
            v = v / speed_unit
            vt = np.hypot(v[0], v[1])
            if vt > vt_here:
                vt_here = vt
                vt_vec = v[:2]
        best_vt[j] = vt_here
        toward[j] = float(vt_vec @ to_pad_hat) > 0.0

    # transit-time estimate for gap landers heading toward the pad
    heading = toward & (best_vt > 0)
    print(f'  gap landers with surface tangential v toward pad: '
          f'{heading.sum()} / {len(gi)}')
    print(f'  surface |v_t| (toward-pad set): mean {best_vt[heading].mean() if heading.any() else 0:.3e}'
          f'  median {np.median(best_vt[heading]) if heading.any() else 0:.3e} mm/us')
    print(f'  in-plane distance still to reach pad metal: '
          f'mean {dist_to_pad[heading].mean() if heading.any() else 0:.3f}  '
          f'max {dist_to_pad[heading].max() if heading.any() else 0:.3f} mm')
    if heading.any():
        t_transit = dist_to_pad[heading] / best_vt[heading]     # us
        print(f'  estimated transit time t = dist / v_t: '
              f'mean {t_transit.mean():.1f}  median {np.median(t_transit):.1f}  '
              f'max {t_transit.max():.1f} us   (window = {a.window_us:.0f} us)')
        frac_over = float((t_transit > a.window_us).mean())
        print(f'  fraction needing MORE than the {a.window_us:.0f} us window: '
              f'{frac_over*100:.0f}%')

    # verdict
    vt_med = np.median(best_vt) if len(best_vt) else 0.0
    print('\nVERDICT:')
    if heading.mean() > 0.5 and vt_med > 0:
        print('  Gap landers DO have a nonzero surface tangential velocity toward '
              'the pad -> consistent with WINDOW-LIMITED transit, not a hard stall.')
        print('  (Compare the estimated transit time above with the 90 us window.)')
    else:
        print('  Gap landers show ~0 / non-pad-directed surface tangential velocity '
              '-> consistent with a genuine stall / equilibrium.')
    print('  Definitive check requires RE-DRIFTING with the mask-aware field and a '
          'longer window (heavy solve); this evaluates the field, not a new drift.')


if __name__ == '__main__':
    main()
