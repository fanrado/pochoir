#!/usr/bin/env python3
"""
E-field components at the endpoints of the enforcement-free task8 drift paths
(pochoir-33w8).

For every grid-node drift path this evaluates the drift E-field E = grad(phi) at
the path's arrival point (index -1) using the SAME ``PotentialField.efield`` the
drift integrated, on ``potential/drift3d`` (solved with the no-flux FR4 Neumann
BC).  Paths are split into GAP landers (arrival (x,y) NOT over pad metal -> not
collected) and PAD landers.  It also prints the vertical E-profile above a
representative gap endpoint so the surface field structure is explicit.

FINDING (contradicts the naive "Ez~0, Ex/Ey nonzero" expectation): with
enforcement removed the paths halt at z ~= 9.85 mm, which is BELOW the FR4 top
face -- in the field-free region behind the collection plane -- so at the actual
endpoints ALL components (Ex, Ey, Ez) ~= 0.  That vanishing field is exactly why
the charge stops (v = mu*E -> 0), no clamp needed.  The strong field lives on the
LAr side AT/above the FR4 surface (z = 9.95-10.0 mm): there the NORMAL field is
large (Ez ~ -100..-200 V/mm, field lines terminating perpendicular on the
charged-up surface) and the tangential (funnelling) field is small but nonzero
(|Etrans| ~ 5-10 V/mm).  So Ez is NOT ~0 at the surface and the tangential field
is the sub-dominant component -- the no-flux condition suppresses the field
INSIDE/below the FR4, it does not make Ez vanish on the LAr side.

Analysis helper only -- no production code, not a test.

Usage: python analyze_gap_endpoint_efield.py [STORE_DIR]
"""
import sys
import numpy as np
from pochoir.main import Main
from pochoir.drift_numpy import PotentialField
from pochoir import units


def main():
    store = sys.argv[1] if len(sys.argv) > 1 else 'store_validate_neumann_fulldepth_01mm'
    m = Main(store)
    dom = m.get_domain('domain/drift_full')
    pot = np.asarray(m.get('potential/drift3d'))
    paths = np.asarray(m.get('paths/drift3d_nodes'))          # (N, T, 3) system units
    ins = np.asarray(m.get('initial/drift_full_insulator')).astype(bool)
    pad = np.asarray(m.get('boundary/drift_full'))[:, :, 100] > 0   # pad metal @node 100
    h = float(dom.spacing[0]) / units.mm
    npix = pad.shape[0]

    pf = PotentialField(dom, pot, 87.0 * units.K, method='linear', insulator=ins)

    N = paths.shape[0]
    end = paths[:, -1, :] / units.mm                          # endpoints (mm)

    def onpad(x, y):
        return pad[int(round(x / h)) % npix, int(round(y / h)) % npix]

    E = np.array([pf.efield(paths[k, -1, :]) / units.V for k in range(N)])   # V/mm
    Ex, Ey, Ez = E[:, 0], E[:, 1], E[:, 2]
    Etr = np.hypot(Ex, Ey)
    gap = np.array([not onpad(end[k, 0], end[k, 1]) for k in range(N)])

    print(f'store = {store}')
    print(f'{N} paths ; gap landers (not collected) = {gap.sum()} ; '
          f'pad landers = {(~gap).sum()}')
    print(f'endpoint z: min {end[:,2].min():.3f}  max {end[:,2].max():.3f} mm '
          f'(FR4 cell node 99 = [9.85, 9.95] mm; pad node 100 top = 10.0 mm)')

    for lbl, msk in [('GAP  (not collected)', gap), ('PAD metal', ~gap)]:
        if not msk.any():
            continue
        print(f'\n{lbl}:  n = {msk.sum()}   E components at arrival (index -1)')
        print(f'  |Ex|   mean {np.abs(Ex[msk]).mean():8.3f}  max {np.abs(Ex[msk]).max():8.3f} V/mm')
        print(f'  |Ey|   mean {np.abs(Ey[msk]).mean():8.3f}  max {np.abs(Ey[msk]).max():8.3f} V/mm')
        print(f'  |Ez|   mean {np.abs(Ez[msk]).mean():8.3f}  max {np.abs(Ez[msk]).max():8.3f} V/mm')
        print(f'  |Etr|  mean {Etr[msk].mean():8.3f}  max {Etr[msk].max():8.3f} V/mm')

    # Vertical E-profile above a representative gap endpoint (nearest gap centre).
    gi = np.where(gap)[0]
    if len(gi):
        kk = gi[np.argmin((end[gi, 0] - 2.15) ** 2 + (end[gi, 1] - 2.15) ** 2)]
        xe, ye, ze = end[kk]
        print(f'\nvertical E-profile above a representative gap endpoint '
              f'(x,y) = ({xe:.3f}, {ye:.3f}) mm  [endpoint z = {ze:.3f} mm]:')
        print('   z(mm)     Ex       Ey       Ez     |Etrans|   (V/mm)')
        for z in [9.70, 9.80, 9.85, 9.90, 9.95, 10.00, 10.05]:
            e = pf.efield(np.array([xe, ye, z])) / units.V
            print(f'  {z:6.2f}  {e[0]:8.3f} {e[1]:8.3f} {e[2]:8.3f}   {np.hypot(e[0], e[1]):8.3f}')
        print('\n=> Field vanishes at/below the endpoint (z<=9.85) -> the charge halts there;')
        print('   the strong NORMAL field and the small tangential funnelling field are on the')
        print('   LAr side at the FR4 surface (z=9.95-10.0), not at the halt point.')


if __name__ == '__main__':
    main()
