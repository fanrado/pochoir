#!/usr/bin/env python3
'''
Compare the induced-charge result of two pochoir stores.

Usage:

    python -m pochoir_Analysis.compare_stores NEW_STORE REF_STORE [-o out.png]

Both arguments are store directories holding Charge_Q.npy,
endpoints.npy and shifted_paths.npy, as written by the induce-pixel
step (e.g. scripts/store_spacing_0pt1_2cmdrift).

This is the validation tool for the weighting-field edge-condition fix
(see issue pochoir-ziqz).  The reference runs solve the weighting field
with reflecting (Neumann mirror) transverse walls, so W never decays
laterally and the drifting electron induces charge from the very first
tick -- 0.02% of the final Q already accrued at z=120mm and 0.21% at
z=88mm in the 0.1mm run, with the electron still ~10cm from the pad.  A
run whose weighting solve uses genuinely fixed walls should instead be
flat at tick 0.  The "fraction of final Q already accrued" numbers
printed below are exactly that metric.
'''

import argparse
import os
import sys

import numpy

import matplotlib
matplotlib.use("Agg")           # no display on the compute nodes
import matplotlib.pyplot as plt


# Column 2 of shifted_paths is the drift coordinate: it starts at the
# cathode (largest z) and decreases toward the pad plane.
ZCOL = 2

DEFAULT_ZS = (150.0, 120.0, 88.0)


def load_store(path):
    '''
    Return (charge, endpoints, paths) read from a store directory.

    charge    (npaths, nticks)      induced charge, cumulative in tick
    endpoints (npaths, 3)           where each path stopped
    paths     (npaths, nticks, 3)   the tiled drift paths
    '''
    def one(name):
        fname = os.path.join(path, name)
        if not os.path.exists(fname):
            raise SystemExit(f'{path}: missing {name}')
        return numpy.load(fname)

    charge = one('Charge_Q.npy')
    endpoints = one('endpoints.npy')
    paths = one('shifted_paths.npy')

    if charge.ndim != 2:
        raise SystemExit(f'{path}: Charge_Q.npy is {charge.shape}, want (npaths, nticks)')
    if paths.shape[:2] != charge.shape:
        raise SystemExit(f'{path}: shifted_paths {paths.shape} does not match '
                         f'Charge_Q {charge.shape}')
    return charge, endpoints, paths


def collecting(charge):
    '''
    Return the boolean mask of paths that actually land on the pad.

    Most paths in a tiled run end on a neighbouring pixel and finish at
    Q=0; "fraction of final Q" is undefined for those, so every metric
    below is computed over this subset only.
    '''
    return charge[:, -1] != 0.0


def norms(charge):
    '''
    Return (final, peak) per-path normalisations.

    Q(t) is not guaranteed monotonic: a path that approaches the pad and
    then swings away leaves a Q that peaks above where it settles.  The
    metric this tool exists for is stated against the FINAL Q, so that
    is what is reported first, but a fraction above 1 would then be a
    confusing way to say "non-monotonic", so the peak-normalised value
    is reported alongside it.
    '''
    return charge[:, -1], numpy.abs(charge).max(axis=1)


def accrued_fraction_at_tick(charge, tick):
    '''
    Return (frac_of_final, frac_of_peak, mask) at a tick, over the
    collecting paths only.
    '''
    final, peak = norms(charge)
    good = collecting(charge)
    ffin = numpy.zeros(charge.shape[0], dtype=float)
    fpk = numpy.zeros(charge.shape[0], dtype=float)
    ffin[good] = charge[good, tick] / final[good]
    nz = good & (peak != 0.0)
    fpk[nz] = charge[nz, tick] / peak[nz]
    return ffin, fpk, good


def accrued_fraction_at_z(charge, paths, z):
    '''
    Return (of_final, of_peak, npaths_used) fraction of Q accrued by the
    time each path has fallen to drift coordinate z, where each of the
    first two is a (mean, max) pair over the collecting paths.

    Paths that never reach z (they start below it, or stop above it)
    are skipped; npaths_used says how many contributed.  None is
    returned when no path reaches z at all -- e.g. asking for z=150mm of
    a 2cm-drift store.
    '''
    zs = paths[..., ZCOL]
    final, peak = norms(charge)

    ffin = []
    fpk = []
    for ipath in range(charge.shape[0]):
        if final[ipath] == 0.0:
            continue
        below = numpy.nonzero(zs[ipath] <= z)[0]
        if below.size == 0 or below[0] == 0:
            # never gets down to z, or already starts at/below it
            continue
        q = charge[ipath, below[0]]
        ffin.append(q / final[ipath])
        if peak[ipath] != 0.0:
            fpk.append(q / peak[ipath])

    if not ffin:
        return None
    ffin = numpy.array(ffin)
    fpk = numpy.array(fpk) if fpk else numpy.zeros(1)
    return ((float(ffin.mean()), float(ffin.max())),
            (float(fpk.mean()), float(fpk.max())),
            ffin.size)


def endpoint_spread(endpoints):
    '''
    Return (rms_radius, max_radius) of the transverse endpoint scatter
    about the mean landing point, plus the mean landing point itself.
    '''
    xy = endpoints[:, :2]
    centre = xy.mean(axis=0)
    r = numpy.linalg.norm(xy - centre, axis=1)
    return float(numpy.sqrt((r**2).mean())), float(r.max()), centre


def report(label, path, charge, endpoints, paths, zlist):
    '''
    Print the metrics for one store.
    '''
    npaths, nticks = charge.shape
    zs = paths[..., ZCOL]

    print(f'--- {label}: {path}')
    print(f'    paths {npaths}, ticks {nticks}, '
          f'drift z {zs.min():.3f} .. {zs.max():.3f} mm')

    final, peak = norms(charge)
    good = collecting(charge)
    print(f'    final Q      mean {final.mean():.6g}  '
          f'std {final.std():.3g}  min {final.min():.6g}  max {final.max():.6g}')
    print(f'    collecting paths (final Q != 0): {int(good.sum())}/{npaths}; '
          'all fractions below are over these only')

    nonmono = good & (peak > numpy.abs(final) * (1.0 + 1e-9))
    if nonmono.any():
        worst = float((peak[nonmono] / numpy.abs(final[nonmono])).max())
        print(f'    NOTE: Q(t) is non-monotonic on {int(nonmono.sum())} of them '
              f'(peak up to {worst:.3g}x the final value), so "of final" can '
              'exceed 1; read "of peak" for those')

    ffin, fpk, good = accrued_fraction_at_tick(charge, 0)
    print(f'    accrued at tick 0     of final mean {ffin[good].mean():.6e} '
          f'max {ffin[good].max():.6e} | of peak mean {fpk[good].mean():.6e} '
          f'max {fpk[good].max():.6e}')

    for z in zlist:
        got = accrued_fraction_at_z(charge, paths, z)
        if got is None:
            print(f'    accrued at z={z:g}mm     n/a (no path reaches this z)')
            continue
        (fm, fx), (pm, px), n = got
        print(f'    accrued at z={z:g}mm     of final mean {fm:.6e} max {fx:.6e} '
              f'| of peak mean {pm:.6e} max {px:.6e}   ({n} paths)')

    rms, mx, centre = endpoint_spread(endpoints)
    print(f'    endpoint spread  rms {rms:.6g} mm  max {mx:.6g} mm  '
          f'about ({centre[0]:.4f}, {centre[1]:.4f})')
    print()


def zoom_ticks(means, frac=0.05, pad=1.6, minimum=20):
    '''
    Return how many leading ticks the zoom panel should span.

    The interesting part of Q(t) is the leading edge, which occupies a
    small fraction of the tick axis; the panel is useless if it spans
    the whole run.  Take the latest tick at which either curve is still
    below `frac` of its own final value, widen it a little so the rise
    itself is visible rather than clipped at the panel edge, and never
    return less than `minimum` ticks.
    '''
    nticks = max(len(m) for m in means)
    reach = [minimum]
    for m in means:
        final = m[-1]
        if final == 0.0:
            continue
        above = numpy.nonzero(m >= frac * final)[0]
        reach.append(int(above[0]) if above.size else nticks)
    return int(min(nticks, max(reach) * pad))


def plot(outfile, new, ref, new_label, ref_label):
    '''
    Write an overlaid mean-Q(t) plot of the two stores.
    '''
    # NOT sharex: the lower panel exists precisely to show a different
    # (much shorter) tick range than the upper one.
    fig, (ax, axz) = plt.subplots(2, 1, figsize=(8, 8))

    for charge, label, style in ((new, new_label, '-'), (ref, ref_label, '--')):
        mean = charge.mean(axis=0)
        ax.plot(mean, style, label=label)
        axz.plot(mean, style, label=label)

    ax.set_ylabel('mean Q (normalised)')
    ax.set_title('Induced charge vs tick')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # The whole point of the comparison lives in the first few percent of
    # the drift, where a reflecting weighting-field wall makes Q rise
    # early.  Zoom in BOTH axes: with the full 4000-tick x range the
    # panel collapses to a vertical line at x=0 and shows nothing.
    means = [new.mean(axis=0), ref.mean(axis=0)]
    nzoom = zoom_ticks(means)
    axz.set_xlim(0, nzoom)
    axz.set_ylabel('mean Q (zoom on the early rise)')
    axz.set_xlabel(f'tick (first {nzoom} of {new.shape[1]})')
    head = numpy.concatenate([m[:nzoom] for m in means])
    axz.set_ylim(0, max(1e-12, 1.05 * head.max()))
    axz.legend()
    axz.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(outfile, dpi=140)
    plt.close(fig)
    print(f'wrote {outfile}')


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('new_store', help='store directory of the new run')
    ap.add_argument('ref_store', help='store directory of the reference run')
    ap.add_argument('-o', '--output', default='compare_stores.png',
                    help='output plot file (def: %(default)s)')
    ap.add_argument('-z', '--zs', type=float, nargs='*', default=list(DEFAULT_ZS),
                    help='drift coordinates (mm) at which to report the '
                    'accrued fraction (def: %(default)s)')
    ap.add_argument('--new-label', default=None, help='legend label for the new run')
    ap.add_argument('--ref-label', default=None, help='legend label for the reference')
    args = ap.parse_args(argv)

    new_label = args.new_label or os.path.basename(args.new_store.rstrip('/'))
    ref_label = args.ref_label or os.path.basename(args.ref_store.rstrip('/'))

    new = load_store(args.new_store)
    ref = load_store(args.ref_store)

    report('new', args.new_store, *new, args.zs)
    report('reference', args.ref_store, *ref, args.zs)

    plot(args.output, new[0], ref[0], new_label, ref_label)
    return 0


if __name__ == '__main__':
    sys.exit(main())
