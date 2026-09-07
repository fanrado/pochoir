#!/usr/bin/env python3
'''
Reference-free smoothness metric for a stitched hybrid near/far solution.

Usage:

    python -m pochoir_Analysis.laplace_residual STORE --key potential/drift3d \
        --interface 19.8 [--coarse-spacing 0.22]

PURPOSE.  We need a measure of how smooth a hybrid solution is that is
computable from the hybrid store ALONE.  Reference-free is the whole
point: at production domain sizes the single-spacing (--hybrid no) solve
is too expensive to run as a comparison, so the metric has to judge the
hybrid output on its own.

The measure is the discrete Laplacian residual.  In free space the true
potential satisfies del^2 phi = 0.  The stitched hybrid array does not:
a gradient kink at the near/far seam and the piecewise-constant gradient
left by the multilinear far upsample both show up as a nonzero residual.

The residual is reported DECOMPOSED PER AXIS -- the x, y and z
second-difference terms separately as well as their sum.  That split is
required, not cosmetic: it separates a longitudinal defect (a kinked
seam, a longitudinal staircase) from a transverse one (the coarse
transverse upsample), and those have different fixes.

UNITS.  The plain stencil sum is reported -- (sum of the 6 neighbours)
minus 6*phi -- and is deliberately NOT divided by h^2.  That makes the
number directly comparable to the fdm solver's own --precision, which is
a maxerr in volts, so the floor has a physical reading: the residual
sits at the solver's own convergence level.  A weighting field is a
dimensionless unit probe, so its residual is dimensionless too and is
labelled accordingly.

KNOWN BLIND SPOT.  This metric measures SMOOTHNESS, not ACCURACY.  A
correct harmonic solution of a slightly WRONG geometry has a residual of
about zero, and this metric will pronounce it clean.  In particular the
coarse grid's pad-area snap (pixelSize 3.5 -> 3.52mm at the 0.22mm
coarse spacing, +1.1% in area) biases the far field smoothly, and a
smooth bias is invisible here by construction.  Bounding that error
needs the --hybrid no comparison, which is a separate step; do not read
a clean residual as evidence that the geometry is right.

This is analysis code.  It reads a store and nothing under pochoir/, and
must never be able to affect either solver.
'''

import argparse
import json
import os
import sys

import numpy

import matplotlib
matplotlib.use("Agg")           # no display on the compute nodes
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm


# Below this interface depth a weighting-field residual is a
# method-confirmation number only -- see label_caveat().
WEIGHTING_INTERFACE_MM = 22.0


def _leaf(key):
    'Trailing path element of a store key: potential/drift3d -> drift3d.'
    return key.rsplit('/', 1)[-1]


def load_array(store, key):
    '''
    Load the single array stored under `key` in store directory `store`.

    pochoir writes one .npz per key holding one array named after the
    key's leaf, so <store>/potential/drift3d.npz holds 'drift3d'.
    '''
    fname = os.path.join(store, key + '.npz')
    if not os.path.exists(fname):
        raise SystemExit(f'{store}: missing {key}.npz')
    dat = numpy.load(fname)
    leaf = _leaf(key)
    if leaf in dat:
        return dat[leaf]
    names = list(dat.keys())
    if len(names) != 1:
        raise SystemExit(f'{fname}: expected one array, found {names}')
    return dat[names[0]]


def load_meta(store, key):
    'Load the sidecar JSON for `key`, or an empty dict if there is none.'
    fname = os.path.join(store, key + '.json')
    if not os.path.exists(fname):
        return dict()
    with open(fname) as fp:
        return json.load(fp)


def companion_keys(store, key):
    '''
    Return (domain_key, boundary_key) for the potential stored at `key`.

    The potential's own sidecar JSON names both -- that is authoritative
    and is used when present, because the boundary leaf does NOT always
    match the potential's: potential/drift3d is solved on boundary/drift.
    Without a sidecar we fall back to the same-leaf key and then to the
    leaf with a trailing '3d' stripped.
    '''
    meta = load_meta(store, key)
    dom = meta.get('domain')
    bnd = meta.get('boundary')

    leaf = _leaf(key)
    guesses = [leaf]
    if leaf.endswith('3d'):
        guesses.append(leaf[:-2])

    def first_present(prefix, given):
        if given:
            return given
        for g in guesses:
            if os.path.exists(os.path.join(store, prefix, g + '.npz')) \
               or os.path.exists(os.path.join(store, prefix, g + '.json')):
                return f'{prefix}/{g}'
        return f'{prefix}/{guesses[-1]}'

    return first_present('domain', dom), first_present('boundary', bnd)


def insulator_key(store, key):
    '''
    Store key of the no-flux insulator mask for `key`, or None.

    The FR4 slab is NOT part of the boundary array.  With
    enableInsulatorFR4 the generator writes it separately, as the
    initial key plus '_insulator', and its cells carry a no-flux
    (Neumann) stencil rather than a Dirichlet value -- so del^2 phi is
    legitimately nonzero on them and the plain 7-point residual is
    meaningless there.  Unmasked, that one slab plane dominates the
    whole volume: on store_pixel_field_no it reads 9.56e+01 V while
    every other plane sits at 1.2e-07.

    The potential's sidecar JSON does not name it (a stitched hybrid
    potential has no 'initial' key at all), so it is looked up by leaf,
    the same way companion_keys resolves its fallbacks.
    '''
    meta = load_meta(store, key)
    leaf = _leaf(key)
    guesses = []
    if meta.get('initial'):
        guesses.append(_leaf(meta['initial']))
    guesses.append(leaf)
    if leaf.endswith('3d'):
        guesses.append(leaf[:-2])
    for g in guesses:
        cand = f'initial/{g}_insulator'
        if os.path.exists(os.path.join(store, cand + '.npz')):
            return cand
    return None


def load_domain(store, domain_key):
    '''
    Return (shape, spacing, origin) as tuples, read from the domain JSON.

    Missing fields fall back to unit spacing at the origin so the metric
    still runs on a store written by an older writer; only the z
    coordinates in the report depend on them.
    '''
    meta = load_meta(store, domain_key)
    shape = tuple(meta.get('shape', ()))
    spacing = tuple(meta.get('spacing', (1.0, 1.0, 1.0)))
    origin = tuple(meta.get('origin', (0.0, 0.0, 0.0)))
    return shape, spacing, origin


def per_axis_residual(phi):
    '''
    Per-axis second differences of `phi` on its interior.

    Returns (dx, dy, dz), each of shape (nx-2, ny-2, nz-2) and aligned
    with phi[1:-1, 1:-1, 1:-1]; the 7-point residual is their sum.

    Each term is the plain stencil sum along that axis -- the two face
    neighbours minus twice the centre -- with NO division by h^2, so the
    total is directly comparable to the solver's --precision in volts.

    Only the interior is computed.  The transverse faces of the drift
    tile are periodic and could be wrapped, but the z faces are real
    Dirichlet planes where the residual is not defined at all, and a
    metric that treats the three axes differently is harder to read than
    one that simply excludes every outermost plane.
    '''
    if phi.ndim != 3:
        raise ValueError(f'expected a 3-D array, got shape {phi.shape}')
    if min(phi.shape) < 3:
        raise ValueError(f'array too small for a 7-point stencil: {phi.shape}')

    c = phi[1:-1, 1:-1, 1:-1]
    dx = phi[2:, 1:-1, 1:-1] + phi[:-2, 1:-1, 1:-1] - 2.0 * c
    dy = phi[1:-1, 2:, 1:-1] + phi[1:-1, :-2, 1:-1] - 2.0 * c
    dz = phi[1:-1, 1:-1, 2:] + phi[1:-1, 1:-1, :-2] - 2.0 * c
    return dx, dy, dz


def stencil_mask(barr):
    '''
    Cells whose 7-point stencil touches an electrode, on the interior.

    `barr` is the store's own boundary array (nonzero = Dirichlet).  A
    cell is excluded if it is itself flagged OR if any of its six face
    neighbours is: del^2 phi is legitimately nonzero there, and the
    stencil of a free cell adjacent to an electrode reaches into it.
    That covers the pads, the FR4 insulator and the cathode.

    Returns a boolean array shaped like the interior, True = EXCLUDE.
    '''
    b = numpy.asarray(barr) != 0
    if b.ndim != 3:
        raise ValueError(f'expected a 3-D boundary array, got {b.shape}')

    m = b[1:-1, 1:-1, 1:-1].copy()
    m |= b[2:, 1:-1, 1:-1]
    m |= b[:-2, 1:-1, 1:-1]
    m |= b[1:-1, 2:, 1:-1]
    m |= b[1:-1, :-2, 1:-1]
    m |= b[1:-1, 1:-1, 2:]
    m |= b[1:-1, 1:-1, :-2]
    return m


def plane_stats(terms, excluded):
    '''
    Per-z-plane max |.| and RMS of each term in `terms`, over unmasked cells.

    `terms` is a dict name -> interior-shaped array; `excluded` is the
    boolean mask from stencil_mask (True = exclude).  Returns a dict with
    'count' (unmasked cells per plane) and, per name, 'max' and 'rms'
    arrays of length nz-2.  Planes with no unmasked cell get 0.0 rather
    than a nan, so the profile stays plottable.
    '''
    keep = ~excluded
    count = keep.sum(axis=(0, 1))
    safe = numpy.maximum(count, 1)

    out = dict(count=count)
    for name, arr in terms.items():
        a = numpy.where(keep, arr, 0.0)
        out[name] = dict(
            max=numpy.abs(a).max(axis=(0, 1)),
            rms=numpy.sqrt((a * a).sum(axis=(0, 1)) / safe),
        )
    return out


def is_weighting(key):
    'True when `key` names a weighting field rather than a drift potential.'
    return 'weight' in _leaf(key).lower()


def units_label(key):
    'Unit name for the residual of `key`.'
    return 'dimensionless' if is_weighting(key) else 'volts'


def label_caveat(key, interface_mm):
    '''
    Warning text when this is a method-confirmation number only, else None.

    A weighting probe is non-periodic with a 22mm transverse scale and
    still carries real transverse structure at a 19.8mm seam -- the
    far-tail notes have centre and edge converging only past ~22mm -- so
    its residual at such an interface is EXPECTED to be worse than at a
    4cm interface.  It must not be read as a regression or quoted as
    production seam quality.
    '''
    if not is_weighting(key) or interface_mm is None:
        return None
    if interface_mm > WEIGHTING_INTERFACE_MM:
        return None
    return (
        f'METHOD-CONFIRMATION NUMBER ONLY: this is a weighting field with the\n'
        f'near/far interface at {interface_mm}mm, at or below the ~'
        f'{WEIGHTING_INTERFACE_MM}mm transverse\n'
        'scale of the probe itself.  The weighting probe is non-periodic and\n'
        'still carries real transverse structure at that seam (centre and edge\n'
        'converge only past ~22mm), so this residual is EXPECTED to be worse\n'
        'than at a 4cm interface.  Do NOT read it as a regression and do NOT\n'
        'quote it as production seam quality.')


def analyse(store, key, interface_mm=None, coarse_spacing=None,
            slice_iy=None):
    '''
    Load `key` from `store` and return everything the report needs.

    Returns a dict with the per-plane statistics, the z coordinate of
    each reported plane, the mask accounting, the resolved companion keys
    and the single x-z slice the heatmap needs.  Only that slice is kept:
    the full interior residual arrays are several times the size of the
    potential itself and are dropped as soon as the statistics are taken.
    '''
    phi = load_array(store, key)
    domain_key, boundary_key = companion_keys(store, key)
    shape, spacing, origin = load_domain(store, domain_key)
    barr = load_array(store, boundary_key)

    if barr.shape != phi.shape:
        raise SystemExit(
            f'{store}: {boundary_key} shape {barr.shape} does not match '
            f'{key} shape {phi.shape}')
    if shape and tuple(shape) != phi.shape:
        raise SystemExit(
            f'{store}: {domain_key} shape {tuple(shape)} does not match '
            f'{key} shape {phi.shape}')

    # The insulator slab is excluded exactly like an electrode: its
    # cells are not Dirichlet, but they are not plain-Laplace either.
    ins_key = insulator_key(store, key)
    n_ins = 0
    solid = numpy.asarray(barr) != 0
    if ins_key is not None:
        ins = numpy.asarray(load_array(store, ins_key)) != 0
        if ins.shape != phi.shape:
            raise SystemExit(
                f'{store}: {ins_key} shape {ins.shape} does not match '
                f'{key} shape {phi.shape}')
        n_ins = int((ins & ~solid).sum())
        solid = solid | ins

    dx, dy, dz = per_axis_residual(phi)
    terms = dict(total=dx + dy + dz, x=dx, y=dy, z=dz)
    excluded = stencil_mask(solid)
    stats = plane_stats(terms, excluded)

    # Keep one x-z slice through the pad centre for the heatmap, then let
    # the full-size residual arrays go.
    iy = _slice_index(phi.shape, slice_iy)
    slice_total = numpy.array(terms['total'][:, iy - 1, :])
    slice_mask = numpy.array(excluded[:, iy - 1, :])
    del terms, dx, dy, dz

    # Interior planes are phi's z indices 1 .. nz-2.
    zindex = numpy.arange(1, phi.shape[2] - 1)
    zmm = origin[2] + zindex * spacing[2]

    interior = excluded.size
    seam_index = None
    if interface_mm is not None and spacing[2]:
        seam_index = int(round((interface_mm - origin[2]) / spacing[2]))

    return dict(
        store=store, key=key, shape=phi.shape,
        domain_key=domain_key, boundary_key=boundary_key,
        spacing=spacing, origin=origin,
        stats=stats, zindex=zindex, zmm=zmm,
        interior=interior, masked=int(excluded.sum()),
        electrodes=int((numpy.asarray(barr) != 0).sum()),
        insulator_key=ins_key, insulator=n_ins,
        slice_iy=iy, slice_total=slice_total, slice_mask=slice_mask,
        slice_x_mm=origin[0] + numpy.arange(1, phi.shape[0] - 1) * spacing[0],
        interface_mm=interface_mm, seam_index=seam_index,
        coarse_spacing=coarse_spacing,
        units=units_label(key), caveat=label_caveat(key, interface_mm),
    )


def report(res, stride=1, out=None):
    '''
    Print the per-plane table and its header to `out` (default stdout).

    The default is resolved HERE rather than in the signature: bound at
    import time it would capture the real stdout, and redirect_stdout or
    pytest's capsys -- which replace sys.stdout afterwards -- could not
    see the output.
    '''
    if out is None:
        out = sys.stdout
    p = lambda *a: print(*a, file=out)

    p(f'store      : {res["store"]}')
    p(f'key        : {res["key"]}  shape {res["shape"]}')
    p(f'domain     : {res["domain_key"]}  spacing {res["spacing"]} '
      f'origin {res["origin"]}')
    p(f'boundary   : {res["boundary_key"]}  '
      f'{res["electrodes"]} flagged cells')
    if res.get('insulator_key'):
        p(f'insulator  : {res["insulator_key"]}  '
          f'{res["insulator"]} no-flux cells, excluded like electrodes')
    else:
        p('insulator  : none found -- if this field was solved with '
          '--insulator, the slab is NOT masked')
    p(f'residual   : 7-point stencil sum, NOT divided by h^2; units '
      f'{res["units"]}')
    kept = res['interior'] - res['masked']
    p(f'masking    : {res["masked"]} of {res["interior"]} interior cells '
      f'excluded ({100.0*res["masked"]/max(res["interior"],1):.2f}%), '
      f'{kept} kept')
    if res['interface_mm'] is not None:
        p(f'interface  : {res["interface_mm"]}mm -> fine z index '
          f'{res["seam_index"]}'
          + (f', coarse spacing {res["coarse_spacing"]}mm'
             if res['coarse_spacing'] else ''))
    if res['caveat']:
        p('')
        p(res['caveat'])
    p('')

    st = res['stats']
    names = ('total', 'x', 'y', 'z')
    head = f'{"iz":>6} {"z/mm":>9} {"ncells":>8}'
    for n in names:
        head += f' {n+"_max":>12} {n+"_rms":>12}'
    p(head)

    for j in range(0, len(res['zindex']), stride):
        row = (f'{res["zindex"][j]:6d} {res["zmm"][j]:9.3f} '
               f'{st["count"][j]:8d}')
        for n in names:
            row += f' {st[n]["max"][j]:12.4e} {st[n]["rms"][j]:12.4e}'
        p(row)

    p('')
    check = transverse_far_check(res)
    if check:
        p(check)
    for n in names:
        p(f'{n:>5}: max over all planes {st[n]["max"].max():.4e}, '
          f'largest plane RMS {st[n]["rms"].max():.4e}')


def far_planes(zmm, interface_mm):
    '''
    Boolean selector for the planes ABOVE the near/far interface.

    These are the planes whose values came from the upsampled coarse
    solution, i.e. where the far grid's own artifacts live.  With no
    interface given, nothing is selected.
    '''
    if interface_mm is None:
        return numpy.zeros(len(zmm), dtype=bool)
    return numpy.asarray(zmm) > interface_mm


def coarse_phase_fold(zmm, values, coarse_spacing, nbins=11):
    '''
    Fold `values` on coarse-cell phase: bin by (z mod coarse) / coarse.

    Returns (centres, means, counts) with `nbins` bins over phase in
    [0,1), or None when there is nothing to fold.

    WHY FOLDING AND NOT AN INTEGER STRIDE.  The coarse:fine ratio here is
    0.22/0.1 = 2.2, NOT an integer, so coarse nodes coincide with fine
    nodes only every 1.1mm and the upsample's piecewise-constant-gradient
    ripple is NON-COMMENSURATE with the fine grid -- a beat pattern, not
    a fixed 4-cell period.  Sampling every Nth plane would work at a 4:1
    ratio and silently find nothing here.  Phase folding is
    ratio-agnostic: a real coarse-cell ripple concentrates in phase and
    shows up as structure across the bins, however the two grids beat.
    '''
    z = numpy.asarray(zmm, dtype=float)
    v = numpy.asarray(values, dtype=float)
    if coarse_spacing is None or coarse_spacing <= 0 or z.size == 0:
        return None

    phase = numpy.mod(z, coarse_spacing) / coarse_spacing
    edges = numpy.linspace(0.0, 1.0, nbins + 1)
    idx = numpy.clip(numpy.digitize(phase, edges) - 1, 0, nbins - 1)

    counts = numpy.bincount(idx, minlength=nbins)
    sums = numpy.bincount(idx, weights=v, minlength=nbins)
    means = numpy.where(counts > 0, sums / numpy.maximum(counts, 1), 0.0)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres, means, counts


def transverse_far_check(res):
    '''
    Free correctness check on the tool, for the DRIFT field only.

    The drift domain is ONE periodic 4.4mm tile, so transverse structure
    decays as exp(-2*pi*z/pitch) and is utterly negligible far from the
    pads.  A drift field showing transverse residual out there is a bug
    in this tool or in stitch-near -- it is not physics.

    The WEIGHTING field is a different object: it is non-periodic with a
    22mm transverse scale and legitimately DOES carry transverse
    structure at a 19.8mm seam, so this check must not be applied to it
    and is skipped.

    Returns a text line, or None when the check does not apply.
    '''
    if is_weighting(res['key']):
        return None
    far = far_planes(res['zmm'], res['interface_mm'])
    if not far.any():
        return None

    st = res['stats']
    trans = numpy.maximum(st['x']['max'], st['y']['max'])
    far_max = trans[far].max()
    everywhere = trans.max()

    # Compared against the transverse term's own maximum, which sits at
    # the pads.  NOT against the z term: for a harmonic field the three
    # terms sum to ~0, so dz is about -(dx+dy) everywhere and their ratio
    # is ~2 by construction -- it says nothing.  The decay is
    # exp(-2*pi*z/pitch), about e^-28 at a 19.8mm seam on a 4.4mm pitch,
    # so 1e-6 is a very loose bar that only a real defect can fail.
    ratio = far_max / everywhere if everywhere > 0 else 0.0
    verdict = ('as expected' if ratio <= 1e-6 else
               'UNEXPECTEDLY LARGE -- suspect this tool or stitch-near, '
               'not physics')
    return (f'drift transverse check above the seam: max |x|,|y| term '
            f'{far_max:.4e}, {ratio:.2e} of its near-pad maximum '
            f'{everywhere:.4e} -- {verdict}')


def _log_ylim(ax, series, headroom=3.0):
    '''
    Clamp a log y axis to the positive data.

    Planes with no unmasked cell report exactly 0 and are floored to
    1e-30 before plotting; without this the axis would span thirty
    decades of nothing and flatten every real feature.
    '''
    vals = numpy.concatenate([numpy.asarray(s, dtype=float).ravel()
                              for s in series])
    vals = vals[vals > 0]
    if vals.size == 0:
        return
    ax.set_ylim(vals.min() / headroom, vals.max() * headroom)


def _slice_index(shape, given):
    'Transverse index of the pad-centre slice, or `given` when supplied.'
    return shape[1] // 2 if given is None else int(given)


def plot(res, outfile):
    '''
    Write the three-panel figure for `res` to `outfile`.

    PLOT 1, residual vs z: per-plane max and RMS of the total residual on
    a log y axis, with the interface marked.  A gradient kink at the seam
    reads as a spike at the interface plane; the far upsample's
    piecewise-constant gradient reads as ripple through the far region.
    The inset folds that far region on coarse-cell phase (see
    coarse_phase_fold) -- the ratio is non-integer, so a ripple is a beat
    and only shows up folded.

    PLOT 2, x-z slice through the pad centre: the seam appears as a
    horizontal line and the staircase as horizontal banding.  Masked
    cells are left blank rather than drawn as zero, so a masking mistake
    cannot pass for a clean region.  The colour scale is symmetric log.

    PLOT 3, per-axis curves: the x, y and z terms versus z on one set of
    axes.  This is what separates "the seam is kinked" (longitudinal)
    from "the transverse upsample is coarse".
    '''
    st = res['stats']
    zmm = res['zmm']
    iface = res['interface_mm']

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 12))

    # ---- PLOT 1: residual vs z --------------------------------------
    ax1.semilogy(zmm, numpy.maximum(st['total']['max'], 1e-30),
                 label='max over plane')
    ax1.semilogy(zmm, numpy.maximum(st['total']['rms'], 1e-30),
                 label='RMS over plane')
    if iface is not None:
        ax1.axvline(iface, color='k', ls='--', lw=1,
                    label=f'interface {iface}mm')
    ax1.set_xlabel('z / mm')
    ax1.set_ylabel(f'|residual| / {res["units"]}')
    ax1.set_title(f'{res["key"]} residual vs z -- {res["store"]}')
    ax1.legend(fontsize='small', loc='upper left')
    ax1.grid(alpha=0.3)
    _log_ylim(ax1, (st['total']['max'], st['total']['rms']))

    far = far_planes(zmm, iface)
    fold = None
    if far.any():
        fold = coarse_phase_fold(zmm[far], st['total']['rms'][far],
                                 res['coarse_spacing'])
    if fold is not None:
        centres, means, counts = fold
        inset = ax1.inset_axes([0.66, 0.14, 0.32, 0.30])
        inset.plot(centres, means, marker='o', ms=3)
        inset.set_title(f'far region folded on {res["coarse_spacing"]}mm '
                        'phase', fontsize='x-small')
        inset.set_xlabel('coarse-cell phase', fontsize='x-small')
        inset.tick_params(labelsize='xx-small')
        inset.grid(alpha=0.3)

    # ---- PLOT 2: x-z heatmap through the pad centre ------------------
    sl = numpy.ma.masked_array(res['slice_total'], mask=res['slice_mask'])
    finite = numpy.abs(sl.compressed())
    finite = finite[finite > 0]
    linthresh = float(numpy.median(finite)) if finite.size else 1e-12
    vmax = float(numpy.abs(sl).max()) if finite.size else 1.0

    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('0.85')        # masked cells blank, never drawn as zero
    im = ax2.pcolormesh(zmm, res['slice_x_mm'], sl,
                        cmap=cmap, shading='auto',
                        norm=SymLogNorm(linthresh=max(linthresh, 1e-30),
                                        vmin=-vmax, vmax=vmax))
    if iface is not None:
        ax2.axvline(iface, color='k', ls='--', lw=1)
    ax2.set_xlabel('z / mm')
    ax2.set_ylabel('x / mm')
    ax2.set_title(f'total residual, x-z slice at iy={res["slice_iy"]} '
                  '(grey = masked)')
    fig.colorbar(im, ax=ax2, label=f'residual / {res["units"]}')

    # ---- PLOT 3: per-axis curves -------------------------------------
    for name, style in (('x', '-'), ('y', '--'), ('z', '-.')):
        ax3.semilogy(zmm, numpy.maximum(st[name]['rms'], 1e-30), style,
                     label=f'{name} term, plane RMS')
    ax3.semilogy(zmm, numpy.maximum(st['total']['rms'], 1e-30), ':',
                 color='k', label='total, plane RMS')
    if iface is not None:
        ax3.axvline(iface, color='k', ls='--', lw=1)
    ax3.set_xlabel('z / mm')
    ax3.set_ylabel(f'|term| / {res["units"]}')
    ax3.set_title('per-axis second-difference terms')
    ax3.legend(fontsize='small')
    ax3.grid(alpha=0.3)
    _log_ylim(ax3, (st['x']['rms'], st['y']['rms'], st['z']['rms'],
                    st['total']['rms']))

    fig.tight_layout()
    fig.savefig(outfile, dpi=110)
    plt.close(fig)
    return outfile


def main(argv=None):
    ap = argparse.ArgumentParser(
        description='Per-axis discrete Laplacian residual of a solved field.')
    ap.add_argument('store', help='store directory')
    ap.add_argument('--key', default='potential/drift3d',
                    help='store key of the potential (default %(default)s)')
    ap.add_argument('--interface', type=float, default=None,
                    help='near/far interface in mm, for labelling the seam')
    ap.add_argument('--coarse-spacing', type=float, default=None,
                    help='coarse spacing in mm, reported for the record')
    ap.add_argument('-o', '--output', default=None,
                    help='write the three-panel figure here (PNG)')
    ap.add_argument('--slice-y', type=int, default=None,
                    help='transverse index of the heatmap slice '
                         '(default: the pad centre, shape[1]//2)')
    ap.add_argument('--stride', type=int, default=1,
                    help='print every Nth z plane (default %(default)s)')
    args = ap.parse_args(argv)

    res = analyse(args.store, args.key, args.interface, args.coarse_spacing,
                  slice_iy=args.slice_y)
    report(res, stride=args.stride)
    if args.output:
        plot(res, args.output)
        print(f'wrote {args.output}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
