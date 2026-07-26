#!/usr/bin/env python3
'''
Task13 iterative hybrid near/far drift-field solver (beads pochoir-nm59).

The bash hybrid runners (run-hybrid-30cm-drift.sh and friends) hard-code a fixed
near/far sequence, so the iteration structure is invisible from the script and
the near->far coupling is one-way.  This module drives the same steps from
Python as an explicit outer iteration, reducing bash to supplying configs and
the store path.

Method (near/far separation at z=20mm, axis 2):

  1. coarse full-volume solve at 0.4mm
  2. `refine` the coarse potential onto the 0.05mm near grid
  3. `near-bc` pins the z=20mm plane to the coarse value
  4. near fine solve at 0.05mm
  5. `coarsen` the near solution back to 0.4mm (stride 8)
  6. `stitch-near` the coarsened near field onto the coarse far field
  7. re-impose exact coarse boundary values, then re-solve the FULL coarse
     volume with the stitched array as INITIAL VALUES ONLY -- the near region
     floats freely
  8. converged when max|phi_k - phi_(k-1)| over the whole coarse volume < tol;
     otherwise back to 2

then a final refinement onto the 0.1mm full grid, a re-solve, and velo/starts/
drift on that 0.1mm field.  The 0.05mm near solve exists only to sharpen the
coarse iteration; it is NOT stitched into the final field.

The floating near region in step 7 is what distinguishes this from
`nearfar.schwarz_solve`, which PINS the near region at the interface.  That
module and the `near-far-solve` command are deliberately untouched by this work.

Every step runs the existing click command in-process via `ctx.invoke`, so it
takes the same code path as the bash scripts and shares one store object.
'''

import json
from pathlib import Path

import numpy


def _kk(k):
    '''Iteration suffix: 0 -> "_k00".'''
    return f'_k{k:02d}'


def _have(ctx, key):
    '''True when `key` is already a dataset in the store.'''
    return (ctx.obj.instore_path / (key + '.npz')).exists()


def _want(ctx, targets, thunk, log=None):
    '''
    The resume guard from run-task10b-drift-2cm-gap09-chamf07.sh: skip when
    every target key is already in the store, otherwise run `thunk` and
    hard-fail if a target is still missing.
    '''
    if isinstance(targets, str):
        targets = [targets]
    targets = list(targets)

    if all(_have(ctx, t) for t in targets):
        if log:
            log(f'have {" ".join(targets)}')
        return

    thunk()

    missing = [t for t in targets if not _have(ctx, t)]
    if missing:
        raise RuntimeError(
            f'step did not produce expected output(s): {" ".join(missing)}')
    if log:
        log(f'made {" ".join(targets)}')


# ---------------------------------------------------------------------------
# Field profiles.
#
# Both fields run the SAME hybrid scheme, the same tolerances and the same final
# 0.1mm grid; only the grids, generator, edges and store-key names differ.
# `--field` selects between them.
#
# DRIFT: one 4.4mm PERIODIC pixel tile (per,per,fix), z = 0..60mm (10mm of
# PCB/pad region below the plane + 50mm drift, cathode at z=60mm).  The 8:1
# coarse/near stride is exact on every axis: 88->11, 401->51.
#
# WEIGHTING: 5x5 pixels = 22mm transverse, NON-periodic (fix,fix,fix) so phi_w
# decays to ~0 at the tile edges rather than wrapping -- a unit probe cannot be
# solved on a single periodic tile.  Strides are exact too: 440/8=55, 401->51,
# 220/4=55, 601->151.
#
# KEY NAMESPACING -- this is load-bearing, not cosmetic.  `_want` skips any step
# whose outputs already exist, and BOTH fields must write into the SAME store so
# `induce-pixel` can see `paths/drift3d` and `potential/weight3d` together.  If
# the weighting profile reused the drift key names, every grid and gen step would
# be skipped as "have" and the weighting field would be solved on the DRIFT
# geometry -- silently, with no error.  So weighting keys carry a `w_` prefix on
# the leaf name, and the drift key names must never change (which also keeps an
# already-solved drift store valid).
#
# `potential/drift3d` and `potential/weight3d` keep exactly task10b's names,
# because `velo` and `induce-pixel` are invoked with them from the shell script.
# ---------------------------------------------------------------------------
FIELDS = {
    'drift': dict(
        generator='pcb_drift_pixel_with_grid',
        edges='per,per,fix',
        prefix='',                       # drift keys are unprefixed (unchanged)
        output='potential/drift3d',
        unit='V',                        # drift potential is in volts
        # transverse extent = ONE pixel pitch (a single periodic tile)
        npixels=1,
    ),
    'weighting': dict(
        generator='pcb_pixel_with_grid',
        edges='fix,fix,fix',
        prefix='w_',
        output='potential/weight3d',
        # the weighting potential is a DIMENSIONLESS unit probe in [0,1], so the
        # convergence delta is not in volts -- do not label it 'V'.
        unit='(dimensionless)',
        # transverse extent = Npixels pitches from the config (5x5 = 22mm), so
        # phi_w has room to decay to ~0 at the edges
        npixels=None,                    # None -> take Npixels from the config
    ),
}

# Default spacings, in mm.  These are the validated Task13 values; the user
# supplies them via --coarse-spacing / --fine-spacing / --full-spacing.
DEFAULT_SPACINGS = dict(coarse=0.4, fine=0.05, full=0.1)

# Minimum physical clearance, in mm, between the electron launch node
# (driftZDepth) and the cathode on the domain's last plane.  Deliberately a
# fixed physical length rather than a multiple of the run's spacings: it is the
# margin the validated 59.9mm/60.0mm geometry actually has, and keeping it
# resolution-independent is what lets the cathode stay at 60.0mm for every
# spacing (see _extents).
MIN_CATHODE_CLEARANCE = 0.1

# The four grids and which of the three user spacings each one uses.  `near_coarse`
# shares the COARSE spacing (it is the coarsen target the near solve is strided
# down onto), which is why three spacings cover four grids.
GRID_SPEC = (
    # leaf,          spacing name,  depth
    ('coarse',       'coarse',      'full'),   # coarse far field, full depth
    ('near',         'fine',        'near'),   # fine near field, to the interface
    ('near_coarse',  'coarse',      'near'),   # coarsen target, to the interface
    ('fine01',       'full',        'full'),   # final full-volume grid
)


def _interface_mm(interface):
    '''
    The interface coordinate as plain mm.

    `interface` arrives as a pochoir units string ('20*mm'); pochoir's internal
    length unit is mm, so unitify gives the number directly.
    '''
    from pochoir.util import unitify
    return float(unitify(interface))


def _extents(prof, cfg, coarse_spacing, interface_mm):
    '''
    Physical extents of the problem, in mm, read off the config.

    transverse : pixel pitch (pixelSize + pixelGap), times Npixels for the
                 multi-pixel weighting probe.
    full depth : the cathode plane -- driftZDepth rounded UP to the next whole
                 coarse cell.  driftZDepth is the electron launch node, defined
                 as the last full-velocity node BELOW the cathode, and the
                 cathode sits on the domain's last plane, so the domain must be
                 at least one cell deeper than the launch node.  Rounding to the
                 coarse cell (not adding one FULL cell) keeps the cathode at a
                 fixed PHYSICAL depth: the electrode must not move when the run's
                 resolution changes.  59.9mm -> 60.0mm for every sane spacing.

                 The rounding target is driftZDepth + MIN_CATHODE_CLEARANCE, not
                 driftZDepth itself: math.ceil leaves an ALREADY-EXACT quotient
                 alone, so rounding the launch node directly put the cathode ON
                 it whenever driftZDepth was a whole multiple of the coarse
                 spacing (0.1 -> 599 cells -> 59.9mm, clearance 0; likewise 0.05
                 and 0.02).  Electrons then launch on the cathode's Dirichlet
                 plane instead of the last full-velocity node below it.  Harmless
                 at the validated 0.4 default (149.75 -> 150 -> 60.0mm) but
                 reachable now that --coarse-spacing is user-settable.
    near depth : the near/far interface coordinate.
    '''
    import math

    pitch = cfg['pixelSize'] + cfg['pixelGap']
    npix = prof['npixels']
    if npix is None:
        npix = int(cfg['Npixels'])

    coarse = coarse_spacing
    launch = cfg['driftZDepth']
    ncoarse = math.ceil((launch + MIN_CATHODE_CLEARANCE) / coarse - 1e-9)
    full_depth = ncoarse * coarse

    return dict(transverse=pitch * npix,
                full=full_depth,
                near=interface_mm)


def _cells(extent_mm, spacing_mm, what, closed):
    '''
    Node count spanning `extent_mm` at `spacing_mm`.

    `closed` distinguishes the two conventions pochoir uses:
      * transverse (periodic or not): N = extent/spacing -- the far node is the
        wrap of the near one, so it is not duplicated;
      * z (fixed edges): N = extent/spacing + 1 -- both faces are real planes
        (pad side and cathode side), so the endpoint is counted.

    The division must be exact: a fractional cell count means the geometry and
    the spacing disagree, which would silently shift electrodes onto the wrong
    plane, so raise rather than round.
    '''
    n = extent_mm / spacing_mm
    if abs(n - round(n)) > 1e-9:
        raise ValueError(
            f'{what}: extent {extent_mm}mm is not a whole number of '
            f'{spacing_mm}mm cells ({n:.6f}) -- adjust the spacing or the '
            f'config geometry so they divide exactly')
    return int(round(n)) + (1 if closed else 0)


def _derive_grids(prof, cfg, spacings, interface_mm):
    '''
    Build the four (key, shape, spacing) triples from the config geometry and the
    user's three spacings -- `--domain yes`.

    Reproduces the validated Task13 grids exactly at the default spacings:
      drift     11,11,151@0.4  88,88,401@0.05  11,11,51@0.4   44,44,601@0.1
      weighting 55,55,151@0.4  440,440,401@0.05 55,55,51@0.4  220,220,601@0.1
    '''
    ext = _extents(prof, cfg, spacings['coarse'], interface_mm)
    grids = []
    for leaf, sp_name, depth_name in GRID_SPEC:
        sp = spacings[sp_name]
        nt = _cells(ext['transverse'], sp, f'{leaf} transverse', closed=False)
        nz = _cells(ext[depth_name], sp, f'{leaf} z', closed=True)
        grids.append((f'domain/{prof["prefix"]}{leaf}',
                      f'{nt},{nt},{nz}', f'{sp}*mm'))
    return tuple(grids)


def _manual_grids(prof, shapes, spacings):
    '''
    Build the four triples from shapes the user typed in -- `--domain no`.

    `shapes` maps each leaf ('coarse', 'near', 'near_coarse', 'fine01') to a
    "nx,ny,nz" string.  Every leaf must be supplied: a missing one cannot be
    guessed without falling back to derivation, which would silently mix the two
    modes.
    '''
    missing = [leaf for leaf, _, _ in GRID_SPEC if not shapes.get(leaf)]
    if missing:
        raise ValueError(
            f'--domain no requires an explicit shape for every grid; missing: '
            f'{", ".join(missing)}')

    grids = []
    for leaf, sp_name, _ in GRID_SPEC:
        shape = str(shapes[leaf]).strip()
        parts = shape.split(',')
        if len(parts) != 3 or not all(p.strip().isdigit() for p in parts):
            raise ValueError(
                f'--domain no: shape for {leaf} must be "nx,ny,nz" integers, '
                f'got {shape!r}')
        grids.append((f'domain/{prof["prefix"]}{leaf}',
                      shape, f'{spacings[sp_name]}*mm'))
    return tuple(grids)

# task10b's fdm flags, applied to every solve here for both fields.
ENGINE = 'torch'
EPOCH = 130000000
NEPOCHS = 10


def _profile(field):
    '''Look up a field profile, failing loudly on an unknown name.'''
    try:
        return FIELDS[field]
    except KeyError:
        raise ValueError(
            f'unknown field {field!r}; expected one of {sorted(FIELDS)}')


def _key(prof, taxon, leaf):
    '''
    Store key for `leaf` in `taxon`, namespaced by the field profile.

    e.g. drift -> "potential/near_k00", weighting -> "potential/w_near_k00".
    '''
    return f'{taxon}/{prof["prefix"]}{leaf}'


def _domains(ctx, grids, log):
    '''The four `domain` invocations for this field.'''
    from pochoir.__main__ import domain

    for key, shape, spacing in grids:
        _want(ctx, key,
              lambda key=key, shape=shape, spacing=spacing: ctx.invoke(
                  domain, domain=key, shape=shape, spacing=spacing),
              log)


def _generate(ctx, prof, coarse_config, fine_config, log):
    '''
    `gen` per domain with the matching config: the coarse 0.4mm transcription
    for the coarse grid, the fine config for both the near and final grids.

    `gen` also writes the no-flux insulator mask under "<initial>_insulator";
    --insulator is an enable signal only, the interface itself is derived from
    the Dirichlet geometry by padplane_noflux_geom.
    '''
    from pochoir.__main__ import gen

    for leaf, cfg in (('coarse', coarse_config),
                      ('near', fine_config),
                      ('fine01', fine_config)):
        dom_key = _key(prof, 'domain', leaf)
        init, bnd = _key(prof, 'initial', leaf), _key(prof, 'boundary', leaf)
        _want(ctx, [init, bnd],
              lambda dom_key=dom_key, init=init, bnd=bnd, cfg=cfg: ctx.invoke(
                  gen, generator=prof['generator'], domain=dom_key,
                  initial=init, boundary=bnd, configs=(cfg,)),
              log)


def _solve(ctx, prof, initial, boundary, insulator, potential, increment,
           precision, log):
    '''
    One `fdm` call with task10b's flags.  `multisteps` left at its default
    ("N", which is what task10b's `--multisteps no` selects).  `edges` comes
    from the field profile: per,per,fix for the periodic drift tile,
    fix,fix,fix for the non-periodic multi-pixel weighting probe.
    '''
    from pochoir.__main__ import fdm

    _want(ctx, [potential, increment],
          lambda: ctx.invoke(
              fdm, initial=initial, boundary=boundary, insulator=insulator,
              edges=prof['edges'], engine=ENGINE, precision=precision,
              epoch=EPOCH, nepochs=NEPOCHS,
              potential=potential, increment=increment),
          log)


def _near_interface(prof, grids):
    '''
    The near/far split plane implied by the resolved near grid, in pochoir units.

    `near_bc` derives the pinned plane itself as the near domain's LAST plane
    (top = ndom.shape[axis]-1) and `stitch_near` takes no interface argument, so
    the split is fixed entirely by the near grid's z extent -- NOT by the
    --interface option.
    '''
    from pochoir.util import unitify

    near_key = _key(prof, 'domain', 'near')
    for key, shape, spacing in grids:
        if key != near_key:
            continue
        nz = int(shape.split(',')[2])
        return (nz - 1) * unitify(spacing)
    raise ValueError(f'field profile has no {near_key} entry')


def _outer_iteration(ctx, prof, grids, k, far_potential, interface, precision,
                     log):
    '''
    Plan steps 2-7 for one pass.  `far_potential` is the current full-volume
    coarse field (the coarse seed for k=0, the previous full_k* after that).
    Returns the new full-volume potential key.

    `interface` is checked against the split the near grid actually implies
    rather than being used to place it: the plane comes from the near grid's z
    extent (see `_near_interface`).  A mismatch used to be silently ignored, so
    `--interface 30*mm` ran happily and still split at 20mm.
    '''
    from pochoir.__main__ import refine, near_bc, coarsen, stitch_near
    from pochoir.util import unitify

    want = unitify(interface)
    have = _near_interface(prof, grids)
    if want != have:
        raise ValueError(
            f'--interface {interface} ({want}) disagrees with the split implied '
            f'by {_key(prof, "domain", "near")} ({have}).  The near/far plane is '
            f'set by the near grid z extent in the field profile, not by this '
            f'option; re-shape the near grid to move it.')

    kk = _kk(k)
    near_refined = _key(prof, 'initial', f'near_refined{kk}')
    near_bc_i = _key(prof, 'initial', f'near_bc{kk}')
    near_bc_b = _key(prof, 'boundary', f'near_bc{kk}')
    near_pot = _key(prof, 'potential', f'near{kk}')
    near_inc = _key(prof, 'increment', f'near{kk}')
    near_coarse = _key(prof, 'potential', f'near_coarse{kk}')
    stitched = _key(prof, 'potential', f'stitched{kk}')
    full_init = _key(prof, 'initial', f'full{kk}')
    full_pot = _key(prof, 'potential', f'full{kk}')
    full_inc = _key(prof, 'increment', f'full{kk}')

    d_near = _key(prof, 'domain', 'near_coarse')
    d_coarse = _key(prof, 'domain', 'coarse')
    i_near, b_near = _key(prof, 'initial', 'near'), _key(prof, 'boundary', 'near')
    i_coarse = _key(prof, 'initial', 'coarse')
    b_coarse = _key(prof, 'boundary', 'coarse')

    # step 2: upsample the current far field onto the 0.05mm near grid
    # (refine also re-imposes the exact fine boundary values).
    _want(ctx, near_refined,
          lambda: ctx.invoke(refine, coarse=far_potential,
                             initial=i_near, boundary=b_near,
                             output=near_refined), log)

    # step 3: pin the z=interface plane Dirichlet from the far field.
    _want(ctx, [near_bc_i, near_bc_b],
          lambda: ctx.invoke(near_bc, initial=near_refined,
                             boundary=b_near, coarse=far_potential,
                             initial_out=near_bc_i, boundary_out=near_bc_b,
                             axis=2), log)

    # step 4: near fine solve, no-flux FR4 BC active.
    _solve(ctx, prof, near_bc_i, near_bc_b, i_near + '_insulator',
           near_pot, near_inc, precision, log)

    # step 5: coarsen the near solution back to 0.4mm (exact stride 8).
    _want(ctx, near_coarse,
          lambda: ctx.invoke(coarsen, input_=near_pot,
                             domain=d_near,
                             output=near_coarse), log)

    # step 6: stitch the coarsened near field onto the coarse far field.
    _want(ctx, stitched,
          lambda: ctx.invoke(stitch_near, near=near_coarse,
                             coarse=far_potential, domain=d_coarse,
                             output=stitched, axis=2), log)

    # step 7a: stitch-near emits a raw potential whose conductor cells now hold
    # coarsened near-field values.  Re-run refine on the SAME domain -- the
    # interpolation is the identity there and it merges the true boundary values
    # back in (refined[bmask] = fi[bmask]) -- rather than writing new code.
    _want(ctx, full_init,
          lambda: ctx.invoke(refine, coarse=stitched,
                             initial=i_coarse,
                             boundary=b_coarse,
                             output=full_init), log)

    # step 7b: full-volume coarse re-solve.  The stitched array is INITIAL
    # VALUES ONLY -- the near region floats freely, which is the whole point of
    # this scheme versus a pinned Schwarz sweep.
    _solve(ctx, prof, full_init, b_coarse, i_coarse + '_insulator',
           full_pot, full_inc, precision, log)

    return full_pot


def _max_abs_delta(ctx, key_a, key_b):
    '''max|a - b| over the whole volume, in volts.'''
    a = numpy.asarray(ctx.obj.get(key_a), dtype=float)
    b = numpy.asarray(ctx.obj.get(key_b), dtype=float)
    return float(numpy.max(numpy.abs(a - b)))


def _final_stage(ctx, prof, converged, precision, log):
    '''
    Final refinement of the converged coarse field onto the 0.1mm full grid,
    then one solve there -> the profile's output key (potential/drift3d for the
    drift field, potential/weight3d for the weighting field -- exactly task10b's
    names, because `velo` and `induce-pixel` are invoked with them from the
    shell script).

    This is where the driver STOPS.  velo / starts / drift / induce-pixel are
    deliberately NOT run here: they live in test/run-task13-hybrid.sh as explicit
    `pochoir` invocations, copied from task10b, so their parameters (temperature,
    starts mode/config, --interp-order, the drift time window, --npixels, --plot)
    can be supplied and tuned by hand on the command line instead of being frozen
    in Python.

    The enforcement-free contract therefore also lives in the shell script:
    velo gets neither --boundary nor --insulator (velocity is pure mu*grad(phi))
    and drift gets no --insulator (paths are never clamped or terminated at a
    surface).  --insulator belongs only to the fdm solves this module runs.
    '''
    from pochoir.__main__ import refine

    seed = _key(prof, 'initial', 'fine01_seed')
    i_fine, b_fine = _key(prof, 'initial', 'fine01'), _key(prof, 'boundary', 'fine01')

    # 0.4mm -> 0.1mm upsample + exact boundary values in one call.
    _want(ctx, seed,
          lambda: ctx.invoke(refine, coarse=converged,
                             initial=i_fine,
                             boundary=b_fine,
                             output=seed), log)

    _solve(ctx, prof, seed, b_fine, i_fine + '_insulator',
           prof['output'], _key(prof, 'increment', 'fine01'), precision, log)


def hybrid_iterate(ctx, coarse_config, fine_config, interface='20*mm',
                   tol=2e-8, max_iters=20, field='drift',
                   coarse_spacing=None, fine_spacing=None, full_spacing=None,
                   derive_domain=True, shapes=None, log=None):
    '''
    Drive the Task13 hybrid FIELD solve: grids, gens, the outer iteration and
    the final 0.1mm refine+solve -> the field's output key.

    `field` selects the profile in FIELDS: 'drift' (periodic single pixel tile ->
    potential/drift3d) or 'weighting' (non-periodic 5x5 unit probe ->
    potential/weight3d).  Both run the identical scheme, tolerances and final
    0.1mm grid; only grids, generator, edges and store-key names differ.

    GRIDS.  The three spacings are always supplied by the caller
    (`coarse_spacing` / `fine_spacing` / `full_spacing`, in mm, defaulting to the
    validated 0.4 / 0.05 / 0.1).  `derive_domain` then chooses where the four grid
    SHAPES come from:

      * True  (`--domain yes`) -- computed from the config geometry: transverse
        extent = (pixelSize + pixelGap) x Npixels-for-this-field, full depth =
        driftZDepth + one full cell, near depth = the interface.  At the default
        spacings this reproduces the validated shapes exactly.
      * False (`--domain no`)  -- taken from `shapes`, a dict of "nx,ny,nz"
        strings keyed by leaf ('coarse', 'near', 'near_coarse', 'fine01').  All
        four are required.

    The chain after the field (velo / starts / drift / induce-pixel) is NOT part
    of this driver -- see `_final_stage`.  It runs as explicit `pochoir` commands
    in test/run-task13-hybrid.sh so its parameters stay hand-settable.

    Convergence is on max|phi_k - phi_(k-1)| over the whole coarse volume.  Two
    guards besides `tol`:

      * iteration cap -- 2e-8 V absolute against a -2500 V bulk is ~1e-11
        relative, near the float64 noise floor of a 151-deep sweep, so stop at
        `max_iters` and report the achieved delta rather than spin;
      * stagnation -- if the delta fails to improve by more than 10% for two
        consecutive iterations the far field has stopped responding to the near
        correction, so log and stop.
    '''
    from pochoir.__main__ import info_msg

    def _log(msg):
        info_msg(msg)
        print(msg)

    if log is None:
        log = _log

    prof = _profile(field)
    log(f'hybrid-iterate: field={field} generator={prof["generator"]} '
        f'edges={prof["edges"]} output={prof["output"]}')

    spacings = dict(
        coarse=DEFAULT_SPACINGS['coarse'] if coarse_spacing is None else float(coarse_spacing),
        fine=DEFAULT_SPACINGS['fine'] if fine_spacing is None else float(fine_spacing),
        full=DEFAULT_SPACINGS['full'] if full_spacing is None else float(full_spacing),
    )

    if derive_domain:
        # The FINE config carries the geometry both the near and final grids use;
        # the coarse config is only the 0.4mm transcription of the same physical
        # object, so extents must come from the fine one.
        cfg = json.loads(Path(fine_config).read_text())
        grids = _derive_grids(prof, cfg, spacings, _interface_mm(interface))
        log(f'hybrid-iterate: domain=yes -- shapes derived from '
            f'{Path(fine_config).name} at spacings coarse={spacings["coarse"]} '
            f'fine={spacings["fine"]} full={spacings["full"]} mm')
    else:
        grids = _manual_grids(prof, shapes or {}, spacings)
        log(f'hybrid-iterate: domain=no -- shapes supplied by the user at '
            f'spacings coarse={spacings["coarse"]} fine={spacings["fine"]} '
            f'full={spacings["full"]} mm')
    for key, shape, spacing in grids:
        log(f'    {key:24s} {shape:16s} {spacing}')

    _domains(ctx, grids, log)
    _generate(ctx, prof, coarse_config, fine_config, log)

    # step 1: the coarse full-volume solve that seeds iteration 0.
    coarse_pot = _key(prof, 'potential', 'coarse')
    _solve(ctx, prof, _key(prof, 'initial', 'coarse'),
           _key(prof, 'boundary', 'coarse'),
           _key(prof, 'initial', 'coarse') + '_insulator',
           coarse_pot, _key(prof, 'increment', 'coarse'), tol, log)

    unit = prof.get('unit', '')
    prev = coarse_pot
    history = []
    criterion = f'max_iters={max_iters}'
    stagnant = 0

    for k in range(max_iters):
        cur = _outer_iteration(ctx, prof, grids, k, prev, interface, tol, log)
        delta = _max_abs_delta(ctx, cur, prev)
        history.append(delta)
        log(f'iter {k}: max|dphi| = {delta:.6e} {unit} (tol {tol:.1e})')

        if delta < tol:
            criterion = f'converged (delta {delta:.6e} < tol {tol:.1e})'
            prev = cur
            break

        # stagnation: less than 10% improvement two iterations running.
        if len(history) >= 2:
            if history[-1] > 0.9 * history[-2]:
                stagnant += 1
            else:
                stagnant = 0
            if stagnant >= 2:
                criterion = (f'stagnated (delta {delta:.6e}, '
                             f'<10% improvement for 2 iterations)')
                prev = cur
                break

        prev = cur

    # --max-iters 0 is reachable from the CLI (click applies no minimum), and
    # then the loop body never runs and history is empty -- guard the summary
    # rather than raising IndexError on history[-1].
    if history:
        log(f'hybrid-iterate: {len(history)} iterations, '
            f'final delta {history[-1]:.6e} {unit}, stopped on {criterion}')
        log('hybrid-iterate: delta history = '
            + ', '.join(f'{d:.6e}' for d in history))
    else:
        log(f'hybrid-iterate: no iterations run ({criterion}); '
            f'proceeding from the coarse seed {prev}')

    _final_stage(ctx, prof, prev, tol, log)

    log(f'hybrid-iterate: done, {field} field {prev} -> {prof["output"]} '
        f'(run the velo/starts/drift/induce chain from the shell script)')
    return prev, history, criterion
