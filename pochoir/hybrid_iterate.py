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
        grids=(
            # key,                 shape,        spacing,   extent
            ('domain/coarse',      '11,11,151',  '0.4*mm'),   # coarse full  0..60mm
            ('domain/near',        '88,88,401',  '0.05*mm'),  # near fine    0..20mm
            ('domain/near_coarse', '11,11,51',   '0.4*mm'),   # coarsen tgt  0..20mm
            ('domain/fine01',      '44,44,601',  '0.1*mm'),   # final full   0..60mm
        ),
    ),
    'weighting': dict(
        generator='pcb_pixel_with_grid',
        edges='fix,fix,fix',
        prefix='w_',
        output='potential/weight3d',
        # the weighting potential is a DIMENSIONLESS unit probe in [0,1], so the
        # convergence delta is not in volts -- do not label it 'V'.
        unit='(dimensionless)',
        grids=(
            ('domain/w_coarse',      '55,55,151',    '0.4*mm'),
            ('domain/w_near',        '440,440,401',  '0.05*mm'),
            ('domain/w_near_coarse', '55,55,51',     '0.4*mm'),
            ('domain/w_fine01',      '220,220,601',  '0.1*mm'),
        ),
    ),
}

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


def _domains(ctx, prof, log):
    '''The four `domain` invocations for this field.'''
    from pochoir.__main__ import domain

    for key, shape, spacing in prof['grids']:
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


def _near_interface(prof):
    '''
    The near/far split plane implied by this profile's near grid, in pochoir
    units.

    `near_bc` derives the pinned plane itself as the near domain's LAST plane
    (top = ndom.shape[axis]-1) and `stitch_near` takes no interface argument, so
    the split is fixed entirely by the near grid's z extent -- NOT by the
    --interface option.
    '''
    from pochoir.util import unitify

    near_key = _key(prof, 'domain', 'near')
    for key, shape, spacing in prof['grids']:
        if key != near_key:
            continue
        nz = int(shape.split(',')[2])
        return (nz - 1) * unitify(spacing)
    raise ValueError(f'field profile has no {near_key} entry')


def _outer_iteration(ctx, prof, k, far_potential, interface, precision, log):
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
    have = _near_interface(prof)
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
                   tol=2e-8, max_iters=20, field='drift', log=None):
    '''
    Drive the Task13 hybrid FIELD solve: grids, gens, the outer iteration and
    the final 0.1mm refine+solve -> the field's output key.

    `field` selects the profile in FIELDS: 'drift' (periodic single pixel tile ->
    potential/drift3d) or 'weighting' (non-periodic 5x5 unit probe ->
    potential/weight3d).  Both run the identical scheme, tolerances and final
    0.1mm grid; only grids, generator, edges and store-key names differ.

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

    _domains(ctx, prof, log)
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
        cur = _outer_iteration(ctx, prof, k, prev, interface, tol, log)
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
