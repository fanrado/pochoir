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
# Grids.  Same 4.4mm periodic pixel tile throughout; z = 0..60mm (10mm of
# PCB/pad region below the plane + 50mm drift, cathode at z=60mm).
# The 8:1 coarse/near stride (0.4/0.05) is exact on every axis: 88->11, 401->51.
# ---------------------------------------------------------------------------
GRIDS = (
    # key,                 shape,          spacing,   extent
    ('domain/coarse',      '11,11,151',    '0.4*mm'),   # coarse full, z=0..60mm
    ('domain/near',        '88,88,401',    '0.05*mm'),  # near fine,   z=0..20mm
    ('domain/near_coarse', '11,11,51',     '0.4*mm'),   # coarsen target, z=0..20
    ('domain/fine01',      '44,44,601',    '0.1*mm'),   # final full,  z=0..60mm
)

GENERATOR = 'pcb_drift_pixel_with_grid'

# task10b's fdm flags, applied to every solve here.
EDGES = 'per,per,fix'
ENGINE = 'torch'
EPOCH = 130000000
NEPOCHS = 10


def _domains(ctx, log):
    '''The four `domain` invocations.'''
    from pochoir.__main__ import domain

    for key, shape, spacing in GRIDS:
        _want(ctx, key,
              lambda key=key, shape=shape, spacing=spacing: ctx.invoke(
                  domain, domain=key, shape=shape, spacing=spacing),
              log)


def _generate(ctx, coarse_config, fine_config, log):
    '''
    `gen` per domain with the matching config: the coarse 0.4mm transcription
    for domain/coarse, the fine config for both domain/near and domain/fine01.

    `gen` also writes the no-flux insulator mask under "<initial>_insulator";
    --insulator is an enable signal only, the interface itself is derived from
    the Dirichlet geometry by padplane_noflux_geom.
    '''
    from pochoir.__main__ import gen

    for dom_key, name, cfg in (
            ('domain/coarse', 'coarse', coarse_config),
            ('domain/near', 'near', fine_config),
            ('domain/fine01', 'fine01', fine_config)):
        init, bnd = f'initial/{name}', f'boundary/{name}'
        _want(ctx, [init, bnd],
              lambda dom_key=dom_key, init=init, bnd=bnd, cfg=cfg: ctx.invoke(
                  gen, generator=GENERATOR, domain=dom_key,
                  initial=init, boundary=bnd, configs=(cfg,)),
              log)


def _solve(ctx, initial, boundary, insulator, potential, increment, precision,
           log):
    '''One `fdm` call with task10b's flags.  `multisteps` left at its default.'''
    from pochoir.__main__ import fdm

    _want(ctx, [potential, increment],
          lambda: ctx.invoke(
              fdm, initial=initial, boundary=boundary, insulator=insulator,
              edges=EDGES, engine=ENGINE, precision=precision,
              epoch=EPOCH, nepochs=NEPOCHS,
              potential=potential, increment=increment),
          log)


def _near_interface():
    '''
    The near/far split plane implied by GRIDS' `domain/near`, in pochoir units.

    `near_bc` derives the pinned plane itself as the near domain's LAST plane
    (top = ndom.shape[axis]-1) and `stitch_near` takes no interface argument, so
    the split is fixed entirely by the near grid's z extent -- NOT by the
    --interface option.
    '''
    from pochoir.util import unitify

    for key, shape, spacing in GRIDS:
        if key != 'domain/near':
            continue
        nz = int(shape.split(',')[2])
        return (nz - 1) * unitify(spacing)
    raise ValueError('GRIDS has no domain/near entry')


def _outer_iteration(ctx, k, far_potential, interface, precision, log):
    '''
    Plan steps 2-7 for one pass.  `far_potential` is the current full-volume
    coarse field (potential/coarse for k=0, the previous potential/full_k* after
    that).  Returns the new full-volume potential key.

    `interface` is checked against the split the near grid actually implies
    rather than being used to place it: the plane comes from `domain/near`'s z
    extent (see `_near_interface`).  A mismatch used to be silently ignored, so
    `--interface 30*mm` ran happily and still split at 20mm.
    '''
    from pochoir.__main__ import refine, near_bc, coarsen, stitch_near
    from pochoir.util import unitify

    want = unitify(interface)
    have = _near_interface()
    if want != have:
        raise ValueError(
            f'--interface {interface} ({want}) disagrees with the split implied '
            f'by domain/near ({have}).  The near/far plane is set by the near '
            f'grid z extent in GRIDS, not by this option; re-shape domain/near '
            f'to move it.')

    kk = _kk(k)
    near_refined = f'initial/near_refined{kk}'
    near_bc_i, near_bc_b = f'initial/near_bc{kk}', f'boundary/near_bc{kk}'
    near_pot, near_inc = f'potential/near{kk}', f'increment/near{kk}'
    near_coarse = f'potential/near_coarse{kk}'
    stitched = f'potential/stitched{kk}'
    full_init = f'initial/full{kk}'
    full_pot, full_inc = f'potential/full{kk}', f'increment/full{kk}'

    # step 2: upsample the current far field onto the 0.05mm near grid
    # (refine also re-imposes the exact fine boundary values).
    _want(ctx, near_refined,
          lambda: ctx.invoke(refine, coarse=far_potential,
                             initial='initial/near', boundary='boundary/near',
                             output=near_refined), log)

    # step 3: pin the z=interface plane Dirichlet from the far field.
    _want(ctx, [near_bc_i, near_bc_b],
          lambda: ctx.invoke(near_bc, initial=near_refined,
                             boundary='boundary/near', coarse=far_potential,
                             initial_out=near_bc_i, boundary_out=near_bc_b,
                             axis=2), log)

    # step 4: near fine solve, no-flux FR4 BC active.
    _solve(ctx, near_bc_i, near_bc_b, 'initial/near_insulator',
           near_pot, near_inc, precision, log)

    # step 5: coarsen the near solution back to 0.4mm (exact stride 8).
    _want(ctx, near_coarse,
          lambda: ctx.invoke(coarsen, input_=near_pot,
                             domain='domain/near_coarse',
                             output=near_coarse), log)

    # step 6: stitch the coarsened near field onto the coarse far field.
    _want(ctx, stitched,
          lambda: ctx.invoke(stitch_near, near=near_coarse,
                             coarse=far_potential, domain='domain/coarse',
                             output=stitched, axis=2), log)

    # step 7a: stitch-near emits a raw potential whose conductor cells now hold
    # coarsened near-field values.  Re-run refine on the SAME domain -- the
    # interpolation is the identity there and it merges the true boundary values
    # back in (refined[bmask] = fi[bmask]) -- rather than writing new code.
    _want(ctx, full_init,
          lambda: ctx.invoke(refine, coarse=stitched,
                             initial='initial/coarse',
                             boundary='boundary/coarse',
                             output=full_init), log)

    # step 7b: full-volume coarse re-solve.  The stitched array is INITIAL
    # VALUES ONLY -- the near region floats freely, which is the whole point of
    # this scheme versus a pinned Schwarz sweep.
    _solve(ctx, full_init, 'boundary/coarse', 'initial/coarse_insulator',
           full_pot, full_inc, precision, log)

    return full_pot


def _max_abs_delta(ctx, key_a, key_b):
    '''max|a - b| over the whole volume, in volts.'''
    a = numpy.asarray(ctx.obj.get(key_a), dtype=float)
    b = numpy.asarray(ctx.obj.get(key_b), dtype=float)
    return float(numpy.max(numpy.abs(a - b)))


def _final_stage(ctx, converged, fine_config, precision, log):
    '''
    Final refinement onto the 0.1mm full grid, then task10b's PART C verbatim.

    velo/starts/drift are ENFORCEMENT-FREE: velo gets neither --boundary nor
    --insulator (velocity is pure mu*grad(phi)) and drift gets no --insulator
    (paths are never clamped or terminated at a surface).
    '''
    from pochoir.__main__ import refine, velo, starts, drift

    # 0.4mm -> 0.1mm upsample + exact boundary values in one call.
    _want(ctx, 'initial/fine01_seed',
          lambda: ctx.invoke(refine, coarse=converged,
                             initial='initial/fine01',
                             boundary='boundary/fine01',
                             output='initial/fine01_seed'), log)

    _solve(ctx, 'initial/fine01_seed', 'boundary/fine01',
           'initial/fine01_insulator',
           'potential/drift3d', 'increment/fine01', precision, log)

    _want(ctx, 'velocity/drift3d',
          lambda: ctx.invoke(velo, temperature='87.0*K',
                             potential='potential/drift3d',
                             velocity='velocity/drift3d'), log)

    _want(ctx, 'starts/drift3d',
          lambda: ctx.invoke(starts, starts='starts/drift3d', mode='yes',
                             configs=(fine_config,), plot=True), log)

    # ~1.5us/mm at 50 V/mm over ~50mm -> ~75us transit; 120us leaves margin.
    _want(ctx, 'paths/drift3d',
          lambda: ctx.invoke(drift, starts='starts/drift3d',
                             velocity='velocity/drift3d',
                             interp_order='linear',
                             paths='paths/drift3d',
                             steps=('0*us,120*us,0.05*us',), plot=True), log)


def hybrid_iterate(ctx, coarse_config, fine_config, interface='20*mm',
                   tol=2e-8, max_iters=20, log=None):
    '''
    Drive the whole Task13 hybrid: grids, gens, the outer iteration, the final
    0.1mm solve and the drift chain.

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

    _domains(ctx, log)
    _generate(ctx, coarse_config, fine_config, log)

    # step 1: the coarse full-volume solve that seeds iteration 0.
    _solve(ctx, 'initial/coarse', 'boundary/coarse', 'initial/coarse_insulator',
           'potential/coarse', 'increment/coarse', tol, log)

    prev = 'potential/coarse'
    history = []
    criterion = f'max_iters={max_iters}'
    stagnant = 0

    for k in range(max_iters):
        cur = _outer_iteration(ctx, k, prev, interface, tol, log)
        delta = _max_abs_delta(ctx, cur, prev)
        history.append(delta)
        log(f'iter {k}: max|dphi| = {delta:.6e} V (tol {tol:.1e})')

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
            f'final delta {history[-1]:.6e} V, stopped on {criterion}')
        log('hybrid-iterate: delta history = '
            + ', '.join(f'{d:.6e}' for d in history))
    else:
        log(f'hybrid-iterate: no iterations run ({criterion}); '
            f'proceeding from the coarse seed {prev}')

    _final_stage(ctx, prev, fine_config, tol, log)

    log(f'hybrid-iterate: done, converged field {prev} -> potential/drift3d')
    return prev, history, criterion
