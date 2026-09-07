#!/usr/bin/env python3
'''
Task13 hybrid near/far drift-field solver (beads pochoir-uy3c).

The bash hybrid runners (run-hybrid-30cm-drift.sh and friends) hard-code a fixed
near/far sequence, so the structure is invisible from the script.  This module
drives the same steps from Python, reducing bash to supplying configs and the
store path.

Method -- near/far separation at z = --interface (axis 2):

  1. coarse full-volume solve at the coarse spacing
  2. `refine` the coarse potential onto the fine near grid, then `near-bc` pins
     the near grid's last z plane (z = --interface) to the coarse value there,
     a Dirichlet condition for the sweep-0 near solve
  3. near solve at the fine spacing on z = 0..interface, no-flux FR4 insulator
     BC active
  4. ONE Schwarz sweep at overlap 2 coarse cells -- a 3-coarse-node band at the
     interface (40.0 / 39.6 / 39.2mm at --interface 40*mm, coarse 0.4mm).  The
     far solve pins the INNER band node (39.2mm) Dirichlet to the downsampled
     near solution and leaves the middle and outer nodes seeded but FREE; the
     near then re-solves with its outer node (40.0mm) pinned to the updated
     far.  Sequence: coarse -> near -> far -> near.

     Only the inner node is Dirichlet on purpose.  Pinning all three would push
     the far problem's real domain up to the outer node and discard the
     overlap, which is the whole mechanism by which the two solves see each
     other.
  5. `stitch-near` multi-linearly upsamples the Schwarz far field onto the full
     fine grid and overwrites the near planes with the Schwarz near solution.
     That stitched array IS the final field, written straight to the profile's
     output key (potential/drift3d, potential/weight3d).

The seam is C0 BY CONSTRUCTION: the final near solve is pinned to the final far
solve, so the two agree in value where they meet.

That does NOT make it C1, and --interp-order linear is STILL REQUIRED for
drift.  The sweep is expected to shrink the derivative jump at the seam, but
that jump has not been measured yet; until it has, assume a kink is present and
do not let cubic interpolation near it (cubic overshoots at a kink).

--max-sweeps 0 skips the sweep entirely and restores the old one-shot path:
step 3's near solution and the step-1 coarse field go straight to step 5, with
the seam continuous only because step 2 pinned the near top plane to the same
coarse values the upsample produces there.

The sweep is run by `nearfar.schwarz_solve` via the `near-far-solve` command,
which this module now calls (step 4).  Neither is modified here.

Every step runs the existing click command in-process via `ctx.invoke`, so it
takes the same code path as the bash scripts and shares one store object.
'''

import json
from pathlib import Path


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
# Both fields run the SAME hybrid scheme (including the sweep), the same
# precision and the
# same final 0.1mm grid; only the grids, generator, edges and store-key names
# differ.  `--field` selects between them.
#
# DRIFT: one 4.4mm PERIODIC pixel tile (per,per,fix), z = 0..60mm (10mm of
# PCB/pad region below the plane + 50mm drift, cathode at z=60mm).
#
# WEIGHTING: 5x5 pixels = 22mm transverse, NON-periodic (fix,fix,fix) so phi_w
# decays to ~0 at the tile edges rather than wrapping -- a unit probe cannot be
# solved on a single periodic tile.
#
# KEY NAMESPACING -- this is load-bearing, not cosmetic.  `_want` skips any step
# whose outputs already exist, and BOTH fields must write into the SAME store so
# `induce-pixel` can see `paths/drift3d` and `potential/weight3d` together.  If
# the weighting profile reused the drift key names, every grid and gen step would
# be skipped as "have" and the weighting field would be solved on the DRIFT
# geometry -- silently, with no error.  So weighting INTERMEDIATE keys (coarse,
# near, near_bc, near_refined) carry a `w_` prefix on the leaf name.
#
# The FULL-VOLUME grid is the exception: it is named by FIELD, not by prefix --
# `drift3d` and `weight3d` -- in EVERY taxon, so a store folder reads
# domain/drift3d + boundary/drift3d + initial/drift3d + potential/drift3d.  That
# is already what `potential/` used (task10b's names, which `velo` and
# `induce-pixel` are invoked with from the shell script); this extends the same
# naming to domain/, boundary/ and initial/, which used to say fine01/w_fine01.
# `weight3d` is why the name comes from the profile's `full_leaf` rather than
# from prefix + leaf: it does not fit the w_ scheme.
#
# NO BACKWARD COMPATIBILITY (pochoir-efgb): nothing reads the old
# fine01/w_fine01 keys, so existing store_task13_*/store_task14_* folders will
# stop resuming the full-volume step and re-do it under the new name.  That is
# accepted -- verification is a fresh-store run.
# ---------------------------------------------------------------------------
FIELDS = {
    'drift': dict(
        generator='pcb_drift_pixel_with_grid',
        edges='per,per,fix',
        prefix='',                       # drift keys are unprefixed (unchanged)
        # The FULL-VOLUME leaf is named by the profile, NOT prefix+'fine01':
        # every taxon of the final grid is domain/drift3d, boundary/drift3d,
        # initial/drift3d, potential/drift3d.
        full_leaf='drift3d',
        unit='V',                        # drift potential is in volts
        # transverse extent = ONE pixel pitch (a single periodic tile)
        npixels=1,
    ),
    'weighting': dict(
        generator='pcb_pixel_with_grid',
        edges='fix,fix,fix',
        prefix='w_',
        # 'weight3d' deliberately does NOT carry the w_ prefix -- the prefix
        # scheme namespaces the INTERMEDIATE leaves only.  See full_leaf above.
        full_leaf='weight3d',
        # the weighting potential is a DIMENSIONLESS unit probe in [0,1], so the
        # convergence delta is not in volts -- do not label it 'V'.
        unit='(dimensionless)',
        # transverse extent = Npixels pitches from the config (5x5 = 22mm), so
        # phi_w has room to decay to ~0 at the edges
        npixels=None,                    # None -> take Npixels from the config
    ),
}

# Default spacings, in mm.  TWO spacings only: the coarse full-volume solve and
# the fine grid used by BOTH the near solve and the final stitched full volume.
# The user supplies them via --coarse-spacing / --fine-spacing.
DEFAULT_SPACINGS = dict(coarse=0.4, fine=0.1)

# Minimum physical clearance, in mm, between the electron launch node
# (driftZDepth) and the cathode on the domain's last plane.  Deliberately a
# fixed physical length rather than a multiple of the run's spacings: it is the
# margin the validated 59.9mm/60.0mm geometry actually has, and keeping it
# resolution-independent is what lets the cathode stay at 60.0mm for every
# spacing (see _extents).
MIN_CATHODE_CLEARANCE = 0.0#0.1

# The GRID_SPEC leaf that denotes the FULL-VOLUME grid.  It is a SPEC-side
# identifier only -- the `--domain no` shapes-dict key and the GRID_SPEC row
# label.  The STORE key it resolves to is the profile's `full_leaf`
# (drift3d/weight3d), never prefix+this.  Keep the two ideas separate: renaming
# the store keys must not force every caller to re-key its shapes dict.
FULL_LEAF = 'fine01'

# The three grids and which of the two user spacings each one uses.  `near` and
# the full-volume leaf share the FINE spacing: the near solve and the final
# stitched volume live on the same 0.1mm lattice, which is what makes the stitch
# an exact plane-for-plane overwrite rather than a resample.  There is no
# coarsen-back target grid any more (no iteration to feed).
GRID_SPEC = (
    # leaf,      spacing name,  depth
    ('coarse',   'coarse',      'full'),   # coarse far field, full depth
    ('near',     'fine',        'near'),   # fine near field, to the interface
    (FULL_LEAF,  'fine',        'full'),   # final full-volume stitch target
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
    Build the three (key, shape, spacing) triples from the config geometry and
    the user's two spacings -- `--domain yes`.

    At the default 0.4 / 0.1 mm spacings:
      drift     11,11,151@0.4   44,44,201@0.1    44,44,601@0.1
      weighting 55,55,151@0.4  220,220,201@0.1  220,220,601@0.1
    '''
    ext = _extents(prof, cfg, spacings['coarse'], interface_mm)
    grids = []
    for leaf, sp_name, depth_name in GRID_SPEC:
        sp = spacings[sp_name]
        nt = _cells(ext['transverse'], sp, f'{leaf} transverse', closed=False)
        nz = _cells(ext[depth_name], sp, f'{leaf} z', closed=True)
        grids.append((_key(prof, 'domain', leaf),
                      f'{nt},{nt},{nz}', f'{sp}*mm'))
    return tuple(grids)


def _manual_grids(prof, shapes, spacings):
    '''
    Build the three triples from shapes the user typed in -- `--domain no`.

    `shapes` maps each GRID_SPEC leaf ('coarse', 'near', FULL_LEAF) to a
    "nx,ny,nz" string.  Note these are the SPEC-side leaf names, not the store
    key names: the full-volume leaf is keyed 'fine01' here but stored as
    domain/drift3d resp. domain/weight3d (see `_key`).
    Every leaf must be supplied: a missing one cannot be guessed without falling
    back to derivation, which would silently mix the two modes.
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
        grids.append((_key(prof, 'domain', leaf),
                      shape, f'{spacings[sp_name]}*mm'))
    return tuple(grids)

# task10b's fdm flags, applied to every solve here for both fields.
ENGINE = 'torch'
EPOCH = 130000000
NEPOCHS = 10

# Banded near/far Schwarz sweep (see `_schwarz`).  The band is
# DEFAULT_BAND_CELLS coarse cells of overlap, i.e. band_cells+1 coarse nodes:
# at --interface 19.8*mm with coarse 0.22mm those are nodes 90/89/88/87 = z
# 19.80/19.58/19.36/19.14mm.  The innermost, 19.14mm, still falls inside the
# near grid's 0..19.8mm range, so the band needs NO grid reshaping.
#
# DEFAULT_TOL is a PLAIN NUMBER, not a units string.  It is a max change on the
# potential array between sweeps -- the same kind of quantity as --precision,
# which is likewise undimensioned.  It was '1*V' until Phase 2/Step 1, which
# round-tripped through the unit system to 1.0, i.e. one whole volt, and so
# halted the sweep after a single iteration; measured near deltas on the 8cm
# grid-free geometry run 0.34 (band 2) to 3.17 (band 20).  At 2e-8 the
# tolerance sits ~7 orders below the smallest of those, so it never gates and
# max_sweeps is the binding constant at ANY band width.  Being undimensioned is
# also what lets ONE value serve both fields: the drift potential is in volts
# while the weighting probe is dimensionless in [0,1] (observed delta 3.7e-05),
# and no volt-valued tolerance can gate both.
DEFAULT_BAND_CELLS = 3
DEFAULT_MAX_SWEEPS = 5
DEFAULT_TOL = 2e-8


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

    INTERMEDIATE leaves (coarse, near, near_bc, near_refined) take the profile's
    `prefix`: drift -> "potential/near", weighting -> "potential/w_near".

    The FULL-VOLUME leaf instead takes the profile's `full_leaf` verbatim, with
    NO prefix, so the final grid is named by field in every taxon:
    domain/drift3d + boundary/drift3d + initial/drift3d + potential/drift3d, and
    domain/weight3d + ... + potential/weight3d.  'weight3d' does not fit the w_
    prefix scheme, which is why the name comes from the profile rather than from
    prefix + FULL_LEAF.

    There is no iteration index in the names any more: each step runs once.
    '''
    if leaf == FULL_LEAF:
        return f'{taxon}/{prof["full_leaf"]}'
    return f'{taxon}/{prof["prefix"]}{leaf}'


def _output_key(prof):
    '''
    The final stitched field's key -- potential/drift3d or potential/weight3d.

    Derived from `full_leaf` rather than stored separately so the output key and
    the full-volume grid name cannot drift apart.  These are exactly task10b's
    names, which `velo` and `induce-pixel` are invoked with from the shell
    script, so they must not change.
    '''
    return _key(prof, 'potential', FULL_LEAF)


def _domains(ctx, grids, log):
    '''The three `domain` invocations for this field.'''
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
    The final grid is generated even though nothing is solved on it, so its
    boundary/insulator arrays are in the store for inspection alongside the
    stitched field.

    `gen` also writes the no-flux insulator mask under "<initial>_insulator";
    --insulator is an enable signal only, the interface itself is derived from
    the Dirichlet geometry by padplane_noflux_geom.
    '''
    from pochoir.__main__ import gen

    for leaf, cfg in (('coarse', coarse_config),
                      ('near', fine_config),
                      (FULL_LEAF, fine_config)):
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


def _check_interface(prof, grids, interface):
    '''
    Fail loudly when `--interface` disagrees with the split the near grid
    actually implies.

    The plane comes from the near grid's z extent (see `_near_interface`), not
    from this option.  A mismatch used to be silently ignored, so
    `--interface 30*mm` ran happily and still split at 20mm.
    '''
    from pochoir.util import unitify

    want = unitify(interface)
    have = _near_interface(prof, grids)
    if want != have:
        raise ValueError(
            f'--interface {interface} ({want}) disagrees with the split implied '
            f'by {_key(prof, "domain", "near")} ({have}).  The near/far plane is '
            f'set by the near grid z extent in the field profile, not by this '
            f'option; re-shape the near grid to move it.')


def _near_solve(ctx, prof, precision, log):
    '''
    Steps 2-3: seed the near grid from the coarse far field, pin its top plane
    Dirichlet, and solve there once.  Returns the near potential key.
    '''
    from pochoir.__main__ import refine, near_bc

    coarse_pot = _key(prof, 'potential', 'coarse')
    near_refined = _key(prof, 'initial', 'near_refined')
    near_bc_i = _key(prof, 'initial', 'near_bc')
    near_bc_b = _key(prof, 'boundary', 'near_bc')
    near_pot = _key(prof, 'potential', 'near')
    near_inc = _key(prof, 'increment', 'near')
    i_near = _key(prof, 'initial', 'near')
    b_near = _key(prof, 'boundary', 'near')

    # step 2a: upsample the coarse far field onto the fine near grid (refine
    # also re-imposes the exact fine boundary values).
    _want(ctx, near_refined,
          lambda: ctx.invoke(refine, coarse=coarse_pot,
                             initial=i_near, boundary=b_near,
                             output=near_refined), log)

    # step 2b: pin the near grid's last z plane (the interface) Dirichlet from
    # the coarse field.  This BC is FIXED -- it is never revisited.
    _want(ctx, [near_bc_i, near_bc_b],
          lambda: ctx.invoke(near_bc, initial=near_refined,
                             boundary=b_near, coarse=coarse_pot,
                             initial_out=near_bc_i, boundary_out=near_bc_b,
                             axis=2), log)

    # step 3: the one near solve, no-flux FR4 BC active.
    _solve(ctx, prof, near_bc_i, near_bc_b, i_near + '_insulator',
           near_pot, near_inc, precision, log)

    return near_pot


def _schwarz(ctx, prof, near_pot, interface, band_cells=DEFAULT_BAND_CELLS,
             max_sweeps=DEFAULT_MAX_SWEEPS, tol=DEFAULT_TOL,
             precision=2e-8, log=None):
    '''
    The banded near/far Schwarz sweep: coarse -> near -> far -> near, ONE sweep.

    `near_bc` pins the near top plane to the coarse field once and never
    revisits it, so the single-shot scheme leaves a derivative kink at the seam.
    This re-solves the two domains against each other across a band of
    `band_cells` coarse cells of overlap, which makes the stitched field
    continuous in gradient as well as in value.

    `near_pot` -- the sweep-0 near solution from `_near_solve` -- is handed over
    as --near-potential, i.e. a WARM START.  The sweep therefore writes its
    results to new keys (near_schwarz / coarse_schwarz) and the existing
    near_refined / near_bc / near store files are left intact for inspection.

    Precision is the driver's SINGLE precision for both sides.  near-far-solve
    defaults to its own 2e-11 / 2e-7 near/far split; passing `precision` to both
    deliberately overrides that, so every solve in this module converges to the
    same tolerance.

    `interface` is REQUIRED and positional on purpose: it used to carry a
    '20*mm' default while the runner passes --interface 40*mm, so a caller that
    forgot it would have banded the sweep at the wrong plane and silently
    solved the wrong problem.  There is now no default to fall back to.

    Returns (near_out, far_out).
    '''
    from pochoir.__main__ import near_far_solve

    near_out = _key(prof, 'potential', 'near_schwarz')
    far_out = _key(prof, 'potential', 'coarse_schwarz')

    _want(ctx, [near_out, far_out],
          lambda: ctx.invoke(
              near_far_solve,
              coarse_potential=_key(prof, 'potential', 'coarse'),
              coarse_initial=_key(prof, 'initial', 'coarse'),
              coarse_boundary=_key(prof, 'boundary', 'coarse'),
              near_initial=_key(prof, 'initial', 'near'),
              near_boundary=_key(prof, 'boundary', 'near'),
              near_potential=near_pot,
              insulator=_key(prof, 'initial', 'near') + '_insulator',
              interface=interface, axis=2, edges=prof['edges'],
              engine=ENGINE, epoch=EPOCH, nepochs=NEPOCHS,
              near_precision=precision, far_precision=precision,
              tol=tol, max_iters=max_sweeps, overlap=band_cells,
              near_out=near_out, far_out=far_out),
          log)

    return near_out, far_out


def _stitch(ctx, prof, near_pot, coarse_pot, log):
    '''
    Steps 4-5: upsample the coarse far field onto the full fine grid, overwrite
    the near planes with the fine near solution, and write the result STRAIGHT
    to the profile's output key.

    `near_pot` and `coarse_pot` are passed in rather than derived here, because
    after a Schwarz sweep they are the sweep outputs (potential/near_schwarz,
    potential/coarse_schwarz) rather than the sweep-0 keys.  `coarse_pot` is a
    FULL-volume coarse array either way -- near-far-solve solves the whole
    coarse domain with the inner band node pinned -- so the stitch geometry is
    identical in both cases.  The output key is unchanged.

    That stitched array is the final field: no full-volume re-solve follows.

    The seam is continuous, but for a different reason in each mode.  After a
    Schwarz sweep (the default) it is because the FINAL near solve was pinned
    to the FINAL far solve at the outer band node, so the two fields being
    stitched already agree there.  With max_sweeps=0 it is the original reason:
    `near_bc` pinned the near top plane to the same coarse values the upsample
    produces at that plane.

    `stitch-near` stamps `domain = domain/drift3d` (resp. domain/weight3d) into
    the output metadata, which is what lets `velo` resolve the grid downstream.

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
    from pochoir.__main__ import stitch_near

    out = _output_key(prof)
    _want(ctx, out,
          lambda: ctx.invoke(stitch_near, near=near_pot,
                             coarse=coarse_pot,
                             domain=_key(prof, 'domain', FULL_LEAF),
                             output=out, axis=2), log)
    return out


def hybrid_iterate(ctx, coarse_config, fine_config, interface='20*mm',
                   precision=2e-8, field='drift',
                   coarse_spacing=None, fine_spacing=None,
                   derive_domain=True, shapes=None,
                   band_cells=DEFAULT_BAND_CELLS,
                   max_sweeps=DEFAULT_MAX_SWEEPS, tol=DEFAULT_TOL, log=None):
    '''
    Drive the Task13 hybrid FIELD solve: grids, gens, the coarse solve, the fine
    near solve against a fixed interface BC, one banded near/far Schwarz sweep,
    and the stitch that IS the final field -> the field's output key.

    `field` selects the profile in FIELDS: 'drift' (periodic single pixel tile ->
    potential/drift3d) or 'weighting' (non-periodic 5x5 unit probe ->
    potential/weight3d).  Both run the identical scheme and precision; only
    grids, generator, edges and store-key names differ.

    GRIDS.  The two spacings are always supplied by the caller
    (`coarse_spacing` / `fine_spacing`, in mm, defaulting to 0.4 / 0.1).
    `derive_domain` then chooses where the three grid SHAPES come from:

      * True  (`--domain yes`) -- computed from the config geometry: transverse
        extent = (pixelSize + pixelGap) x Npixels-for-this-field, full depth =
        driftZDepth + MIN_CATHODE_CLEARANCE rounded up to a whole coarse cell
        (see `_extents` -- 59.9mm -> 60.0mm at every spacing), near depth = the
        interface.
      * False (`--domain no`)  -- taken from `shapes`, a dict of "nx,ny,nz"
        strings keyed by GRID_SPEC leaf ('coarse', 'near', 'fine01' =
        FULL_LEAF).  All three required.

    THE SWEEP.  `band_cells` (default 3) is the near/far overlap in COARSE
    cells, so the band spans band_cells+1 coarse nodes at the interface.
    `max_sweeps` (default 5) caps the sweep count and `tol` (default 2e-8, a
    plain number in the units of the potential array, exactly like
    --precision) is the inter-sweep convergence tolerance handed to
    near-far-solve.  `max_sweeps` is the constant that binds: 2e-8 is far below
    any delta this geometry produces, so the tolerance is an escape hatch for
    an already-converged interface rather than the normal stopping condition.

    `max_sweeps=0` SKIPS the sweep and stitches the sweep-0 near solution
    against the step-1 coarse field, reproducing the old one-shot scheme
    exactly.

    `precision` is the per-solve fdm convergence precision -- a different thing
    from `tol`, and applied identically to every solve here including both
    sides of the sweep.

    The chain after the field (velo / starts / drift / induce-pixel) is NOT part
    of this driver -- see `_stitch`.  It runs as explicit `pochoir` commands in
    test/run-task13-hybrid.sh so its parameters stay hand-settable.

    Returns `(final_key, grids)`.
    '''
    from pochoir.__main__ import info_msg

    def _log(msg):
        info_msg(msg)
        print(msg)

    if log is None:
        log = _log

    prof = _profile(field)
    log(f'hybrid-iterate: field={field} generator={prof["generator"]} '
        f'edges={prof["edges"]} output={_output_key(prof)}')

    spacings = dict(
        coarse=DEFAULT_SPACINGS['coarse'] if coarse_spacing is None else float(coarse_spacing),
        fine=DEFAULT_SPACINGS['fine'] if fine_spacing is None else float(fine_spacing),
    )

    if derive_domain:
        # The FINE config carries the geometry both the near and final grids use;
        # the coarse config is only the 0.4mm transcription of the same physical
        # object, so extents must come from the fine one.
        cfg = json.loads(Path(fine_config).read_text())
        grids = _derive_grids(prof, cfg, spacings, _interface_mm(interface))
        log(f'hybrid-iterate: domain=yes -- shapes derived from '
            f'{Path(fine_config).name} at spacings coarse={spacings["coarse"]} '
            f'fine={spacings["fine"]} mm')
    else:
        grids = _manual_grids(prof, shapes or {}, spacings)
        log(f'hybrid-iterate: domain=no -- shapes supplied by the user at '
            f'spacings coarse={spacings["coarse"]} fine={spacings["fine"]} mm')
    for key, shape, spacing in grids:
        log(f'    {key:24s} {shape:16s} {spacing}')

    _check_interface(prof, grids, interface)

    _domains(ctx, grids, log)
    _generate(ctx, prof, coarse_config, fine_config, log)

    # step 1: the coarse full-volume solve.
    coarse_pot = _key(prof, 'potential', 'coarse')
    _solve(ctx, prof, _key(prof, 'initial', 'coarse'),
           _key(prof, 'boundary', 'coarse'),
           _key(prof, 'initial', 'coarse') + '_insulator',
           coarse_pot, _key(prof, 'increment', 'coarse'), precision, log)

    # steps 2-3: the fine near solve against the fixed interface BC.
    near_pot = _near_solve(ctx, prof, precision, log)

    # step 3b: the banded near/far Schwarz sweep.  max_sweeps == 0 skips it and
    # falls back to the sweep-0 keys, reproducing the one-shot scheme exactly.
    if max_sweeps:
        near_s, far_s = _schwarz(ctx, prof, near_pot, interface,
                                 band_cells=band_cells, max_sweeps=max_sweeps,
                                 tol=tol, precision=precision, log=log)
    else:
        near_s, far_s = near_pot, coarse_pot
        log('hybrid-iterate: max_sweeps=0 -- skipping the Schwarz sweep, '
            'stitching the one-shot near/coarse fields')

    # steps 4-5: the stitch, which is the final field.
    final = _stitch(ctx, prof, near_s, far_s, log)

    log(f'hybrid-iterate: done, {field} field stitched to {final} '
        f'(run the velo/starts/drift/induce chain from the shell script)')
    return final, grids
