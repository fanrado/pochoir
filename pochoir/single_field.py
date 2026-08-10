#!/usr/bin/env python3
'''
The SINGLE-SPACING field driver -- one grid, one solve, no near/far, no stitch.

This is `hybrid_iterate` with the hybrid taken out.  Same field profiles, same
generator, same edges, same fdm flags, same resume guard, and -- deliberately --
the same OUTPUT KEYS and the same final LATTICE, so a single-spacing run and a
hybrid run can be compared plane for plane:

    domain/<full_leaf> -> gen -> one fdm -> potential/<full_leaf>

where <full_leaf> is drift3d resp. weight3d, exactly as in the hybrid path.
`velo` and `induce-pixel` are invoked against those names from the runner
scripts, so they must not change.

EVERYTHING REUSED, NOTHING COPIED.  The field profiles and every step helper are
imported from `pochoir.hybrid_iterate`; this module contributes only the
single-grid SEQUENCE.  `hybrid_iterate` is not modified -- it drives the
validated task13 hybrid run.

Why the depth is still quantised to the COARSE spacing (see `_derive_shape`):
the cathode is a physical electrode and must not move when the run's resolution
changes, and the single-spacing grid is meant to be the SAME lattice as the
hybrid fine grid.  Both fall out of rounding the cathode plane to a whole
0.4mm cell regardless of the spacing actually being solved on.
'''

import json
from pathlib import Path

# Imported, never redefined.  These are hybrid_iterate's internals on purpose:
# the single-spacing path must stay bit-for-bit the same machinery as the
# hybrid path, so a divergence in behaviour is impossible by construction.
from pochoir.hybrid_iterate import (
    DEFAULT_SPACINGS,
    FULL_LEAF,
    _cells,
    _domains,
    _extents,
    _key,
    _output_key,
    _profile,
    _solve,
    _want,
)

# The single-spacing default: the fine 0.1mm lattice.
DEFAULT_SPACING = DEFAULT_SPACINGS['fine']

# The quantum the cathode plane is rounded up to, in mm -- the COARSE spacing,
# even though nothing here is solved at 0.4mm.  Two reasons, both load-bearing:
#
#   * the cathode must sit at a fixed PHYSICAL depth, independent of the
#     spacing the run happens to use (`_extents` makes the same argument);
#   * it is what makes the derived grid IDENTICAL to the hybrid run's fine grid
#     (44,44,701 / 220,220,701 at 0.1mm), which is the whole point of having a
#     single-spacing mode to compare against.
#
# Quantising to the solve spacing instead would put the cathode at 69.9mm and
# give 700 z-nodes at 0.1mm -- a different electrode position and a lattice the
# hybrid output could not be diffed against.
DEPTH_QUANTUM = DEFAULT_SPACINGS['coarse']

# `_extents` wants a near/far interface coordinate.  There is no near field
# here, so the value is unused; 0.0 keeps it out of the derivation.
NO_INTERFACE = 0.0


def _derive_shape(prof, cfg, spacing):
    '''
    The single full-depth grid derived from the config geometry, "nx,ny,nz".

    Transverse extent is the pixel pitch times the field's pixel count (1 for
    the periodic drift tile, Npixels for the weighting probe) and is NOT closed
    -- the far node is the wrap of the near one.  z spans the pad side to the
    cathode plane and IS closed, both faces being real planes.  Both conventions
    come from `_cells`; the cathode plane comes from `_extents` at DEPTH_QUANTUM.

    At the task13 validation geometry (driftZDepth 69.9 -> 70.0mm depth, 4.4mm
    pitch, Npixels 5) and 0.1mm this gives 44,44,701 for drift and
    220,220,701 for weighting.
    '''
    ext = _extents(prof, cfg, DEPTH_QUANTUM, NO_INTERFACE)
    nt = _cells(ext['transverse'], spacing, 'single transverse', closed=False)
    nz = _cells(ext['full'], spacing, 'single z', closed=True)
    return f'{nt},{nt},{nz}'


def _check_shape(shape):
    '''
    Validate a user-supplied "nx,ny,nz" and return it verbatim.

    Verbatim matters: the runner's shape table is the authority in that mode, so
    the string is checked for form and then passed through untouched rather than
    being re-derived or 'corrected' against the config.
    '''
    shape = str(shape).strip()
    parts = shape.split(',')
    if len(parts) != 3 or not all(p.strip().isdigit() for p in parts):
        raise ValueError(
            f'shape must be "nx,ny,nz" integers, got {shape!r}')
    return shape


def _generate_one(ctx, prof, config, log):
    '''
    `gen` for the single full-volume grid.

    hybrid_iterate._generate cannot be reused as-is: it loops over all three
    GRID_SPEC leaves (coarse, near, full) and would demand coarse/near domains
    that do not exist in this mode.  So the LOOP is what differs -- the step
    itself is the same `gen` invocation, guarded by the same imported `_want`
    and keyed by the same imported `_key`.

    `gen` also writes the no-flux insulator mask under "<initial>_insulator";
    --insulator is an enable signal only, the interface itself is derived from
    the Dirichlet geometry by padplane_noflux_geom.
    '''
    from pochoir.__main__ import gen

    dom = _key(prof, 'domain', FULL_LEAF)
    init, bnd = _key(prof, 'initial', FULL_LEAF), _key(prof, 'boundary', FULL_LEAF)
    _want(ctx, [init, bnd],
          lambda: ctx.invoke(gen, generator=prof['generator'], domain=dom,
                             initial=init, boundary=bnd, configs=(config,)),
          log)
    return init, bnd


def single_field(ctx, config, field='drift', spacing=DEFAULT_SPACING,
                 precision=2e-8, derive_domain=False, shape=None, log=None):
    '''
    Solve one field on ONE grid at ONE spacing -> the field's output key.

    `field` selects the profile in FIELDS: 'drift' (periodic single pixel tile
    -> potential/drift3d) or 'weighting' (non-periodic Npixels probe ->
    potential/weight3d).  Generator, edges and key names all follow from it, so
    the two fields can share a store without colliding.

    THE GRID.  `shape` is the PRIMARY input: when given it is used verbatim,
    which is the path the runner script takes (its shape table is the
    authority).  `derive_domain=True` instead computes the shape from the config
    geometry the way the hybrid driver does -- available, but not what the
    runner relies on.  Supplying neither is an error: silently falling back to
    derivation would let a typo'd shape flag change the lattice unnoticed.

    `precision` is the fdm convergence precision, the same knob and default as
    the hybrid path so the two modes converge to the same criterion.

    The chain after the field (velo / starts / drift / induce-pixel) is NOT part
    of this driver; it runs as explicit `pochoir` commands from the runner so
    its parameters stay hand-settable.

    Returns `(final_key, grid)` where grid is the single
    `(key, shape, spacing)` triple -- a one-element analogue of the hybrid
    driver's `grids`, so callers can report both modes the same way.
    '''
    from pochoir.__main__ import info_msg

    def _log(msg):
        info_msg(msg)
        print(msg)

    if log is None:
        log = _log

    prof = _profile(field)
    spacing = float(spacing)

    if shape:
        # Checked BEFORE derive_domain on purpose: an explicit shape is the
        # authority and overrides derivation, so passing both is not an error
        # and cannot silently give the caller a grid it did not ask for.
        shape_str = _check_shape(shape)
        source = 'supplied by the caller'
        if derive_domain:
            log('single-field: explicit shape given -- it overrides '
                'derive_domain, no shape derived from the config')
    elif derive_domain:
        cfg = json.loads(Path(config).read_text())
        shape_str = _derive_shape(prof, cfg, spacing)
        source = f'derived from {Path(config).name}'
    else:
        raise ValueError(
            'single_field needs an explicit shape, or derive_domain=True to '
            'compute one from the config; refusing to guess the grid')

    grid = (_key(prof, 'domain', FULL_LEAF), shape_str, f'{spacing}*mm')

    log(f'single-field: field={field} generator={prof["generator"]} '
        f'edges={prof["edges"]} output={_output_key(prof)}')
    log(f'single-field: one grid at {spacing}mm, shape {source}')
    log(f'    {grid[0]:24s} {grid[1]:16s} {grid[2]}')

    # The same three steps the hybrid driver runs, minus near/far/stitch.
    # `_domains` already takes a sequence of triples, so the single grid goes
    # through it unchanged.
    _domains(ctx, (grid,), log)
    init, bnd = _generate_one(ctx, prof, config, log)

    final = _key(prof, 'potential', FULL_LEAF)
    _solve(ctx, prof, init, bnd, init + '_insulator', final,
           _key(prof, 'increment', FULL_LEAF), precision, log)

    log(f'single-field: done, {field} field solved to {final} '
        f'(run the velo/starts/drift/induce chain from the shell script)')
    return final, grid
