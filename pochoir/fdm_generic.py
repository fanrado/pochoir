import sys

from pochoir import arrays


def edge_condition(arr, *periodic, info_msg=None):
    '''
    Apply N edge conditions (periodic if True, else fixed) to N-D array.
    '''
    np = len(periodic)
    na = len(arr.shape)
    if np != na:
        raise ValueError(f"dimension mismatch: {np} != {na}")

    # whole array slice
    slices = [slice(0,s) for s in arr.shape]
    for dim, per in enumerate(periodic):
        n = arr.shape[dim]
        src1 = list(slices)
        src2 = list(slices)
        dst1 = list(slices)
        dst2 = list(slices)

        dst1[dim] = slice(0,1)
        src1[dim] = slice(n-2, n-1)

        dst2[dim] = slice(n-1,n)
        src2[dim] = slice(1,2)

        if per:
            arr[tuple(dst1)] = arr[tuple(src1)]
            arr[tuple(dst2)] = arr[tuple(src2)]
        else:                   # fixed
            arr[tuple(dst1)] = arr[tuple(src2)]
            arr[tuple(dst2)] = arr[tuple(src1)]


def stencil(array, res=None):
    '''
    Return sum of 2N views of N-D array.

    Each view for a dimension is offset by +/- one cell.

    The shape of the returned array is reduced by two indices in each
    dimension.  If res is given, it must be of reduced size and it
    will be used to hold the result.
    '''
    slices = [slice(1, s-1) for s in array.shape]
    nd = len(slices)
    norm = 1/(2*nd)

    if res is None:
        core_shape = [s-2 for s in array.shape]
        amod = arrays.module(array)
        if arrays.is_torch(array):
            res = amod.zeros(core_shape, dtype=array.dtype, device=array.device)
        else:
            res = amod.zeros(core_shape)
    else:
        res[:] = 0

    for dim, n in enumerate(array.shape):
        pos = list(slices)
        pos[dim] = slice(2, n)
        res += array[tuple(pos)]

        neg = list(slices)
        neg[dim] = slice(0, n-2)
        res += array[tuple(neg)]
    res *= norm
    return res


def stencil_poisson(array, source=None, spacing=1.0, res=None):
    '''
    Return the Poisson-equation update for an N-D array.

    Solves  ∇²φ = -f  in the finite-difference sense.  Each interior
    cell is updated to

        φ_new = (1/(2N)) * Σ(neighbours) + (spacing² / (2N)) * f

    which is the standard Gauss-Seidel / Jacobi relaxation step for
    the Poisson equation.

    Parameters
    ----------
    array   : N-D array of the current field φ  (including boundary halo)
    source  : N-D array of the right-hand-side f at **interior cells only**
              (shape matches the reduced core shape, i.e. each dimension is
              ``array.shape[d] - 2``).  ``fdm_torch.solve`` constructs it
              this way from ``(s - non_padded_phi0) * mutable_core``.
    spacing : grid spacing h  [default 1.0]
    res     : optional pre-allocated output array of the reduced (interior)
              shape; created if not supplied

    Returns
    -------
    res : array of shape (s0-2, s1-2, …) with the updated interior values
    '''
    slices = [slice(1, s - 1) for s in array.shape]
    nd = len(slices)
    norm = 1.0 / (2 * nd)
    amod = arrays.module(array)

    if res is None:
        core_shape = [s - 2 for s in array.shape]
        if arrays.is_torch(array):
            res = amod.zeros(core_shape, dtype=array.dtype, device=array.device)
        else:
            res = amod.zeros(core_shape)
    else:
        res[:] = 0

    for dim, n in enumerate(array.shape):
        pos = list(slices)
        pos[dim] = slice(2, n)
        res += array[tuple(pos)]

        neg = list(slices)
        neg[dim] = slice(0, n - 2)
        res += array[tuple(neg)]
    res *= norm

    # Subtract the source term contribution: h² * f / (2N).
    # Sign convention: solving ∇²φ = -f, so the source contribution is
    # subtracted from the stencil result.
    if source is not None:
        res -= (spacing ** 2) * source * norm

    return res


def stencil_poisson_harmonic(phi, eps, res=None):
    '''
    Return the Poisson-equation update for phi with harmonically averaged
    interface permittivities, operating on the full interior at once.

    Computes for each interior cell:

        φ[i,j,k] = numerator / denominator

    where:
        numerator   = Σ_d  ( ε[i+½,…] * φ[i+1,…] + ε[i-½,…] * φ[i-1,…] )
        denominator = Σ_d  ( ε[i+½,…] + ε[i-½,…] )

    and ε[i±½,…] = 2*ε[i]*ε[i±1] / (ε[i] + ε[i±1])  (harmonic average).

    Parameters
    ----------
    phi : N-D array of the current potential field (including boundary halo)
    eps : N-D array of permittivities (same shape as phi)
    res : optional pre-allocated output array of reduced (interior) shape

    Returns
    -------
    res : array of shape (s0-2, s1-2, …) with the updated interior phi values
    '''
    slices = [slice(1, s - 1) for s in phi.shape]
    nd = len(slices)
    amod = arrays.module(phi)

    if res is None:
        core_shape = [s - 2 for s in phi.shape]
        if arrays.is_torch(phi):
            res = amod.zeros(core_shape, dtype=phi.dtype, device=phi.device)
        else:
            res = amod.zeros(core_shape)
    else:
        # res is reused across relaxation steps (fdm_torch passes tmp_core);
        # it must be zeroed each call or the `res += ...` accumulation below
        # grows without bound and the solve diverges. Mirrors stencil_poisson.
        res[:] = 0

    denom = amod.zeros_like(res)

    eps_core = eps[tuple(slices)]  # ε[i, j, k] for all interior points

    for dim, n in enumerate(phi.shape):
        # --- positive neighbour: i+1 along this dim ---
        pos = list(slices)
        pos[dim] = slice(2, n)
        eps_pos = eps[tuple(pos)]                              # ε[i+1, …]
        phi_pos = phi[tuple(pos)]                              # φ[i+1, …]
        e_pos = (2 * eps_core * eps_pos) / (eps_core + eps_pos)  # ε[i+½, …]

        # --- negative neighbour: i-1 along this dim ---
        neg = list(slices)
        neg[dim] = slice(0, n - 2)
        eps_neg = eps[tuple(neg)]                              # ε[i-1, …]
        phi_neg = phi[tuple(neg)]                              # φ[i-1, …]
        e_neg = (2 * eps_core * eps_neg) / (eps_core + eps_neg)  # ε[i-½, …]

        res   += e_pos * phi_pos + e_neg * phi_neg
        denom += e_pos + e_neg
    res /= denom
    return res


def neumann_coeff(solid_mask, like=None):
    '''
    Pre-compute the per-interior-cell multiplicative coefficient for the
    masked no-flux (Neumann) stencil :func:`stencil_poisson_neumann`.

    For a free cell adjacent to ``k`` insulator ("solid") neighbours the
    no-flux update averages over only its ``(2N - k)`` non-solid neighbours
    (mirror / ghost cell => zero normal derivative).  The coefficient is the
    reciprocal of that active-neighbour count::

        coeff = 1 / (number of non-solid neighbours)

    It depends only on the geometry, so it is built once and reused for every
    relaxation step -- the per-step stencil then stays branch-free and
    torch.compile-safe.

    Multiplying by this reciprocal, rather than dividing by the count, is what
    makes the all-``False`` mask case *bit-identical* to :func:`stencil_poisson`:
    with no solid cells every interior cell keeps its full ``2N`` neighbours, so
    ``coeff == 1/(2N) == norm`` uniformly and the update reduces exactly to the
    plain-Laplace relaxation.

    Parameters
    ----------
    solid_mask : N-D bool array (including boundary halo), True where the cell
                 is an excluded insulator ("solid") cell.
    like       : optional reference array (e.g. the field ``phi``) supplying the
                 dtype / device / backend of the returned coefficient.  When
                 omitted the coefficient follows ``solid_mask``'s backend in the
                 default float type.

    Returns
    -------
    coeff : array of the reduced (interior) shape holding 1 / active-count.

    Raises
    ------
    AssertionError
        If any *free* interior cell is fully enclosed by solid cells (zero
        active neighbours): such a cell is decoupled from the solve and the
        geometry is ill-posed.
    '''
    ref = solid_mask if like is None else like
    amod = arrays.module(ref)
    slices = [slice(1, s - 1) for s in solid_mask.shape]
    core_shape = [s - 2 for s in solid_mask.shape]

    # active neighbour = a cell that is NOT solid.  The coefficient dtype must
    # match the dtype the stencil's `res` accumulator uses, so that the final
    # `res *= coeff` reproduces stencil_poisson's `res *= norm` bit-for-bit:
    # torch res follows phi.dtype, while numpy `zeros` defaults to float64
    # regardless of the input dtype.
    if arrays.is_torch(ref):
        dtype = ref.dtype if like is not None else amod.float64
        count = amod.zeros(core_shape, dtype=dtype, device=ref.device)
        active_f = (~solid_mask).to(dtype)
    else:
        dtype = amod.float64
        count = amod.zeros(core_shape, dtype=dtype)
        active_f = (~solid_mask).astype(dtype)

    for dim, n in enumerate(solid_mask.shape):
        pos = list(slices)
        pos[dim] = slice(2, n)
        count += active_f[tuple(pos)]

        neg = list(slices)
        neg[dim] = slice(0, n - 2)
        count += active_f[tuple(neg)]

    # A free interior cell with zero active neighbours is fully enclosed by
    # insulator and cannot be relaxed; reject such geometries up front so the
    # per-step stencil never has to branch on it.
    free_interior = ~solid_mask[tuple(slices)]
    enclosed = free_interior & (count == 0)
    assert not bool(amod.any(enclosed)), \
        'insulator mask leaves fully-enclosed free cell(s); geometry ill-posed'

    # Clamp the count to >= 1 so solid cells (whose result is discarded
    # downstream) never produce inf/nan; free cells are guaranteed >= 1 above.
    denom = count + (count == 0)
    return 1.0 / denom


def stencil_poisson_neumann(phi, solid_mask, coeff, res=None):
    '''
    Return the source-free (Laplace) relaxation update for ``phi`` with a masked
    no-flux (Neumann) boundary on the ``solid_mask`` region.

    Each free interior cell is updated to the average of its **non-solid**
    neighbours::

        phi_new = ( sum over non-solid neighbours of phi ) * coeff

    where ``coeff = 1 / (number of non-solid neighbours)`` is pre-computed once
    by :func:`neumann_coeff`.  Dropping the solid neighbours from both the sum
    and the normalisation is a mirror / ghost-cell realisation of a zero normal
    derivative (dphi/dn = 0) on every face of the excluded region: field lines
    run tangential to the insulator and can only terminate on the conductors.

    NO permittivity / epsilon enters here -- this is plain Laplace on E with a
    reflecting mask, distinct from :func:`stencil_poisson_harmonic`.

    When ``solid_mask`` is all-``False`` the result is bit-identical to
    :func:`stencil_poisson` with ``source=None`` (every cell keeps its full
    ``2N`` neighbours and ``coeff == 1/(2N)``): the non-negotiable plain-Laplace
    regression.

    All operations are array-level with no data-dependent branching, so the
    step is torch.compile-safe.

    Parameters
    ----------
    phi        : N-D field array (including boundary halo).
    solid_mask : N-D bool array (same shape as ``phi``), True on excluded
                 insulator cells whose bonds to free cells are cut.
    coeff      : pre-computed reciprocal-active-neighbour-count array of the
                 reduced (interior) shape, from :func:`neumann_coeff`.
    res        : optional pre-allocated output of the reduced shape; created if
                 not supplied and zeroed on reuse (mirrors ``stencil_poisson``).

    Returns
    -------
    res : array of shape (s0-2, s1-2, ...) with the updated interior values.
    '''
    slices = [slice(1, s - 1) for s in phi.shape]
    amod = arrays.module(phi)

    if res is None:
        core_shape = [s - 2 for s in phi.shape]
        if arrays.is_torch(phi):
            res = amod.zeros(core_shape, dtype=phi.dtype, device=phi.device)
        else:
            res = amod.zeros(core_shape)
    else:
        # res is reused across relaxation steps (fdm_torch passes tmp_core); it
        # must be zeroed each call or the `res += ...` accumulation grows without
        # bound.  Mirrors stencil_poisson / stencil_poisson_harmonic.
        res[:] = 0

    # active neighbour value = phi where the neighbour is NOT solid, else 0.
    if arrays.is_torch(phi):
        active = (~solid_mask).to(phi.dtype)
    else:
        active = (~solid_mask).astype(phi.dtype)

    for dim, n in enumerate(phi.shape):
        pos = list(slices)
        pos[dim] = slice(2, n)
        res += phi[tuple(pos)] * active[tuple(pos)]

        neg = list(slices)
        neg[dim] = slice(0, n - 2)
        res += phi[tuple(neg)] * active[tuple(neg)]

    res *= coeff
    return res


def mirror_masks(insulator, lar):
    '''
    Pre-compute the geometry-only masks for the no-flux insulator boundary
    realised as an interface *mirror projection* (see
    :func:`mirror_project`).

    Unlike the :func:`neumann_coeff` / :func:`stencil_poisson_neumann` pair --
    which drops insulator neighbours from the free-cell relaxation and freezes
    the insulator region -- this scheme relaxes the whole volume with plain
    Laplace (:func:`stencil_poisson`) and imposes the no-flux face by copying
    each insulator *surface* cell to the average of its adjacent LAr cells.
    Setting the two nodes straddling an insulator--LAr face equal makes the
    normal derivative across that face zero (normal E ~ 0) while leaving the
    transverse variation of phi -- and hence the transverse E -- untouched.

    A cell is an insulator *surface* cell iff it is an insulator cell with at
    least one LAr (free, non-insulator, non-Dirichlet) neighbour.  Insulator
    cells buried under a conductor (LAr count 0) and interior insulator cells
    are left to plain Laplace so they still carry the field into the region
    below the slab -- no shielding.

    The masks depend only on the geometry, so they are built once and reused
    every relaxation step; the per-step projection is then branch-free and
    torch.compile-safe.

    Parameters
    ----------
    insulator : N-D bool array (including boundary halo), True on insulator
                cells.
    lar       : N-D bool array (same shape), True on free LAr cells
                (``(~insulator) & (~conductor)``), i.e. the cells whose phi is
                relaxed and which an insulator face mirrors against.

    Returns
    -------
    masks : dict with keys
            ``'dirs'`` : list of ``(axis, shift, m)`` where ``m`` is the bool
                         mask of insulator cells whose neighbour along
                         ``roll(., shift, axis)`` is LAr;
            ``'cnt'``  : float array, per-cell count of LAr neighbours;
            ``'surf'`` : bool array, True on insulator surface cells
                         (``cnt > 0``) -- the only cells the projection writes.

    Raises
    ------
    ValueError
        If ``insulator`` and ``lar`` disagree in shape, or overlap (a cell
        flagged both insulator and LAr is contradictory).
    '''
    if tuple(insulator.shape) != tuple(lar.shape):
        raise ValueError(
            f"shape mismatch: insulator {tuple(insulator.shape)} != "
            f"lar {tuple(lar.shape)}")

    amod = arrays.module(insulator)
    if bool(amod.any(insulator & lar)):
        raise ValueError("insulator and lar masks overlap; they must be disjoint")

    if arrays.is_torch(insulator):
        cnt = amod.zeros(insulator.shape, dtype=amod.float64, device=insulator.device)
    else:
        cnt = amod.zeros(insulator.shape, dtype=amod.float64)

    dirs = []
    for axis in range(insulator.ndim):
        for shift in (-1, 1):
            # neighbour along (axis, shift) is LAr?  roll(lar, shift, axis)[i]
            # is lar[i - shift]; iterating shift in (-1, +1) covers both the
            # +axis and -axis face neighbours.
            neigh_is_lar = amod.roll(lar, shift, axis)
            m = insulator & neigh_is_lar
            dirs.append((axis, shift, m))
            if arrays.is_torch(insulator):
                cnt = cnt + m.to(amod.float64)
            else:
                cnt = cnt + m.astype(amod.float64)

    surf = cnt > 0
    return {'dirs': dirs, 'cnt': cnt, 'surf': surf}


def mirror_project(phi, masks):
    '''
    Apply the insulator no-flux mirror projection to ``phi`` in one branch-free
    array pass, using the geometry masks from :func:`mirror_masks`.

    Each insulator *surface* cell is set to the average of its LAr neighbours::

        phi[surf] = ( sum over LAr-neighbour directions of phi_neighbour ) / cnt

    All other cells -- LAr, Dirichlet, buried/interior insulator -- are returned
    unchanged, so the plain-Laplace relaxation carries the field into the region
    below the slab and the transverse phi profile (transverse E) is preserved.

    The function is pure (returns a new array; does not mutate ``phi``), so it is
    torch.compile-safe and composes cleanly with the relaxation sweep.  Apply it
    each iteration after the plain :func:`stencil_poisson` update and before
    :func:`edge_condition`.

    Parameters
    ----------
    phi   : N-D field array (including boundary halo).
    masks : dict returned by :func:`mirror_masks` for this geometry.

    Returns
    -------
    phi_new : array of the same shape as ``phi`` with insulator surface cells
              overwritten by the mirror average and every other cell unchanged.
    '''
    amod = arrays.module(phi)
    acc = amod.zeros_like(phi)

    for axis, shift, m in masks['dirs']:
        if arrays.is_torch(phi):
            mf = m.to(phi.dtype)
        else:
            mf = m.astype(phi.dtype)
        # value of the LAr neighbour along (axis, shift): roll(phi, shift, axis).
        acc = acc + amod.roll(phi, shift, axis) * mf

    cnt = masks['cnt']
    surf = masks['surf']
    if arrays.is_torch(phi):
        cnt = cnt.to(phi.dtype)
    else:
        cnt = cnt.astype(phi.dtype)
    # clamp cnt to >= 1 so non-surface cells (cnt == 0) never divide by zero;
    # their result is discarded by the `surf` select below.
    denom = cnt + (cnt == 0)

    return amod.where(surf, acc / denom, phi)


def padplane_noflux_geom(barr, insulator=None):
    '''
    Locate the pad-plane no-flux (Neumann) interface for the *node-centered*
    mirror and pre-compute the geometry-only masks used by
    :func:`padplane_noflux`.

    This SUPERSEDES the face-centered :func:`mirror_masks` / :func:`mirror_project`
    pair for the pixel-plane case.  The face-centered mirror equates an insulator
    *body* cell to its LAr neighbour (phi[i] = phi[i+1]); that only zeros the
    normal field at the virtual half-node *between* two grid nodes -- a location
    the nodal central-difference E never samples -- so the field the drift
    integrator sees at the node is unchanged.  Here the interface is placed *on* a
    node (the pixel/pad plane): the two nodes flanking it along the normal are set
    equal (phi[z_pad-1] = phi[z_pad+1] on the gap columns), so the centered
    derivative ``-(phi[z_pad+1]-phi[z_pad-1])/(2*dz)`` is exactly zero AT the pad
    node itself.

    The interface is auto-derived from the Dirichlet geometry (no hand-crafted
    mask):

    - ``z_pad`` is the unique *partially* Dirichlet plane along the last axis
      (mix of fixed pad/grid cells and free gap cells).  A fully-Dirichlet plane
      (the cathode) or a fully-free plane is not an interface.
    - ``gap2d`` are the free (gap) nodes on that plane -- the nodes made no-flux.
    - ``drift_sign`` (+1/-1) points from the pad plane toward the LAr drift
      volume, taken as the side of the fully-Dirichlet (cathode) plane.  The node
      on the *opposite* (non-drift) side is the ghost that gets overwritten, so
      the drift-side field is never corrupted.

    Parameters
    ----------
    barr : N-D bool array (UNPADDED), True on Dirichlet (fixed) cells.  The normal
           to the pad plane is the last axis.
    insulator : N-D bool array (UNPADDED) or None.  The FR4 no-flux slab mask
           (the same array supplied to the solver).  Used ONLY to disambiguate
           the pad electrode when a shield grid is present: a shield grid is a
           SECOND, detached partially-Dirichlet plane (its apertures leave free
           cells), so the partial planes are no longer a single contiguous block.
           The no-flux interface belongs to the pad/FR4, so ``partial`` is
           restricted to the contiguous block carrying the FR4 slab; any other
           partial plane (the shield grid) is left as an ordinary Dirichlet
           electrode, untouched by the mirror.  ``None`` (no shield-grid path) is
           byte-identical to the original single-electrode behaviour.

    Returns
    -------
    masks : dict with keys
            ``'z_pad'``      : int, pad-plane index on the last axis;
            ``'gap2d'``      : bool array over the transverse axes, gap nodes;
            ``'drift_sign'`` : int, +1/-1 toward the drift (cathode) side;
            ``'ghost_mask'`` : full-shape bool, True on the ghost plane
                               (``z_pad - drift_sign``) at gap nodes -- the only
                               cells :func:`padplane_noflux` overwrites;
            ``'roll_shift'`` : int (``-2*drift_sign``); rolling phi by this along
                               ``axis`` brings the drift-side plane
                               (``z_pad + drift_sign``) onto the ghost plane;
            ``'axis'``       : int, the normal axis (last).

    Raises
    ------
    ValueError
        If there is no partially-Dirichlet plane, if the partial planes are not
        a contiguous block (a single electrode structure), or if the drift-facing
        pad plane is on the domain edge (ghost/drift neighbour out of range).
    '''
    amod = arrays.module(barr)
    axis = barr.ndim - 1
    plane_size = 1
    for d in range(barr.ndim - 1):
        plane_size *= barr.shape[d]

    # Dirichlet count per index along the normal axis (as a plain python list so
    # comparisons below are backend-agnostic).
    counts = barr.sum(axis=tuple(range(barr.ndim - 1)))
    counts = [int(c) for c in counts.tolist()]

    # Partially-Dirichlet planes = the electrode plane(s) mixing fixed pad/grid
    # cells with free gap cells.  A single-cell pad gives one such plane; a pad
    # given real thickness (>=2 cells) gives a CONTIGUOUS block of them.  Require
    # the block to be contiguous (one electrode structure) and take the no-flux
    # interface to be its DRIFT-FACING face -- the gap cells there sit next to the
    # drift volume, so that is where E_z must vanish.
    partial = sorted(z for z in range(barr.shape[axis])
                     if 0 < counts[z] < plane_size)
    if not partial:
        raise ValueError(
            f"padplane_noflux_geom: no partially-Dirichlet plane (the pad plane) "
            f"found; per-plane Dirichlet counts {counts} (plane_size={plane_size})")

    # Shield-grid disambiguation.  A shield grid (GridHoleShape square/circular)
    # is drawn as a solid plane with apertures well ABOVE the pad, so it appears
    # as a second partially-Dirichlet plane detached from the pad block -- the
    # partial planes are then non-contiguous.  The no-flux interface is the
    # pad/FR4, so when an insulator mask is supplied keep only the contiguous
    # run of `partial` that carries the FR4 slab (the insulator's z-plane(s),
    # which are bonded to the pad); every other partial plane (the shield grid)
    # is left out of the interface derivation and stays an ordinary Dirichlet
    # electrode fixed at its own potential, untouched by the mirror.  With
    # insulator=None, or when `partial` is already one block, this is a no-op so
    # non-shield-grid runs are byte-identical.
    if insulator is not None and partial != list(range(partial[0], partial[-1] + 1)):
        icounts = insulator.sum(axis=tuple(range(barr.ndim - 1)))
        insul_z = [z for z in range(barr.shape[axis]) if int(icounts[z]) > 0]
        if insul_z:
            # split `partial` into maximal contiguous runs
            runs, run = [], [partial[0]]
            for z in partial[1:]:
                if z == run[-1] + 1:
                    run.append(z)
                else:
                    runs.append(run); run = [z]
            runs.append(run)
            # the pad block is the run overlapping the FR4 slab (or, if the slab
            # sits on a fully-free plane, the run immediately adjacent to it).
            pad_runs = [r for r in runs
                        if any((iz in r) or (iz + 1 in r) or (iz - 1 in r)
                               for iz in insul_z)]
            if len(pad_runs) == 1:
                partial = pad_runs[0]

    if partial != list(range(partial[0], partial[-1] + 1)):
        raise ValueError(
            f"padplane_noflux_geom: partially-Dirichlet planes {partial} are not "
            f"contiguous; expected a single electrode block (per-plane counts "
            f"{counts})")
    zlo, zhi = partial[0], partial[-1]

    full = [int(z) for z in range(barr.shape[axis]) if counts[z] == plane_size]
    above = [z for z in full if z > zhi]
    below = [z for z in full if z < zlo]
    if above and not below:
        drift_sign = 1
    elif below and not above:
        drift_sign = -1
    elif above and below:
        # both sides bounded by a Dirichlet plane: drift is the larger free gap.
        drift_sign = 1 if (min(above) - zhi) >= (zlo - max(below)) else -1
    else:
        # no fully-Dirichlet plane found; assume the drift volume is toward +z.
        drift_sign = 1

    # drift-facing face of the electrode block: its high-z end if drift is +z.
    z_pad = zhi if drift_sign > 0 else zlo

    ghost_z = z_pad - drift_sign
    src_z = z_pad + drift_sign
    if not (0 <= ghost_z < barr.shape[axis]) or not (0 <= src_z < barr.shape[axis]):
        raise ValueError(
            f"padplane_noflux_geom: pad plane z_pad={z_pad} too close to the "
            f"domain edge for a node-centered mirror (ghost={ghost_z}, "
            f"src={src_z}, n={barr.shape[axis]})")

    # gap nodes = free (non-Dirichlet) cells on the pad plane.
    pad_idx = [slice(None)] * barr.ndim
    pad_idx[axis] = z_pad
    gap2d = ~barr[tuple(pad_idx)]

    # full-shape ghost mask: True only on the ghost plane at gap nodes.
    ghost_mask = amod.zeros_like(barr)
    ghost_idx = [slice(None)] * barr.ndim
    ghost_idx[axis] = ghost_z
    ghost_mask[tuple(ghost_idx)] = gap2d

    return {'z_pad': z_pad, 'gap2d': gap2d, 'drift_sign': drift_sign,
            'ghost_mask': ghost_mask, 'roll_shift': -2 * drift_sign, 'axis': axis}


def padplane_noflux(phi, masks):
    '''
    Apply the node-centered pad-plane no-flux mirror to ``phi`` in one branch-free
    array pass, using the geometry from :func:`padplane_noflux_geom`.

    Each ghost node (the non-drift-side neighbour of a gap node on the pad plane)
    is overwritten with the drift-side neighbour's value::

        phi[.., z_pad - drift_sign] = phi[.., z_pad + drift_sign]   (on gap nodes)

    so the centered z-derivative at the pad node is exactly zero (E_z = 0 at the
    node the drift integrator samples) while the transverse phi profile -- and
    hence the transverse E -- is untouched.  All other cells are unchanged, so the
    plain-Laplace relaxation carries the field into the region below the plane (no
    shielding).

    The function is pure (returns a new array; does not mutate ``phi``) and so is
    torch.compile-safe.  ``masks['ghost_mask']`` must match ``phi`` in shape,
    backend and device (the solver pads/moves it once at set-up); the drift-side
    value is fetched by rolling ``phi`` along the normal axis by
    ``masks['roll_shift']``.

    Parameters
    ----------
    phi   : N-D field array (including boundary halo).
    masks : dict returned by :func:`padplane_noflux_geom` (with ``ghost_mask``
            conformed to ``phi``).

    Returns
    -------
    phi_new : array of the same shape as ``phi`` with the ghost plane's gap nodes
              overwritten by the drift-side value and every other cell unchanged.
    '''
    amod = arrays.module(phi)
    rolled = amod.roll(phi, masks['roll_shift'], masks['axis'])
    return amod.where(masks['ghost_mask'], rolled, phi)
