import sys

from pochoir import arrays
import time
def edge_condition(arr, *periodic, info_msg=None):
    '''
    Apply N edge conditions (periodic if True, else fixed) to N-D array.
    '''
    # e0_time = time.time()
    np = len(periodic)
    na = len(arr.shape)
    if np != na:
        raise ValueError(f"dimension mismatch: {np} != {na}")
    
    # whole array slice
    slices = [slice(0,s) for s in arr.shape]
    # e1_time = time.time()
    # info_msg(f'edge_condition init = {e1_time - e0_time:.2f} seconds')
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
    # info_msg(f'edge_condition = {time.time() - e1_time:.4f} seconds')


def stencil(array, res = None):
    '''
    Return sum of 2N views of N-D array.

    Each view for a dimension is offset by +/- one cell.

    The shape of the returned array is reduced by two indices in each
    dimension.  If res is given, it must be of reduced size and it
    will be used to hold the result.
    '''
    # stencil0_time = time.time()
    # whole array slice
    slices = [slice(1,s-1) for s in array.shape]
    nd = len(slices)
    norm = 1/(2*nd)

    if res is None:
        core_shape = [s-2 for s in array.shape]
        # core_shape = [s for s in array.shape]
        amod = arrays.module(array)
        if arrays.is_torch(array):
            res = amod.zeros(core_shape, dtype=array.dtype, device=array.device)
        else:
            res = amod.zeros(core_shape)
    else:
        res[:] = 0
    # stencil1_time = time.time()
    # info_msg(f'tmp_core init = {stencil1_time - stencil0_time:.2f} seconds')
    for dim, n in enumerate(array.shape):
        pos = list(slices)
        pos[dim] = slice(2,n)
        res += array[tuple(pos)]

        neg = list(slices)
        neg[dim] = slice(0,n-2)
        res += array[tuple(neg)]
    # s = time.time()
    # info_msg(f'stencil compute = {s - stencil1_time:.4f} seconds')
    res *= norm
def stencil(array, res = None):
    '''
    Return sum of 2N views of N-D array.

    Each view for a dimension is offset by +/- one cell.

    The shape of the returned array is reduced by two indices in each
    dimension.  If res is given, it must be of reduced size and it
    will be used to hold the result.
    '''
    # stencil0_time = time.time()
    # whole array slice
    slices = [slice(1,s-1) for s in array.shape]
    nd = len(slices)
    norm = 1/(2*nd)

    if res is None:
        core_shape = [s-2 for s in array.shape]
        # core_shape = [s for s in array.shape]
        amod = arrays.module(array)
        if arrays.is_torch(array):
            res = amod.zeros(core_shape, dtype=array.dtype, device=array.device)
        else:
            res = amod.zeros(core_shape)
    else:
        res[:] = 0
    # stencil1_time = time.time()
    # info_msg(f'tmp_core init = {stencil1_time - stencil0_time:.2f} seconds')
    for dim, n in enumerate(array.shape):
        pos = list(slices)
        pos[dim] = slice(2,n)
        res += array[tuple(pos)]

        neg = list(slices)
        neg[dim] = slice(0,n-2)
        res += array[tuple(neg)]
    # s = time.time()
    # info_msg(f'stencil compute = {s - stencil1_time:.4f} seconds')
    res *= norm
    # info_msg(f'stencil norm = {time.time() - s:.4f} seconds')
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
    source  : N-D array of the right-hand-side f  (same shape as array)
              The core region (interior cells) is used; boundary values
              are ignored.
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
        # core_shape = [s for s in array.shape]
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
    
    # Subtract the source term contribution: h² * f / (2N)
    if source is not None:
        res -= (spacing ** 2) * source*norm ## corrected the sign here, since we are solving ∇²φ = -f = source, 
                        #so the source term contribution should be subtracted from the stencil result to get the updated φ values.

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
    print(f'stencil_poisson_harmonic: max denom = {denom.max()}, min denom = {denom.min()}')
    sys.exit()
    res /= denom
    return res