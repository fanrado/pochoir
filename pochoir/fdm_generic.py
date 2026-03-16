from pochoir import arrays

# Optional torch import for compiled conv-based stencil path
_HAS_TORCH = False
try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    pass

# Kernel cache: maps (ndim, dtype, device) → averaging-Laplacian kernel tensor
_kernel_cache = {}


def _get_stencil_kernel(ndim, dtype, device):
    """
    Build and cache the averaging-neighbour stencil kernel for F.conv{2,3}d.

    The kernel has weight 1/(2*ndim) at each of the 2*ndim face-adjacent
    neighbours and 0 at the centre, matching the slicing-based stencil.
    """
    key = (ndim, dtype, device)
    if key not in _kernel_cache:
        shape = [1, 1] + [3] * ndim
        k = torch.zeros(shape, dtype=dtype, device=device)
        center = [0, 0] + [1] * ndim
        for d in range(ndim):
            lo = list(center); lo[2 + d] = 0
            hi = list(center); hi[2 + d] = 2
            k[tuple(lo)] = 1.0
            k[tuple(hi)] = 1.0
        k /= (2 * ndim)
        _kernel_cache[key] = k
    return _kernel_cache[key]


if _HAS_TORCH:
    @torch.compile(fullgraph=True)
    def _stencil_conv2d(array, kernel):
        """Single fused conv2d averaging stencil."""
        return F.conv2d(array.unsqueeze(0).unsqueeze(0), kernel).squeeze(0).squeeze(0)

    @torch.compile(fullgraph=True)
    def _stencil_conv3d(array, kernel):
        """Single fused conv3d averaging stencil."""
        return F.conv3d(array.unsqueeze(0).unsqueeze(0), kernel).squeeze(0).squeeze(0)

    @torch.compile(fullgraph=True)
    def _stencil_poisson_conv2d(array, kernel, source_core):
        """Poisson relaxation update via conv2d: (1/2N)*Σneighbours − source_core."""
        return F.conv2d(array.unsqueeze(0).unsqueeze(0), kernel).squeeze(0).squeeze(0) - source_core

    @torch.compile(fullgraph=True)
    def _stencil_poisson_conv3d(array, kernel, source_core):
        """Poisson relaxation update via conv3d: (1/2N)*Σneighbours − source_core."""
        return F.conv3d(array.unsqueeze(0).unsqueeze(0), kernel).squeeze(0).squeeze(0) - source_core


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
    if arrays.is_torch(array):
        ndim = array.dim()
        kernel = _get_stencil_kernel(ndim, array.dtype, array.device)
        if ndim == 2:
            result = _stencil_conv2d(array, kernel)
        else:
            result = _stencil_conv3d(array, kernel)
        if res is not None:
            res.copy_(result)
            return res
        return result

    # numpy / cupy / numba path (unchanged)
    slices = [slice(1,s-1) for s in array.shape]
    nd = len(slices)
    norm = 1/(2*nd)

    if res is None:
        core_shape = [s-2 for s in array.shape]
        amod = arrays.module(array)
        res = amod.zeros(core_shape)
    else:
        res[:] = 0

    for dim, n in enumerate(array.shape):
        pos = list(slices)
        pos[dim] = slice(2,n)
        res += array[tuple(pos)]

        neg = list(slices)
        neg[dim] = slice(0,n-2)
        res += array[tuple(neg)]

    res *= norm
    return res


def stencil_poisson(array, source=None, spacing=1.0, res=None):
    '''
    Return the Poisson-equation update for an N-D array.

    Solves  ∇²φ = f  in the finite-difference sense.  Each interior
    cell is updated to

        φ_new = (1/(2N)) * Σ(neighbours) - f_core

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
    if arrays.is_torch(array):
        ndim = array.dim()
        kernel = _get_stencil_kernel(ndim, array.dtype, array.device)
        if source is not None:
            slices = tuple(slice(1, s - 1) for s in array.shape)
            source_core = source[slices]
            if ndim == 2:
                result = _stencil_poisson_conv2d(array, kernel, source_core)
            else:
                result = _stencil_poisson_conv3d(array, kernel, source_core)
        else:
            if ndim == 2:
                result = _stencil_conv2d(array, kernel)
            else:
                result = _stencil_conv3d(array, kernel)
        if res is not None:
            res.copy_(result)
            return res
        return result

    # numpy / cupy / numba path (unchanged)
    slices = [slice(1, s - 1) for s in array.shape]
    nd = len(slices)
    norm = 1.0 / (2 * nd)
    source_scaled = 2*nd * source / (spacing ** 2) if source is not None else source
    amod = arrays.module(array)

    if res is None:
        core_shape = [s - 2 for s in array.shape]
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

    if source_scaled is not None:
        res -= (spacing ** 2) * source_scaled[tuple(slices)]

    res *= norm
    return res
