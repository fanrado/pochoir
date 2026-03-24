"""
3D Weighting Field IDW Initialisation — PyTorch / GPU
=======================================================
Computes an Inverse Distance Weighting (IDW) initial guess for the weighting
field on a 3D grid, using PyTorch for GPU acceleration.

Key ideas
---------
* All heavy work (distance computation, weighting) runs on the GPU via torch.
* Electrode pixels are processed in batches to stay within VRAM limits.
* Falls back to CPU automatically if no CUDA device is available.
* The final tensor can be handed directly to a GPU-based iterative solver.
"""

import torch
import numpy as np
import sys

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_dynamic_batch_size(N: int, dtype: torch.dtype, dev: torch.device) -> int:
    """Compute GPU-memory-aware batch size for the electrode voxel loop.

    Targets 1/8 of total VRAM for the (batch × N) distance matrix so that
    large grids do not exhaust memory.  Falls back to 1024 on CPU.
    """
    bytes_per_elem = torch.finfo(dtype).bits // 8
    if dev.type == "cuda":
        props = torch.cuda.get_device_properties(dev)
        available = props.total_memory // 8  # use at most 1/8 of total VRAM
        return max(256, int(available // (N * bytes_per_elem)))
    return 1024


def _idw_dist_batch(
    elec_batch: torch.Tensor,
    grid_coords: torch.Tensor,
    grid_chunk: int = 65536,
) -> torch.Tensor:
    """Min distance from each grid voxel to a batch of electrode voxels.

    Parameters
    ----------
    elec_batch  : (B, D) float — electrode coords in this batch
    grid_coords : (N, D) float — all grid voxel coords
    grid_chunk  : max number of grid voxels to process at once.
                  Limits peak memory to O(B * grid_chunk) instead of O(B * N).
                  Set to 0 to disable chunking (materialises the full matrix).
    Returns     : (N,)   float — per-voxel minimum distance to this batch

    Note: @torch.compile is intentionally omitted.  The inductor backend
    benchmarks pad_mm candidates by allocating full (B, N) trial tensors,
    which OOMs when N is large (e.g. z_chunk * ny * nx).  torch.cdist is
    already dispatched to an optimised CUDA kernel without compilation.
    """
    N = grid_coords.shape[0]
    if grid_chunk <= 0 or N <= grid_chunk:
        return torch.cdist(elec_batch, grid_coords).min(dim=0).values  # (B, N) → (N,)

    d_min = torch.empty(N, device=grid_coords.device, dtype=grid_coords.dtype)
    for start in range(0, N, grid_chunk):
        end = min(start + grid_chunk, N)
        d_min[start:end] = torch.cdist(elec_batch, grid_coords[start:end]).min(dim=0).values
    return d_min


def _mask_to_device(mask, dev: torch.device) -> torch.Tensor:
    """Convert a mask (numpy or torch) to *dev*, pinning memory when CUDA."""
    if isinstance(mask, np.ndarray):
        cpu_mask = torch.from_numpy(mask)
    else:
        cpu_mask = mask.cpu() if mask.device.type == "cuda" else mask
    if dev.type == "cuda":
        cpu_mask = cpu_mask.pin_memory()
    return cpu_mask.to(dev)


# ---------------------------------------------------------------------------
# Core IDW function
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 2-D IDW core
# ---------------------------------------------------------------------------

def init_idw_2d_torch(
    grid_shape: tuple,
    electrodes: list,
    power: float = 1.0,
    batch_size: int = 0,
    device: str | None = None,
    dtype: torch.dtype = torch.float32,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    IDW initial guess for a 2D weighting field using PyTorch.

    Parameters
    ----------
    grid_shape : (ny, nx) — size of the 2D grid
    electrodes : list of dicts, each with:
                   'mask'    — (ny, nx) bool numpy array or torch BoolTensor
                   'voltage' — float
    power      : IDW exponent
    batch_size : number of electrode pixels per GPU batch.
                 0 (default) = auto-size from GPU memory.
    device     : 'cuda', 'cpu', or None (auto-detect)
    dtype      : torch float dtype (float32 recommended for GPU)

    Returns
    -------
    phi : torch.Tensor of shape (ny, nx) on `device`
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    print(f"Running 2D IDW on: {dev}"
          + (f" [{torch.cuda.get_device_name(dev)}]"
             if dev.type == "cuda" else ""))

    ny, nx = grid_shape

    jj, ii = torch.meshgrid(
        torch.arange(ny, device=dev, dtype=dtype),
        torch.arange(nx, device=dev, dtype=dtype),
        indexing="ij",
    )
    jj_flat = jj.reshape(-1)   # (N,)
    ii_flat = ii.reshape(-1)
    N = jj_flat.shape[0]

    # Pre-stack grid coords once for reuse across electrodes
    grid_coords = torch.stack([jj_flat, ii_flat], dim=1)  # (N, 2)

    # Dynamic batch size from GPU memory
    if batch_size <= 0:
        batch_size = _compute_dynamic_batch_size(N, dtype, dev)

    numerator   = torch.zeros(N, device=dev, dtype=dtype)
    denominator = torch.zeros(N, device=dev, dtype=dtype)

    for elec in electrodes:
        mask    = _mask_to_device(elec["mask"], dev)
        voltage = float(elec["voltage"])

        ej, ei = torch.where(mask)
        n_elec = ej.shape[0]
        if n_elec == 0:
            continue

        ej = ej.to(dtype)
        ei = ei.to(dtype)

        d_min = torch.full((N,), float("inf"), device=dev, dtype=dtype)

        for start in range(0, n_elec, batch_size):
            end = min(start + batch_size, n_elec)
            elec_batch = torch.stack(
                [ej[start:end], ei[start:end]], dim=1
            )  # (B, 2)
            d_min = torch.minimum(d_min, _idw_dist_batch(elec_batch, grid_coords))

        d_min = torch.clamp(d_min, min=1e-10)

        w = 1.0 / d_min**power
        numerator   += w * voltage
        denominator += w

    phi_flat = torch.where(denominator > 0, numerator / denominator,
                           torch.zeros_like(numerator))

    if out is not None:
        out.view(-1).copy_(phi_flat)
        phi = out
    else:
        phi = phi_flat.reshape(ny, nx)

    for elec in electrodes:
        mask = _mask_to_device(elec["mask"], dev)
        phi[mask] = float(elec["voltage"])

    return phi


# ---------------------------------------------------------------------------
# PCB pixel geometry helpers  (mirror of gen_pcb_pixel_with_grid.trimCorner)
# ---------------------------------------------------------------------------

def _apply_trim_corner(
    mask: np.ndarray,
    x: int, y: int,
    z1: int, z2: int,
    corner: int,
) -> None:
    """
    Zero-out (set False) the rounded corner voxels of a pixel in a bool mask.
    Replicates trimCorner() from gen_pcb_pixel_with_grid.py exactly.
    """
    if corner == 0:          # top-right  (x_max, y_max)
        mask[x-3:x+1, y,     z1:z2] = False
        mask[x,     y-3:y+1, z1:z2] = False
        mask[x,     y,       z1:z2] = False
        mask[x-1,   y-1,     z1:z2] = False
    elif corner == 1:        # top-left   (x_max, y_min)
        mask[x-3:x+1, y,     z1:z2] = False
        mask[x,     y:y+4,   z1:z2] = False
        mask[x,     y,       z1:z2] = False
        mask[x-1,   y+1,     z1:z2] = False
    elif corner == 2:        # bottom-left (x_min, y_min)
        mask[x:x+4, y,       z1:z2] = False
        mask[x,     y:y+4,   z1:z2] = False
        mask[x,     y,       z1:z2] = False
        mask[x+1,   y+1,     z1:z2] = False
    elif corner == 3:        # bottom-right (x_min, y_max)
        mask[x:x+4, y,       z1:z2] = False
        mask[x,     y-3:y+1, z1:z2] = False
        mask[x,     y,       z1:z2] = False
        mask[x+1,   y-1,     z1:z2] = False


## 3D
def init_idw_3d_torch(
    grid_shape: tuple,
    electrodes: list,
    power: float = 1.0,
    batch_size: int = 0,
    device: str | None = None,
    dtype: torch.dtype = torch.float32,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    IDW initial guess for a 3D weighting field using PyTorch.

    Parameters
    ----------
    grid_shape : (nz, ny, nx) — size of the 3D grid
    electrodes : list of dicts, each with:
                   'mask'    — (nz, ny, nx) bool numpy array or torch BoolTensor
                   'voltage' — float
    power      : IDW exponent (1 = linear, 2 = quadratic drop-off)
    batch_size : number of electrode voxels per GPU batch.
                 0 (default) = auto-size from GPU memory.
    device     : 'cuda', 'cpu', or None (auto-detect)
    dtype      : torch float dtype (float32 recommended for GPU)

    Returns
    -------
    phi : torch.Tensor of shape (nz, ny, nx) on `device`
    """
    # ---- Device selection ------------------------------------------------
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    print(f"Running IDW on: {dev}"
          + (f" [{torch.cuda.get_device_name(dev)}]"
             if dev.type == "cuda" else ""))

    nz, ny, nx = grid_shape

    # ---- Build coordinate grids (on GPU) ---------------------------------
    # Each has shape (nz, ny, nx)
    kk, jj, ii = torch.meshgrid(
        torch.arange(nz, device=dev, dtype=dtype),
        torch.arange(ny, device=dev, dtype=dtype),
        torch.arange(nx, device=dev, dtype=dtype),
        indexing="ij",
    )
    # Flatten to (N,) for vectorised distance computation
    kk_flat = kk.reshape(-1)   # (N,)
    jj_flat = jj.reshape(-1)
    ii_flat = ii.reshape(-1)
    N = kk_flat.shape[0]       # total number of voxels = nz*ny*nx

    # Pre-stack grid coords once for reuse across electrodes
    grid_coords = torch.stack([kk_flat, jj_flat, ii_flat], dim=1)  # (N, 3)

    # Dynamic batch size from GPU memory
    if batch_size <= 0:
        batch_size = _compute_dynamic_batch_size(N, dtype, dev)

    numerator   = torch.zeros(N, device=dev, dtype=dtype)
    denominator = torch.zeros(N, device=dev, dtype=dtype)

    # ---- Loop over electrodes --------------------------------------------
    for elec in electrodes:
        mask    = _mask_to_device(elec["mask"], dev)
        voltage = float(elec["voltage"])

        # Electrode voxel coordinates  shape (n_elec,)
        ek, ej, ei = torch.where(mask)
        n_elec = ek.shape[0]
        if n_elec == 0:
            continue
        print(f"Processing electrode with voltage {voltage:.2f} V, "
              f"{n_elec} voxels")
        ek = ek.to(dtype)
        ej = ej.to(dtype)
        ei = ei.to(dtype)

        # Batch over electrode voxels to bound peak memory to
        # O(batch_size * N) instead of O(n_elec * N).
        d_min = torch.full((N,), float("inf"), device=dev, dtype=dtype)
        for start in range(0, n_elec, batch_size):
            end = min(start + batch_size, n_elec)
            elec_batch = torch.stack(
                [ek[start:end], ej[start:end], ei[start:end]], dim=1
            )  # (B, 3)
            d_min = torch.minimum(d_min, _idw_dist_batch(elec_batch, grid_coords))

        # Avoid division by zero directly on electrode voxels
        d_min = torch.clamp(d_min, min=1e-10)

        w = 1.0 / d_min**power         # (N,)
        numerator   += w * voltage
        denominator += w

    # ---- Normalise -------------------------------------------------------
    phi_flat = torch.where(denominator > 0, numerator / denominator,
                           torch.zeros_like(numerator))

    if out is not None:
        out.view(-1).copy_(phi_flat)
        phi = out
    else:
        phi = phi_flat.clone()

    # ---- Enforce exact boundary values on electrode voxels ---------------
    for elec in electrodes:
        mask = elec["mask"]
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).to(dev)
        if phi.dim() == 1:
            phi[mask.reshape(-1)] = float(elec["voltage"])
        else:
            phi[mask] = float(elec["voltage"])

    return phi
# ---------------------------------------------------------------------------
# PCB pixel IDW initialisation
# ---------------------------------------------------------------------------

def init_idw_pcb_pixel(
    arr: "torch.Tensor | np.ndarray",
    p_size: int,
    p_gap: int,
    n_pix: int,
    target_voltage: float = 1.0,
    ground_voltage: float = 0.0,
    power: float = 1.0,
    batch_size: int = 0,
    device: str | None = None,
    dtype: torch.dtype = torch.float32,
    dim: int = 2,
    z_electrode_start: int = 0,
    z_electrode_end: int | None = None,
) -> torch.Tensor:
    """
    IDW initial guess for a 2D or 3D weighting field using the PCB pixel geometry.

    The target electrode is the centre pixel of the n_pix × n_pix grid
    (index i == j == int(n_pix / 2)).  All other pixels form the ground
    electrode.

    Parameters
    ----------
    arr              : array whose shape defines the domain:
                         dim=2 → (nx, ny)
                         dim=3 → (nx, ny, nz)
    p_size           : pixel size  in grid units
    p_gap            : pixel gap   in grid units
    n_pix            : number of pixels per side
    target_voltage   : voltage on the centre pixel   [default 1.0]
    ground_voltage   : voltage on all other pixels   [default 0.0]
    power            : IDW exponent
    batch_size       : electrode-pixel batch size for GPU (0 = auto from VRAM)
    device           : 'cuda', 'cpu', or None (auto-detect)
    dtype            : torch float dtype
    z_electrode_start: (dim=3 only) first z index of the electrode slab
    z_electrode_end  : (dim=3 only) one-past-last z index of the electrode slab.
                       None = nz (electrode spans the full z range — not
                       recommended; gives z-independent results).

    Returns
    -------
    phi : torch.Tensor of shape (nx, ny) or (nx, ny, nz) on `device`
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    print(f"Running {dim}D IDW on: {dev}"
          + (f" [{torch.cuda.get_device_name(dev)}]" if dev.type == "cuda" else ""))

    # Convert arr to a torch tensor on device — this is the output buffer.
    # .to() is a no-op when arr is already the correct dtype/device.
    if isinstance(arr, np.ndarray):
        arr_t = torch.from_numpy(arr).to(dev, dtype=dtype)
    else:
        arr_t = arr.to(dev=dev, dtype=dtype)

    N = arr_t.numel()
    center = int(n_pix / 2)

    # Build flat grid coordinates once; reused across all electrodes.
    # For dim==3 the IDW is computed on the 2D electrode plane (x, y only) and
    # then scaled by a linear z-decay, so N = nx*ny regardless of nz.
    if dim == 2:
        nx, ny = arr_t.shape[0], arr_t.shape[1]
        gx, gy = torch.meshgrid(
            torch.arange(nx, device=dev, dtype=dtype),
            torch.arange(ny, device=dev, dtype=dtype),
            indexing="ij",
        )
        grid_coords = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)  # (N, 2)
        N = nx * ny
    else:  # dim == 3
        nx, ny, nz = arr_t.shape[0], arr_t.shape[1], arr_t.shape[2]
        z_end = z_electrode_end if z_electrode_end is not None else nz
        gx, gy = torch.meshgrid(
            torch.arange(nx, device=dev, dtype=dtype),
            torch.arange(ny, device=dev, dtype=dtype),
            indexing="ij",
        )
        grid_coords = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)  # (nx*ny, 2)
        N = nx * ny

    if batch_size <= 0:
        batch_size = _compute_dynamic_batch_size(N, dtype, dev)

    # IDW accumulators — written in-place throughout.
    numerator   = torch.zeros(N, device=dev, dtype=dtype)
    denominator = torch.zeros(N, device=dev, dtype=dtype)

    # Single mask tensor — zeroed and refilled per electrode, never reallocated.
    mask = torch.zeros(arr_t.shape, dtype=torch.bool, device=dev)

    # Collect 2-D boundary indices per electrode for boundary enforcement.
    # For dim==3 we store (nx*ny)-space indices derived from the xy projection.
    boundaries: list[tuple[torch.Tensor, float]] = []

    for is_target in (True, False):
        voltage = target_voltage if is_target else ground_voltage

        mask.zero_()
        for i in range(n_pix):
            for j in range(n_pix):
                x0 = int(p_gap / 2) + i * (p_size + p_gap)
                x1 = x0 + p_size
                y0 = int(p_gap / 2) + j * (p_size + p_gap)
                y1 = y0 + p_size
                if (i == center and j == center) == is_target:
                    if dim == 2:
                        mask[x0:x1, y0:y1] = True
                    else:
                        mask[x0:x1, y0:y1, z_electrode_start:z_end] = True

        if dim == 2:
            flat_idx = mask.view(-1).nonzero(as_tuple=False).squeeze(1)
            coords = torch.where(mask)
        else:
            # Project the 3-D electrode slab onto the xy plane for 2-D IDW.
            mask_2d = mask.any(dim=2)                                          # (nx, ny)
            flat_idx = mask_2d.view(-1).nonzero(as_tuple=False).squeeze(1)    # (nx*ny)-space
            coords = torch.where(mask_2d)

        boundaries.append((flat_idx, voltage))

        n_elec = coords[0].shape[0]
        if n_elec == 0:
            continue
        print(f"  Electrode voltage={voltage:.2f} V, {n_elec} voxels")

        elec_coords = torch.stack([c.to(dtype) for c in coords], dim=1)  # (n_elec, 2)

        d_min = torch.full((N,), float("inf"), device=dev, dtype=dtype)
        for start in range(0, n_elec, batch_size):
            end = min(start + batch_size, n_elec)
            torch.minimum(d_min, _idw_dist_batch(elec_coords[start:end], grid_coords), out=d_min)

        d_min.clamp_(min=1e-10)
        d_min.pow_(-power)              # in-place: d_min now holds weights = 1/d^power
        numerator.add_(d_min, alpha=voltage)
        denominator.add_(d_min)

    if dim == 2:
        # Write 2-D IDW result directly into arr_t.
        flat = arr_t.view(-1)
        valid = denominator > 0
        flat[valid]  = numerator[valid] / denominator[valid]
        flat[~valid] = 0.0
        for flat_idx, voltage in boundaries:
            flat[flat_idx] = voltage
    else:
        # 2-D IDW gives the lateral (x,y) distribution at the electrode plane.
        # Multiply by a linear z-decay (1 at z_electrode_start → 0 at z=nz-1)
        # so the field decreases monotonically with depth.
        valid = denominator > 0
        numerator[valid] = numerator[valid] / denominator[valid]
        numerator[~valid] = 0.0
        idw_2d = numerator.reshape(nx, ny)                                     # (nx, ny)

        nz_span = max(nz - 1 - z_electrode_start, 1)
        decay = ((nz - 1 - torch.arange(nz, device=dev, dtype=dtype)) / nz_span).clamp_(min=0.0)
        # decay[z_electrode_start] == 1, decay[nz-1] == 0

        arr_t[:, :, :z_electrode_start] = 0.0
        arr_t[:, :, z_electrode_start:] = idw_2d.unsqueeze(-1) * decay[z_electrode_start:]

        # Enforce exact electrode voltages on the electrode slab.
        for flat_idx, voltage in boundaries:
            mask_2d = torch.zeros(N, dtype=torch.bool, device=dev)
            mask_2d[flat_idx] = True
            arr_t[:, :, z_electrode_start:z_end][mask_2d.reshape(nx, ny)] = voltage

    return arr_t
        



# ---------------------------------------------------------------------------
# Plot helper for init_idw_pcb_pixel output
# ---------------------------------------------------------------------------

def plot_idw_pcb_pixel(
    phi: "torch.Tensor | np.ndarray",
    title: str = "init_idw_pcb_pixel — IDW initial guess",
    save_path: str | None = None,
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap: str = "RdBu_r",
) -> None:
    """
    Plot the 2D weighting field returned by init_idw_pcb_pixel.

    Parameters
    ----------
    phi       : torch.Tensor or numpy array of shape (nx, ny)
    title     : figure title
    save_path : if given, save the figure to this path
    vmin/vmax : colour-scale limits
    cmap      : matplotlib colourmap
    """
    import matplotlib.pyplot as plt

    if hasattr(phi, "cpu"):
        phi_np = phi.cpu().numpy()
    else:
        phi_np = np.asarray(phi)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(phi_np.T, origin="lower", cmap=cmap,
                   vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xlabel("x-index")
    ax.set_ylabel("y-index")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="φ (V)")
    plt.title(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved → {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Test for init_idw_pcb_pixel
# ---------------------------------------------------------------------------

def test_init_idw_pcb_pixel(save_path: str = "idw_pcb_pixel_slices.png") -> None:
    """
    Test init_idw_pcb_pixel on a small PCB pixel geometry and plot the
    resulting 2D phi array.
    """
    import time

    p_size = 38
    p_gap  = 6
    n_pix  = 5

    xy_size = int(p_gap / 2) + n_pix * (p_size + p_gap) + int(p_gap / 2)
    arr = np.zeros((xy_size, xy_size), dtype=np.float32)

    center    = int(n_pix / 2)
    x0_tgt    = int(p_gap / 2) + center * (p_size + p_gap)
    x_tgt_mid = x0_tgt + p_size // 2
    y_tgt_mid = x_tgt_mid

    print("=" * 60)
    print("test_init_idw_pcb_pixel (2D)")
    print(f"  grid_shape = {arr.shape}")
    print(f"  n_pix={n_pix}  p_size={p_size}  p_gap={p_gap}")
    print(f"  target pixel centre: x={x_tgt_mid}, y={y_tgt_mid}")
    print("=" * 60)

    t0  = time.time()
    phi = init_idw_pcb_pixel(
        arr,
        p_size=p_size,
        p_gap=p_gap,
        n_pix=n_pix,
        target_voltage=1.0,
        ground_voltage=0.0,
        power=1.0,
        batch_size=128,
    )
    elapsed = time.time() - t0

    phi_np = phi.cpu().numpy()
    print(f"  Elapsed   : {elapsed:.2f} s")
    print(f"  phi range : [{phi_np.min():.4f}, {phi_np.max():.4f}]")
    print(f"  phi at target centre pixel: "
          f"{phi_np[x_tgt_mid, y_tgt_mid]:.4f}  (expected ~1.0)")

    assert phi_np[x_tgt_mid, y_tgt_mid] > 0.9, \
        "Target electrode centre should be close to target_voltage=1.0"
    print("  Assertions passed.")

    plot_idw_pcb_pixel(
        phi,
        title=(
            "init_idw_pcb_pixel — IDW initial guess (2D)\n"
            f"n_pix={n_pix}, p_size={p_size}, p_gap={p_gap}"
        ),
        save_path=save_path,
    )
