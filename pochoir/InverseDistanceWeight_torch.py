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


@torch.compile
def _idw_dist_batch(elec_batch: torch.Tensor, grid_coords: torch.Tensor) -> torch.Tensor:
    """Min distance from each grid voxel to a batch of electrode voxels.

    Uses ``torch.cdist`` (fused CUDA kernel) instead of manual broadcast ops.

    Parameters
    ----------
    elec_batch  : (B, D) float — electrode coords in this batch
    grid_coords : (N, D) float — all grid voxel coords
    Returns     : (N,)   float — per-voxel minimum distance to this batch
    """
    dist = torch.cdist(elec_batch, grid_coords)  # (B, N)
    return dist.min(dim=0).values


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

def init_idw_3d_torch(
    grid_shape: tuple,
    electrodes: list,
    power: float = 1.0,
    batch_size: int = 0,
    device: str | None = None,
    dtype: torch.dtype = torch.float32,
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

        ek = ek.to(dtype)
        ej = ej.to(dtype)
        ei = ei.to(dtype)

        # Minimum distance from every grid voxel to this electrode,
        # computed in batches over electrode voxels to save VRAM.
        d_min = torch.full((N,), float("inf"), device=dev, dtype=dtype)

        for start in range(0, n_elec, batch_size):
            end = min(start + batch_size, n_elec)
            # Stack electrode batch coords: (B, 3)
            elec_batch = torch.stack(
                [ek[start:end], ej[start:end], ei[start:end]], dim=1
            )
            d_min = torch.minimum(d_min, _idw_dist_batch(elec_batch, grid_coords))

        # Avoid division by zero directly on electrode voxels
        d_min = torch.clamp(d_min, min=1e-10)

        w = 1.0 / d_min**power         # (N,)
        numerator   += w * voltage
        denominator += w

    # ---- Normalise -------------------------------------------------------
    phi_flat = torch.where(denominator > 0, numerator / denominator,
                           torch.zeros_like(numerator))

    phi = phi_flat.reshape(nz, ny, nx)

    # ---- Enforce exact boundary values on electrode voxels ---------------
    for elec in electrodes:
        mask = _mask_to_device(elec["mask"], dev)
        phi[mask] = float(elec["voltage"])

    return phi


# ---------------------------------------------------------------------------
# Convenience wrapper: numpy in → numpy out  (useful for existing code)
# ---------------------------------------------------------------------------

def init_idw_3d_torch_numpy(grid_shape, electrodes, **kwargs) -> np.ndarray:
    """Same as init_idw_3d_torch but returns a numpy array."""
    phi = init_idw_3d_torch(grid_shape, electrodes, **kwargs)
    return phi.cpu().numpy()


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
) -> torch.Tensor:
    """
    IDW initial guess for a 2D weighting field using the PCB pixel geometry.

    The target electrode is the centre pixel of the n_pix × n_pix grid
    (index i == j == int(n_pix / 2)).  All other pixels form the ground
    electrode.

    Parameters
    ----------
    arr             : 2D array whose shape (nx, ny) defines the domain
    p_size          : pixel size  in grid units
    p_gap           : pixel gap   in grid units
    n_pix           : number of pixels per side
    target_voltage  : voltage on the centre pixel   [default 1.0]
    ground_voltage  : voltage on all other pixels   [default 0.0]
    power           : IDW exponent
    batch_size      : electrode-pixel batch size for GPU (0 = auto from VRAM)
    device          : 'cuda', 'cpu', or None (auto-detect)
    dtype           : torch float dtype

    Returns
    -------
    phi : torch.Tensor of shape (nx, ny) on `device`
    """
    nx, ny = arr.shape[0], arr.shape[1]
    center = int(n_pix / 2)

    target_mask = torch.zeros((nx, ny), dtype=torch.bool)
    ground_mask = torch.zeros((nx, ny), dtype=torch.bool)

    for i in range(n_pix):
        for j in range(n_pix):
            x0 = int(p_gap / 2) + i * (p_size + p_gap)
            x1 = x0 + p_size
            y0 = int(p_gap / 2) + j * (p_size + p_gap)
            y1 = y0 + p_size

            mask = target_mask if (i == center and j == center) else ground_mask
            mask[x0:x1, y0:y1] = True

    electrodes = [
        {"mask": target_mask, "voltage": target_voltage},
        {"mask": ground_mask, "voltage": ground_voltage},
    ]

    return init_idw_2d_torch(
        (nx, ny),
        electrodes,
        power=power,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
    )


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
