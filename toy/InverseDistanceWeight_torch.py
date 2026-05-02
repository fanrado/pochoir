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


# ---------------------------------------------------------------------------
# Core IDW function
# ---------------------------------------------------------------------------

def init_idw_3d_torch(
    grid_shape: tuple,
    electrodes: list,
    power: float = 1.0,
    batch_size: int = 1024,
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
    batch_size : number of electrode voxels per GPU batch (tune for VRAM)
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

    numerator   = torch.zeros(N, device=dev, dtype=dtype)
    denominator = torch.zeros(N, device=dev, dtype=dtype)

    # ---- Loop over electrodes --------------------------------------------
    for elec in electrodes:
        mask    = elec["mask"]
        voltage = float(elec["voltage"])

        # Convert mask to torch if needed
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        mask = mask.to(dev)

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
        # Shape at each step: (batch, N)  →  reduce min over electrode dim.
        d_min = torch.full((N,), float("inf"), device=dev, dtype=dtype)

        for start in range(0, n_elec, batch_size):
            end = min(start + batch_size, n_elec)

            # Electrode voxels in this batch: shape (B,)
            ek_b = ek[start:end]   # (B,)
            ej_b = ej[start:end]
            ei_b = ei[start:end]

            # Broadcast:  (B, 1) - (1, N)  →  (B, N)
            dk = ek_b.unsqueeze(1) - kk_flat.unsqueeze(0)
            dj = ej_b.unsqueeze(1) - jj_flat.unsqueeze(0)
            di = ei_b.unsqueeze(1) - ii_flat.unsqueeze(0)

            dist = torch.sqrt(dk**2 + dj**2 + di**2)  # (B, N)

            # Update running minimum across this electrode's voxels
            d_min = torch.minimum(d_min, dist.min(dim=0).values)

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
        mask = elec["mask"]
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).to(dev)
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
    arr: np.ndarray,
    p_size: int,
    p_gap: int,
    n_pix: int,
    pp_loweredge: int,
    pp_width: int,
    target_voltage: float = 1.0,
    ground_voltage: float = 0.0,
    power: float = 1.0,
    batch_size: int = 1024,
    device: str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    IDW initial guess for a 3D weighting field using the PCB pixel geometry
    from gen_pcb_pixel_with_grid.py.

    The target electrode is the centre pixel of the n_pix × n_pix grid
    (index i == j == int(n_pix / 2), exactly as in draw_pixel_plane).
    All other pixels plus the bottom face (z = 0) form the ground electrode.
    Corner trimming is applied identically to trimCorner() in the original.

    Parameters
    ----------
    arr             : 3D numpy array whose shape (nx, ny, nz) defines the domain
    p_size          : pixel size   in grid units  (= round(pixelSize / spacing))
    p_gap           : pixel gap    in grid units  (= round(pixelGap  / spacing))
    n_pix           : number of pixels per side
    pp_loweredge    : lower z-edge of the pixel plane in grid units
                      (= int(pixelPlaneLowEdgePosition / spacing))
    pp_width        : thickness of the pixel plane in grid units
                      (= int(pixelPlaneWidth / spacing))
    target_voltage  : voltage on the centre pixel   [default 1.0]
    ground_voltage  : voltage on all other pixels + bottom face  [default 0.0]
    power           : IDW exponent
    batch_size      : electrode-voxel batch size for GPU (tune for VRAM)
    device          : 'cuda', 'cpu', or None (auto-detect)
    dtype           : torch float dtype

    Returns
    -------
    phi : torch.Tensor of shape grid_shape on `device`
    """
    grid_shape = arr.shape
    center = int(n_pix / 2)          # centre pixel index (same as original)
    z1 = pp_loweredge
    z2 = pp_width + pp_loweredge + 1  # exclusive upper z bound (original uses z2)

    target_mask = np.zeros(grid_shape, dtype=bool)
    ground_mask = np.zeros(grid_shape, dtype=bool)

    for i in range(n_pix):
        for j in range(n_pix):
            x0 = int(p_gap / 2) + i * (p_size + p_gap)
            x1 = x0 + p_size            # exclusive end in slice notation
            y0 = int(p_gap / 2) + j * (p_size + p_gap)
            y1 = y0 + p_size

            mask = target_mask if (i == center and j == center) else ground_mask

            # Fill pixel body
            mask[x0:x1, y0:y1, z1:z2] = True

            # # Trim all four corners (mirrors exact call order in draw_pixel_plane)
            # _apply_trim_corner(mask, x1 - 1, y1 - 1, z1, z2, corner=0)
            # _apply_trim_corner(mask, x1 - 1, y0,     z1, z2, corner=1)
            # _apply_trim_corner(mask, x0,     y1 - 1, z1, z2, corner=3)
            # _apply_trim_corner(mask, x0,     y0,     z1, z2, corner=2)

    # Ground plane at z = 0 (mirrors barr[:, :, 0] = 1 in generator())
    ground_mask[:, :, 0] = True

    electrodes = [
        {"mask": target_mask, "voltage": target_voltage},
        {"mask": ground_mask, "voltage": ground_voltage},
    ]

    return init_idw_3d_torch(
        grid_shape,
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
    ix: int | None = None,
    iy: int | None = None,
    iz: int | None = None,
    title: str = "init_idw_pcb_pixel — IDW initial guess",
    save_path: str | None = None,
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap: str = "RdBu_r",
) -> None:
    """
    Plot the XY, XZ, and YZ orthogonal slices of the 3-D field returned by
    init_idw_pcb_pixel.

    Parameters
    ----------
    phi       : torch.Tensor or numpy array of shape (nx, ny, nz)
    ix        : x-index of the YZ slice  (defaults to nx // 2)
    iy        : y-index of the XZ slice  (defaults to ny // 2)
    iz        : z-index of the XY slice  (defaults to nz // 2)
    title     : overall figure title
    save_path : if given, save the figure to this path
    vmin/vmax : colour-scale limits
    cmap      : matplotlib colourmap
    """
    import matplotlib.pyplot as plt

    # Accept both torch tensors and numpy arrays
    if hasattr(phi, "cpu"):
        phi_np = phi.cpu().numpy()
    else:
        phi_np = np.asarray(phi)

    nx, ny, nz = phi_np.shape
    if ix is None:
        ix = nx // 2
    if iy is None:
        iy = ny // 2
    if iz is None:
        iz = nz // 2

    views = [
        (f"XY  (z = {iz})", phi_np[:, :, iz],  "y-index", "x-index"),
        (f"XZ  (y = {iy})", phi_np[:, iy, :],  "z-index", "x-index"),
        (f"YZ  (x = {ix})", phi_np[ix, :, :],  "z-index", "y-index"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (label, data, xlabel, ylabel) in zip(axes, views):
        im = ax.imshow(data, origin="lower", cmap=cmap,
                       vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="φ (V)")

    plt.suptitle(title, fontsize=11, fontweight="bold")
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
    XY, XZ, and YZ views of the resulting phi array.

    Geometry (all in grid units)
    ----------------------------
    n_pix = 3      → 3×3 grid, centre pixel at i = j = 1
    p_size = 10    → each pixel is 10 × 10 voxels
    p_gap  = 4     → 4-voxel gap between pixels (2-voxel half-gap at border)
    pp_loweredge=5 → pixel plane starts at z = 5
    pp_width = 2   → pixel plane is 3 voxels thick (z = 5..7 inclusive)

    Grid size: x,y need at least int(p_gap/2) + n_pix*(p_size+p_gap) = 2+42 = 44
    → use 50×50×20 so there is comfortable padding on all sides.
    """
    import time
    import matplotlib.pyplot as plt

    # ---- Parameters ------------------------------------------------------
    # p_size      = 38
    # p_gap       = 6
    # n_pix       = 5
    # pp_loweredge = 100
    # pp_width    = 1    # z = pp_loweredge ... pp_loweredge+pp_width (inclusive)
    p_size=38
    p_gap=6
    n_pix=5
    pp_width=1
    pp_loweredge=100
    # Grid large enough to hold the 3×3 pixel array with margin
    xy_size = int(p_gap / 2) + n_pix * (p_size + p_gap) + int(p_gap / 2)  # = 44
    nz      = pp_loweredge + pp_width + 10                                  # = 17
    grid_shape = (xy_size, xy_size, nz)   # (nx, ny, nz) = (44, 44, 17)
    arr = np.zeros(grid_shape, dtype=np.float32)

    # Coordinates of the centre pixel for slice labels
    center    = int(n_pix / 2)          # = 1
    x0_tgt    = int(p_gap / 2) + center * (p_size + p_gap)   # = 16
    x_tgt_mid = x0_tgt + p_size // 2                          # = 21
    y0_tgt    = x0_tgt                                         # symmetric
    y_tgt_mid = x_tgt_mid
    z_tgt_mid = pp_loweredge + pp_width // 2                   # = 6

    print("=" * 60)
    print("test_init_idw_pcb_pixel")
    print(f"  grid_shape   = {grid_shape}")
    print(f"  n_pix={n_pix}  p_size={p_size}  p_gap={p_gap}")
    print(f"  pp_loweredge={pp_loweredge}  pp_width={pp_width}")
    print(f"  target pixel centre voxel: "
          f"x={x_tgt_mid}, y={y_tgt_mid}, z={z_tgt_mid}")
    print("=" * 60)

    # ---- Run -------------------------------------------------------------
    t0  = time.time()
    phi = init_idw_pcb_pixel(
        arr,
        p_size=p_size,
        p_gap=p_gap,
        n_pix=n_pix,
        pp_loweredge=pp_loweredge,
        pp_width=pp_width,
        target_voltage=1.0,
        ground_voltage=0.0,
        power=1.0,
        batch_size=128,
    )
    elapsed = time.time() - t0

    phi_np = phi.cpu().numpy()
    print(f"  Elapsed   : {elapsed:.2f} s")
    print(f"  phi range : [{phi_np.min():.4f}, {phi_np.max():.4f}]")
    print(f"  phi at target centre voxel: "
          f"{phi_np[x_tgt_mid, y_tgt_mid, z_tgt_mid]:.4f}  (expected ~1.0)")
    print(f"  phi at z=0 boundary: "
          f"{phi_np[x_tgt_mid, y_tgt_mid, 0]:.4f}  (expected ~0.0)")

    # ---- Basic assertions ------------------------------------------------
    assert phi_np[x_tgt_mid, y_tgt_mid, z_tgt_mid] > 0.9, \
        "Target electrode centre should be close to target_voltage=1.0"
    assert phi_np[x_tgt_mid, y_tgt_mid, 0] < 0.1, \
        "Ground plane (z=0) should be close to ground_voltage=0.0"
    print("  Assertions passed.")

    # ---- Plot ------------------------------------------------------------
    plot_idw_pcb_pixel(
        phi,
        ix=x_tgt_mid,
        iy=y_tgt_mid,
        iz=z_tgt_mid,
        title=(
            "init_idw_pcb_pixel — IDW initial guess\n"
            f"n_pix={n_pix}, p_size={p_size}, p_gap={p_gap}, "
            f"pp_loweredge={pp_loweredge}, pp_width={pp_width}"
        ),
        save_path=save_path,
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt

    # ---- Geometry --------------------------------------------------------
    NZ, NY, NX = 60, 80, 80     # modest size for a quick demo

    # Target electrode: a flat rectangular pad near the top (z=5 slice)
    target_mask = np.zeros((NZ, NY, NX), dtype=bool)
    target_mask[5:8, 30:50, 30:50] = True   # 1 V

    # Ground electrode: bottom plate
    ground_mask = np.zeros((NZ, NY, NX), dtype=bool)
    ground_mask[52:55, 10:70, 10:70] = True  # 0 V

    # Outer boundary (6 faces of the box) = 0 V
    boundary_mask = np.zeros((NZ, NY, NX), dtype=bool)
    boundary_mask[0,  :, :] = True
    boundary_mask[-1, :, :] = True
    boundary_mask[:,  0, :] = True
    boundary_mask[:, -1, :] = True
    boundary_mask[:, :,  0] = True
    boundary_mask[:, :, -1] = True

    electrodes = [
        {"mask": target_mask,   "voltage": 1.0},
        {"mask": ground_mask,   "voltage": 0.0},
        {"mask": boundary_mask, "voltage": 0.0},
    ]

    # ---- Run IDW ---------------------------------------------------------
    t0 = time.time()
    phi = init_idw_3d_torch(
        (NZ, NY, NX),
        electrodes,
        power=1,
        batch_size=512,       # increase if you have lots of VRAM
    )
    elapsed = time.time() - t0
    print(f"\nIDW completed in {elapsed:.2f} s")
    print(f"phi range: [{phi.min():.4f}, {phi.max():.4f}]")
    print(f"Output tensor: shape={tuple(phi.shape)}, device={phi.device}, "
          f"dtype={phi.dtype}")

    # ---- Visualise three orthogonal slices -------------------------------
    phi_np = phi.cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    slices = [
        ("XY  (z = NZ//2)", phi_np[NZ // 2, :, :]),
        ("XZ  (y = NY//2)", phi_np[:, NY // 2, :]),
        ("YZ  (x = NX//2)", phi_np[:, :, NX // 2]),
    ]
    for ax, (title, data) in zip(axes, slices):
        im = ax.imshow(data, origin="upper", cmap="RdBu_r",
                       vmin=0, vmax=1, aspect="equal")
        ax.set_title(title, fontsize=12)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="V")

    plt.suptitle("3D Weighting Field — IDW Initial Guess (PyTorch)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = "idw_3d_torch_slices.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out_path}")

    # ---- Test init_idw_pcb_pixel -----------------------------------------
    print("\n" + "=" * 60)
    test_init_idw_pcb_pixel(save_path="idw_pcb_pixel_slices.png")