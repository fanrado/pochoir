"""
Weighting Field Initial Value Strategies
=========================================
Implements three initial guess methods for iterative weighting field solvers:
  1. Zero initialization (baseline)
  2. Inverse Distance Weighting (IDW)
  3. Coarse grid solve + interpolation (multigrid-style)

Assumes:
  - 2D grid (easily extendable to 3D)
  - Target electrode at 1V, all other electrodes at 0V
  - Dirichlet boundary conditions enforced externally
"""

import numpy as np
from scipy.ndimage import zoom
from scipy.ndimage import laplace


# ---------------------------------------------------------------------------
# Helper: simple Gauss-Seidel relaxation (used for coarse grid solve)
# ---------------------------------------------------------------------------

def gauss_seidel(phi, mask_fixed, n_iter=500, omega=1.5):
    """
    Solve Laplace's equation via successive over-relaxation (SOR).

    Parameters
    ----------
    phi       : 2D array, initial potential (boundary values already set)
    mask_fixed: 2D bool array, True where values are fixed (electrodes/boundary)
    n_iter    : number of iterations
    omega     : relaxation factor (1 = Gauss-Seidel, 1<omega<2 = SOR)

    Returns
    -------
    phi : 2D array, relaxed potential
    """
    phi = phi.copy()
    ny, nx = phi.shape
    for _ in range(n_iter):
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                if not mask_fixed[j, i]:
                    phi_new = 0.25 * (phi[j+1, i] + phi[j-1, i] +
                                      phi[j, i+1] + phi[j, i-1])
                    phi[j, i] = phi[j, i] + omega * (phi_new - phi[j, i])
    return phi


def gauss_seidel_fast(phi, mask_fixed, n_iter=500, omega=1.5):
    """
    Vectorised checkerboard SOR — much faster than the nested loop version.
    Suitable for moderate grid sizes.
    """
    phi = phi.copy()
    ny, nx = phi.shape
    free = ~mask_fixed

    for _ in range(n_iter):
        for parity in (0, 1):          # checkerboard red-black sweep
            j_idx, i_idx = np.meshgrid(range(1, ny-1), range(1, nx-1),
                                        indexing='ij')
            mask = ((j_idx + i_idx) % 2 == parity) & free[1:-1, 1:-1]

            avg = 0.25 * (phi[0:-2, 1:-1] + phi[2:, 1:-1] +
                          phi[1:-1, 0:-2] + phi[1:-1, 2:])
            update = phi[1:-1, 1:-1].copy()
            update[mask] = (update[mask]
                            + omega * (avg[mask] - update[mask]))
            phi[1:-1, 1:-1] = update
    return phi


# ===========================================================================
# Method 1 — Zero initialisation (baseline)
# ===========================================================================

def init_zeros(grid_shape):
    """
    Simplest possible initial guess: set everything to 0 V.
    Boundary conditions must be applied separately after this call.

    Parameters
    ----------
    grid_shape : (ny, nx)

    Returns
    -------
    phi : 2D array of zeros
    """
    return np.zeros(grid_shape)


# ===========================================================================
# Method 2 — Inverse Distance Weighting (IDW)
# ===========================================================================

def init_idw(grid_shape, electrodes, power=1):
    """
    Geometry-aware initial guess via Inverse Distance Weighting.

    For the weighting field (target = 1 V, all others = 0 V) this reduces to:

        phi(r) = (1/d_target) / sum_k(1/d_k)

    Parameters
    ----------
    grid_shape : (ny, nx)
    electrodes : list of dicts with keys:
                   'mask'    — 2D bool array, True on electrode pixels
                   'voltage' — float, boundary voltage for this electrode
    power      : IDW exponent (1 = linear falloff, 2 = quadratic, …)

    Returns
    -------
    phi : 2D array with IDW-interpolated initial values
    """
    ny, nx = grid_shape
    # Pixel coordinate arrays
    jj, ii = np.mgrid[0:ny, 0:nx]          # jj = row index, ii = col index

    numerator   = np.zeros((ny, nx))
    denominator = np.zeros((ny, nx))

    for elec in electrodes:
        emask   = elec['mask']              # (ny, nx) bool
        voltage = elec['voltage']

        # Distance from every grid point to the nearest electrode pixel
        ej, ei = np.where(emask)            # electrode pixel coordinates

        if len(ej) == 0:
            continue

        # Broadcast: shape (n_elec_pixels, ny, nx, nz)
        dj = jj[np.newaxis, :, :] - ej[:, np.newaxis, np.newaxis]
        di = ii[np.newaxis, :, :] - ei[:, np.newaxis, np.newaxis]
        dist_to_pixels = np.sqrt(dj**2 + di**2)   # (n_pixels, ny, nx)

        # Minimum distance to this electrode
        d_min = dist_to_pixels.min(axis=0)          # (ny, nx)

        # Avoid division by zero on electrode pixels themselves
        d_min = np.where(d_min == 0, 1e-10, d_min)

        w = 1.0 / d_min**power
        numerator   += w * voltage
        denominator += w

    phi = np.where(denominator > 0, numerator / denominator, 0.0)

    # Enforce exact boundary values on electrode pixels
    for elec in electrodes:
        phi[elec['mask']] = elec['voltage']

    return phi

def init_idw_3d(grid_shape, electrodes, power=1):
    """
    Geometry-aware initial guess via Inverse Distance Weighting.

    For the weighting field (target = 1 V, all others = 0 V) this reduces to:

        phi(r) = (1/d_target) / sum_k(1/d_k)

    Parameters
    ----------
    grid_shape : (ny, nx)
    electrodes : list of dicts with keys:
                   'mask'    — 2D bool array, True on electrode pixels
                   'voltage' — float, boundary voltage for this electrode
    power      : IDW exponent (1 = linear falloff, 2 = quadratic, …)

    Returns
    -------
    phi : 3D array with IDW-interpolated initial values
    """
    ny, nx, nz = grid_shape
    # Pixel coordinate arrays
    jj, ii, kk = np.mgrid[0:ny, 0:nx, 0:nz]          # jj = row index, ii = col index, kk = depth index

    numerator   = np.zeros((ny, nx, nz))
    denominator = np.zeros((ny, nx, nz))

    for elec in electrodes:
        emask   = elec['mask']              # (ny, nx) bool
        voltage = elec['voltage']

        # Distance from every grid point to the nearest electrode pixel
        ej, ei, ek = np.where(emask)            # electrode pixel coordinates

        if len(ej) == 0:
            continue

        # Broadcast: shape (n_elec_pixels, ny, nx, nz)
        dj = jj[np.newaxis, :, :, :] - ej[:, np.newaxis, np.newaxis, np.newaxis]
        di = ii[np.newaxis, :, :, :] - ei[:, np.newaxis, np.newaxis, np.newaxis]
        dk = kk[np.newaxis, :, :, :] - ek[:, np.newaxis, np.newaxis, np.newaxis]
        dist_to_pixels = np.sqrt(dj**2 + di**2 + dk**2)   # (n_pixels, ny, nx, nz)

        # Minimum distance to this electrode
        d_min = dist_to_pixels.min(axis=0)          # (ny, nx, nz)

        # Avoid division by zero on electrode pixels themselves
        d_min = np.where(d_min == 0, 1e-10, d_min)

        w = 1.0 / d_min**power
        numerator   += w * voltage
        denominator += w    

    phi = np.where(denominator > 0, numerator / denominator, 0.0)

    # Enforce exact boundary values on electrode pixels
    for elec in electrodes:
        phi[elec['mask']] = elec['voltage']

    return phi
# ===========================================================================
# Method 3 — Coarse grid solve + interpolation
# ===========================================================================

def init_coarse_grid(grid_shape, electrodes, coarsen_factor=4, n_iter=300,
                     omega=1.5):
    """
    Solve the same problem on a coarser grid, then interpolate to fine grid.
    This is the most accurate initial guess and mimics the first level of a
    multigrid V-cycle.

    Parameters
    ----------
    grid_shape     : (ny, nx) — fine grid size
    electrodes     : same format as init_idw()
    coarsen_factor : integer downscaling factor (e.g. 4 → 4× coarser grid)
    n_iter         : SOR iterations on the coarse grid
    omega          : SOR relaxation factor

    Returns
    -------
    phi_fine : 2D array, initial guess on the fine grid
    """
    ny, nx = grid_shape
    cy = max(2, ny // coarsen_factor)
    cx = max(2, nx // coarsen_factor)
    coarse_shape = (cy, cx)
    scale_y = cy / ny
    scale_x = cx / nx

    # --- Build coarse grid ---
    phi_c      = np.zeros(coarse_shape)
    mask_c     = np.zeros(coarse_shape, dtype=bool)

    for elec in electrodes:
        # Downsample electrode mask with nearest-neighbour
        emask_c = zoom(elec['mask'].astype(float),
                       (scale_y, scale_x), order=0) > 0.5
        phi_c[emask_c]  = elec['voltage']
        mask_c[emask_c] = True

    # Fix outer boundary (assume Dirichlet = 0 on border)
    phi_c[0, :]  = 0; phi_c[-1, :] = 0
    phi_c[:, 0]  = 0; phi_c[:, -1] = 0
    mask_c[0, :] = True; mask_c[-1, :] = True
    mask_c[:, 0] = True; mask_c[:, -1] = True

    # --- Relax on coarse grid ---
    phi_c = gauss_seidel_fast(phi_c, mask_c, n_iter=n_iter, omega=omega)

    # --- Interpolate back to fine grid ---
    phi_fine = zoom(phi_c, (ny / cy, nx / cx), order=1)   # bilinear
    phi_fine = np.clip(phi_fine, 0.0, 1.0)

    # Re-enforce exact electrode values on the fine grid
    for elec in electrodes:
        phi_fine[elec['mask']] = elec['voltage']

    return phi_fine


# ===========================================================================
# Convergence checker (optional utility)
# ===========================================================================

def residual_rms(phi, mask_fixed):
    """
    RMS of the Laplacian residual at free interior points.
    Useful for monitoring convergence speed of different initialisations.
    """
    lap = np.abs(laplace(phi))
    free_interior = (~mask_fixed)
    free_interior[[0, -1], :] = False
    free_interior[:, [0, -1]] = False
    return np.sqrt(np.mean(lap[free_interior]**2))


# ===========================================================================
# Demo / usage example
# ===========================================================================
def set_idw(grid_shape, power=1):
    NX, NY = grid_shape
    # Target electrode: horizontal strip near top centre (1 V)
    target_mask = np.zeros((NY, NX, 1), dtype=bool)
    target_mask[10:15, 35:65, 0] = True          # rows 10-14, cols 35-64

    # Ground electrode: bottom strip (0 V)
    ground_mask = np.zeros((NY, NX, 1), dtype=bool)
    ground_mask[85:90, 20:80, 0] = True

    # Outer boundary (0 V) — represented as four edge rows/cols
    boundary_mask = np.zeros((NY, NX, 1), dtype=bool)
    boundary_mask[0, :, 0]  = True
    boundary_mask[-1, :, 0] = True
    boundary_mask[:, 0, 0]  = True
    boundary_mask[:, -1, 0] = True

    electrodes = [
        {'mask': target_mask,   'voltage': 1.0},
        {'mask': ground_mask,   'voltage': 0.0},
        {'mask': boundary_mask, 'voltage': 0.0},
    ]
    return init_idw(grid_shape, electrodes, power=power)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # ---- Define a simple 2-D geometry ----
    NY, NX = 100, 100

    # Target electrode: horizontal strip near top centre (1 V)
    target_mask = np.zeros((NY, NX), dtype=bool)
    target_mask[10:15, 35:65] = True          # rows 10-14, cols 35-64

    # Ground electrode: bottom strip (0 V)
    ground_mask = np.zeros((NY, NX), dtype=bool)
    ground_mask[85:90, 20:80] = True

    # Outer boundary (0 V) — represented as four edge rows/cols
    boundary_mask = np.zeros((NY, NX), dtype=bool)
    boundary_mask[0, :]  = True
    boundary_mask[-1, :] = True
    boundary_mask[:, 0]  = True
    boundary_mask[:, -1] = True

    electrodes = [
        {'mask': target_mask,   'voltage': 1.0},
        {'mask': ground_mask,   'voltage': 0.0},
        {'mask': boundary_mask, 'voltage': 0.0},
    ]

    fixed_mask = target_mask | ground_mask | boundary_mask

    # ---- Compute initial guesses ----
    phi_zero   = init_zeros((NY, NX))
    # phi_idw    = init_idw((NY, NX), electrodes, power=1)
    phi_idw   = set_idw((NY, NX), power=1)
    phi_coarse = init_coarse_grid((NY, NX), electrodes,
                                  coarsen_factor=5, n_iter=400)

    # Apply boundary conditions to zero-init (IDW / coarse do it internally)
    for elec in electrodes:
        phi_zero[elec['mask']] = elec['voltage']

    # ---- Report initial residuals ----
    for name, phi in [("Zero init", phi_zero),
                      ("IDW init",  phi_idw),
                      ("Coarse grid init", phi_coarse)]:
        r = residual_rms(phi, fixed_mask)
        print(f"{name:20s}  initial RMS residual = {r:.6f}")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    titles = ["Zero Init", "IDW Init", "Coarse Grid Init"]
    fields = [phi_zero, phi_idw, phi_coarse]

    for ax, title, phi in zip(axes, titles, fields):
        im = ax.imshow(phi, origin='upper', cmap='RdBu_r',
                       vmin=0, vmax=1, aspect='equal')
        ax.set_title(title, fontsize=13)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Voltage (V)')

    plt.suptitle("Weighting Field — Initial Guess Comparison", fontsize=14,
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig("initial_guess_comparison.png",
                dpi=150, bbox_inches='tight')
    plt.show()
    print("\nPlot saved to initial_guess_comparison.png")