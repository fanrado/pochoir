#!/usr/bin/env python3

import os
import numpy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def _plot_dir():
    """Directory for diagnostic plots, following the run's output folder.

    Mirrors ``pochoir/__main__.py`` (``POCHOIR_STORE`` env, default ``store``)
    so these side-saved figures land in the *renamed* output folder instead of
    a hardcoded ``store/``. The outer folder name is configurable; only the
    inner store layout is fixed.
    """
    d = os.environ.get('POCHOIR_STORE', 'store')
    os.makedirs(d, exist_ok=True)
    return d

def draw_quarter_circle(x0,y0,r):
    """
    Sorted by x-index IDs of an apper-right quarter-circle on a given grid
    x0,y0,r are in index units
    
    draws IV quadrant
    """
    x=0
    y=r
    d=3-2*r
    id_circ1=[]
    shifted=[]
    id_circ1.append((x,y))
    while x<=y :
        if d<0:
            d=d+4*x+6
            x=x+1
            id_circ1.append((x,y))
            id_circ1.append((y,x))
        if d>=0:
            d=d+4*(x-y)+6
            x=x+1
            y=y-1
            id_circ1.append((x,y))
            id_circ1.append((y,x))
    id_circ1.sort(key = lambda x: x[0])
    for id in id_circ1:
        sh=(id[0]+x0,id[1]+y0)
        shifted.append(sh)
    return shifted

def mirror_xaxis(id_circ1,x0,y0,r):
    """
    mirror quarter-circle of Xaxis
    """
    id_circ2=[]
    for id in id_circ1:
        id_circ2.append((id[0],y0-(id[1]-y0)))
    return id_circ2

def mirror_yaxis(id_circ1,x0,y0,r):
    """
    mirror quarter-circle of Yaxis
    """
    id_circ2=[]
    for id in id_circ1:
        id_circ2.append((x0-(id[0]-x0),id[1]))
    return id_circ2


def mirror_center(id_circ1,x0,y0):
    """
    mirror quarter-circle of the center point
    """
    id_circ2=[]
    for id in id_circ1:
        id_circ2.append((x0-(id[0]-x0),y0-(id[1]-y0)))
    return id_circ2

def fill_area(arr,barr,val):
    """
    fill 2D area inside the boundary
    
    barr should be constructed such, [x,[y_start,y_stop]] sorted in increasing x
    """
    for b in barr:
        arr[b[0],b[1][0]:b[1][1]+1]=val

def draw_plane(arr,z,val):
    """
    Fill 1 plane
    """
    arr[:,:,z]=val

def form_quarter_boundary(indx,x0,y0):
    dx = indx[0][0]-x0 # x index from center
    dy = indx[0][1]-y0   # y index from center
    barr = []
    if dx>=0 and dy>0:
        for idx in indx:
            yarr = (y0,idx[1])
            xarr = (idx[0],yarr)
            barr.append(xarr)
    if dx>=0 and dy<0:
        for idx in indx:
            yarr = (idx[1],y0)
            xarr = (idx[0],yarr)
            barr.append(xarr)
    if dx<0 and dy>=0:
        for idx in indx:
            yarr = (y0,idx[1])
            xarr = (idx[0],yarr)
            barr.append(xarr)
    if dx<0 and dy<=0:
        for idx in indx:
            yarr = (idx[1],y0)
            xarr = (idx[0],yarr)
            barr.append(xarr)
    return barr


def draw_pcb_plane(shape, arr, barr, z, r1, gridPotential):
    # Draw grid plane boundary in barr, and set potential in arr
    Nx, Ny = shape
    xi, yi = numpy.mgrid[0:Nx, 0:Ny]
    # Start with the full plane as boundary
    barr[:, :, z] = 1
    arr[:, :, z] = gridPotential
    # Zero out the 4 quarter-holes at the corners
    for cx, cy in [(0, 0), (Nx-1, 0), (0, Ny-1), (Nx-1, Ny-1)]:
        mask = (xi - cx)**2 + (yi - cy)**2 <= r1**2
        barr[:, :, z][mask] = 0
        arr[:, :, z][mask] = 0


def trimCorner(arr, x, y, z1, z2, corner, val=0, chamfer_r=4):
    """Trim (or fill) one rounded corner of a square hole or pad.

    Carves a quarter-disk-shaped notch at one inner corner of a square
    aperture so that the 90° tip is replaced by an arc of radius
    ``chamfer_r`` (in grid-index units).  The chamfer box is an
    ``r × r`` square anchored at the inner corner ``(x, y)`` and
    extending into the pad along the directions implied by ``corner``.
    Cells of that box whose distance from the box's inward corner exceeds
    ``r`` are written with ``val``.

    Parameters
    ----------
    arr       : ndarray
        3-D boundary or potential array modified in-place.
    x, y      : int
        Grid indices of the inner corner of the square aperture.
    z1, z2    : int
        z-slice range ``[z1, z2)`` over which the stencil is applied.
    corner    : int
        Which of the four inner corners to process, numbered by quadrant:
          0 – bottom-left  (chamfer box toward -x, -y)
          1 – top-left     (toward -x, +y)
          2 – top-right    (toward +x, +y)
          3 – bottom-right (toward +x, -y)
    val       : int or float, optional
        Value written into the carved cells.  ``0`` carves the corner out
        of a solid region (pixel-plane use case); ``1`` fills the corner
        back into a void (PCB-shield use case).
    chamfer_r : int, optional
        Radius of the rounded corner in grid-index units.  Larger values
        produce more strongly rounded corners; ``0`` or negative is a
        no-op (sharp 90° corner).
    """
    r = int(chamfer_r)
    if r <= 0:
        return
    if corner == 0:
        x0, x1, y0, y1 = x - r + 1, x + 1, y - r + 1, y + 1
        cx, cy = x - r + 1, y - r + 1
    elif corner == 1:
        x0, x1, y0, y1 = x - r + 1, x + 1, y,         y + r
        cx, cy = x - r + 1, y + r - 1
    elif corner == 2:
        x0, x1, y0, y1 = x,         x + r, y,         y + r
        cx, cy = x + r - 1, y + r - 1
    elif corner == 3:
        x0, x1, y0, y1 = x,         x + r, y - r + 1, y + 1
        cx, cy = x + r - 1, y - r + 1
    else:
        return
    xi, yi = numpy.mgrid[x0:x1, y0:y1]
    mask = (xi - cx) ** 2 + (yi - cy) ** 2 > r ** 2
    arr[x0:x1, y0:y1, z1:z2][mask] = val


def trimCorner_forpix(arr, x, y, z1, z2, corner):
    """Fixed-cell corner trim, copied VERBATIM from branch ``for_pix``.

    This is the original, non-parametric chamfer: it carves a hardcoded
    ~4-cell staircase notch at an inner corner, *independent of the grid
    spacing*.  It always writes ``0`` (carve a pad).  The reach along each
    edge is fixed at 4 grid cells, so the physical size of the removed
    region scales with the spacing (0.4 mm at 0.1 mm, 0.2 mm at 0.05 mm).

    Used only when ``chamferMode == 'fixed_cell'`` to reproduce the exact
    footprint of the ``for_pix`` branch.  Kept byte-for-byte identical to
    that branch's ``trimCorner`` so the removed cells match exactly.
    """
    if corner == 0:
        arr[x-3:x+1, y, z1:z2] = 0
        arr[x, y-3:y+1, z1:z2] = 0
        arr[x, y, z1:z2] = 0
        arr[x-1, y-1, z1:z2] = 0
    if corner == 1:
        arr[x-3:x+1, y, z1:z2] = 0
        arr[x, y:y+4, z1:z2] = 0
        arr[x, y, z1:z2] = 0
        arr[x-1, y+1, z1:z2] = 0
    if corner == 2:
        arr[x:x+4, y, z1:z2] = 0
        arr[x, y:y+4, z1:z2] = 0
        arr[x, y, z1:z2] = 0
        arr[x+1, y+1, z1:z2] = 0
    if corner == 3:
        arr[x:x+4, y, z1:z2] = 0
        arr[x, y-3:y+1, z1:z2] = 0
        arr[x, y, z1:z2] = 0
        arr[x+1, y-1, z1:z2] = 0


def _apply_rounded_corners(barr, p_size, p_gap, z1, z2, val, chamfer_r):
    """Apply ``trimCorner`` to all four inner corners of a square aperture.

    Encodes the corner positions and their quadrant indices once so that both
    ``draw_pixel_plane`` and ``draw_pcb_plane_rounded_sq_drift`` call the same
    geometry without repeating the coordinate arithmetic.

    Parameters
    ----------
    barr          : ndarray
        3-D boundary array modified in-place.
    p_size, p_gap : int
        Pixel size and gap in grid-index units.
    z1, z2        : int
        z-slice range ``[z1, z2)`` passed through to ``trimCorner``.
    val           : int or float
        Fill value forwarded to ``trimCorner`` (0 to carve, 1 to fill)
    """
    half = (p_size+1) // 2 ## This change here is related to the comment by Brett about a broken symmetry in the quarter pixels corners/rounded
    corners = [
        (half - 1,     half - 1,     0),
        (half - 1,     half + p_gap, 1),
        (half + p_gap, half - 1,     3),
        (half + p_gap, half + p_gap, 2),
    ]
    for x, y, corner in corners:
        trimCorner(barr, x, y, z1, z2, corner, val=val, chamfer_r=chamfer_r)


## Draw shield plane with square holes, rounded corners
def draw_pcb_plane_rounded_sq_drift(arr, barr, p_gap, p_size, pcb_width, pp_loweredge, gridPotential, chamfer_r):
    """Draw the PCB shield plane as a solid layer with rounded-square holes.

    The shield is modelled as a single z-plane that is initially set to solid
    (boundary = 1, potential = ``gridPotential``).  Four rectangular quadrants
    around the pixel-hole centre are then cleared to open the apertures, and
    ``_apply_rounded_corners`` restores the corner cells that were
    over-cleared, producing apertures with approximately rounded corners at
    grid resolution.

    This function is the complement of ``draw_pixel_plane``: where
    ``draw_pixel_plane`` starts from void and fills in solid pixel pads (then
    carves rounded corners out with ``val=0``), this function starts from solid
    and cuts square holes (then fills rounded corners back in with ``val=1``).
    Both delegate corner geometry to the unified ``trimCorner`` /
    ``_apply_rounded_corners`` helpers.

    Parameters
    ----------
    arr           : ndarray, shape (Nx, Ny, Nz)
        Potential array modified in-place.
    barr          : ndarray, shape (Nx, Ny, Nz)
        Boundary mask array modified in-place (1 = boundary, 0 = free).
    p_gap         : int
        Gap between pixel edges in grid-index units.
    p_size        : int
        Pixel side length in grid-index units.
    pcb_width     : int
        Thickness of the PCB layer in grid-index units.
    pp_loweredge  : int
        z-index of the lower edge of the pixel plane.
    gridPotential : float
        Electric potential applied to the PCB shield plane (V).

    Modification history
    --------------------
    ``trimCorner_pcb`` (a verbatim copy of ``trimCorner`` with 1 instead of 0)
    has been removed.  Corner filling is now handled by ``trimCorner(...,
    val=1)`` via ``_apply_rounded_corners``, eliminating the code duplication.

    Use case
    --------
    Called from ``generator`` before ``draw_pixel_plane`` to set the upper
    boundary of the drift volume::

        draw_pcb_plane_rounded_sq_drift(arr, barr, p_gap, p_size,
                                        pcb_width, pp_loweredge,
                                        gridPotential)
    """
    z = pp_loweredge + pcb_width
    z1, z2 = z, z + 1
    barr[:, :, z] = 1
    arr[:, :, z]  = gridPotential
    half = (p_size + 1) // 2
    barr[0:half,        0:half,        z] = 0
    barr[0:half,        half+p_gap:,   z] = 0
    barr[half+p_gap:,   0:half,        z] = 0
    barr[half+p_gap:,   half+p_gap:,   z] = 0
    _apply_rounded_corners(barr, p_size, p_gap, z1, z2, val=1, chamfer_r=chamfer_r)
##----

import sys
def draw_pixel_plane(arr, barr, p_size, p_gap, n_pix, pp_loweredge, pp_width, cathodePotential, gridPotential, epsilon=None, chamfer_r=0.7, chamferMode='dynamic', fr4_bottom=False, n_fr4=0):
    """Draw the pixel collection plane as solid pads with rounded-square corners.

    Initialises the full volume with the cathode potential and a solid boundary
    mask, then marks the four corner quadrants of the pixel-hole region as
    boundary over the pixel-plane z-range.  ``_apply_rounded_corners`` then
    carves the over-filled corner cells back out (``val=0``) so that the inner
    edges of each pixel pad approximate a rounded square at grid resolution.

    This function is the complement of ``draw_pcb_plane_rounded_sq_drift``:
    where the PCB function starts from solid and opens holes (filling corners
    back with ``val=1``), this function starts from void pads and fills them
    solid (then carves rounded corners out with ``val=0``).  Both share the
    same corner geometry via the unified ``trimCorner`` /
    ``_apply_rounded_corners`` helpers, replacing the former duplicated pair
    ``trimCorner`` / ``trimCorner_pcb``.

    Parameters
    ----------
    arr              : ndarray, shape (Nx, Ny, Nz)
        Potential array modified in-place.
    barr             : ndarray, shape (Nx, Ny, Nz)
        Boundary mask array modified in-place (1 = boundary, 0 = free).
    p_size           : int
        Pixel side length in grid-index units.
    p_gap            : int
        Gap between pixel edges in grid-index units.
    n_pix            : int
        Number of pixels along one axis of the detector.
    pp_loweredge     : int
        z-index of the lower edge of the pixel plane.
    pp_width         : int
        Thickness of the pixel plane in grid-index units.
    cathodePotential : float
        Electric potential applied to the cathode (back plane) (V).
    gridPotential    : float
        Electric potential applied to the grid / pixel plane (V).

    Use case
    --------
    Called from ``generator`` after ``draw_pcb_plane_rounded_sq_drift`` to set
    the pixel-collection boundary of the drift volume::

        draw_pixel_plane(arr, barr, p_size, p_gap, n_pix,
                         pp_loweredge, pp_width,
                         cathodePotential, gridPotential)
    """
    draw_plane(arr,-1,cathodePotential) # This line sets the initial values
    # ## Set the initial values to be linear along z
    # for i in range(pp_loweredge, arr.shape[2]):
    #     arr[:, :, i] = (-7000/1399)*(i-100)
    # ## Set the initial values to be random between -7000 and 0 for z=101 and z=1498
    # for i in range(pp_loweredge+2, arr.shape[2]-1):
    #     arr[:, :, i] = numpy.random.uniform(-7000, 0, size=(arr.shape[0], arr.shape[1]))
    draw_plane(barr,-1,1) # This line sets the boundary values

    dims = p_size*n_pix+p_gap*(n_pix-1)
    half = (p_size + 1) // 2 ## This change here is related to the comment by Brett about a broken symmetry in the quarter pixels corners/rounded
    z1, z2 = pp_loweredge, pp_width + pp_loweredge + 1
    # When the pixel-plane laminate FR4 is activated (fr4_bottom), the pad
    # conductor covers only the TOP of the pp_width layer; the bottom n_fr4
    # cell(s) are left FREE so their FR4 permittivity enters the Poisson solve.
    # For fr4_bottom=False (n_fr4=0) this is byte-identical to the old behaviour.
    zp1 = z1 + n_fr4 if fr4_bottom else z1
    barr[0:half,        0:half,        zp1:z2] = 1
    barr[0:half,        half+p_gap:,   zp1:z2] = 1
    barr[half+p_gap:,   0:half,        zp1:z2] = 1
    barr[half+p_gap:,   half+p_gap:,   zp1:z2] = 1
    if chamferMode == 'fixed_cell':
        # Reproduce the for_pix branch EXACTLY: fixed-cell staircase chamfer
        # (spacing-independent 4-cell reach), applied at the same anchor
        # positions and quadrant order as for_pix's draw_pixel_plane.
        h = p_size // 2
        trimCorner_forpix(barr, h - 1,     h - 1,     zp1, z2, 0)
        trimCorner_forpix(barr, h - 1,     h + p_gap, zp1, z2, 1)
        trimCorner_forpix(barr, h + p_gap, h - 1,     zp1, z2, 3)
        trimCorner_forpix(barr, h + p_gap, h + p_gap, zp1, z2, 2)
    else:
        _apply_rounded_corners(barr, p_size, p_gap, zp1, z2, val=0, chamfer_r=chamfer_r)
    # arr[(p_size+p_gap):(p_size+p_gap)+p_size,(p_size+p_gap):(p_size+p_gap)+p_size,pp_loweredge:pp_width+pp_loweredge+1]=1
    # draw pixel plane for drift field
    # 3D scatter plot of arr (non-zero voxels colored by potential)
    mask_arr = arr != 0
    if mask_arr.any():
        xi, yi, zi = numpy.where(mask_arr)
        vals = arr[mask_arr]
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(xi, yi, zi, c=vals, cmap='RdBu_r', s=2, alpha=0.6)
        fig.colorbar(sc, ax=ax, label='Potential (V)', shrink=0.6)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title('arr (drift field boundary conditions)')
        plt.tight_layout()
        plt.savefig(os.path.join(_plot_dir(), 'domain_drift_arr_3d.png'), dpi=150)
        plt.close()

    # 3D scatter plot of barr (boundary mask, non-zero voxels)
    mask_barr = barr != 0
    if mask_barr.any():
        xi, yi, zi = numpy.where(mask_barr)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(xi, yi, zi, c=zi, cmap='viridis', s=2, alpha=0.4, marker=',')
        plt.colorbar(sc, ax=ax, label='z index')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title('barr (drift field boundary mask)')
        plt.tight_layout()
        plt.savefig(os.path.join(_plot_dir(), 'domain_drift_barr_3d.png'), dpi=150)
        plt.close()

    # Same plot but clipped to first 150 z-planes, with large markers to prove surface-like render
    barr_clipped = barr[:, :, :150]
    mask_clip = barr_clipped != 0
    if epsilon is not None:
        mask_eps = (epsilon[:,:,:150] != 1.5) & (epsilon[:,:,:150] != 0)
    if mask_clip.any():
        xi, yi, zi = numpy.where(mask_clip)
        if epsilon is not None:
            xii, yii, zii = numpy.where(mask_eps)
        for marker_size, fname in [(2, 'domain_drift_barr_3d_clipped150_s2.png'),
                                   (20, 'domain_drift_barr_3d_clipped150_s20.png'),
                                   (200, 'domain_drift_barr_3d_clipped150_s200.png')]:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            if epsilon is not None:
                scc = ax.scatter(xii, yii, zii, cmap='viridis', s=marker_size, marker=',', edgecolors='k', linewidth=1)
            sc = ax.scatter(xi, yi, zi, c=zi, cmap='viridis', s=marker_size, marker=',', alpha=0.3)
            plt.colorbar(sc, ax=ax, label='z index')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.set_title(f'barr drift (first 150 z) - s={marker_size}')
            plt.tight_layout()
            plt.savefig(os.path.join(_plot_dir(), fname), dpi=150)
            plt.close()

    # plt.figure(figsize=(10,10))
    # plt.imshow(arr[:, :, pp_loweredge], origin='lower')
    # plt.title('pixel plane')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.savefig('store/domain_drift_initial_cond.png')
    # plt.close()
    
def generator(dom, cfg, info_msg=None):
    r1 = int(cfg['HoleRadius']/dom.spacing[0]-1)
    pcb_width = int(cfg['PcbWidth']/dom.spacing[2])
    gridHoleShape = cfg['GridHoleShape']

    gridPotential = cfg['GridPotential']
    cathodePotential = cfg['CathodePotential']
    pp_loweredge = int(cfg['pixelPlaneLowEdgePosition']/dom.spacing[0])
    p_size=int(round(cfg["pixelSize"]/dom.spacing[0]))
    p_gap=int(round(cfg["pixelGap"]/dom.spacing[0]))
    #chamfer_r=int(cfg["chamfer_r"]/dom.spacing[0])
    chamfer_r=int(round(cfg["chamfer_r"]/dom.spacing[0]))
    chamferMode = cfg.get('chamferMode', 'dynamic')
    n_pix = cfg['Npixels']
    pp_width = int(cfg['pixelPlaneWidth']/dom.spacing[0])

    arr = numpy.zeros(dom.shape)
    barr = numpy.zeros(dom.shape)

    ## epsilon is an array of the dielectric constants
    epsilon = None
    LArPermittivity = cfg.get('LArPermittivity', None)
    FR4Permittivity = cfg.get('FR4Permittivity', None)
    if gridHoleShape in ['circular', 'square'] and LArPermittivity is not None and FR4Permittivity is not None:
        epsilon = numpy.zeros(dom.shape)
        ## This is correct if there was no hole in the FR4
        # epsilon[:, :, pp_loweredge+pp_width+1:pp_loweredge+pp_width+pcb_width-1] = FR4Permittivity
        epsilon[:, :, pp_loweredge+pp_width+pcb_width+1:] = LArPermittivity

    # Pixel-plane laminate FR4 WITHOUT a shield grid (Task7a): default LAr
    # everywhere, then make the BOTTOM n_fr4 cell(s) of the pixel-plane layer
    # FR4.  The pad conductor sits on TOP of the layer (see draw_pixel_plane
    # fr4_bottom); the FR4 cell(s) stay FREE so their permittivity enters the
    # harmonic-mean Poisson solve.  LAr fills the drift gap above the pad and
    # the region below the FR4.  Default off -> epsilon stays None for every
    # non-FR4 run (byte-unchanged).
    #
    # Physically-correct laminate geometry: the pad and FR4 thicknesses are
    # given explicitly (in mm) via 'padThickness'/'FR4Thickness' -- e.g. a
    # 1oz-copper pad (~0.0348 mm) over a 1.6 mm FR4 substrate.  Because 1oz Cu
    # is thinner than one 0.05 mm cell, n_pad is clamped to a 1-cell minimum;
    # n_fr4 likewise.  The pixel-plane layer thickness pp_width is then derived
    # so the conductor occupies exactly n_pad cells on top and FR4 the n_fr4
    # cells below (pad cells = z2 - zp1 = pp_width + 1 - n_fr4 = n_pad).  If the
    # thickness keys are absent, fall back to the legacy half-and-half split
    # (n_fr4 = pp_width // 2) so older FR4 configs are unchanged.
    enableFR4 = cfg.get('enableFR4', False)
    fr4_bottom = False
    n_fr4 = 0
    if enableFR4 and LArPermittivity is not None and FR4Permittivity is not None:
        fr4_thickness = cfg.get('FR4Thickness', None)
        pad_thickness = cfg.get('padThickness', None)
        if fr4_thickness is not None and pad_thickness is not None:
            n_fr4 = max(1, int(round(fr4_thickness / dom.spacing[0])))
            n_pad = max(1, int(round(pad_thickness / dom.spacing[0])))
            pp_width = n_pad + n_fr4 - 1
            if info_msg is not None:
                info_msg(f'FR4 laminate: pad={pad_thickness}mm ({n_pad} cell(s)), '
                         f'FR4={fr4_thickness}mm ({n_fr4} cell(s)), pp_width={pp_width}')
        else:
            n_fr4 = max(1, pp_width // 2)
        epsilon = numpy.full(dom.shape, LArPermittivity)
        epsilon[:, :, pp_loweredge:pp_loweredge + n_fr4] = FR4Permittivity
        fr4_bottom = True
    if gridHoleShape == 'circular':
        draw_pcb_plane((len(arr),len(arr[0])), arr, barr, pp_loweredge+pcb_width, r1, gridPotential) # Draw the PCB plane with holes circular
        ## We need to use a mask to define the holes in the FR4 and set the permittivity to LAr in those holes
        ## round hole
        Nx, Ny = len(arr),len(arr[0])
        xi, yi = numpy.mgrid[0:Nx, 0:Ny]
        # Zero out the 4 quarter-holes at the corners
        for cx, cy in [(0, 0), (Nx-1, 0), (0, Ny-1), (Nx-1, Ny-1)]:
            mask = (xi - cx)**2 + (yi - cy)**2 <= r1**2
            epsilon[:, :, pp_loweredge+pp_width+1:pp_loweredge+pp_width+pcb_width-1][mask] = LArPermittivity

    elif gridHoleShape == 'square':
        draw_pcb_plane_rounded_sq_drift(arr, barr, p_gap, p_size, pcb_width, pp_loweredge, gridPotential, chamfer_r=chamfer_r) # Draw the PCB plane with holes rounded square
        ## We need to use a mask to define the holes in the FR4 and set the permittivity to LAr in those holes
        ## square hole
    barr[arr==0]=0


    draw_pixel_plane(arr,barr,p_size,p_gap,n_pix,pp_loweredge,pp_width,cathodePotential,gridPotential, epsilon=epsilon, chamfer_r=chamfer_r, chamferMode=chamferMode, fr4_bottom=fr4_bottom, n_fr4=n_fr4)
    

    if info_msg is not None:
        info_msg(f'cathode potential : {cathodePotential} V')
        info_msg(f'arr[:, :, 0] = {arr[:, :, 0]}')
        info_msg(f'arr[:, :, -1] = {arr[:, :, -1]}')
        info_msg('----------------------------------')
        info_msg(f'arr[:, :, 10] = {arr[:, :, 10]}')
        info_msg(f'arr.shape = {arr.shape}')
        info_msg(f'arr[22, 22, :] = {arr[22, 22, :]}')
        info_msg('----------------------------------')
        info_msg(f'barr[:, :, 0] = {barr[:, :, 0]}')
        info_msg(f'barr[:, :, -1] = {barr[:, :, -1]}')
        info_msg(f'barr.shape = {barr.shape}')
        info_msg(f'barr[22, 22, :] = {barr[22, 22, :]}')
        info_msg(f'pixelPlaneLowEdgePosition = {pp_loweredge}')
        info_msg(f'barr[:, :, pp_loweredge] = {barr[:, :, pp_loweredge]}')
        
        info_msg(f'p_size = {p_size}, p_gap = {p_gap}, n_pix = {n_pix}, pp_width = {pp_width}')
    return arr,barr, epsilon
