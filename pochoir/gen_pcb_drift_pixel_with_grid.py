#!/usr/bin/env python3

import numpy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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


def trimCorner(arr,x,y,z1,z2,corner):
    if corner==0:
        arr[x-3:x+1,y,z1:z2]=0
        arr[x,y-3:y+1,z1:z2]=0
        arr[x,y,z1:z2]=0
        arr[x-1,y-1,z1:z2]=0
    if corner==1:
        arr[x-3:x+1,y,z1:z2]=0
        arr[x,y:y+4,z1:z2]=0
        arr[x,y,z1:z2]=0
        arr[x-1,y+1,z1:z2]=0
    if corner==2:
        arr[x:x+4,y,z1:z2]=0
        arr[x,y:y+4,z1:z2]=0
        arr[x,y,z1:z2]=0
        arr[x+1,y+1,z1:z2]=0
    if corner==3:
        arr[x:x+4,y,z1:z2]=0
        arr[x,y-3:y+1,z1:z2]=0
        arr[x,y,z1:z2]=0
        arr[x+1,y-1,z1:z2]=0

## Draw shield plane with square holes, rounded corners
def trimCorner_pcb(arr,x,y,z1,z2,corner):
    if corner==0:
        arr[x-3:x+1,y,z1:z2]=1
        arr[x,y-3:y+1,z1:z2]=1
        arr[x,y,z1:z2]=1
        arr[x-1,y-1,z1:z2]=1
    if corner==1:
        arr[x-3:x+1,y,z1:z2]=1
        arr[x,y:y+4,z1:z2]=1
        arr[x,y,z1:z2]=1
        arr[x-1,y+1,z1:z2]=1
    if corner==2:
        arr[x:x+4,y,z1:z2]=1
        arr[x,y:y+4,z1:z2]=1
        arr[x,y,z1:z2]=1
        arr[x+1,y+1,z1:z2]=1
    if corner==3:
        arr[x:x+4,y,z1:z2]=1
        arr[x,y-3:y+1,z1:z2]=1
        arr[x,y,z1:z2]=1
        arr[x+1,y-1,z1:z2]=1
def draw_pcb_plane_rounded_sq_drift(arr, barr, p_gap, p_size, pcb_width, pp_loweredge, gridPotential):
    barr[:, :, pcb_width+pp_loweredge] = 1
    arr[:, :, pcb_width+pp_loweredge] = gridPotential
    barr[0:int(p_size/2),0:int(p_size/2),pp_loweredge+pcb_width]=0
    barr[0:int(p_size/2),int(p_size/2)+p_gap:,pp_loweredge+pcb_width]=0
    barr[int(p_size/2)+p_gap:,0:int(p_size/2),pp_loweredge+pcb_width]=0
    barr[int(p_size/2)+p_gap:,int(p_size/2)+p_gap:,pp_loweredge+pcb_width]=0
    
    
    trimCorner_pcb(barr,int(p_size/2)-1,int(p_size/2)-1,pp_loweredge+pcb_width,pcb_width+pp_loweredge+1,0)
    
    trimCorner_pcb(barr,int(p_size/2)-1,int(p_size/2)+p_gap,pp_loweredge+pcb_width,pcb_width+pp_loweredge+1,1)
    
    trimCorner_pcb(barr,int(p_size/2)+p_gap,int(p_size/2)-1,pp_loweredge+pcb_width,pcb_width+pp_loweredge+1,3)
    
    trimCorner_pcb(barr,int(p_size/2)+p_gap,int(p_size/2)+p_gap,pp_loweredge+pcb_width,pcb_width+pp_loweredge+1,2)
##----

import sys
def draw_pixel_plane(arr,barr,p_size,p_gap,n_pix,pp_loweredge,pp_width,cathodePotential,gridPotential):
    draw_plane(arr,-1,cathodePotential) # This line sets the initial values
    # ## Set the initial values to be linear along z
    # for i in range(pp_loweredge, arr.shape[2]):
    #     arr[:, :, i] = (-7000/1399)*(i-100)
    # ## Set the initial values to be random between -7000 and 0 for z=101 and z=1498
    # for i in range(pp_loweredge+2, arr.shape[2]-1):
    #     arr[:, :, i] = numpy.random.uniform(-7000, 0, size=(arr.shape[0], arr.shape[1]))
    # print(f'arr[:, :, 100] = {arr[:, :, 100]}')
    # print(f'arr[:, :, 101] = {arr[:, :, 101]}')
    # print(f'arr[:, :, 1498] = {arr[:, :, 1498]}')
    # print(f'arr[:, :, 1499] = {arr[:, :, 1499]}')
    # sys.exit()
    draw_plane(barr,-1,1) # This line sets the boundary values

    dims = p_size*n_pix+p_gap*(n_pix-1)
    barr[0:int(p_size/2),0:int(p_size/2),pp_loweredge:pp_width+pp_loweredge+1]=1
    barr[0:int(p_size/2),int(p_size/2)+p_gap:,pp_loweredge:pp_width+pp_loweredge+1]=1
    barr[int(p_size/2)+p_gap:,0:int(p_size/2),pp_loweredge:pp_width+pp_loweredge+1]=1
    barr[int(p_size/2)+p_gap:,int(p_size/2)+p_gap:,pp_loweredge:pp_width+pp_loweredge+1]=1
    
    
    trimCorner(barr,int(p_size/2)-1,int(p_size/2)-1,pp_loweredge,pp_width+pp_loweredge+1,0)
    
    trimCorner(barr,int(p_size/2)-1,int(p_size/2)+p_gap,pp_loweredge,pp_width+pp_loweredge+1,1)
    
    trimCorner(barr,int(p_size/2)+p_gap,int(p_size/2)-1,pp_loweredge,pp_width+pp_loweredge+1,3)
    
    trimCorner(barr,int(p_size/2)+p_gap,int(p_size/2)+p_gap,pp_loweredge,pp_width+pp_loweredge+1,2)
    # arr[(p_size+p_gap):(p_size+p_gap)+p_size,(p_size+p_gap):(p_size+p_gap)+p_size,pp_loweredge:pp_width+pp_loweredge+1]=1
    # draw pixel plane for drift field
    # print(f'pp_loweredge={pp_loweredge}')
    # print(f'barr shape={barr.shape}')
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
        plt.savefig('store/domain_drift_arr_3d.png', dpi=150)
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
        plt.savefig('store/domain_drift_barr_3d.png', dpi=150)
        plt.close()

    # Same plot but clipped to first 150 z-planes, with large markers to prove surface-like render
    barr_clipped = barr[:, :, :150]
    mask_clip = barr_clipped != 0
    if mask_clip.any():
        xi, yi, zi = numpy.where(mask_clip)
        for marker_size, fname in [(2, 'domain_drift_barr_3d_clipped150_s2.png'),
                                   (20, 'domain_drift_barr_3d_clipped150_s20.png'),
                                   (200, 'domain_drift_barr_3d_clipped150_s200.png')]:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            sc = ax.scatter(xi, yi, zi, c=zi, cmap='viridis', s=marker_size, alpha=0.4, marker=',')
            plt.colorbar(sc, ax=ax, label='z index')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.set_title(f'barr drift (first 150 z) - s={marker_size}')
            plt.tight_layout()
            plt.savefig(f'store/{fname}', dpi=150)
            plt.close()

    # plt.figure(figsize=(10,10))
    # plt.imshow(arr[:, :, pp_loweredge], origin='lower')
    # print(f'kdlsjfew arr[43, 43, 1499]={arr[:, :, 1499]}')
    # plt.title('pixel plane')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.savefig('store/domain_drift_initial_cond.png')
    # plt.close()
    
def generator(dom, cfg, info_msg=None):
    
    r1 = int(cfg['HoleRadius']/dom.spacing[0]-1)
    pcb_width = int(cfg['PcbWidth']/dom.spacing[2])
    gridPotential = cfg['GridPotential']
    cathodePotential = cfg['CathodePotential']
    pp_loweredge = int(cfg['pixelPlaneLowEdgePosition']/dom.spacing[0])
    p_size=int(round(cfg["pixelSize"]/dom.spacing[0]))
    p_gap=int(round(cfg["pixelGap"]/dom.spacing[0]))
    n_pix = cfg['Npixels']
    pp_width = int(cfg['pixelPlaneWidth']/dom.spacing[0])

    arr = numpy.zeros(dom.shape)
    barr = numpy.zeros(dom.shape)
    # draw_pcb_plane((len(arr),len(arr[0])), arr, barr, pp_loweredge+pcb_width, r1, gridPotential) # Draw the PCB plane with holes circular
    draw_pcb_plane_rounded_sq_drift(arr, barr, p_gap, p_size, pcb_width, pp_loweredge, gridPotential) # Draw the PCB plane with holes rounded square
    barr[arr==0]=0
    
    draw_pixel_plane(arr,barr,p_size,p_gap,n_pix,pp_loweredge,pp_width,cathodePotential,gridPotential)
    

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
    return arr,barr
