#!/usr/bin/env python3

import numpy
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .gen_pcb_drift_pixel_with_grid import draw_quarter_circle as draw_quarter
# from pochoir.InverseDistanceWeight_torch import init_idw_pcb_pixel, plot_idw_pcb_pixel

def fill_area(arr,barr,val):
    for b in barr:
        arr[b[0],b[1][0]:b[1][1]+1]=val
        
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


    
def draw_plane(arr,z,val):
    """
    Fill 1 plane
    """
    arr[:,:,z]=val

def mirror_center(id_circ1,x0,y0):
    """
    mirror quarter-circle of the center point
    """
    id_circ2=[]
    for id in id_circ1:
        id_circ2.append((x0-(id[0]-x0),y0-(id[1]-y0)))
    return id_circ2

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


def draw_3Dstrips(arr,barr,Nstrips,z,r1,thickness=1):

    shape = (int(len(barr)/Nstrips),int(len(barr[0])/Nstrips))
    
    xc=int(shape[0]/2-1)
    yc=int(shape[1]/2-1)
    id_circ3 = draw_quarter(xc,yc,r1)
    id_circ2 = mirror_xaxis(id_circ3,xc,yc,r1)
    id_circ4 = mirror_yaxis(id_circ3,xc,yc,r1)
    id_circ1 = mirror_center(id_circ3,xc,yc)
    barr1=form_quarter_boundary(id_circ1,xc,yc)
    barr2=form_quarter_boundary(id_circ2,xc,yc)
    barr3=form_quarter_boundary(id_circ3,xc,yc)
    barr4=form_quarter_boundary(id_circ4,xc,yc)
    for iz in range(z, z+thickness):
        draw_plane(barr,iz,1)
        fill_area(barr[:,:,iz],barr1,0)
        fill_area(barr[:,:,iz],barr2,0)
        fill_area(barr[:,:,iz],barr3,0)
        fill_area(barr[:,:,iz],barr4,0)
        for i in range(Nstrips):
            for j in range(Nstrips):
                barr[i*shape[0]:(i+1)*shape[0],j*shape[1]:(j+1)*shape[1],iz]=barr[:shape[0],:shape[1],iz]

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


def draw_pixel_plane(arr,barr,p_size,p_gap,n_pix,pp_loweredge,pp_width):
    dims = p_size*n_pix+p_gap*(n_pix-1)
    for i in range(0,n_pix):
        for j in range(0,n_pix):
            barr[int(p_gap/2)+i*(p_size+p_gap):int(p_gap/2)+i*(p_size+p_gap)+p_size,int(p_gap/2)+j*(p_size+p_gap):int(p_gap/2)+j*(p_size+p_gap)+p_size,pp_loweredge:pp_width+pp_loweredge+1]=1
            trimCorner(barr,int(p_gap/2)+i*(p_size+p_gap)+p_size-1,int(p_gap/2)+j*(p_size+p_gap)+p_size-1,pp_loweredge,pp_width+pp_loweredge+1,0)
    
            trimCorner(barr,int(p_gap/2)+i*(p_size+p_gap)+p_size-1,int(p_gap/2)+j*(p_size+p_gap),pp_loweredge,pp_width+pp_loweredge+1,1)
    
            trimCorner(barr,int(p_gap/2)+i*(p_size+p_gap),int(p_gap/2)+j*(p_size+p_gap)+p_size-1,pp_loweredge,pp_width+pp_loweredge+1,3)
    
            trimCorner(barr,int(p_gap/2)+i*(p_size+p_gap),int(p_gap/2)+j*(p_size+p_gap),pp_loweredge,pp_width+pp_loweredge+1,2)
            if i==int(n_pix/2) and i==j:
                    arr[int(p_gap/2)+i*(p_size+p_gap):int(p_gap/2)+i*(p_size+p_gap)+p_size,int(p_gap/2)+j*(p_size+p_gap):int(p_gap/2)+j*(p_size+p_gap)+p_size,pp_loweredge:pp_width+pp_loweredge+1]=1
                    trimCorner(arr,int(p_gap/2)+i*(p_size+p_gap)+p_size-1,int(p_gap/2)+j*(p_size+p_gap)+p_size-1,pp_loweredge,pp_width+pp_loweredge+1,0)
    
                    trimCorner(arr,int(p_gap/2)+i*(p_size+p_gap)+p_size-1,int(p_gap/2)+j*(p_size+p_gap),pp_loweredge,pp_width+pp_loweredge+1,1)
    
                    trimCorner(arr,int(p_gap/2)+i*(p_size+p_gap),int(p_gap/2)+j*(p_size+p_gap)+p_size-1,pp_loweredge,pp_width+pp_loweredge+1,3)
    
                    trimCorner(arr,int(p_gap/2)+i*(p_size+p_gap),int(p_gap/2)+j*(p_size+p_gap),pp_loweredge,pp_width+pp_loweredge+1,2)
    # # draw pixel plane
    # plt.figure(figsize=(10,10))
    # plt.imshow(barr[:,:,pp_loweredge],origin='lower')
    # plt.title('pixel plane')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.savefig('store/pixel_plane_bc.png')
    # plt.close()

    plt.figure(figsize=(10,10))
    plt.imshow(arr[:,:,pp_loweredge],origin='lower')
    plt.title('pixel plane')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.savefig('store/pixel_plane_initialcond.png')
    plt.close()
def generator(dom, cfg):
    
    
    r1 = int(round(cfg['HoleRadius']/dom.spacing[0])-1)
    pcb_width = int(cfg['PcbWidth']/dom.spacing[0])

    arr = numpy.zeros(dom.shape)
    barr = numpy.zeros(dom.shape)
    
    
    p_size=int(round(cfg["pixelSize"]/dom.spacing[0]))
    p_gap=int(round(cfg["pixelGap"]/dom.spacing[0]))
    n_pix = cfg['Npixels']
    pp_width = int(cfg['pixelPlaneWidth']/dom.spacing[0])
    pp_loweredge = int(cfg['pixelPlaneLowEdgePosition']/dom.spacing[0])

    draw_pixel_plane(arr,barr,p_size,p_gap,n_pix,pp_loweredge,pp_width)

    print(f'p_size={p_size}, p_gap={p_gap}, n_pix={n_pix}, pp_width={pp_width}, pp_loweredge={pp_loweredge}')

    # Target pixel centre coordinates (used for iterative target_voltage seeding)
    center    = int(n_pix / 2)
    x0_tgt    = int(p_gap / 2) + center * (p_size + p_gap)
    x_tgt_mid = x0_tgt + p_size // 2
    y0_tgt    = x0_tgt
    y_tgt_mid = x_tgt_mid
    z_tgt_mid = pp_loweredge + pp_width // 2

    # Iterative IDW initialisation along z-axis in chunks of chunk_size.
    # For the first chunk the target voltage is 1.0 (the electrode voltage).
    # For every subsequent chunk the target voltage is taken from the value at
    # the target pixel's (x, y) position at the last z-plane of the previous chunk.
    # _device = 'cuda:1'
    # chunk_size = 110
    # # nz_total = arr.shape[2]
    # nz_total = 300
    # target_voltage = 1.0
    # out = init_idw_pcb_pixel(
    #     arr[:, :, pp_loweredge],
    #     p_size=p_size,
    #     p_gap=p_gap,
    #     n_pix=n_pix,
    #     target_voltage=target_voltage,
    #     ground_voltage=0.0,
    #     power=1.0,
    #     batch_size=1024,
    #     device=_device,
    #     dtype=torch.float32
    # )
    # arr[:, :, pp_loweredge] = out.cpu().numpy()

    # plot_idw_pcb_pixel(
    #     phi=arr[:, :, pp_loweredge],
    #     title="init_idw_pcb_pixel — IDW initial guess",
    #     save_path='idw_init_pcb_pixel.png',
    #     vmin=0.0,
    #     vmax=1.0,
    #     cmap="RdBu_r",
    # ) 
    draw_3Dstrips(arr,barr,n_pix,pp_loweredge+pcb_width,r1,thickness=pcb_width)
    # draw_pixel_plane(arr,barr,p_size,p_gap,n_pix,pp_loweredge,pp_width)

    barr[:,:,0]=1
    # # draw pixel plane
    # plt.figure(figsize=(10,10))
    # plt.imshow(barr[:,:,pp_loweredge],origin='lower')
    # plt.title('pixel plane')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.savefig('store/pixel_plane_bc.png')
    # plt.close()

    # 3D voxel plot of the whole volume
    z_pcb_start = pp_loweredge + pcb_width
    z_pcb_end   = z_pcb_start + pcb_width

    # Build a boolean volume restricted to the z-range we care about
    filled = barr.astype(bool)

    # Colour map: assign RGBA per voxel
    colors = numpy.empty(filled.shape + (4,), dtype=float)
    # default transparent
    colors[..., :] = 0
    # ground plane: gray
    colors[:, :, 0:2, :] = [0.5, 0.5, 0.5, 0.9]
    # pixel plane: orange
    colors[:, :, pp_loweredge:pp_loweredge + pp_width + 1, :] = [1.0, 0.6, 0.1, 0.9]
    # PCB strip: steelblue
    colors[:, :, z_pcb_start:z_pcb_end, :] = [0.27, 0.51, 0.71, 0.9]

    fig3d = plt.figure(figsize=(12, 10))
    ax3d = fig3d.add_subplot(111, projection='3d')
    ax3d.voxels(filled, facecolors=colors, edgecolors='none')
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')
    ax3d.set_title('Full geometry: ground plane, pixel plane, PCB strip')
    plt.tight_layout()
    # plt.show()
    plt.savefig('store/full_geometry.png')
    plt.close()

    return arr,barr
