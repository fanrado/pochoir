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


def draw_pcb_plane(shape,arr,z,r1,val):
    draw_plane(arr,z,val)
    id_circ1=draw_quarter_circle(0,0,r1)
    id_circ2=draw_quarter_circle(shape[0]-1,shape[1]-1,r1)
    id_circ2_m=mirror_center(id_circ2,shape[0]-1,shape[1]-1)
    id_circ3=draw_quarter_circle(shape[0]-1,0,r1)
    id_circ3_m=mirror_yaxis(id_circ3,shape[0]-1,0,r1)
    id_circ4=draw_quarter_circle(0,shape[1]-1,r1)
    id_circ4_m=mirror_xaxis(id_circ4,0,shape[1]-1,r1)
    barr1=form_quarter_boundary(id_circ1,0,0)
    barr2=form_quarter_boundary(id_circ2_m,shape[0]-1,shape[1])
    barr3=form_quarter_boundary(id_circ3_m,shape[0]-1,0)
    barr4=form_quarter_boundary(id_circ4_m,0,shape[1]-1)
    fill_area(arr,barr1,0)
    fill_area(arr,barr2,0)
    fill_area(arr,barr3,0)
    fill_area(arr,barr4,0)

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

def draw_pixel_plane(arr,barr,p_size,p_gap,n_pix,pp_loweredge,pp_width,cathodePotential,gridPotential):
    draw_plane(arr,-1,cathodePotential)
    draw_plane(barr,-1,1)
    dims = p_size*n_pix+p_gap*(n_pix-1)
    barr[0:int(p_size/2),0:int(p_size/2),pp_loweredge:pp_width+pp_loweredge+1]=1
    barr[0:int(p_size/2),int(p_size/2)+p_gap:,pp_loweredge:pp_width+pp_loweredge+1]=1
    barr[int(p_size/2)+p_gap:,0:int(p_size/2),pp_loweredge:pp_width+pp_loweredge+1]=1
    barr[int(p_size/2)+p_gap:,int(p_size/2)+p_gap:,pp_loweredge:pp_width+pp_loweredge+1]=1
    
    
    trimCorner(barr,int(p_size/2)-1,int(p_size/2)-1,pp_loweredge,pp_width+pp_loweredge+1,0)
    
    trimCorner(barr,int(p_size/2)-1,int(p_size/2)+p_gap,pp_loweredge,pp_width+pp_loweredge+1,1)
    
    trimCorner(barr,int(p_size/2)+p_gap,int(p_size/2)-1,pp_loweredge,pp_width+pp_loweredge+1,3)
    
    trimCorner(barr,int(p_size/2)+p_gap,int(p_size/2)+p_gap,pp_loweredge,pp_width+pp_loweredge+1,2)
        #arr[i*(p_size+p_gap):i*(p_size+p_gap)+p_size,i*(p_size+p_gap):i*(p_size+p_gap)+p_size,pp_loweredge:pp_width+pp_loweredge+1]=1
    ## draw pixel plane for drift field
    plt.figure(figsize=(10,10))
    plt.imshow(barr[:,:,pp_loweredge],origin='lower')
    plt.title('pixel plane')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('store/domain_drift.png')
    
def generator(dom, cfg):
    
    r1 = int(cfg['HoleRadius']/dom.spacing[0]-1)
    pcb_width = int(cfg['PcbWidth']/dom.spacing[2])
    gridPotential = cfg['GridPotential']
    cathodePotential = cfg['CathodePotential']
    pp_loweredge = int(cfg['pixelPlaneLowEdgePosition']/dom.spacing[0])

    arr = numpy.zeros(dom.shape)
    barr = numpy.ones(dom.shape)
    #draw_pcb_plane((len(arr),len(arr[0])),arr,pp_loweredge+pcb_width,r1,-1000)
    barr[arr==0]=0
    p_size=int(round(cfg["pixelSize"]/dom.spacing[0]))
    p_gap=int(round(cfg["pixelGap"]/dom.spacing[0]))
    n_pix = cfg['Npixels']
    pp_width = int(cfg['pixelPlaneWidth']/dom.spacing[0])
    draw_pixel_plane(arr,barr,p_size,p_gap,n_pix,pp_loweredge,pp_width,cathodePotential,gridPotential)

    return arr,barr
