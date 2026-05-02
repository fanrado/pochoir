#!/usr/bin/env python3
'''
Make plots.
'''
from . import arrays
from . import units
from pathlib import Path
import numpy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# fixme: instead of passing in a file name just so this can be called,
# the caller in __main__.py should handle saving.
def savefig(fname):
    path = Path(fname)
    if not path.parent.exists():
        path.parent.mkdir(parents=True)
    plt.savefig(path.resolve(), dpi=600)

def signedlog(arr, eps = 1e-5, scale=None):
    '''
    Apply the "signed log" transform to an array.

    Result is +/-log10(|arr|*scale) with the sign of arr preserved in
    the result and any values that are in eps of zero set to zero.

    If scale is not given then 1/eps is used.
    '''
    if not scale:
        scale = 1/eps

    shape = arr.shape
    arr = numpy.array(arr).reshape(-1)
    arr[numpy.logical_and(arr < eps, arr > -eps)] = 0.0
    pos = arr>eps
    neg = arr<-eps
    arr[pos] = numpy.log10(arr[pos]*scale)
    arr[neg] = -numpy.log10(-arr[neg]*scale)
    return arr.reshape(shape)

def image(arr, fname, domain, title="", scale="linear"):
    if len(arr.shape) != 2:
        raise ValueError("image plots take 2D arrays")

    arr = arrays.to_numpy(arr)
    if scale == "signedlog":
        arr = signedlog(arr)

    plt.clf()

    # extent = None
    # if domain:
    #     extent = domain.imshow_extent()
    #X,Y = domain.meshgrid
    Y,X = numpy.meshgrid(*domain.linspaces, indexing="ij")

    plt.title(title)
    print(f'plotting: {arr.shape} {fname}')
    #plt.imshow(arr, interpolation='none', aspect='auto',
    #           extent = extent)
    plt.pcolormesh(arr, shading='auto')
    #plt.imshow(arr, interpolation='none', aspect='auto')
    plt.colorbar()
    plt.show()
    savefig(fname)

def set_limits(limits):
    if not limits:
        return
    xlim, ylim = limits
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)


def quiver(varr, fname, domain, step=100, limits=None, scale=1.0):
    '''
    Plot a vector field.

    step determines the amount of decimation.
    '''
    varr = [arrays.to_numpy(a) for a in varr]
    ndim = len(varr)
    if ndim not in (2,3):
        raise ValueError("quiver plots take vector of 2D or 3D arrays")

    
    mg = numpy.meshgrid(*domain.linspaces, indexing="ij")
    print (f'meshgrid: {mg[0].shape} -> {mg[1].shape}')

    # possibly decimate
    slcs = tuple([slice(0,s,step) for s in varr[0].shape])
    skip = (slice(0,25,1),slice(0,17,1),slice(140,220,4))
    #skip = (slice(None,None,2),6,slice(50,400,1))
    plt.clf()
    #ndim=4
    if ndim == 4:               # 2D
        plt.quiver(mg[2][skip], mg[0][skip],
                   varr[2][skip], varr[0][skip],
                   scale=scale, units='xy')
        set_limits(limits)
            
    else:                       # 3D
        fig = plt.figure()
        # matplotlib >= 3.5 removed kwargs on Figure.gca(); the documented
        # replacement for `gca(projection='3d')` is `add_subplot(projection='3d')`.
        ax = fig.add_subplot(projection='3d')
        #ax.set_xlim3d(0,domain.shape[0]*domain.spacing[0])
        #ax.set_ylim3d(0,domain.shape[1]*domain.spacing[1])
        #ax.set_zlim3d(0,domain.shape[2]*domain.spacing[2])
        ax.set_xlim3d(0,2.5)
        ax.set_ylim3d(0,1.75)
        ax.set_zlim3d(130*0.1,260*0.1)
        ax.quiver(mg[0][skip], mg[1][skip], mg[2][skip],
                  varr[0][skip], varr[1][skip], varr[2][skip],
                  length=domain.spacing[0], normalize=True)
        set_limits(limits)
    plt.show()
    savefig(fname)

def drift2d(paths, output, domain, trajectory):
    '''
    Plot 2D drift paths
    '''
    for path in paths:
        plt.scatter(path[:,1], path[:,0])
    savefig(output)

def drift3d(varr, fname, domain, trajectory):
    '''
    Plot 3D drift paths
    '''
    varr = [arrays.to_numpy(a) for a in varr]
    ndim = len(varr)
    mg = domain.meshgrid
    plt.clf()
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlim3d(0,domain.shape[0]*domain.spacing[0])
    ax.set_ylim3d(0,domain.shape[1]*domain.spacing[1])
    ax.set_zlim3d(0,domain.shape[2]*domain.spacing[2])
    if(len(varr)<trajectory):
        raise ValueError("Not enough trajectories to plot")
    if(trajectory==-1):
        xdata,ydata,zdata = varr[0][:,0],varr[0][:,1],varr[0][:,2]
        ax.plot3D(xdata,ydata,zdata)
    else:
        for i in range(trajectory):
            xdata,ydata,zdata = varr[11*5+i][:,0],varr[11*5+i][:,1],varr[11*5+i][:,2]
            ax.plot3D(xdata,ydata,zdata)
    savefig(fname)
    
def drift3d_b(varr, barr, fname, domain, trajectory,zoom,gif,title=""):
    '''
    Plot 3D drift paths and boundary array
    '''
    arr1 = numpy.array(varr)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim3d(0,domain.shape[0]*domain.spacing[0])
    ax.set_ylim3d(0,domain.shape[1]*domain.spacing[1])
    if zoom == "yes":
        barr[:,:,0]=0
        ax.set_zlim3d(10,35)#domain.shape[2])
        #ax.set_zlim3d(10.0,20)
        #ax.set_title(title+"(zoomed)")
        ax.set_title("")
    else:
        ax.set_title(title)
        ax.set_zlim3d(0,domain.shape[2]*domain.spacing[2])
    
    arr2 = numpy.array(barr)
    x,y,z = arr2.nonzero()
    ax.scatter(x*domain.spacing[0],y*domain.spacing[1],z*domain.spacing[2],s=1)
    ax.set_xlabel('X, mm')
    ax.set_ylabel('Y, mm')
    ax.set_zlabel('Z, mm')
    
    if(len(varr)<trajectory):
        raise ValueError("Not enough trajectories to plot")
    if(trajectory==-1):
        xdata,ydata,zdata = arr1[0][:,0],arr1[0][:,1],arr1[0][:,2]
        ax.scatter(xdata,ydata,zdata,s=0.85)
    else:
        for i in range(trajectory):
            st=0
            xdata,ydata,zdata = arr1[st+i][:,0],arr1[st+i][:,1],arr1[st+i][:,2]
            print(xdata)
            ax.scatter(xdata,ydata,zdata,s=1)
    plt.show()
    #savefig(fname)
    fname2 = fname[:-4]
    if gif == "yes":
        def rotate(angle):
            ax.view_init(azim=angle)
        import matplotlib.animation as animation
        print("Making animation")
        rot_animation = animation.FuncAnimation(fig, rotate, frames=numpy.arange(0, 362, 2), interval=100)
        rot_animation.save(fname2+'.gif', dpi=80, writer='imagemagick')
    
def scatt3d(varr,fname,domain,gif,title=""):
    arr = numpy.array(varr)
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim3d(0,domain.shape[0]*domain.spacing[0])
    ax.set_ylim3d(0,domain.shape[1]*domain.spacing[1])
    ax.set_zlim3d(0,domain.shape[2]*domain.spacing[2])
    x,y,z = arr.nonzero()
    ax.scatter(x*domain.spacing[0],y*domain.spacing[1],z*domain.spacing[2],s=0.7)
    ax.set_xlabel('X, mm')
    ax.set_ylabel('Y, mm')
    ax.set_zlabel('Z, mm')
    ax.set_title(title);
    savefig(fname)
    #plt.show()
    fname2 = fname[:-4]
    if gif == "yes":
        def rotate(angle):
            ax.view_init(azim=angle)
        import matplotlib.animation as animation
        print("Making animation")
        rot_animation = animation.FuncAnimation(fig, rotate, frames=numpy.arange(0, 362, 2), interval=100)
        rot_animation.save(fname2+'.gif', dpi=80, writer='imagemagick')
    
def slice3d(varr,fname,domain,scale,dim,index,title=""):
    
    arr = arrays.to_numpy(varr)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if scale == "signedlog":
        arr = signedlog(arr)
    plt.title(title+"(scale:"+scale+", slice:"+dim+f', index: {index})')
    print(f'plotting: {arr.shape} {fname}')
    if dim == "x":
        plt.pcolormesh(arr[index,:,:], shading='auto')
    if dim == "y":
        plt.pcolormesh(arr[:,index,:], shading='auto')
    if dim == "z":
        plt.pcolormesh(arr[:,:,index], shading='auto')
    plt.colorbar()
    #ax.set_aspect('equal')
    import sys
    numpy.set_printoptions(threshold=sys.maxsize)
    #print(arr[408:515,0:28,340])
    plt.show()
    savefig(fname)

def mirror_arr_yaxis(arr):
    result = numpy.empty_like(arr)
    result[:,::-1]=arr[:,:]
    return result

def slice3d_two(varr,varr2,fname,domain,scale,dim,index,title=""):
    
    arr_c = arrays.to_numpy(varr[:,:,0:1600])
    arr_c_m = mirror_arr_yaxis(arr_c)
    print(arr_c.shape,varr.shape,varr2.shape)
    arr1 = numpy.zeros(varr2.shape)
    print(varr[19,14,340])
    for i in range(0,arr_c.shape[0]):
        for j in range(0,arr_c.shape[1]):
            arr1[i,j,:]=arr_c_m[i,j,:]
            arr1[i,j,:]=arr_c_m[i,j,:]
            arr1[i,j+29,:]=arr_c[i,j,:]
            arr1[i,j+29,:]=arr_c[i,j,:]
    arr2 = arrays.to_numpy(varr2)
    arr = arr1+arr2
    arr[19,14+29,340]=arr[19,14+29,340]+1
    arr[20,14+29,340]=arr[20,14+29,340]+1
    arr[19,15+29,340]=arr[19,15+29,340]+1
    arr[20,15+29,340]=arr[20,15+29,340]+1

    fig = plt.figure()
    ax = fig.add_subplot(111)
    if scale == "signedlog":
        arr = signedlog(arr)
    plt.title(title+"(scale:"+scale+", slice:"+dim+f', index: {index})')
    print(f'plotting: {arr.shape} {fname}')
    if dim == "x":
        plt.pcolormesh(arr[index,:,:], shading='auto')
    if dim == "y":
        plt.pcolormesh(arr[:,index,:], shading='auto')
    if dim == "z":
        plt.pcolormesh(arr[:,:,index], shading='auto')
    plt.colorbar()
    #ax.set_aspect('equal')
    print(arr[0,:,:])
    plt.show()
    #savefig(fname)
    
def electronics_no_gain_scale(time, gain, shaping=2.2, elec_type="cold"):
    '''
    This version takes gain parameter already scaled such that the
    gain actually desired is obtained.
    Both the "cold" and "warm" electroinics reponse functions are adapted from
    wire-cell-toolkit/util/src/Response.cxx.
    '''
    domain=(0, 10)
    #time = time/units.us
    #st = shaping/units.us
    if time <= domain[0] or time >= domain[1]:
        return 0.0

    time = time/units.us
    st = shaping/units.us
        
    from math import sin, cos, exp
    ret = 0
    if elec_type == "warm":
        reltime = time / st
        leftArg = (0.9 * reltime) * (0.9 * reltime)
        rightArg = (0.5 * reltime) * (0.5 * reltime)
        ret = ((1. - exp(-0.5 * leftArg)) * exp(-0.5 * rightArg))
    else:
        ret = 4.31054*exp(-2.94809*time/st) \
              -2.6202*exp(-2.82833*time/st)*cos(1.19361*time/st) \
              -2.6202*exp(-2.82833*time/st)*cos(1.19361*time/st)*cos(2.38722*time/st) \
              +0.464924*exp(-2.40318*time/st)*cos(2.5928*time/st) \
              +0.464924*exp(-2.40318*time/st)*cos(2.5928*time/st)*cos(5.18561*time/st) \
              +0.762456*exp(-2.82833*time/st)*sin(1.19361*time/st) \
              -0.762456*exp(-2.82833*time/st)*cos(2.38722*time/st)*sin(1.19361*time/st) \
              +0.762456*exp(-2.82833*time/st)*cos(1.19361*time/st)*sin(2.38722*time/st) \
              -2.6202*exp(-2.82833*time/st)*sin(1.19361*time/st)*sin(2.38722*time/st)  \
              -0.327684*exp(-2.40318*time/st)*sin(2.5928*time/st) +  \
              +0.327684*exp(-2.40318*time/st)*cos(5.18561*time/st)*sin(2.5928*time/st) \
              -0.327684*exp(-2.40318*time/st)*cos(2.5928*time/st)*sin(5.18561*time/st) \
              +0.464924*exp(-2.40318*time/st)*sin(2.5928*time/st)*sin(5.18561*time/st)
    ret *= gain
    return ret

def electronics(time, peak_gain=14*units.mV/units.fC, shaping=2.2, elec_type="cold"): #*units.mV/units.fC
    '''
    Electronics response function.
        - gain :: the peak gain value in [voltage]/[charge]
        - shaping :: the shaping time in Wire Cell system of units
        - domain :: outside this pair, the response is identically zero
    '''
    # see wirecell.sigproc.plots.electronics() for these magic numbers.
    gain = peak_gain
    if elec_type=="cold":
        if shaping <= 0.5:
            gain = peak_gain*10.146826
        elif shaping <= 1.0:
            gain = peak_gain*10.146828
        elif shaping <= 2.0:
            gain = peak_gain*10.122374
        else:
            gain = peak_gain*10.120179
    return electronics_no_gain_scale(time, gain, shaping, elec_type)

electronics = numpy.vectorize(electronics)

def convolve(f1, f2):
    '''
    Return the simple convolution of the two arrays using FFT+mult+invFFT method.
    '''
    # fftconvolve adds an unwanted time shift
    #from scipy.signal import fftconvolve
    #return fftconvolve(field, elect, "same")
    s1 = numpy.fft.fft(f1)
    s2 = numpy.fft.fft(f2)
    sig = numpy.fft.ifft(s1*s2)

    return numpy.real(sig)

def _convolve(f1, f2):
    '''
    Return the simple convolution of the two arrays using FFT+mult+invFFT method.
    '''
    from scipy.signal import fftconvolve
    return fftconvolve(f1, f2, "same")


def current(cur,cur2, output):
    '''
    Plot Current
    '''
    plt.figure(figsize=(10,6))
    #print(cur)
    x = numpy.linspace(0,6000,5999)
    #for c in cur:
    #import sys
    #numpy.set_printoptions(threshold=sys.maxsize)
    integral=0
    c_temp = numpy.zeros((5999))
    print("Number Of Currents=",len(cur))
    elecResp = electronics(x)#*units.us)
    shift_all=11
    #plt.plot(x,convolve(elecResp/(units.mV/units.fC),(cur[shift_all+671]+cur[shift_all+672]+cur[shift_all+673]+cur[shift_all+674]+cur[shift_all+675]+cur[shift_all+676]+cur[shift_all+677]+cur[shift_all+678]+cur[shift_all+679]+cur[shift_all+680]+cur[shift_all+681])*0.1*units.us/(11*units.fC)),color="black")
    for num,c in enumerate(cur):
        #if num<660 or num>725:
        #    continue
        #if num!=683:
        #    continue
        #if num<698 or num>702:
        #    continue
        #if num<231 or num>241:
        #if num<242 or num>252:
        #if num<253 or num>263:
        #if num!=670:
        #    continue
        c_pa = c#/units.picoampere
        c_temp=c_temp+numpy.asarray(c)
        for cc in c_pa:
            integral=integral+cc*0.1*units.us
        #if num==18:
        print("IntegralChargefor ",num," is ",integral)
        integral=0
        plt.plot(x,c_pa)
        #print(num,max(convolve(elecResp/(units.mV/units.fC),c_pa*0.1*units.us/units.fC)))
        #plt.plot(x,convolve(elecResp/(units.mV/units.fC),c_pa*0.01*units.us/units.fC))
    #c_temp=c_temp/11.0
    #plt.plot(x,c_temp)
    #print(elecResp)
    convRes=convolve(elecResp/(units.mV/units.fC),cur[61]*0.1*units.us/units.fC)
    #plt.plot(x,convRes,label="2.5mm hole")
    #plt.plot(x,convRes*1.03,'--',label="2.0mm hole")
    import math
    print("TotResp=",sum(convRes))
    #ctot=(cur2[0]+cur2[1]+cur2[2]+cur2[3]+cur2[4]+cur2[5]+cur2[6]+cur2[7]+cur2[8])/9
    #convRes2=convolve(elecResp/(units.mV/units.fC),ctot*0.1*units.us/units.fC)
    #shift=0
    #for i in range(0,shift):
    #    convRes2=numpy.insert(convRes2,0,0)
    #plt.plot(x,convRes2[:]*0.97,'--',label="2.0mm hole")
    #plt.plot(x,cur[19])#elecResp/(units.mV/units.fC))
    #plt.plot(x,(cur[22])*0.1*units.us/units.fC) # e/0.1us
    #plt.plot(x,cur[2]/320000)
    plt.xlim([1160, 1230])
    plt.title("Induction Plane",fontsize=14)
    plt.legend(fontsize=14)
    #plt.xlabel('Time, us')
    plt.ylabel('Response, mV',fontsize=14)
    plt.xlabel('DriftTime, us',fontsize=14)
    
    #plt.ylabel('Current on the Strip, picoAmp')
    #plt.ylabel('Current on the Strip, picoAmp')
    #plt.pcolormesh(cur[0][0:3,-3:-1,201], shading='auto')
    #plt.colorbar()
    plt.show()
    #plt.savefig(output)



def currentpixel(cur, output):
    '''
    Plot Current
    '''
    plt.figure(figsize=(10,6))
    print(len(cur),len(cur[0]))
    x = numpy.linspace(0,320,6399)
    #for c in cur:
    #import sys
    #numpy.set_printoptions(threshold=sys.maxsize)
    integral=0
    c_temp = numpy.zeros((6399))
    print("Number Of Currents=",len(cur))
    elecResp = electronics(x)#*units.us)
    shift_all=11
    #plt.plot(x,convolve(elecResp/(units.mV/units.fC),(cur[shift_all+671]+cur[shift_all+672]+cur[shift_all+673]+cur[shift_all+674]+cur[shift_all+675]+cur[shift_all+676]+cur[shift_all+677]+cur[shift_all+678]+cur[shift_all+679]+cur[shift_all+680]+cur[shift_all+681])*0.1*units.us/(11*units.fC)),color="black")
    for num,c in enumerate(cur):
        #if num<149 or num>150:
        if num<191-12-10 or num>191-13-5:#-6-12-12-12 or num>186-12-12-7:
            continue
        #if num!=683:
        #    continue
        #if num<698 or num>702:
        #    continue
        #if num<231 or num>241:
        #if num<242 or num>252:
        #if num<253 or num>263:
        #if num!=670:
        #    continue
        c_pa = c#/units.picoampere
        c_temp=c_temp+numpy.asarray(c)
        for cc in c_pa:
            integral=integral+cc*0.05*units.us
        #if num==18:
        print("IntegralChargefor ",num," is ",integral)
        integral=0
        plt.plot(x,c_pa)
        #print(num,max(convolve(elecResp/(units.mV/units.fC),c_pa*0.1*units.us/units.fC)))
        #plt.plot(x,convolve(elecResp/(units.mV/units.fC),c_pa*0.01*units.us/units.fC))
    #c_temp=c_temp/11.0
    #plt.plot(x,c_temp)
    #print(elecResp)
    #convRes=convolve(elecResp/(units.mV/units.fC),cur[61]*0.02*units.us/units.fC)
    #plt.plot(x,convRes,label="2.5mm hole")
    #plt.plot(x,convRes*1.03,'--',label="2.0mm hole")
    import math
    #print("TotResp=",sum(convRes))
    #ctot=(cur2[0]+cur2[1]+cur2[2]+cur2[3]+cur2[4]+cur2[5]+cur2[6]+cur2[7]+cur2[8])/9
    #convRes2=convolve(elecResp/(units.mV/units.fC),ctot*0.1*units.us/units.fC)
    #shift=0
    #for i in range(0,shift):
    #    convRes2=numpy.insert(convRes2,0,0)
    #plt.plot(x,convRes2[:]*0.97,'--',label="2.0mm hole")
    #plt.plot(x,cur[19])#elecResp/(units.mV/units.fC))
    #plt.plot(x,(cur[22])*0.1*units.us/units.fC) # e/0.1us
    #plt.plot(x,cur[2]/320000)
    plt.xlim([141-6,141])
    plt.title("Pixel",fontsize=14)
    plt.legend(fontsize=14)
    #plt.xlabel('Time, us')
    #plt.ylabel('Response, mV',fontsize=14)
    plt.ylabel('Indused Current, A.U.',fontsize=14)
    plt.xlabel('DriftTime, us',fontsize=14)
    #plt.legend(loc="upper left")

    #plt.ylabel('Current on the Strip, picoAmp')
    #plt.ylabel('Current on the Strip, picoAmp')
    #plt.pcolormesh(cur[0][0:3,-3:-1,201], shading='auto')
    #plt.colorbar()
    plt.show()
    #plt.savefig(output)
