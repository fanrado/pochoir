#!/usr/bin/env python
'''
Low-level functions for arrays.  
'''

from . import units

# Ideally, this is only module to import these two:
import numpy

# import limited numpy api.
# fixme: need to rethink this....
ones = numpy.ones
zeros = numpy.zeros
linspace = numpy.linspace
meshgrid = numpy.meshgrid

def is_numpy(arr):
    return isinstance(arr, numpy.ndarray)
def is_torch(arr):
    try:
        import torch
    except ImportError:
        return False
    return isinstance(arr, torch.Tensor)
def is_cupy(arr):
    try:
        import cupy
    except ImportError:
        return False
    return isinstance(arr, cupy._core.core.ndarray)

def module(array):
    '''
    Return either numpy or torch module matching type of array
    '''
    if is_torch(array):
        import torch
        return torch
    if is_numpy(array):
        return numpy
    if is_cupy(array):
        import cupy
        return cupy


def fromstr1(string, dtype=float):
    '''
    Parse string as 1d list of numbers, return array

    Numbers may have unit names multiplied
    '''
    s = [dtype(eval(s.strip(), units.__dict__)) for s in string.split(",") if s.strip()]
    return to_numpy(s)


def to_device(arr, device):
    '''
    Return a new array on the given device.
    '''
    if device == 'numpy':
        return to_numpy(arr)
    arr = to_torch(arr, device)
    #print (arr.device)
    return arr

def to_numpy(array):
    '''
    Return array or a new numpy array if not already one.
    '''
    if is_numpy(array):
        return numpy.array(array)
    if is_torch(array):
        return array.to('cpu').numpy()
    if is_cupy(array):
        return array.get()
    return numpy.array(array)   # hail mary

def to_torch(array, device='cpu'):
    '''
    Return array or a new torch tensor if not already one.
    '''
    import torch
    return torch.tensor(array, device=device)
    

def to_like(array, like):
    '''
    Return data in array in the form like like.
    '''
    if is_torch(like):
        return to_torch(array, device=like.device)
    return numpy.array(array)


def gradient(array, *spacing):
    '''
    Return the finite difference gradient of the array.
    '''
    # Compatibility shim for numpy >= 1.23: numpy.gradient no longer accepts a
    # single 1-D array as "one scalar per axis"; passing one length-N array is
    # interpreted as coordinates for axis 0 and must match that axis length.
    # Callers (e.g. test_dipole, test_velo) pass dom.spacing as a single arg,
    # so unpack a length-N sequence into N positional scalars before forwarding.
    if len(spacing) == 1 and hasattr(spacing[0], '__len__'):
        spacing = tuple(spacing[0])
    if isinstance(array, numpy.ndarray):
        return numpy.array(numpy.gradient(array, *spacing))

    # Amazingly, PyTorch has no equivalent.  An alternative solution
    # is to reimplment numpy.gradient() in terms of tensor slicing and
    # arithmetic operations.  At the cost of possible GPU->CPU->GPU
    # transit, for now we do the dirty:
    a = array.to('cpu').numpy()
    # Same numpy>=1.23 caveat as above: spread spacings as positional scalars.
    gvec = numpy.gradient(a, *spacing)
    g = numpy.array(gvec)
    return to_torch(g, device=array.device)
    
def vmag(vfield):
    '''
    Return magnitude of vector field as scalar field.

    The vfield is an N-list of N-d arrays, each giving one dimension's component.
    '''
    c2s = [c*c for c in vfield]
    tot = numpy.zeros_like(c2s[0])
    for c2 in c2s:
        tot += c2
    return numpy.sqrt(tot)


def dup(array):
    '''
    Return a copy of the array
    '''
    if is_torch(array):
        import torch
        return torch.clone(array, requires_grad=False)
    return numpy.copy(array)


def core_slices1(array):
    '''
    Return slices with core shape of and array made one cell in each
    direction.
    '''
    return tuple([slice(1,s-1) for s in array.shape])


def core1(array):
    '''
    Return core part of array, removing a 1 element pad
    '''
    return array[core_slices1(array)]


def pad1(array):
    '''
    Return a new array with every dimension increased by 1 on either
    edge and central value holding array.
    '''
    mod = module(array)
    shape = [s+2 for s in array.shape]
    padded = mod.zeros(shape, dtype=array.dtype)
    padded[core_slices1(padded)] = array
    return padded

def rgi(points, values):
    '''
    Return a "regular grid interpolator".

    Points are an N-tuple of arrays, each holding the grid points on
    one axis.

    Values are the values on the en-meshgrid-ment of the grid points.

    The array type of values determines the interpolation engine.
    '''
    if is_torch(values):
        from torch_interpolations import RegularGridInterpolator as RGI
    else:
        from scipy.interpolate import RegularGridInterpolator as RGI
    return RGI(points, values, method="linear")

def invert(arr):
    if is_torch(arr):
        return arr.logical_not()
    return numpy.invert(arr)

def pr2array(pr, nimperwire = 6, nbinsperwire = 10):
    '''
    Convert a schema.PlaneResponse to a numpy array
    '''
    nwires = len(pr.paths) // nimperwire
    midwire = nwires//2

    nticks = pr.paths[0].current.size
    nimps = int(nwires*nbinsperwire)
    res = numpy.zeros((nimps, nticks))
    pitches = numpy.zeros(nimps)

    for iwire in range(nwires):
        ibin0 = iwire * nimperwire
        for ind in range(nimperwire-1):
            a = pr.paths[ibin0 + ind+0].current
            b = pr.paths[ibin0 + ind+1].current
            m = 0.5 * (a+b)

            p1 = pr.paths[ibin0 + ind+0].pitchpos
            p2 = pr.paths[ibin0 + ind+1].pitchpos
            pm = 0.5*(p1+p2)
            
            obin = iwire * nbinsperwire + ind;

            res[obin] = m
            pitches[obin] = pm

    res = res + numpy.flipud(res)
    pitches = pitches - numpy.flip(pitches)
        
    # for path in pr.paths:
    #     print ("%.3f mm"%(path.pitchpos/units.mm))
    return res,pitches

def fr2arrays(fr, gain=0, shaping=0):
    '''
    Return a dict of Numpy arrays.  IF gain and shaping are nonzero,
    convolve with corresponding electronics response.
    '''
    nplanes = len(fr.planes)
    planeid = numpy.zeros(nplanes)

    #print (type(fr.tstart), fr.tstart, type(fr.period), fr.period, fr.speed)
    dat = dict(origin=fr.origin,
               tstart=fr.tstart,
               period=fr.period,
               speed=fr.speed,
               locations=numpy.zeros(nplanes),
               pitches=numpy.zeros(nplanes))


    responses = list();
    for iplane, pr in enumerate(fr.planes):
        r,p = pr2array(pr)
        last_r = r

        #dat['resp%d' % pr.planeid] = r
        responses.append(r)
        dat['bincenters%d' % pr.planeid] = p
        dat['locations'][iplane] = pr.location
        dat['pitches'][iplane] = pr.pitch
    

    if gain != 0.0 and shaping != 0.0:
        from . import electronics
        
        dat["gain"] = gain;
        dat["shaping"] = shaping;

        ncols = 0
        eresp = None
        espec = None
        smeared = list()
        for r in responses:
            if ncols != r.shape[1]:
                ncols = r.shape[1]
                times = [units.ns*(fr.tstart + fr.period * ind) for ind in range(ncols)]
                eresp = electronics(times, gain, shaping)
                espec = numpy.fft.fft(eresp)

            nrows = r.shape[0]
            #print ("shaping %d x %s" % (nrows, ncols))
            for irow in range(nrows):
                rspec = numpy.fft.fft(r[irow])
                r[irow] = numpy.real(numpy.fft.ifft(rspec*espec))
        dat['eresp'] = eresp
        dat['espec'] = espec
    for ind, pr in enumerate(fr.planes):
        dat['resp%d' % pr.planeid] = responses[ind]
        
    return dat;
