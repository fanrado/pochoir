#!/usr/bin/env python3
'''
Solve initial value problem to get drift paths using pytorch
'''
import math
import numpy
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator as RGI
from pochoir import units

class Simple:
    '''
    Simple ODE calable
    '''

    def __init__(self, domain, vfield, verbose=False):
        '''
        The vfield give vector feild on domain.
        '''
        shape = domain.shape
        spacing = domain.spacing
        origin = domain.origin
        points = list()
        self.bb = domain.bb
        self.verbose = verbose
        self.calls = 0

        for dim in range(len(domain.shape)):
            start = origin[dim]
            stop  = origin[dim] + shape[dim] * spacing[dim]
            stop_f = float("{:.2f}".format(stop))
            spacing_f = float("{:.2f}".format(spacing[dim]))
            #rang = numpy.arange(start, stop, spacing[dim])
            rang = numpy.arange(start, stop_f, spacing_f)
            #rang = numpy.arange(start, stop, spacing[dim])
            #print ("interp dim:", dim, rang.shape, vfield[dim].shape)
            points.append(rang)

        self.interp = [
            RGI(points, component, fill_value=0.0)
            for component in vfield]

    def inside(self, point):
        for i,p in enumerate(point):
            if p < self.bb[0][i] or p > self.bb[1][i]:
                return False
        return True

    def interpolate(self, pos):
        velo = numpy.zeros_like(pos)        
        for ind, inter in enumerate(self.interp):
            try:
                got = inter([pos])
            except ValueError as err:
                print(f'Interpolation failed at:\n\tv_{ind}(r=@{pos/units.mm} mm)')
                print(f'\tdomain: {self.bb}')
                raise

            velo[ind] = got[0]
        return velo

    def extrapolate(self, pos):
        return numpy.zeros_like(pos)

    def __call__(self, time, pos):
        '''
        Return velocity vector at location (time independent).
        '''
        self.calls += 1
        speed_unit = units.mm/units.us
        if self.inside(pos):
            velo = self.interpolate(pos)
            what = "interp"
        else:
            velo = self.extrapolate(pos)
            what = "extrap"

        vmag = math.sqrt(sum([v*v for v in velo]))
        #if self.verbose:
            #print(f'{what}:{self.calls:4d}: t={time/units.us:.3f} us, r={pos/units.mm} mm v={velo} vmag={vmag} mm/us')
            #print(f'{what}:{self.calls:4d}: t={time/units.us:.3f} us, r={pos/units.mm} mm v={velo/speed_unit} vmag={vmag/speed_unit:.3f} mm/us')
        #print(pos,velo)
        return velo



def solve(domain, start, velocity, times, verbose=False):
    '''
    Return the path of points at times from start through velocity field.
    '''
    start = numpy.array(start)
    speed_unit = units.mm/units.us
    velocity = [numpy.array(v) for v in velocity]
    #print(velocity[2][:,:,95:100])
    times = numpy.array(times)
   # skip = (slice(None,None,2),slice(None,None,2),slice(None,None,50))
    
    #for i in range(150,250,1):
     #   for j in range(0,17,2):
      #      for k in range(0,25,2):
       #         print(k,j,i,"=>",velocity[0][k][j][i],velocity[1][k][j][i],velocity[2][k][j][i])
    
    print(f'start @{start}')#, times={times/units.us}')
    func = Simple(domain, velocity, True)
    #res = odeint(func, start, times, rtol=0.01, atol=0.01)
    res = solve_ivp(func, [times[0], times[-1]], start, t_eval=times,
                    rtol=0.0000000001, atol=0.0000000001,
                    method='Radau', #Radau
                    #first_step=0.001,
                    #max_step=0.001
                    )
    print("Last Point=",res['y'].T[-1]/units.mm)
    #print(len(res['y'].T))
    #print(f"function called {func.calls} times")
    return res['y'].T



class ScalarField:
    '''
    Scalar field interpolator (for dl or dt) on the same domain grid.
    '''

    def __init__(self, domain, field_array):
        shape = domain.shape
        spacing = domain.spacing
        origin = domain.origin
        self.bb = domain.bb

        points = []
        for dim in range(len(shape)):
            start = origin[dim]
            stop  = origin[dim] + shape[dim] * spacing[dim]
            stop_f = float("{:.2f}".format(stop))
            spacing_f = float("{:.2f}".format(spacing[dim]))
            points.append(numpy.arange(start, stop_f, spacing_f))

        self.interp = RGI(points, field_array, fill_value=0.0)

    def inside(self, point):
        for i, p in enumerate(point):
            if p < self.bb[0][i] or p > self.bb[1][i]:
                return False
        return True

    def __call__(self, pos):
        if self.inside(pos):
            return float(self.interp([pos])[0])
        return 0.0


def solve_sde(domain, start, velocity, dl, dt, times, verbose=False, rng=None):
    # Convert units (keep your original scaling)
    dl = 1000 * dl / units.m / units.us     # longitudinal diffusion (m^2/us)
    dt = 1000 * dt / units.m / units.us     # transverse diffusion (m^2/us)

    # Diagnostics (optional)
    # print("Velocity z-axis =", velocity[2][25,15,3000], " (units m/us)")
    # print("DL =", dl[25,15,3000], " (units m^2/us)")
    # print("DT =", dt[25,15,3000], " (units m^2/us)")

    start = numpy.array(start, dtype=float)
    velocity = [numpy.array(v, dtype=float) for v in velocity]
    times = numpy.array(times, dtype=float)
    dt_array = numpy.diff(times)              # time step per iteration (us)

    # Interpolators
    vel_interp = Simple(domain, velocity, verbose=verbose)
    dl_interp  = ScalarField(domain, dl)
    dt_interp  = ScalarField(domain, dt)

    # RNG
    if rng is None:
        rng = numpy.random.default_rng()

    # Integrate path
    pos = start.copy()
    path = [pos.copy()]

    for i, dt_time in enumerate(dt_array):
        t = times[i]

        # Drift
        v_drift = vel_interp(t, pos)          # length/time
        delta_pos = v_drift * dt_time         # deterministic step (length)

        # Diffusion coefficients at current location (ensure nonnegative)
        d_long = max(float(dl_interp(pos)), 0.0)  # length^2/time
        d_tran = max(float(dt_interp(pos)), 0.0)  # length^2/time

        # Build anisotropic diffusion relative to local drift direction
        vnorm = numpy.linalg.norm(v_drift)
        if vnorm > 0.0:
            u = v_drift / vnorm                # unit vector
        else:
            # If drift is zero, anisotropy axis is undefined; pick a fixed axis
            u = numpy.array([0.0, 0.0, 1.0])

        if (d_long > 0.0) or (d_tran > 0.0):
            # Draw a scalar for the parallel direction
            z_par = rng.normal(0.0, 1.0)

            # Draw a 3D standard normal and project to the plane ⟂ u
            w = rng.normal(0.0, 1.0, size=3)
            z_perp = w - (w @ u) * u          # Cov[z_perp] = I - u u^T

            # Compose anisotropic Gaussian increment with Cov = 2*dt_time*[d_long uu^T + d_tran (I - uu^T)]
            noise = (math.sqrt(2.0 * dt_time * d_long) * z_par) * u \
                    +  math.sqrt(2.0 * dt_time * d_tran) * z_perp

            delta_pos = delta_pos + noise

        pos = pos + delta_pos
        path.append(pos.copy())
    #for n,p in enumerate(path):
    #    print(n,p,dt_interp(p))
    return numpy.stack(path)
