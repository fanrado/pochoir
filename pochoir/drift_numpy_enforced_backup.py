#!/usr/bin/env python3
'''
Solve initial value problem to get drift paths using pytorch
'''
import math
import numpy
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator as RGI
from pochoir import units
from pochoir import lar

# Drift-path ending classification (P4 insul-bc, EPIC pochoir-ktj0):
DRIFT_NONE = 0      # still drifting / uncollected at the end of the time window
DRIFT_PAD = 1       # parked on a conductor pad -> normal charge collection
DRIFT_SURFACE = 2   # terminated at the FR4 insulator surface -> SURFACE CHARGE
                    # (terminal, but NOT counted as a pad collection)

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



class PotentialField:
    '''
    ODE callable that computes drift velocity from the *scalar* drift
    potential rather than from a precomputed velocity/E-field vector.

    The scalar potential is interpolated once (a genuine scalar function),
    and the electric field at an arbitrary position is obtained by
    differentiating the interpolant: E(x,y,z) = grad(phi_interp).  Because
    E is the gradient of a scalar, it is curl-free by construction and thus
    satisfies the static Maxwell equation, unlike a component-wise
    interpolation of the vector E-field.  The drift velocity is then
    v = mu(|E|,T) * E, using the same arithmetic as the `velo` command.
    '''

    def __init__(self, domain, potential, temperature,
                 method='linear', verbose=False, insulator=None):
        '''
        domain      : pochoir Domain (shape/spacing/origin).
        potential   : scalar potential array on the domain grid.
        temperature : LAr temperature in system-of-units.
        method      : RGI interpolation order ('linear' or 'cubic').
        insulator   : optional bool mask (domain shape) marking excluded FR4
                      cells; drift velocity is zeroed inside them so a charge
                      that reaches the insulator sticks as surface charge (also
                      a safety net against near-surface numerical residual).
        '''
        self.bb = domain.bb
        self.spacing = numpy.array(domain.spacing, dtype=float)
        self.temp = temperature
        self.verbose = verbose
        self.calls = 0

        # geometry for mapping a physical position to a grid cell (insulator)
        self.insulator = None
        if insulator is not None:
            self.insulator = numpy.asarray(insulator).astype(bool)
            self._origin = numpy.array(domain.origin, dtype=float)
            self._ishape = numpy.array(domain.shape, dtype=int)

        # Use the exact grid coordinate axes (shape-length) so the axes
        # match the potential array shape exactly.
        points = domain.linspaces

        potential = numpy.asarray(potential)
        # fill_value=None + bounds_error=False -> extrapolate/clamp instead
        # of injecting a spurious 0 that would create a huge false gradient
        # at the (periodic) transverse edges.
        self.interp = RGI(points, potential, method=method,
                          bounds_error=False, fill_value=None)

    def inside(self, point):
        for i, p in enumerate(point):
            if p < self.bb[0][i] or p > self.bb[1][i]:
                return False
        return True

    def potential_at(self, pos):
        return float(self.interp([pos])[0])

    def _cell(self, pos):
        '''Nearest grid cell index for a physical position, clamped in-range.'''
        idx = numpy.round((numpy.asarray(pos, float) - self._origin)
                          / self.spacing).astype(int)
        idx = numpy.clip(idx, 0, self._ishape - 1)
        return tuple(idx)

    def in_insulator(self, pos):
        '''True when ``pos`` falls inside an excluded FR4 (insulator) cell.'''
        if self.insulator is None:
            return False
        return bool(self.insulator[self._cell(pos)])

    def efield(self, pos):
        '''
        E = grad(phi_interp) via central finite differences, using the same
        (+grad phi) sign convention and units.V scaling as the `velo` command.
        Sample points are clamped inside the bounding box; a one-sided
        difference is used when a neighbour would fall outside.
        '''
        pos = numpy.asarray(pos, dtype=float)
        lo = numpy.array(self.bb[0], dtype=float)
        hi = numpy.array(self.bb[1], dtype=float)
        efield = numpy.zeros_like(pos)
        for dim in range(len(pos)):
            h = 0.5 * self.spacing[dim]
            pp = pos.copy()
            pm = pos.copy()
            # clamp the +/- sample points inside the domain
            phigh = min(pos[dim] + h, hi[dim])
            plow = max(pos[dim] - h, lo[dim])
            pp[dim] = phigh
            pm[dim] = plow
            denom = phigh - plow
            if denom <= 0.0:
                efield[dim] = 0.0
                continue
            efield[dim] = (self.potential_at(pp) - self.potential_at(pm)) / denom
        return efield * units.V

    def __call__(self, time, pos):
        '''
        Return the drift velocity vector at location (time independent).
        '''
        self.calls += 1
        if not self.inside(pos):
            return numpy.zeros_like(numpy.asarray(pos, dtype=float))
        if self.in_insulator(pos):
            # Inside the reflecting FR4: zero drift velocity so the charge
            # sticks (surface charge) instead of diving through the insulator.
            return numpy.zeros_like(numpy.asarray(pos, dtype=float))
        efield = self.efield(pos)
        emag = math.sqrt(sum([e*e for e in efield]))
        mu = lar.mobility(emag, self.temp)
        return numpy.array([e*mu/units.mm**2 for e in efield])


def solve_potential(domain, start, potential, temperature, times,
                    method='linear', verbose=False, insulator=None):
    '''
    Return ``(path, endtag)`` for one charge drifting from ``start`` through
    the velocity field derived on-the-fly from the interpolated scalar
    potential.  ``path`` is the (len(times), ndims) array of positions;
    ``endtag`` is one of ``DRIFT_NONE`` / ``DRIFT_PAD`` / ``DRIFT_SURFACE``.

    With ``insulator=None`` the behaviour (and the returned ``path``) is
    identical to before -- the full time window is integrated with no
    termination -- and ``endtag`` is ``DRIFT_NONE``.

    With an insulator mask the drift terminates when the charge reaches the
    FR4 top surface: the path is padded with that terminal point for the
    remaining ticks and tagged ``DRIFT_SURFACE`` (surface charge, NOT a pad
    collection).  A charge that instead parks on a conductor (drift speed
    collapses) is tagged ``DRIFT_PAD``; one still moving at the end is
    ``DRIFT_NONE``.
    '''
    start = numpy.array(start, dtype=float)
    potential = numpy.asarray(potential)
    times = numpy.array(times)

    print(f'start @{start}')
    func = PotentialField(domain, potential, temperature,
                          method=method, verbose=verbose, insulator=insulator)

    if insulator is None:
        res = solve_ivp(func, [times[0], times[-1]], start, t_eval=times,
                        rtol=0.0000000001, atol=0.0000000001,
                        method='Radau',
                        )
        print("Last Point=", res['y'].T[-1]/units.mm)
        return res['y'].T, DRIFT_NONE

    # --- insulating-surface termination + ending classification ---
    # FR4 top face (facing the drift gap).  The highest insulator z-index is the
    # FR4 cell whose CENTRE sits at that node; its top face is half a cell above
    # the node centre, i.e. (max index + 0.5)*spacing -- NOT (index + 1), which
    # would place the plane a full cell too high (at the pad node centre) and
    # leave a half-cell gap the descending electron never crosses.  This 9.95mm
    # face coincides with the pad bottom and with the in_insulator rounding
    # boundary (positions below it round to the FR4 node).  Charges drift DOWN
    # (decreasing z) toward the pad plane; in the inter-pad gaps they reach this
    # surface and must stop there.
    ins = func.insulator
    max_fr4_index = int(numpy.max(numpy.where(ins.any(axis=(0, 1)))[0]))
    z_surface = float(domain.origin[2] + (max_fr4_index + 0.5) * domain.spacing[2])

    def hit_surface(t, y):
        return y[2] - z_surface
    hit_surface.terminal = True
    hit_surface.direction = -1   # fire only when descending through the surface

    res = solve_ivp(func, [times[0], times[-1]], start, t_eval=times,
                    rtol=0.0000000001, atol=0.0000000001,
                    method='Radau', events=hit_surface,
                    )

    ys = res['y'].T
    nt = len(times)
    ndim = start.shape[0]
    path = numpy.empty((nt, ndim))
    n = min(len(ys), nt)
    path[:n] = ys[:n]

    surfaced = (res.status == 1) and len(res.t_events) and len(res.t_events[0])
    if surfaced:
        term = numpy.asarray(res.y_events[0][-1], dtype=float)  # point on surface
        endtag = DRIFT_SURFACE
    else:
        term = ys[-1] if len(ys) else start
        # PAD if the drift speed collapsed near a conductor, else still drifting
        sp_end = float(numpy.sqrt((func(0.0, term) ** 2).sum()))
        sp0 = float(numpy.sqrt((func(0.0, start) ** 2).sum()))
        endtag = DRIFT_PAD if (sp0 > 0.0 and sp_end < 1e-2 * sp0) else DRIFT_NONE
    if n < nt:
        path[n:] = term
    print("Last Point=", path[-1] / units.mm, "endtag=", endtag)
    return path, endtag


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
