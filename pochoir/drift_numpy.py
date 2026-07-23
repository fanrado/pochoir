#!/usr/bin/env python3
'''
Solve initial value problem to get drift paths using pytorch

NOTE (all drift-side enforcement removed, pochoir-w3x9): earlier revisions
imposed artificial conditions on the potential-based drift -- a terminal event
that force-stopped a path at the FR4 top face, a v=0 zeroing inside the
insulator, and a mask-aware (ghost/mirror) correction of the E-field at the FR4
surface.  All are deleted.  The drift now simply integrates v = mu*grad(phi) on
the SOLVED potential; E = grad(phi) is taken plainly, with no insulator mask or
field correction here.  The ONLY insulator condition is the no-flux (Neumann)
FR4 boundary that the Laplace SOLVER imposed when producing the potential --
whatever the field does near the frozen FR4 cells is reported honestly.
'''
import math
import numpy
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator as RGI
from pochoir import units
from pochoir import lar

# Drift-path ending classification (P4 insul-bc, EPIC pochoir-ktj0):
DRIFT_NONE = 0      # still drifting / uncollected at the end of the time window

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

        # --- periodic transverse wrap (per,per,fix drift tile), pochoir-9rjv ---
        # The drift tile is periodic in x,y with period shape*spacing (e.g. 4.4mm)
        # and the pixel pad is centered ON the seam (x=4.4 == x=0).  Append the
        # 0-slice as a wrap node on the transverse axes so the interpolator is
        # valid across the seam; positions are wrapped modulo the period in
        # __call__.  Without this a strong transverse (focusing) field pushes
        # near-seam electrons past the domain edge, where RGI fill_value=0 freezes
        # them ~1mm above the pad instead of letting them wrap onto the pad center.
        self.period = numpy.array([shape[d]*spacing[d]
                                   for d in range(len(shape))], dtype=float)
        self.periodic = (0, 1)  # transverse axes of the per,per,fix drift tile
        vfield = list(vfield)
        for d in self.periodic:
            points[d] = numpy.append(points[d], points[d][0] + self.period[d])
            vfield = [numpy.concatenate([c, numpy.take(c, [0], axis=d)], axis=d)
                      for c in vfield]

        self.interp = [
            RGI(points, component, fill_value=0.0)
            for component in vfield]

    def inside(self, point):
        for i,p in enumerate(point):
            if i in self.periodic:   # wrapped into [0,period) in __call__; always in range
                continue
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
        # wrap transverse coords onto the periodic tile before field lookup so an
        # electron crossing the seam (pad center) re-enters and reaches the pad
        # instead of freezing at the domain edge (pochoir-9rjv).
        pos = numpy.array(pos, dtype=float)
        for d in self.periodic:
            pos[d] = pos[d] % self.period[d]
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
                 method='linear', verbose=False):
        '''
        domain      : pochoir Domain (shape/spacing/origin).
        potential   : scalar potential array on the domain grid.
        temperature : LAr temperature in system-of-units.
        method      : RGI interpolation order ('linear' or 'cubic').

        The drift E-field is the plain gradient of the interpolated potential,
        E = grad(phi_interp).  No insulator mask, ghost/mirror fill, or other
        manipulation is applied here: the ONLY condition on the field is the
        no-flux (Neumann) FR4 boundary that the Laplace SOLVER imposed when it
        produced ``potential``.  Whatever the field does near the frozen FR4
        cells is reported honestly, not corrected.
        '''
        self.bb = domain.bb
        self.spacing = numpy.array(domain.spacing, dtype=float)
        self.temp = temperature
        self.verbose = verbose
        self.calls = 0

        # Use the exact grid coordinate axes (shape-length) so the axes
        # match the potential array shape exactly.
        points = list(domain.linspaces)

        potential = numpy.asarray(potential)
        # --- periodic transverse wrap (per,per,fix drift tile), pochoir-9rjv ---
        # The drift tile is periodic in x,y with period shape*spacing (e.g. 4.4mm)
        # and the pixel pad is centered ON the seam (x=4.4 == x=0).  Append the
        # 0-slice as a wrap node on the transverse axes so phi (and its gradient)
        # are correct ACROSS the seam; positions and finite-difference samples are
        # wrapped modulo the period below.  Without this, an electron pushed past
        # the transverse edge by a strong (focusing) field freezes ~1mm above the
        # pad instead of wrapping onto the pad center.
        self.period = numpy.array([domain.shape[d]*domain.spacing[d]
                                   for d in range(len(domain.shape))], dtype=float)
        self.periodic = (0, 1)  # transverse axes of the per,per,fix drift tile
        for d in self.periodic:
            points[d] = numpy.append(points[d], points[d][0] + self.period[d])
            potential = numpy.concatenate(
                [potential, numpy.take(potential, [0], axis=d)], axis=d)
        # fill_value=None + bounds_error=False -> extrapolate/clamp instead
        # of injecting a spurious 0 that would create a huge false gradient
        # at the (periodic) transverse edges.
        self.interp = RGI(points, potential, method=method,
                          bounds_error=False, fill_value=None)

    def inside(self, point):
        for i, p in enumerate(point):
            if i in self.periodic:   # wrapped into [0,period) in __call__; always in range
                continue
            if p < self.bb[0][i] or p > self.bb[1][i]:
                return False
        return True

    def potential_at(self, pos):
        return float(self.interp([pos])[0])

    def efield(self, pos):
        '''
        E = grad(phi_interp) via central finite differences, using the same
        (+grad phi) sign convention and units.V scaling as the `velo` command.
        Sample points are clamped inside the bounding box; a one-sided
        difference is used when a neighbour would fall outside.  This is the
        plain gradient of the solved potential -- no insulator/mask correction.
        '''
        pos = numpy.asarray(pos, dtype=float)
        lo = numpy.array(self.bb[0], dtype=float)
        hi = numpy.array(self.bb[1], dtype=float)
        efield = numpy.zeros_like(pos)
        for dim in range(len(pos)):
            h = 0.5 * self.spacing[dim]
            pp = pos.copy()
            pm = pos.copy()
            if dim in self.periodic:
                # periodic central difference across the seam (pochoir-9rjv):
                # wrap the +/- samples so the gradient at the pad center (seam)
                # uses the true neighbour, not a clamped one-sided value.
                pp[dim] = (pos[dim] + h) % self.period[dim]
                pm[dim] = (pos[dim] - h) % self.period[dim]
                denom = 2.0 * h
            else:
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
        # wrap transverse coords onto the periodic tile before field lookup so an
        # electron crossing the seam (pad center) re-enters and reaches the pad
        # instead of freezing at the domain edge (pochoir-9rjv).
        pos = numpy.asarray(pos, dtype=float).copy()
        for d in self.periodic:
            pos[d] = pos[d] % self.period[d]
        if not self.inside(pos):
            # Outside the solved domain there is simply no field data; this is a
            # data-availability limit, NOT an imposed physics condition.
            return numpy.zeros_like(numpy.asarray(pos, dtype=float))
        efield = self.efield(pos)
        emag = math.sqrt(sum([e*e for e in efield]))
        mu = lar.mobility(emag, self.temp)
        return numpy.array([e*mu/units.mm**2 for e in efield])


def solve_potential(domain, start, potential, temperature, times,
                    method='linear', verbose=False):
    '''
    Return ``(path, endtag)`` for one charge drifting from ``start`` through
    the velocity field derived on-the-fly from the interpolated scalar
    potential.  ``path`` is the (len(times), ndims) array of positions.

    The drift is a plain integration of the velocity field v = mu*E, with
    E = grad(phi) taken from the potential that was solved WITH the no-flux
    FR4 Neumann boundary condition.  NO path-termination event, NO in-insulator
    velocity zeroing, and NO mask-aware field correction are imposed: whatever
    the charge does near the FR4 surface (decelerate, slide tangentially, pile
    up) must emerge from the SOLVED field itself -- we simulate the physics, we
    do not dictate where the charge is allowed to go.  The solver's Neumann FR4
    boundary is the only condition on the field.  If a trajectory fails to
    converge near the surface that is a NUMERICAL problem for the integrator
    settings (step size / method / tolerances), not a reason to force a stop.

    ``endtag`` is always ``DRIFT_NONE`` (kept for call-site compatibility).
    '''
    start = numpy.array(start, dtype=float)
    potential = numpy.asarray(potential)
    times = numpy.array(times)

    print(f'start @{start}')
    func = PotentialField(domain, potential, temperature,
                          method=method, verbose=verbose)

    res = solve_ivp(func, [times[0], times[-1]], start, t_eval=times,
                    rtol=0.0000000001, atol=0.0000000001,
                    method='Radau',
                    )
    print("Last Point=", res['y'].T[-1]/units.mm)
    return res['y'].T, DRIFT_NONE


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
