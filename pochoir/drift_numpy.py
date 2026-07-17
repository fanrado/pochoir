#!/usr/bin/env python3
'''
Solve initial value problem to get drift paths using pytorch

NOTE (enforcement removed): earlier revisions imposed two artificial conditions
on the potential-based drift -- a terminal event that force-stopped a path at
the FR4 top face, and a v=0 zeroing inside the insulator so charges "stuck" as
surface charge.  Those dictated the outcome instead of letting it follow from
the field.  They have been deleted: the drift now simply integrates the
velocity field derived from the (Neumann-BC) potential.  The pre-removal version
is preserved verbatim in drift_numpy_enforced_backup.py.
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

        Mask-aware ghost/mirror at the FR4 surface (pochoir-9keo): the FDM
        solve implements the no-flux (Neumann) insulator BC by dropping stencil
        bonds to the frozen FR4 cells (stencil_poisson_neumann), which is a
        mirror reflection (dphi/dn = 0) across the surface.  A NAIVE central
        difference here instead reaches into those frozen cells (phi = 0),
        fabricating a spurious normal-field spike just above the surface and
        reading (0,0,0) inside the solid -- so gap electrons get shoved through
        the FR4 and halt with no field.  To stay consistent with the solve, a
        +/- sample that lands inside an insulator cell is mirrored back across
        the surface (its value replaced by the reflected active-side sample),
        which drives the NORMAL component to zero at the FR4 face while leaving
        the TANGENTIAL components (whose samples stay in the LAr, off the solid)
        intact.  Gap electrons then slide toward the pad instead of being
        pushed into the insulator -- no clamp, just the correct field.
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
            phi_plus = self.potential_at(pp)
            phi_minus = self.potential_at(pm)
            # ghost/mirror reflection at the insulator surface.  The half-cell
            # sample points straddle a cell face, so test the ADJACENT CELL in
            # each direction (the same neighbour the FDM stencil drops its bond
            # to) rather than rounding a point sitting exactly on the face.  If
            # exactly one neighbour cell is solid, reflect that sample about pos
            # (ghost value = active-side value) so the normal derivative
            # vanishes there; if both neighbours are solid there is no field.
            if self.insulator is not None:
                base = numpy.asarray(self._cell(pos), dtype=int)
                cp = base.copy(); cp[dim] = min(cp[dim] + 1, self._ishape[dim] - 1)
                cm = base.copy(); cm[dim] = max(cm[dim] - 1, 0)
                plus_solid = bool(self.insulator[tuple(cp)])
                minus_solid = bool(self.insulator[tuple(cm)])
                if plus_solid and not minus_solid:
                    phi_plus = phi_minus
                elif minus_solid and not plus_solid:
                    phi_minus = phi_plus
                elif plus_solid and minus_solid:
                    efield[dim] = 0.0
                    continue
            efield[dim] = (phi_plus - phi_minus) / denom
        return efield * units.V

    def __call__(self, time, pos):
        '''
        Return the drift velocity vector at location (time independent).
        '''
        self.calls += 1
        if not self.inside(pos):
            # Outside the solved domain there is simply no field data; this is a
            # data-availability limit, NOT an imposed physics condition.
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
    potential.  ``path`` is the (len(times), ndims) array of positions.

    The drift is a plain integration of the velocity field v = mu*E, with
    E = grad(phi) taken from the potential that was solved WITH the no-flux
    FR4 Neumann boundary condition.  NO path-termination event and NO
    in-insulator velocity zeroing are imposed: whatever the charge does near
    the FR4 surface (decelerate, slide tangentially, pile up) must emerge from
    the field itself -- we simulate the physics, we do not dictate where the
    charge is allowed to go.  If a trajectory fails to converge near the
    surface that is a NUMERICAL problem to be solved by the integrator
    settings (step size / method / tolerances), not by forcing the path to
    stop.

    ``endtag`` is always ``DRIFT_NONE`` (kept for call-site compatibility).
    The ``insulator`` argument is accepted for CLI compatibility but does NOT
    alter the drift.
    '''
    start = numpy.array(start, dtype=float)
    potential = numpy.asarray(potential)
    times = numpy.array(times)

    print(f'start @{start}')
    func = PotentialField(domain, potential, temperature,
                          method=method, verbose=verbose, insulator=insulator)

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
