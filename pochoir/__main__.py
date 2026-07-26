#!/usr/bin/env python3
'''
CLI to pochoir

pochoir [global options] <command> [command options] [arguments]

Most commands here can be thought of nodes in a DAG joined by entries
in the pochoir data store.

For consistency the following conventions are followed:

    - both input from the store and named output destined for the
      store are given as command options.

    - input has lower-case short options, output has upper.

    - options may be used for non-store related names

    - arguments are for non-store related information

Every array output to the store should have metadata which describes
what inputs were used to create it.  Commands which need an ancestor
array do not be explicitly require it named on the command line but
instead it is resovled via metadta based on a more immediate input
array.

There is a data type taxonomy and its names are used in a manner
constient between CLI option names and store keys.  These are:

    - domain :: a grid of points in space

    - initial :: a scalar field holding an Initial Value Array

    - boundary :: a scalar field holding a Boundary Value Array

    - potential :: a scalar field such as solved by FDM

    - increment :: a scalar field holding the difference between two
      potentials from subsequent FDM steps aka the "error" in a
      potential solution.

    - velocity :: a vector velocity field

    - gradient :: a vector gradient field

    - points :: points in space

    - ...

Additional metadata may be stored such as:

    - taxon :: name the taxonomy type of the array

    - command :: name the command that produced the array
'''

import sys, os
import json
import click
import pochoir
import torch
from . import units
# no others than click and pochoir!
import logging
# Directory for auxiliary artifacts (PNGs, legacy .npy dumps) written by
# commands that side-save outside the ctx.obj store.  Respect POCHOIR_STORE so
# a debug/alternate run (e.g. run-full-3d-pixel-debug.sh) does not clobber the
# default ./store.  The ctx.obj store itself is resolved separately via the
# --store/POCHOIR_STORE click option.
STORE_DIR = os.environ.get('POCHOIR_STORE', 'store')
if not os.path.exists(STORE_DIR):
    os.makedirs(STORE_DIR)
log_filename = os.environ.get('POCHOIR_LOG', os.path.join(STORE_DIR, 'pochoir.log'))
logging.basicConfig(
    level=logging.INFO,
    filename=log_filename,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filemode='a'
)
info_msg = logging.info
err_msg = logging.error
debug_msg = logging.debug

@click.group()
@click.option("-s","--store",type=click.Path(),
              envvar="POCHOIR_STORE",
              help="File for primary data storage (input and maybe output)")
@click.option("-o","--outstore",type=click.Path(),
              help="File for output (primary only input)")
@click.pass_context
def cli(ctx, store, outstore):
    '''
    pochoir command line interface
    '''
    if not store:
        store = "."
    ctx.obj = pochoir.main.Main(store, outstore)

@cli.command()
def version():
    '''
    Print the version
    '''
    click.echo(pochoir.__version__)


@cli.command()
@click.option("-m", "--multi", default=None,
              type=click.Path(file_okay=False, dir_okay=True),
              help="Specify a directory to receive multi-file output")
@click.option("-o", "--output", default="/dev/stdout",
              type=click.Path(file_okay=True, dir_okay=False),
              help="Specify output file, default to stdout")
@click.argument("filename")
@click.pass_context
def gencfg(ctx, multi, output, filename):
    '''
    Generate JSON configuration files from master file.
    '''
    import json
    import pochoir.gencfg as gc
    if multi:
        gc.multi(filename, multi, output)
        return
    data = gc.loadf(filename)
    open(output,'wb').write(json.dumps(data).encode())

@cli.command()
@click.option("-s","--shape", type=str, required=True,
              help="The number of grid points in each dimension")
@click.option("-o","--origin", default=None, type=str,
              help="The spatial location of zero index grid point (def=0's)")
@click.option("-S","--spacing", default=None, type=str,
              help="The grid spacing as scalar or vector (def=1's)")
@click.option("-D", "--domain", type=str,
              help="Generated domain name") 
@click.pass_context
def domain(ctx, shape, origin, spacing, domain):
    '''
    Produce a "domain" and store it to the named dataset.

    A domain describes a finite, uniform grid in N-D space in these
    terms:

        - shape :: an N-D integer vector giving the number of grid
          points in each dimension.  Required.

        - origin :: an N-D spatial vector identifying the location of
          the grid point with all indices zero.  These may use spatial
          units.

        - spacing :: a scalar or N-D vector in same distance units as
          used in origin and which gives a common or a per-dimension
          spacing between neighboring grid points.  This may use
          spatial units.

    A vector is given as a comma-separated list of numbers.

    Spatial units are applied by multiplying a unit symbol such as

        - 10*mm

        - 2.4*cm

    If no spatial unit is given, mm is assumed.

    Note: this description corresponds to vtk/paraview uniform
    rectilinear grid, aka an "image".
    '''
    # info_msg("generating domain with shape={}, origin={}, spacing={}".format(shape, origin, spacing))
    shape = pochoir.arrays.fromstr1(shape, int)
    ndim = shape.size

    if spacing:
        val = pochoir.arrays.fromstr1(spacing)
        if "," in spacing:
            spacing = val
        else:
            spacing = pochoir.arrays.zeros(ndim) + val[0]
    else:
        spacing = pochoir.arrays.ones(ndim)

    if origin:
        origin = pochoir.arrays.fromstr1(origin)
    else:
        origin = pochoir.arrays.zeros(ndim)

    dom = pochoir.domain.Domain(shape, spacing, origin)

    ctx.obj.put_domain(domain, dom)


@cli.command()
@click.option("-d", "--domain", type=str, 
              help="Use named dataset for the domain, (def: indices)")
@click.option("-I","--initial", type=str,
              help="Initial value array to generate")
@click.option("-B","--boundary", type=str,
              help="Boundary array array to generate")
@click.option("-g","--generator", type=str, default=None,
              help="The generator method")
@click.argument("configs", nargs=-1)
@click.pass_context
def gen(ctx, domain, generator, initial, boundary, configs):
    '''
    Generate initial and boundary value arrays from a high-level
    generator.
    '''
    if generator is None:
        info_msg("available geometry generators:")
        for one in pochoir.gen.__dict__:
            if one[0] == "_":
                continue
            info_msg('\t'+one)
        return

    cfg = dict()
    for config in configs:
        cfg.update(json.loads(open(config,'rb').read().decode()))
    cfg = pochoir.util.unitify(cfg)
    meth = getattr(pochoir.gen, generator)

    dom = ctx.obj.get_domain(domain)
    # info_msg("domain={}".format(dom))
    # info_msg("cfg={}".format(cfg))
    
    # Generators return (iarr, barr, epsilon); those that build a no-flux
    # insulator mask append it as an optional 4th element (see EPIC
    # pochoir-ktj0).  Unpack tolerantly so 3-tuple generators are unaffected.
    result = meth(dom, cfg)#, info_msg) # iarr is initial array, barr is boundary array
    iarr, barr, epsilon = result[0], result[1], result[2]
    insulator = result[3] if len(result) > 3 else None

    # info_msg("initial array shape={}, boundary array shape={}".format(iarr.shape, barr.shape))
    # info_msg("initial array dtype={}, boundary array dtype={}".format(iarr.dtype, barr.dtype))

    params = dict(domain=domain, generator=generator,
                  command="gen", config=','.join(configs))
    ctx.obj.put(initial, iarr, taxon="initial", **params)
    ctx.obj.put(boundary, barr, taxon="boundary", **params)
    if epsilon is not None:
        ctx.obj.put(initial+"_epsilon", epsilon, taxon="permittivity", **params)
    if insulator is not None:
        ctx.obj.put(initial+"_insulator", insulator, taxon="insulator", **params)

    
@cli.command()
@click.option("-i","--initial", type=str,
              help="Name initial value array")
@click.option("-b","--boundary", type=str,
              help="Name the boundary array")
@click.option("-a","--ambient", type=float, default=0.0,
              help="Ambient potential")
@click.option("-d","--domain", default=None, type=str,
              help="Use domain for the plot")
@click.argument("filenames", nargs=-1)
@click.pass_context
def init(ctx, initial, boundary, ambient, domain, filenames):
    '''
    Initialize a problem with a shape file.

    This produces named initial and boundary value arrays.

    Filename arguments give JSON files which are progressively
    loadeded to update a configuration of shapes and their potentils.

    The full data structure is:

        {
            shapes: [ordered list of shapes],
            values: {shape name to value map},
        }

    Each element of the shapes array holds attributes:

        {
            name: "unique name of shape",
            type: "shape type name",
            ....: parameters depending on shape type
        }

    2D shapes and their args are:

    - rectangle :: "point1" and "point2" giving opposite corners as 2-element lists
    - circle :: "center" as 2 element list and "radius" 

    3D shapes and their args are:

    - box :: "point1" and "point2" giving opposite corners as 3-element lists
    - cylinder :: "center" and "radius" and "hheight" (half height) and axis of symmetry

    All spatial distances are given in the same unit as the domain spacing.
    The domain sets the allowed dimensionality.

    See also the "gen" command for a high-level way to init.
    '''
    dom = ctx.obj.get_domain(domain)

    cfg = dict()
    for fname in filenames:
        cfg.update(json.loads(open(fname,'rb').read().decode()))

    iarr, barr = pochoir.geom.init(dom, cfg, ambient)

    fnames = ",".join(filenames)
    ctx.obj.put(initial, iarr, result="initial",
                geom=fnames, domain=domain)
    ctx.obj.put(boundary, barr, result="boundary",
                geom=fnames, domain=domain)

@cli.command()
@click.option("-i","--initial", type=str,
              help="Input initial value array")
@click.option("-b","--boundary", type=str,
              help="Input the boundary array")
@click.option('--epsilon', type=str, default=None,
              help="Input the permittivity array for Poisson equation")
@click.option('--insulator', type=str, default=None,
              help="Input the no-flux (Neumann) insulator mask array (NO epsilon)")
@click.option("-e","--edges", type=str,
              help="Comma separated list of 'fixed' or 'periodic' giving domain edge conditions")
@click.option("--precision", type=float, default=0.0,
              help="Finish when no changes larger than precision")
@click.option("--epoch", type=int, default=1000,
              help="Number of iterations before any check")
@click.option("-n", "--nepochs", type=int, default=1,
              help="Limit number of epochs (def: one epoch)")
@click.option("--engine",
              type=click.Choice(["numpy", "numba", "torch", "cupy", "cumba"]),
              default="numpy",
              help="The FDM engine to use")
@click.option("-P", "--potential", type=str,
              help="Output array holding solution for potential")
@click.option("-I", "--increment", type=str,
              help="Output array holding increment (error) on the solution")
@click.option("-M", "--multisteps", type=str, default="N",
              help="Whether to use multistep method (Y/N)")
@click.pass_context
def fdm(ctx, initial, boundary,
        edges, precision, epoch, nepochs, engine,
        potential, increment, multisteps, epsilon, insulator):
    '''
    Apply finite-difference method.

    Solve Laplace equation given initial/boundary value arrays to
    produce a scalar potential array.
    '''
    import pochoir.fdm
    try:
        solve = getattr(pochoir.fdm, f'solve_{engine}')
        info_msg(f"using FDM engine {engine}")
    except AttributeError as err:
        click.echo(f'no fdm solver engine {engine}')
        info_msg(f'no fdm solver engine {engine}')
        click.echo(err)
        sys.exit(-1)

    iarr, imd = ctx.obj.get(initial, True)
    barr, bmd = ctx.obj.get(boundary, True)
    eps = None
    if epsilon is not None:
        eps, bmd = ctx.obj.get(epsilon, True) if epsilon else (None, None)
    ins = None
    if insulator is not None:
        ins, _ = ctx.obj.get(insulator, True)
    if not "domain" in bmd:
        click.echo(f'failed to get domain for {boundary}')
        info_msg(f'failed to get domain for {boundary}')
        click.echo(bmd)
        sys.exit(-1)
    # info_msg(f'got initial and boundary arrays with shapes {iarr.shape} and {barr.shape}')
    # info_msg(f'got initial and boundary arrays with dtypes {iarr.dtype} and {barr.dtype}')
    # info_msg(f'got domain: {bmd["domain"]}')

    domain = bmd['domain']

    bool_edges = [e.startswith("per") for e in edges.split(",")]
    if len(bool_edges) != iarr.ndim:
        raise ValueError("the number of periodic condition do not match problem dimensions")

    params = dict(operation="fdm", domain=domain,
                  initial=initial, boundary=boundary,
                  edges=edges, epoch=epoch, nepochs=nepochs,
                  precision=precision, command="fdm")
    if multisteps.lower() in ["y", "yes"]:
        # first step is to solve \nabla^2 \phi_0 = 0 with given boundary conditions, using float32. 
        phi_0, err_phi0 = solve(iarr, barr, bool_edges,
                        precision, epoch, nepochs, info_msg=info_msg, ctx=ctx, potential=potential, increment=increment, params=params, phi0=None, _dtype=torch.float32) # , ctx=ctx, potential=potential, increment=increment : arguments to save checkpoints during the solve
        potential_float32 = potential+"_float32"
        increment_float32 = increment+"_float32"
        ctx.obj.put(potential_float32, phi_0, taxon="potential", **params)
        ctx.obj.put(increment_float32, err_phi0, taxon="increment", **params)

        # second step is to solve \nabla^2 \delta = -\nabla^2 \phi_0 with given boundary conditions, using float64.
        ## cast phi_0 to float64 for the second step, and use it as source term in the poisson equation.

        phi_0 = phi_0.to(torch.float64) # remove this first to check the laplacian of phi_0 in float32. It should give me zero
        # precision = 3e-5
        delta_phi, err_delta_phi0 = solve(iarr*0, barr, bool_edges,
                        precision, epoch, nepochs, info_msg=info_msg, ctx=ctx, potential=potential, increment=increment, params=params, phi0=phi_0, _dtype=torch.float64)
        potential_float64 = potential+"_float64_delta" ## delta
        increment_float64 = increment+"_float64_delta" ## error on delta
        ctx.obj.put(potential_float64, delta_phi, taxon="potential", **params)
        ctx.obj.put(increment_float64, err_delta_phi0, taxon="increment", **params)
        ## cast delta_phi back to float64 and add it to phi_0 to get the final solution.
        delta_phi = delta_phi.to(torch.float64)
        arr = phi_0 + delta_phi
        err = torch.sqrt(err_phi0**2 + err_delta_phi0**2)

        ctx.obj.put(potential, arr, taxon="potential", **params)
        ctx.obj.put(increment, err, taxon="increment", **params)
    else:
        # Pass the insulator mask only when present so non-torch engines and
        # flag-off runs (which do not accept an `insulator` kwarg) are unchanged.
        extra = {}
        if ins is not None:
            extra['insulator'] = ins
        phi_0, err_phi0 = solve(iarr, barr, bool_edges,
                        precision, epoch, nepochs, info_msg=info_msg, ctx=ctx, potential=potential, increment=increment, params=params, phi0=None, _dtype=torch.float64, epsilon=eps, **extra) # , ctx=ctx, potential=potential, increment=increment : arguments to save checkpoints during the solve
        ctx.obj.put(potential, phi_0, taxon="potential", **params)
        ctx.obj.put(increment, err_phi0, taxon="increment", **params)



@cli.command()
@click.option("-t", "--temperature", type=str, default="89*K",
              help="LAr temperature")
@click.option("-p", "--potential", type=str,
              help="Input potential array")
@click.option("-b", "--boundary", type=str, default=None,
              help="Input boundary array (def: resolve via potential metadata)")
@click.option('--insulator', type=str, default=None,
              help="Accepted for compatibility; INERT (velocity is pure grad(phi))")
@click.option("-V", "--velocity", type=str,
              help="Output velocity array")
@click.option("-L", "--diff-longitudinal", "dl_key", type=str, default=None,
              help="Output key for longitudinal diffusion (dl)")
@click.option("-T", "--diff-transverse", "dt_key", type=str, default=None,
              help="Output key for transverse diffusion (dt)")
@click.pass_context
def velo(ctx, temperature, potential, boundary, insulator, velocity,dl_key,dt_key):
    '''
    Calculate a velocity field from a potential field
    '''
    temp = pochoir.arrays.fromstr1(temperature)[0]
    pot, md = ctx.obj.get(potential, True)
    domain = md['domain']
    dom = ctx.obj.get_domain(domain)
    import numpy
    pot = pot#*units.V
    debug_msg(f"TEST={pot[1,1,-1]}; Spacing={dom.spacing}")
    # Pure drift field: E = grad(phi) of the solved potential.  No electrode
    # zeroing, tile-edge zeroing, or in-insulator manipulation is applied here
    # -- the ONLY condition on the field is the no-flux Neumann FR4 boundary
    # that the Laplace SOLVER imposed when it produced the potential.  The
    # --boundary and --insulator options are accepted for CLI backward
    # compatibility but no longer alter the field (pochoir-w3x9).
    efield = pochoir.arrays.gradient(pot, *dom.spacing)
    efield = efield*units.V

    #temp=87.7
    debug_msg(f"temp={temp}")
    emag = pochoir.arrays.vmag(efield)
    mu = pochoir.lar.mobility(emag, temp)
    if dl_key is not None:
        dl = pochoir.lar.diff_longit(emag,temp)
    if dt_key is not None:
        dt = pochoir.lar.diff_tran(emag,temp)
    varr = [e*mu/units.mm**2 for e in efield]
    varr=numpy.array(varr)
    # NOTE (enforcement removed): no in-insulator velocity zeroing here either;
    # the saved velocity follows directly from the Neumann-BC field.

    params = dict(domain=domain, command="velo",
                  potential=potential, temperature=temp)
    # save velocity
    efield_path = '/'.join([velocity.split('/')[0], 'efield'])
    ctx.obj.put(efield_path, efield, **{**params, "taxon": "efield"})
    ctx.obj.put(velocity, varr, **{**params, "taxon": "velocity"})
    debug_msg(f"velocity shape={varr.shape}")
    # save dl/dt if keys provided
    if dl_key is not None:
        ctx.obj.put(dl_key, dl, **{**params, "taxon": "diffusion_longitudinal"})
    debug_msg(f"dl shape={dl_key}")
    if dt_key is not None:
        ctx.obj.put(dt_key, dt, **{**params, "taxon": "diffusion_transverse"})
    debug_msg(f"dt shape={dt_key}")

@cli.command()
@click.option("-s", "--scalar", type=str,
              help="Input scalar array")
@click.option("-G", "--gradient", type=str,
              help="Output gradient array")
@click.pass_context
def grad(ctx, scalar, gradient):
    '''
    Calculate the gradient of a scalar field.
    '''
    pot, md = ctx.obj.get(scalar, True)
    domain = md['domain']
    dom = ctx.obj.get_domain(domain)
    field = pochoir.arrays.gradient(pot, *dom.spacing)
    ctx.obj.put(gradient, field, taxon="gradient",
                domain=domain, scalar=scalar, command="grad")




def make_pixel_start_points(z_depth=148.0, ngridpoints=10, pitch=4.4, spacing=None):
    """
    Generate a regular ngridpoints x ngridpoints grid of drift starting points
    inside one pixel cell, cell-centred (first point at spacing/2).

    Parameters
    ----------
    z_depth : float
        Starting z position in grid-index units. z=148 corresponds to 14.8 mm
        from the anode (for a 0.1 mm/cell domain), just above the pixel
        collection plane. All starting points share this fixed depth.
        JSON config key: ``driftZDepth``.
    ngridpoints : int
        Number of grid points per side: 10 for 10x10, 8 for 8x8, 6 for 6x6.
        JSON config key: ``nGridPoints``.
    pitch : float
        Center-to-center pixel spacing in mm, equal to pixelSize + pixelGap.
        Derived automatically from existing JSON keys ``pixelSize`` and
        ``pixelGap`` — no dedicated config key required.
    spacing : float or None
        Grid pitch in mm. If None, computed as pitch / ngridpoints
        (e.g. 4.4/10 = 0.44 mm). The first point is placed at spacing/2
        so the grid is cell-centred, matching the convention in
        test-full-3d-pixel.sh: dist=(0.22 0.66 1.10 ... 4.18).
        JSON config key: ``gridSpacing`` (optional).

    Returns
    -------
    points : list of [x, y, z_depth]
        Exactly ngridpoints**2 points, no duplicates.

    JSON config keys summary
    ------------------------
    driftZDepth  : float  — maps to z_depth
    nGridPoints  : int    — maps to ngridpoints
    gridSpacing  : float  — maps to spacing (optional; derived if absent)
    pixelSize    : float  — existing key, used to derive pitch
    pixelGap     : float  — existing key, used to derive pitch
    """
    if spacing is None:
        spacing = pitch / ngridpoints  # e.g. 4.4/10 = 0.44 mm
    half = spacing / 2.0                    # cell-centred offset: 0.22 mm
    
    points = []
    for j in range(ngridpoints):
        x = half + j * spacing
        for i in range(ngridpoints):
            y = half + i * spacing
            points.append([x, y, z_depth])
    return points


@cli.command()
@click.option("-S","--starts", default=None, type=str,
              help="Output starts points array")
@click.option("-m","--mode", default="no", type=str,
              help="enable hardcodede array input")
@click.option("-c","--config", "configs", type=click.Path(exists=True), multiple=True,
              help="JSON config(s) holding driftZDepth, nGridPoints, gridSpacing (optional), "
                   "pixelSize, pixelGap.  Used when --mode yes to parameterise the pixel "
                   "start-point grid.  Same files passed to `gen` and `induce-pixel`.")
@click.option("--plot/--no-plot", default=False,
              help="If set, write a scatter PNG of the starting points to store/starting_points.png")
@click.argument("points", nargs=-1)
@click.pass_context
def starts(ctx, starts, mode, configs, plot, points):
    '''
    Store "starting" points.
    '''
    import numpy
    if mode=="yes":
        params = _load_start_point_config(configs)
        kwargs = {"z_depth": params["z_depth"], "ngridpoints": params["ngridpoints"]}
        if params["pitch"] is not None:
            kwargs["pitch"] = params["pitch"]
        if params["spacing"] is not None:
            kwargs["spacing"] = params["spacing"]
        points = make_pixel_start_points(**kwargs)
    else:
        npoints = len(points)
        if not npoints:
            raise ValueError("require at least one point")
        points = [pochoir.arrays.fromstr1(p) for p in points]
    
    debug_msg(f"POINTS (n={len(points)}): {points}")
    
    # import numpy
    arr = numpy.asarray(points)
    if plot:
        import os
        import matplotlib.pyplot as plt
        os.makedirs(STORE_DIR, exist_ok=True)
        plt.figure(figsize=(10,10))
        plt.scatter(arr[:,0],arr[:,1])
        plt.title('starting points')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(os.path.join(STORE_DIR, 'starting_points.png'))
        plt.close()
    debug_msg(f"whatever we save: {arr}")
    ctx.obj.put(starts, arr, taxon="points", command="starts")


@cli.command("drift")
@click.option("-P", "--paths", type=str,
              help="Output paths array")
@click.option("--starts", type=str,
              help="Input starting points")
@click.option("--velocity", type=str,
              help="Intput velocity array")
@click.option('--insulator', type=str, default=None,
              help="Accepted for compatibility; INERT (drift is pure grad(phi))")
@click.option("-L", "--diff-longitudinal", "dl_key", type=str, default=None,
              help="(Optional) Input longitudinal diffusion field")
@click.option("-T", "--diff-transverse", "dt_key", type=str, default=None,
              help="(Optional) Input transverse diffusion field")
@click.option("--verbose/--no-verbose", default=False,
              help="Verbose print during calculation")
@click.option("--engine", type=click.Choice(["numpy", "torch","numpyold"]),
              default="numpy",
              help="The IVP engine to use")
@click.option("--plot/--no-plot", default=False,
              help="If set, write a 3D plot of the drift paths to store/drift_paths_3d.png")
@click.option("--interp-order", type=click.Choice(["linear", "cubic"]),
              default="linear",
              help="Interpolation order for the scalar potential (potential-based drift)")
@click.argument("steps", nargs=-1)
@click.pass_context
def drift(ctx, paths, starts, velocity, insulator, dl_key, dt_key, verbose, engine, plot, interp_order, steps):
    '''
    Calculate drift paths.

    The (non-SDE, numpy) drift velocity is derived by interpolating the
    scalar drift potential and differentiating the interpolant, so that
    E = grad(phi) stays curl-free (satisfies the static Maxwell equation).
    The potential array and temperature are resolved from the velocity
    array's metadata.  Stores that lack this metadata fall back to the
    legacy component-wise velocity interpolation.
    '''
    debug_msg('START DRIFT')
    start_points = ctx.obj.get(starts)
    debug_msg(f"all start_points: {start_points}")
    if start_points is None:
        click.echo(f'no starts: {starts}')
        return -1

    steps = ','.join(steps)
    start, stop, step = pochoir.arrays.fromstr1(steps)
    nsteps = int((stop-start)/step)

    ticks = pochoir.arrays.linspace(start, stop, nsteps,
                                    endpoint=False)

    drifter = getattr(pochoir.drift, f'solve_{engine}')
    velo, md = ctx.obj.get(velocity, True)
    domain = md['domain']
    dom = ctx.obj.get_domain(domain)
    
    use_sde = (dl_key is not None) and (dt_key is not None)
    dl = dt = None
    if use_sde:
        dl, md_dl = ctx.obj.get(dl_key, True)
        dt, md_dt = ctx.obj.get(dt_key, True)

    # Prefer deriving the velocity from the interpolated scalar potential
    # (curl-free E = grad(phi)).  Resolve the potential array + temperature
    # from the velocity metadata written by the `velo` command.  Older
    # stores lacking this metadata fall back to velocity interpolation.
    from pochoir import drift_numpy
    use_potential = (engine == "numpy" and not use_sde
                     and 'potential' in md and 'temperature' in md)
    pot = temp = None
    if use_potential:
        pot = ctx.obj.get(md['potential'])
        temp = md['temperature']
        info_msg(f'drift: potential-based (key={md["potential"]}, '
                 f'T={temp}, interp={interp_order})')
    elif not use_sde:
        info_msg('drift: legacy velocity-interpolation '
                 '(no potential/temperature metadata found)')

    # --insulator is accepted for CLI backward compatibility but is INERT: the
    # drift is a pure integration of v = mu*grad(phi) on the solved potential,
    # whose only insulator condition is the solver's no-flux Neumann FR4 BC.
    # No mask-based path termination or field/velocity correction is applied
    # (pochoir-w3x9).
    if insulator is not None:
        info_msg('drift: --insulator is inert (drift follows the solved '
                 'Neumann-BC field; no mask-based termination or correction)')

    # shape: (nstarts, nticks, ndims); endtags: per-path ending classification
    thepaths = pochoir.arrays.zeros((len(start_points), len(ticks),
                                     len(dom.shape)))
    endtags = pochoir.arrays.zeros(len(start_points))
    for ind, point in enumerate(start_points):
        #print("input point: ",point)
        endtag = drift_numpy.DRIFT_NONE
        if use_sde:
            path = drift_numpy.solve_sde(dom, point, velo, dl, dt , ticks, verbose=verbose)
        elif use_potential:
            path, endtag = drift_numpy.solve_potential(dom, point, pot, temp, ticks,
                                               method=interp_order, verbose=verbose)
        else:
            path = drifter(dom, point, velo, ticks, verbose=verbose)
        thepaths[ind]=path
        endtags[ind]=endtag

    if plot:
        import os
        import matplotlib.pyplot as plt
        os.makedirs(STORE_DIR, exist_ok=True)
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection='3d')
        for i in range(0,thepaths.shape[0]):
            ax.plot(thepaths[i,:,0],thepaths[i,:,1],thepaths[i,:,2])
        plt.title('drift paths')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.savefig(os.path.join(STORE_DIR, 'drift_paths_3d.png'))
        plt.close()
    params=dict(taxon="paths", command="drift", domain=domain,
                tstart=start, tstop=stop, nsteps=nsteps)
    ctx.obj.put(paths, thepaths, **params)
    # Companion per-path ending tags (0=none/drifting, 1=pad collection,
    # 2=FR4 surface charge).  Stored under a separate key so consumers of the
    # positions array are unaffected; only written for the potential engine.
    if use_potential:
        ctx.obj.put(paths + "_endtag", endtags,
                    **{**params, "taxon": "paths_endtag"})


@cli.command("bc-interp")
@click.option("-x","--xcoord", type=str, default="17.5*mm",
              help="Name distance from the center along Xaxis to setup BC")

@click.option("-p", "--potential2d", type=str,
              help="The input 2D scalar potential array")
@click.option("-i", "--initial3d", type=str,
              help="The input 3D scalar initial values array")
@click.option("-b", "--boundary3d", type=str,
              help="The input 3D scalar boundary value array")

@click.option("-I","--initial", type=str,
              help="The output interpolated 3D initial values array")
@click.option("-B","--boundary", type=str,
              help="The output interpolated 3D boundary array")
@click.pass_context
def bc_interp(ctx, xcoord,                        # option
              potential2d, initial3d, boundary3d, # input
              initial, boundary                   # output
              ):
    '''
    Interpolate 2D solution into 3D boundary condition
    '''
    sol2D, md2d = ctx.obj.get(potential2d, True)
    barr3D, md3d = ctx.obj.get(boundary3d, True)
    arr3D = ctx.obj.get(initial3d)
    domain2d = md2d['domain']
    domain3d = md3d['domain']
    dom2D = ctx.obj.get_domain(domain2d)
    dom3D = ctx.obj.get_domain(domain3d)
    xcoord = pochoir.util.unitify(xcoord)

    from pochoir.bc_interp import interp

    arr, barr= interp(sol2D, arr3D, barr3D, dom2D, dom3D, xcoord)
    params = dict(command="bc-interp",
                  potential2d=potential2d, initial3d=initial3d, boundary3d=boundary3d,
                  domain2d=domain2d, domain3d=domain3d, domain=domain3d,
                  xcoord=xcoord)
    ctx.obj.put(initial, arr, taxon="initial", **params)
    ctx.obj.put(boundary, barr, taxon="boundary", **params)
    
@cli.command("extendwf")
@click.option("-p", "--potential2d", type=str,
              help="The input 2D scalar potential array")
@click.option("-P", "--potential3d", type=str,
              help="The input 3D scalar potential array")
@click.option("-n", "--nstrips", default=10.0,
              help="Max number of strips from the central for the extension")
@click.option("-o","--output", type=str,
              help="Output enlarged weighting field")
@click.pass_context
def extendwf(ctx,
              potential2d, potential3d,nstrips,
              output
              ):
    '''
    extend 3D weigting filed using 2D weighting field to full 2D volume
    '''
    sol2D, md2d = ctx.obj.get(potential2d, True)
    sol3D, md3d = ctx.obj.get(potential3d, True)
    domain2d = md2d['domain']
    domain3d = md3d['domain']
    dom2D = ctx.obj.get_domain(domain2d)
    dom3D = ctx.obj.get_domain(domain3d)
    import numpy
    #at the moment assume that 2d and 3d simulations have same properties with some magic numbers (including shifts in 2D solution)
    #NEEDS FIX for better calculation
    cut_z=1600 #this is the number we cut 3Dweight sim along drift
    horizontal = "yes"
    if horizontal=="yes":
        onestrip = dom3D.shape[0]/7.0
        newXdim = int((nstrips*2+1)*onestrip)
        arr = numpy.zeros((newXdim,dom3D.shape[1],dom2D.shape[1]))
        for i in range(0,newXdim):
            if i<onestrip*7:
                for j in range(0,dom3D.shape[1]):
                    arr[i,j,:] = sol2D[i,:]
            if i>=onestrip*7 and i<onestrip*14:
                for j in range(0,dom3D.shape[1]):
                    arr[i,j,:cut_z] = sol3D[i-dom3D.shape[0],j,:]
                    arr[i,j,cut_z:] = sol2D[i,cut_z:]
            if i>=onestrip*14:
                for j in range(0,dom3D.shape[1]):
                    arr[i,j,:] = sol2D[i,:]
    #arr = numpy.zeros((dom3D.shape[0],dom3D.shape[1],dom2D.shape[1]))
    #arr[:,:,0:int(dom3D.shape[2])]=sol3D

    #for i in range(0,dom3D.shape[1]):
    #    arr[:,i,int(dom3D.shape[2]):int(dom2D.shape[1])]=sol2D[742:1442,int(dom3D.shape[2]):int(dom2D.shape[1])]
    #for i in range(0,dom3D.shape[1]):
        #arr[:,i,int(dom3D.shape[2]-2):int(dom3D.shape[2])]=(sol3D[:,i,int(dom3D.shape[2]-2):int(dom3D.shape[2])]+arr[:,i,int(dom3D.shape[2]-2):int(dom3D.shape[2])])/2
   #     arr[:,i,int(dom3D.shape[2])]=(sol3D[:,i,int(dom3D.shape[2]-1)]+arr[:,i,int(dom3D.shape[2])])/2
   #     arr[:,i,int(dom3D.shape[2]+1)]=(sol3D[:,i,int(dom3D.shape[2]-1)]+arr[:,i,int(dom3D.shape[2]+1)])/2
   #     arr[:,i,int(dom3D.shape[2]+2)]=(sol3D[:,i,int(dom3D.shape[2]-1)]+arr[:,i,int(dom3D.shape[2]+2)])/2
   #     arr[:,i,int(dom3D.shape[2]+3)]=(sol3D[:,i,int(dom3D.shape[2]-1)]+arr[:,i,int(dom3D.shape[2]+3)])/2
   #     arr[:,i,int(dom3D.shape[2]+4)]=(sol3D[:,i,int(dom3D.shape[2]-1)]+arr[:,i,int(dom3D.shape[2]+4)])/2
    print("final domain:",arr.shape)
    dom = pochoir.domain.Domain(arr.shape, 0.05, [0.0,0.0,0.0])
    domain = "domain/weight3dextend"
    ctx.obj.put_domain(domain, dom)
    params = dict(command="extendwf",domain=domain,
                  potential2d=potential2d,potential3d=potential3d,nstrips=nstrips, output=output )
    ctx.obj.put(output, arr, taxon="output", **params)


@cli.command()
@click.option("-i", "--input", type=str, required=True,
              help="The input paths array")
@click.option("-t", "--translation", type=str, required=True,
              help="A spacial vector along which to move the paths")
@click.option("-O", "--output", type=str, required=True,
              help="The output array name")
@click.pass_context
def move_paths(ctx, input, translation, output):
    '''
    Move paths along offset vector.
    '''
    arr, arrmd = ctx.obj.get(input, True)
    try:
        atype = arrmd['taxon']
    except KeyError:
        click.echo(f'array "{input}" has no type')
        click.exit(-1)
    
    if atype != "paths":
        click.echo(f'array "{input}" is not of type "paths"')
        click.exit(-1)

    translation = pochoir.arrays.fromstr1(translation)

    from pochoir.arrays import to_like
    newarr = arr + to_like(translation, arr)
    ctx.obj.put(output, newarr, **arrmd)



@cli.command()
@click.option("-q","--charge", default=1.0,
              help="The amount of drifting charge")
@click.option("-w","--weighting", type=str,
              help="The input scalar weighting potential")
@click.option("-p","--paths", type=str,
              help="The input drift paths array")
@click.option("-a","--average", default=0.0,
              help="Average N paths along strip")
@click.option("-n","--nstrips", default=1.0,
              help="Calculate current for n strips from the central as well ")
@click.option("-O", "--output", type=str,
              help="Output array holding induced current waveforms")
@click.pass_context
def induce(ctx, charge, weighting, paths, average,nstrips, output):
    '''
    Calculate induced current.

    The current is that induced by the given charge moving along the
    paths and in the presence of a scalar weighting potential.
    
    Note: Paths are assumed to be provided in order of averaging blocks
    current for the strips culculated based on 50L detector mirror symmetry
    '''
    wpot, wmd = ctx.obj.get(weighting, True)
    try:
        domain = wmd['domain']
    except KeyError:
        click.echo(f'no domain for {weighting}.  metadata:\n{wmd}')
        return -1
    import numpy
    dom = ctx.obj.get_domain(domain)
    the_paths, pmd = ctx.obj.get(paths, True)
    npaths, nsteps, ndim = the_paths.shape
    ticks = pochoir.arrays.linspace(pmd['tstart'], pmd['tstop'],
                                    pmd['nsteps'], endpoint=False)
    rgi = pochoir.arrays.rgi(dom.linspaces, wpot)
    shift_x = dom.shape[0]*dom.spacing[0]/2.0
    shift_y = 0#dom.shape[1]*dom.spacing[1]/2.0
    shifted_paths = []
    if nstrips>1:
        dx = dom.shape[0]*dom.spacing[0]/nstrips
        print("dx=",dx)
        for i in range(0,int(nstrips)):
            for j in range(0,len(the_paths)):
                #newpath = [[45+(2.0*i+1.0)*dx/2.0-x[0],x[1],x[2]] for x in the_paths[j]]
                #2view
                #newpath = [[(2.0*i+1.0)*dx/2.0+x[0],x[1],x[2]] for x in the_paths[j]]
                #eview coll
                newpath = [[x[0]+i*1.0*dx,x[1]+1.45,x[2]] for x in the_paths[j]]
                xdata_old = [x[0] for x in the_paths[j]]
                #print(newpath)
                xdata_new = [x[0] for x in newpath]
                #flip if needed
                #newpaths_f = [[x[0]-dx/2.0,x[1],x[2]] for x in newpaths[j]]
                shifted_paths.append(newpath)
    if nstrips<=1:
        for i in range(0,len(the_paths)):
            newpath = [[shift_x+x[0],x[1]+shift_y,x[2]] for x in the_paths[i]]
            shifted_paths.append(newpath)
        shifted_paths=numpy.array(shifted_paths)
    print("TotalPaths=",len(shifted_paths))
    Q = charge * rgi(shifted_paths) #/ units.V
    print(wpot[325+714,14,340],wpot[325+714,15,340],wpot[326+714,14,340],wpot[325+714,15,340])
    assert len(Q.shape) == 2
    #assert Q.shape[0] == npaths
    assert Q.shape[1] == nsteps
    numpy.set_printoptions(threshold=sys.maxsize)
    for i in range(0,len(shifted_paths[644])):
        print(shifted_paths[699][i],shifted_paths[698][i],Q[699,i],Q[698,i])


    dQ = Q[:, 1:] - Q[:, :-1]
    dT = ticks[1:] - ticks[:-1]
    I = []
    I_tot = dQ/dT

    if average>0:
        print("Average ", average," paths along the stip")
        tot_paths = int(len(I_tot)/average)
        for i in range(0,len(I_tot),int(average)):
            I_temp = numpy.zeros(I_tot[0].shape)
            for p in range(0,int(average)):
                I_temp=I_temp+numpy.asarray(I_tot[i+p])
            I_temp = I_temp/average
            I.append(I_temp.tolist())
    if average<=0:
        print("No Averaging")
        I=I_tot
    ctx.obj.put(output, I, command="induce", taxon="current",
                charge = charge,
                domain=domain, paths=paths,average=average,nsteps=nsteps, weighting=weighting)

def _shift_paths_pixel_grid(the_paths, npaths=10, npixels=5, pixel_pitch=4.4, pixel_gap=0.6, pixel_size=3.8, pad_center=None):
    """Replicate drift paths across a 2D pixel grid by applying spatial offsets.

    Takes a set of drift paths defined relative to a single pixel and tiles them
    across a (npix+1) x (npix+1) grid of pixels, where npix = npixels // 2.
    Each path is translated so its origin aligns with the center of the
    corresponding pixel in the grid.

    Args:
        the_paths: List of paths, each a list of [x, y, z] coordinate triplets.
            Must contain at least npaths * npaths entries.
        npaths: Number of paths per pixel level.
        npixels: Total number of pixels along one axis; half of this (npix)
            determines the grid extent in each quadrant.
        pixel_pitch: Center-to-center distance between adjacent pixels (mm).
        pixel_gap: Gap between adjacent pixel edges (mm).
        pixel_size: Size of a single pixel (mm).
        pad_center: Optional (x, y) physical center of the collecting pad in the
            weighting domain.  When given, the tiled base pixel is aligned to it
            so on-metal endpoints sample the pinned W=1 cells exactly.  When None,
            the center is derived from the pitch formula (legacy behaviour), which
            can be off by up to one cell from where the generator rasterized the
            pad.

    Returns:
        List of shifted paths covering the full pixel grid, each path being a
        list of [x, y, z] coordinate triplets translated to its pixel position.
    """
    npix = int(npixels/2)
    nedge = npaths // 2  # scales with npaths; averaging by (npaths//10) gives same output as npaths=10
    if pad_center is None:
        center_pos_x = npix*pixel_pitch + pixel_gap/2 + pixel_size/2
        center_pos_y = npix*pixel_pitch + pixel_gap/2 + pixel_size/2
    else:
        center_pos_x, center_pos_y = pad_center
    new_shifted_paths = []
    for ix_pix in range(npix):
        for lvl in range(npaths):
            for iy_pix in range(npix):
                for i in range(npaths):
                    newpath = [[center_pos_x+x[0]+ix_pix*pixel_pitch, center_pos_y+x[1]+iy_pix*pixel_pitch, x[2]] for x in the_paths[i+lvl*npaths]]
                    new_shifted_paths.append(newpath)

            for i in range(nedge):
                newpath = [[center_pos_x+x[0]+ix_pix*pixel_pitch, center_pos_y+x[1]+npix*pixel_pitch, x[2]] for x in the_paths[i+lvl*npaths]]
                new_shifted_paths.append(newpath)

    for lvl in range(nedge):
        for iy_pix in range(npix):
            for i in range(npaths):
                newpath = [[center_pos_x+x[0]+npix*pixel_pitch, center_pos_y+x[1]+iy_pix*pixel_pitch, x[2]] for x in the_paths[i+lvl*npaths]]
                new_shifted_paths.append(newpath)

        for i in range(nedge):
            newpath = [[center_pos_x+x[0]+npix*pixel_pitch, center_pos_y+x[1]+npix*pixel_pitch, x[2]] for x in the_paths[i+lvl*npaths]]
            new_shifted_paths.append(newpath)
    return new_shifted_paths

_PIXEL_GEOMETRY_REQUIRED_KEYS = ("pixelSize", "pixelGap", "Npixels")


def _load_pixel_geometry(config_paths):
    """Load pixel geometry from one or more JSON configs.

    Returns a dict with float values for ``pixel_size``, ``pixel_gap``,
    ``pixel_pitch`` (derived as ``pixelSize + pixelGap``) and int ``npixels``.
    Raises ``KeyError`` naming the missing key if any required field is absent,
    and ``ValueError`` if no config path is supplied.
    """
    if not config_paths:
        raise ValueError(
            "induce-pixel requires --config pointing at the JSON used to "
            "generate the weighting potential (keys: "
            f"{', '.join(_PIXEL_GEOMETRY_REQUIRED_KEYS)})"
        )
    cfg = {}
    for path in config_paths:
        with open(path, "rb") as fh:
            cfg.update(json.loads(fh.read().decode()))
    missing = [k for k in _PIXEL_GEOMETRY_REQUIRED_KEYS if k not in cfg]
    if missing:
        raise KeyError(
            f"pixel geometry config missing required key(s) {missing} "
            f"(loaded from {list(config_paths)})"
        )
    pixel_size = float(cfg["pixelSize"])
    pixel_gap = float(cfg["pixelGap"])
    return {
        "pixel_size": pixel_size,
        "pixel_gap": pixel_gap,
        "pixel_pitch": pixel_size + pixel_gap,
        "npixels": int(cfg["Npixels"]),
    }


_START_POINT_DEFAULTS = {
    "driftZDepth": 28.0,
    "nGridPoints": 10,
}


def _load_start_point_config(config_paths):
    """Load make_pixel_start_points parameters from one or more JSON configs.

    Required JSON keys (with fallback defaults if no config is supplied):
      driftZDepth  (float) — z starting position in grid-index units
      nGridPoints  (int)   — grid points per side

    Optional JSON keys:
      gridSpacing  (float) — grid pitch in mm; derived from
                             (pixelSize + pixelGap) / nGridPoints if absent
      pixelSize    (float) — used to derive pitch (center-to-center spacing)
      pixelGap     (float) — used to derive pitch

    Returns a dict with keys: z_depth, ngridpoints, pitch, spacing.
    ``pitch`` is None when neither pixelSize/pixelGap nor gridSpacing are
    present in the config (make_pixel_start_points will then use its own
    default pitch of 4.4 mm).
    """
    cfg = {}
    for path in config_paths:
        with open(path, "rb") as fh:
            cfg.update(json.loads(fh.read().decode()))

    z_depth = float(cfg.get("driftZDepth", _START_POINT_DEFAULTS["driftZDepth"]))
    ngridpoints = int(cfg.get("nGridPoints", _START_POINT_DEFAULTS["nGridPoints"]))

    if "pixelSize" in cfg and "pixelGap" in cfg:
        pitch = float(cfg["pixelSize"]) + float(cfg["pixelGap"])
    else:
        pitch = None

    if "gridSpacing" in cfg:
        spacing = float(cfg["gridSpacing"])
    elif pitch is not None:
        spacing = pitch / ngridpoints
    else:
        spacing = None

    return {
        "z_depth": z_depth,
        "ngridpoints": ngridpoints,
        "pitch": pitch,
        "spacing": spacing,
    }


@cli.command()
@click.option("-q","--charge", default=1.0,
              help="The amount of drifting charge")
@click.option("-w","--weighting", type=str,
              help="The input scalar weighting potential")
@click.option("-p","--paths", type=str,
              help="The input drift paths array")
@click.option("-a","--average", default=0.0,
              help="Average N paths along strip")
@click.option("-n","--npixels", default=1.0,
              help="Calculate current for n pixels from the central as well ")
@click.option("-c","--config", "configs", type=click.Path(exists=True), multiple=True,
              help="JSON config(s) holding pixelSize, pixelGap, Npixels (same files passed to `gen`)")
@click.option("-O", "--output", type=str,
              help="Output array holding induced current waveforms")
@click.option("--plot/--no-plot", default=False,
              help="If set, write a charge waveform PNG to store/charge.png")
@click.pass_context
def induce_pixel(ctx, charge, weighting, paths, average, npixels, configs, output, plot):
    '''
    Calculate induced current.

    For a single pixel
    '''
    wpot, wmd = ctx.obj.get(weighting, True)
    try:
        domain = wmd['domain']
    except KeyError:
        click.echo(f'no domain for {weighting}.  metadata:\n{wmd}')
        return -1
    import numpy
    dom = ctx.obj.get_domain(domain)
    the_paths, pmd = ctx.obj.get(paths, True)
    npaths, nsteps, ndim = the_paths.shape
    ticks = pochoir.arrays.linspace(pmd['tstart'], pmd['tstop'],
                                    pmd['nsteps'], endpoint=False)
    rgi = pochoir.arrays.rgi(dom.linspaces, wpot)
    print(f'dom.linspaces : {dom.linspaces}')
    shift_x = dom.shape[0]*dom.spacing[0]/2.0
    shift_y = 0#dom.shape[1]*dom.spacing[1]/2.0
    shifted_paths = []
    print("input paths shape : ",the_paths.shape)
    # sys.exit()
    if npixels>1:
        geom = _load_pixel_geometry(configs)
        print(f'geom: {geom}')
        # Align the tiled collecting pixel to where the weighting generator
        # actually pinned the collecting pad (the only electrode held at W=1),
        # read from the solved weighting potential via the domain axes.  This
        # removes the sub-cell offset between the pitch formula and the pinned
        # pad that otherwise samples on-metal edge/corner endpoints just past
        # the W=1 edge (Ramo requires W=1 for any charge on the collecting pad).
        _on = numpy.isclose(wpot, 1.0).any(axis=2)
        _ix, _iy = numpy.where(_on)
        pad_center = None
        if _ix.size:
            _xs, _ys = dom.linspaces[0], dom.linspaces[1]
            pad_center = (0.5*(_xs[_ix.min()] + _xs[_ix.max()]),
                          0.5*(_ys[_iy.min()] + _ys[_iy.max()]))
            print(f'collecting-pad center (aligned to pinned W=1 mask): {pad_center}')
        shifted_paths = _shift_paths_pixel_grid(
            the_paths=the_paths, npaths=10, # change back to 10 after checking the many paths
            npixels=geom["npixels"],
            pixel_pitch=geom["pixel_pitch"],
            pixel_gap=geom["pixel_gap"],
            pixel_size=geom["pixel_size"],
            pad_center=pad_center,
        )
    # numpy.save('store/shifted_paths.npy', shifted_paths)
    # numpy.save('store/old_paths.npy', the_paths)
    # print(f'Shifted_paths[0,0] : {shifted_paths[0][0]}') 
    # print(f'old_paths[0,0] : {old_paths[0][0]}')
    # print("TotalPaths after shifting=",len(shifted_paths))
    # for i in range(0,len(shifted_paths)):
    #     print(f'shifted_paths[0,{i}] : {shifted_paths[i][0]} \t old_paths[0,{i}] : {old_paths[i][0]}') 
    # sys.exit()   
    if npixels<=1:
        for i in range(0,len(the_paths)):
            newpath = [[shift_x+x[0],x[1]+shift_y,x[2]] for x in the_paths[i]]
            shifted_paths.append(newpath)
        shifted_paths=numpy.array(shifted_paths)

    print("TotalPaths=",len(shifted_paths))
    import numpy as _np
    _sp = _np.array(shifted_paths)
    startpoints=[]
    endpoints=[]
    for num,p in enumerate(shifted_paths):
        startpoints.append(p[0])
        endpoints.append(p[-1])
    Q = charge * rgi(shifted_paths) #/ units.V
    print(f'Charge Q : {Q}')
    assert len(Q.shape) == 2
    #assert Q.shape[0] == npaths
    assert Q.shape[1] == nsteps
    numpy.set_printoptions(threshold=sys.maxsize)

    dQ = Q[:, 1:] - Q[:, :-1]
    if plot:
        import matplotlib.pyplot as plt
        os.makedirs(STORE_DIR, exist_ok=True)
        plt.figure(figsize=(10,6))
        plt.plot(ticks[1:], Q[0,1:], label="Charge")
        plt.savefig(os.path.join(STORE_DIR, 'charge.png'))
        plt.close()
    dT = ticks[1:] - ticks[:-1]
    # print(f'dT [:10] : {dT[:10]}r')
    # print(f'len(dT) : {len(dT)}')
    # sys.exit()
    I = []
    I_tot = dQ/dT
    print(f'Induced current I_tot : {I_tot}')
    if average>0:
        print("Average ", average," paths along the stip")
        tot_paths = int(len(I_tot)/average)
        for i in range(0,len(I_tot),int(average)):
            I_temp = numpy.zeros(I_tot[0].shape)
            for p in range(0,int(average)):
                I_temp=I_temp+numpy.asarray(I_tot[i+p])
            I_temp = I_temp/average
            I.append(I_temp.tolist())
    if average<=0:
        print("No Averaging")
        I=I_tot
        # print("I shape=",I.shape)
        # print("I=",I)
    import numpy as np
    #np.save('fr_4p4pitch_3.8pix_circgrid_1p9.npy', I)
    np.save(os.path.join(STORE_DIR, 'fr_4p4pitch_3.8pix_nogrid_10pathsperpixel.npy'), I)
    np.save(os.path.join(STORE_DIR, 'startpoints.npy'), startpoints)
    np.save(os.path.join(STORE_DIR, 'endpoints.npy'), endpoints)
    ctx.obj.put(output, I, command="induce", taxon="current",
                charge = charge,
                domain=domain, paths=paths,average=average,nsteps=nsteps, weighting=weighting)

        
        
@cli.command()
@click.option("-q","--charge", default=1.0,
              help="The amount of drifting charge")
@click.option("-w","--weighting", type=str,
              help="The input scalar weighting potential")
@click.option("-p","--paths", type=str,
              help="The input drift paths array")
@click.option("-a","--average", default=0.0,
              help="Average N paths along strip")
@click.option("-n","--nstrips", default=1.0,
              help="Calculate current for n strips from the central as well ")
@click.option("-O", "--output", type=str,
              help="Output array holding induced current waveforms")
@click.pass_context
def induce_30deg(ctx, charge, weighting, paths, average,nstrips, output):
    '''
    Calculate induced current for 30deg config.

    The current is that induced by the given charge moving along the
    paths and in the presence of a scalar weighting potential.
    
    Note: Paths are assumed to be provided in order of averaging blocks
    current for the strips culculated based on 50L detector mirror symmetry
    '''
    wpot, wmd = ctx.obj.get(weighting, True)
    try:
        domain = wmd['domain']
    except KeyError:
        click.echo(f'no domain for {weighting}.  metadata:\n{wmd}')
        return -1
    import numpy
    dom = ctx.obj.get_domain(domain)
    the_paths, pmd = ctx.obj.get(paths, True)
    npaths, nsteps, ndim = the_paths.shape
    ticks = pochoir.arrays.linspace(pmd['tstart'], pmd['tstop'],
                                    pmd['nsteps'], endpoint=False)
    rgi = pochoir.arrays.rgi(dom.linspaces, wpot)
    shift_x = dom.shape[0]*dom.spacing[0]/2.0
    shift_y = 0#dom.shape[1]*dom.spacing[1]/2.0
    shifted_paths = []
    if nstrips>1:
        dx = dom.shape[0]*dom.spacing[0]/nstrips
        for i in range(0,int(nstrips)):
            counter=0
            print("Process Strip: ",i)
            if(i%2==0):
                print("Direction 1")
                for j in range(0,int(len(the_paths)/2)):
                #newpath = [[45+(2.0*i+1.0)*dx/2.0-x[0],x[1],x[2]] for x in the_paths[j]]
                #newpath = [[(2.0*i+1.0)*dx/2.0+x[0],x[1],x[2]] for x in the_paths[j]] config for 5mm strip looks like I put it on right sided of the strip
                #for 30deg we sim paths as following first 4x11 paths are just shift and last 2x11 need a reversal as well
                    if counter<4*11:
                        # v1 newpath = [[i*1.0*dx+x[0],x[1],x[2]] for x in the_paths[j]]
                        newpath = [[i*1.0*dx+x[0],x[1],x[2]] for x in the_paths[j]]
                    else:
                        # v1 newpath = [[2*2.55-x[0]+i*1.0*dx,x[1],x[2]] for x in the_paths[j]]
                        newpath = [[2.55+x[0]+i*1.0*dx,x[1]+1.45,x[2]] for x in the_paths[j]]
                    counter=counter+1
                #flip if needed
                #newpaths_f = [[x[0]-dx/2.0,x[1],x[2]] for x in newpaths[j]]
                    shifted_paths.append(newpath)
            else:
                print("Direction 2")
                for j in range(int(len(the_paths)/2),len(the_paths)):
                    if counter<4*11:
                    # v1 newpath = [[2.55-x[0]+i*1.0*dx,x[1],x[2]] for x in the_paths[j]]
                        newpath = [[x[0]+i*1.0*dx,x[1]+1.45,x[2]] for x in the_paths[j]]
                    else:
                    # v1 newpath = [[2.55+x[0]+i*1.0*dx,x[1],x[2]] for x in the_paths[j]]
                        newpath = [[2.55+x[0]+i*1.0*dx,x[1],x[2]] for x in the_paths[j]]
                    counter=counter+1
                    shifted_paths.append(newpath)
    if nstrips<=1:
        for i in range(0,len(the_paths)):
            newpath = [[shift_x+x[0],x[1]+shift_y,x[2]] for x in the_paths[i]]
            shifted_paths.append(newpath)
        shifted_paths=numpy.array(shifted_paths)
    print("TotalPaths=",len(shifted_paths))
    Q = charge * rgi(shifted_paths) #/ units.V
    assert len(Q.shape) == 2
    #assert Q.shape[0] == npaths
    assert Q.shape[1] == nsteps

    dQ = Q[:, 1:] - Q[:, :-1]
    dT = ticks[1:] - ticks[:-1]
    I = []
    I_tot = dQ/dT

    if average>0:
        print("Average ", average," paths along the stip")
        tot_paths = int(len(I_tot)/average)
        for i in range(0,len(I_tot),int(average)):
            print("Averaging processed: ",i," out of",len(I_tot))
            I_temp = numpy.zeros(I_tot[0].shape)
            for p in range(0,int(average)):
                I_temp=I_temp+numpy.asarray(I_tot[i+p])
            I_temp = I_temp/average
            I.append(I_temp.tolist())
    if average<=0:
        print("No Averaging")
        I=I_tot
    ctx.obj.put(output, I, command="induce_30deg", taxon="current_deg",
                charge = charge,
                domain=domain, paths=paths,average=average,nsteps=nsteps, weighting=weighting)

@cli.command("convertfr")
@click.option("-u","--uinput", type=str,
              help="Input averaged current for indcution1")
@click.option("-v","--vinput", type=str,
              help="Input averaged current for induction2")
@click.option("-w","--winput", type=str,
              help="Input averaged current for collection")
@click.option("-O", "--output", type=str,
              help="Output file name")
@click.argument("configs", nargs=-1)
@click.pass_context
def convertfr(ctx, uinput,vinput,winput,output, configs):
    '''
    Convert Field Responce in json WireCell format
    
    Takes current as an input
    
    Note: Due to symmetry in 50L detector strip configuration current version requires following input:
     Current from 6 paths it transverse strip direction. Paths cover only half of the strip. Curent should be averaged along the strip prior to conversion. Paths should be chosen such with the same distance between them + first should start in the ~middle on the strip and last one as close to the middle of the pitch between strips as possible
    '''
    from . import schema
    from . import persist
    import numpy
    
    cfg = dict()
    for config in configs:
        cfg.update(json.loads(open(config,'rb').read().decode()))
    curr_I1, curr_I1md = ctx.obj.get(uinput, True)
    curr_I2, curr_I2md = ctx.obj.get(vinput, True)
    curr_C, curr_Cmd = ctx.obj.get(winput, True)
    anti_drift_axis = (1.0, 0.0, 0.0)
    origin = cfg["origin"]
    speed = cfg["speed"]
    tstart = cfg["tstart"]
    period = cfg["period"]
    pathR_I1 = []
    pathR_I2 = []
    pathR_C = []
    pitch_I1=cfg["planeUpitch"]
    pitch_I2=cfg["planeVpitch"]
    pitch_C=cfg["planeWpitch"]
    nstrips = cfg["totstrip"]
    npaths = cfg["npaths"]
    #float("{:.2f}".format(stop))
    pitchpos_I1 = float("{:.3f}".format(-1*pitch_I1*nstrips/2))
    pitchpos_I2 = float("{:.3f}".format(-1*pitch_I2*nstrips/2))
    pitchpos_C = float("{:.3f}".format(-1*pitch_C*nstrips/2))
    in_strip_shift_I1 = float("{:.3f}".format(cfg["planeUpathdist"]))
    in_strip_shift_I2 = float("{:.3f}".format(cfg["planeVpathdist"]))
    in_strip_shift_C = float("{:.3f}".format(cfg["planeWpathdist"]))
    between_strip_shift_I1 = float("{:.3f}".format(pitch_I1/2))
    between_strip_shift_I2 = float("{:.3f}".format(pitch_I2/2))
    between_strip_shift_C = float("{:.3f}".format(pitch_C/2))
    import matplotlib.pyplot as plt
    for i in range(0,npaths*nstrips,npaths):
        for j in range(0,npaths):
            pr_I1 = schema.PathResponse(-1*curr_I1[i+j][0:1325],pitchpos_I1,wirepos=0)
            pr_I2 = schema.PathResponse(-1*curr_I2[i+j][0:1325],pitchpos_I2,wirepos=0)
            pr_C = schema.PathResponse(-1*curr_C[i+j][0:1325],pitchpos_C,wirepos=0)
            if i+j>0:
                x = numpy.linspace(0,1325,1325)
                plt.plot(x,curr_I2[i+j][0:1325])
            pathR_I1.append(pr_I1)
            pathR_I2.append(pr_I2)
            pathR_C.append(pr_C)
            if j<npaths-1:
                pitchpos_I1 = float("{:.3f}".format(pitchpos_I1+in_strip_shift_I1))
                pitchpos_I2 = float("{:.3f}".format(pitchpos_I2+in_strip_shift_I2))
                pitchpos_C = float("{:.3f}".format(pitchpos_C+in_strip_shift_C))
        pitchpos_I1 = float("{:.3f}".format(pitchpos_I1+between_strip_shift_I1))
        pitchpos_I2 = float("{:.3f}".format(pitchpos_I2+between_strip_shift_I2))
        pitchpos_C = float("{:.3f}".format(pitchpos_C+between_strip_shift_C))
    plt.show()
    planes=[]
    planeid=0
    location=cfg["planeUlocation"]
    pitch=cfg["planeUpitch"]
    plr_I1 = schema.PlaneResponse(pathR_I1,planeid,location,pitch)
    planes.append(plr_I1)
    planeid=1
    location=cfg["planeVlocation"]
    pitch=cfg["planeVpitch"]
    plr_I2 = schema.PlaneResponse(pathR_I2,planeid,location,pitch)
    planes.append(plr_I2)
    planeid=2
    location=cfg["planeWlocation"]
    pitch=cfg["planeWpitch"]
    plr_C = schema.PlaneResponse(pathR_C,planeid,location,pitch)
    planes.append(plr_C)
    fr = schema.FieldResponse(planes,anti_drift_axis, origin, tstart, period, speed)
    persist.dumpfr(output,fr)
    
@cli.command()
@click.option("-w","--weighting", type=str,
              help="Input 3D weighting potential")
@click.option("-p","--paths", type=str,
              help="Input 3D drift paths")
@click.option("-v","--velocity", type=str,
              help="Input velocity array")
@click.option("-C", "--current", type=str,
              help="Output current array")
@click.pass_context
def srdot(ctx, weighting, paths, velocity, current):
    '''
    Apply Ramo theorem dot product.
    '''
    pot, potmd = ctx.obj.get(weighting, True)
    pot_domain = potmd['domain']
    dom_Ew = ctx.obj.get_domain(pot_domain)


    sol_Ew = pochoir.arrays.gradient(pot, dom_Ew.spacing)
    sol_Drift, pathmd = ctx.obj.get(paths, True)
    path_domain = pathmd['domain']
    dom_Drift = ctx.obj.get_domain(path_domain)

    velo = ctx.obj.get(velocity)
    res = pochoir.srdot.dotprod(dom_Ew, dom_Drift, sol_Ew, sol_Drift, velo)
    params = dict(operation="srdot", 
                  weight_domain=pot_domain, path_domain=path_domain,
                  weighting=weighting, paths=paths, velocity=velocity)
    ctx.obj.put(current, res, command="srdot", taxon="response", **params)



@cli.command("plot-image")
@click.option("-a", "--array", type=str, required=True,
              help="Input array to plot")
@click.option("-o", "--output",
              type=click.Path(exists=False, dir_okay=False),
              help="Output graphics file")
@click.option("-s", "--scale", default="linear",
              type=click.Choice(["linear","signedlog"]),
              help="Output graphics file")
@click.option("-u", "--units", type=str, default=None,
              help="The units in which to display magnitude")
@click.pass_context
def plot_image(ctx, array, output, scale, units):
    '''
    Visualize a dataset as 2D image
    '''
    arr, md = ctx.obj.get(array, True)
    domain = md.get("domain")
    if domain:
        dom = ctx.obj.get_domain(domain)
    if units is not None:
        u = pochoir.arrays.fromstr1(units)
        arr = arr/u
    title = f'{array}'
    if units:
        title += f' [{units}]'
    pochoir.plots.image(arr, output, dom, title, scale=scale)
    
    
    
@cli.command("plot-current")
@click.option("-c", "--current", type=str, required=True,
              help="Input current array to plot")
@click.option("-C", "--currentcomp", type=str, required=False,
              help="Input current array to plot")
@click.option("-o", "--output",
              type=click.Path(exists=False, dir_okay=False),
              help="Output graphics file")
@click.pass_context
def plot_current(ctx, current,currentcomp, output):
    '''
    Visualize a dataset as 2D image
    '''
    arr, md = ctx.obj.get(current, True)
    arr2, md2 = ctx.obj.get(currentcomp, True)
    pochoir.plots.current(arr,arr2, output)
    
    
@cli.command("plot-current-pixel")
@click.option("-c", "--current", type=str, required=True,
              help="Input current array to plot")
@click.option("-o", "--output",
              type=click.Path(exists=False, dir_okay=False),
              help="Output graphics file")
@click.pass_context
def plot_current_pixel(ctx, current, output):
    '''
    Visualize a dataset as 2D image
    '''
    arr, md = ctx.obj.get(current, True)
    pochoir.plots.currentpixel(arr, output)

@cli.command("plot-scatter3d")
@click.option("-a", "--array", type=str, required=True,
              help="Input array to plot")
@click.option("-o", "--output",
              type=click.Path(exists=False, dir_okay=False),
              help="Output graphics file")
@click.option("-g", "--gif",
              default="no", type=click.Choice(["yes","no"]),
              help="create gif image ")
@click.pass_context
def plot_scatter3d(ctx, array, output,gif):
    '''
    Visualize a dataset as 3D image pdf + gif
    '''
    arr, md = ctx.obj.get(array, True)
    domain = md.get("domain")
    if domain:
        dom = ctx.obj.get_domain(domain)
    else:
        dom = ctx.obj.get_domain(domain)
    title = f'{array}'
    pochoir.plots.scatt3d(arr, output, dom,gif,title)
    
    
@cli.command("plot-slice3d")
@click.option("-a", "--array", type=str, required=True,
              help="Input array to plot")
@click.option("-o", "--output",
              type=click.Path(exists=False, dir_okay=False),
              help="Output graphics file")
@click.option("-s", "--scale", default="linear",
              type=click.Choice(["linear","signedlog"]),
              help="Output graphics file")
@click.option("-d", "--dim", default="z",
              type=click.Choice(["x","y","z"]),
              help="choose axis to slice")
@click.option("-m", "--magnitude", default="no", type=click.Choice(["yes","no"]),
              help="calc magnitude")
@click.option("-i","--index",type=int, default=1.0,
              help="choose index to slice")
@click.option("-u", "--units", type=str, default=None,
              help="The units in which to display magnitude")
@click.pass_context
def plot_slice3d(ctx, array, output, scale,dim, magnitude, index, units):
    '''
    Visualize a dataset as 2D image
    '''
    parr, md = ctx.obj.get(array, True)
    domain = md.get("domain")
    if magnitude == "yes" :
        import numpy
        arr = numpy.sqrt(parr[0]*parr[0]+parr[1]*parr[1]+parr[2]*parr[2])
    else:
        arr = parr
    if domain:
        dom = ctx.obj.get_domain(domain)
    if units is not None:
        u = pochoir.arrays.fromstr1(units)
        arr = arr/u
    title = f'{array}'
    if units:
        title += f' [{units}]'
    pochoir.plots.slice3d(arr, output, dom,scale,dim,index,title)

@cli.command("plot-slice3d-twoarr")
@click.option("-a", "--array", type=str, required=True,
              help="Input array to plot")
@click.option("-c", "--comparray", type=str, required=True,
              help="Input array to plot")
@click.option("-o", "--output",
              type=click.Path(exists=False, dir_okay=False),
              help="Output graphics file")
@click.option("-s", "--scale", default="linear",
              type=click.Choice(["linear","signedlog"]),
              help="Output graphics file")
@click.option("-d", "--dim", default="z",
              type=click.Choice(["x","y","z"]),
              help="choose axis to slice")
@click.option("-m", "--magnitude", default="no", type=click.Choice(["yes","no"]),
              help="calc magnitude")
@click.option("-i","--index",type=int, default=1.0,
              help="choose index to slice")
@click.option("-u", "--units", type=str, default=None,
              help="The units in which to display magnitude")
@click.pass_context
def plot_slice3d_twoarr(ctx, array,comparray, output, scale,dim, magnitude, index, units):
    '''
    Visualize a dataset as 2D image
    '''
    parr, md = ctx.obj.get(array, True)
    parr2, md2 = ctx.obj.get(comparray, True)
    domain = md.get("domain")
    if magnitude == "yes" :
        import numpy
        arr = numpy.sqrt(parr[0]*parr[0]+parr[1]*parr[1]+parr[2]*parr[2])
        arr2=parr2
    else:
        arr = parr
        arr2= parr2
    if domain:
        dom = ctx.obj.get_domain(domain)
    if units is not None:
        u = pochoir.arrays.fromstr1(units)
        arr = arr/u
        arr2 = arr2/u
    title = f'{array}'
    if units:
        title += f' [{units}]'
    pochoir.plots.slice3d_two(arr,arr2, output, dom,scale,dim,index,title)

@cli.command("plot-mag")
@click.option("-a", "--array", type=str, required=True,
              help="Input array to plot")
@click.option("-o", "--output",
              type=click.Path(exists=False, dir_okay=False),
              help="Output graphics file")
@click.option("-u", "--units", type=str, default=None,
              help="The units in which to display magnitude")
@click.pass_context
def plot_mag(ctx, array, output, units):
    '''
    Plot magnitude of a vector field
    '''
    arr, md = ctx.obj.get(array, True)
    domain = md.get("domain")
    if domain:
        dom = ctx.obj.get_domain(domain)
    mag = pochoir.arrays.vmag(arr)
    from pochoir import units
    import numpy
    import matplotlib.pyplot as plt

    speed_unit = units.mm/units.us
    speed= mag/speed_unit
    speed_z = arr[2][:,:,:]#/speed_unit
    x = numpy.linspace(0,200,4000)
    for i in range(0,25):
        for j in range(0,17):
            plt.plot(x,speed_z[i,j,:])
    plt.title('velocity in the middle')
    plt.xlabel('vertical drift , mm')
    plt.ylabel('velocity, mm/mus')
    plt.show()
    #if units is not None:
    #    u = pochoir.arrays.fromstr1(units)
    #    mag = mag/u
    #title = f'{array}'
    #if units:
    #    title += f' [{units}]'
    #pochoir.plots.image(mag, output, dom, title)


@cli.command("plot-quiver")
@click.option("-a", "--array", type=str, required=True,
              help="Input array to plot")
@click.option("-o", "--output",
              type=click.Path(exists=False, dir_okay=False),
              help="Output graphics file")
@click.option("--step", default=1,
              help="Step over which to sample the array")
@click.option("--scale", default=None, type=float,
              help="Scale the arrows, larger number makes smaller arrows")
@click.option("--xlim", default=None, type=str,
              help="Limit X plot range")
@click.option("--ylim", default=None, type=str,
              help="Limit Y plot range")
@click.pass_context
def plot_quiver(ctx, array, output, step, scale, xlim, ylim):
    '''
    Visualize a 2D or 3D vector field as a "quiver" plot.
    '''
    arr, md = ctx.obj.get(array, True)
    domain = md.get("domain")
    if domain:
        dom = ctx.obj.get_domain(domain)
    if xlim:
        xlim = pochoir.arrays.fromstr1(xlim)
    if ylim:
        ylim = pochoir.arrays.fromstr1(ylim)

    pochoir.plots.quiver(arr, output, domain=dom, step=step,
                         limits=(xlim, ylim), scale=scale)


@cli.command("plot-drift")
@click.option("-t", "--trajectory", type=int, default=-1,
              help="Number of trajectories to plot (def: plot only traj 0)")
@click.option("-p", "--paths", type=str,
              help="The paths array to plot")
@click.option("-o", "--output",
              type=click.Path(exists=False, dir_okay=False),
              help="Output graphics file")
@click.pass_context
def plot_drift(ctx, trajectory, paths, output):
    '''
    Visualize 2D or 3D paths
    '''
    arr, md = ctx.obj.get(paths, True)
    domain = md.get("domain")
    dom = None
    if domain:
        dom = ctx.obj.get_domain(domain)
    if arr.shape[-1] == 2:
        pochoir.plots.drift2d(arr, output, dom, trajectory)
        return
    if arr.shape[-1] == 3:
        pochoir.plots.drift3d(arr, output, dom, trajectory,gif)
    click.echo(f'unsupported array of shape {arr.shape}')
    return -1

@cli.command("plot-drift3d")
@click.option("-t", "--trajectory", type=int, default=-1,
              help="Number of trajectories to plot (def: plot only traj 0)")
@click.option("-p", "--paths", type=str,
              help="The paths array to plot")
@click.option("-b", "--boundary", type=str,
              help="boundary array")
@click.option("-z", "--zoom", default="no", type=click.Choice(["yes","no"]),
              help="boundary array")
@click.option("-g", "--gif",
              default="no", type=click.Choice(["yes","no"]),
              help="create gif image ")
@click.option("-o", "--output",
              type=click.Path(exists=False, dir_okay=False),
              help="Output graphics file")
@click.pass_context
def plot_drift3d(ctx, trajectory, paths,boundary,zoom,gif, output):
    '''
    '''
    barr, mdb = ctx.obj.get(boundary, True)
    arr, md = ctx.obj.get(paths, True)
    domain = md.get("domain")
    dom = None
    if domain:
        dom = ctx.obj.get_domain(domain)
    title = f'{paths}'
    pochoir.plots.drift3d_b(arr,barr, output, dom, trajectory,zoom,gif,title)


@cli.command("export-vtk-image")
@click.argument("name")
@click.pass_context
def export_vtk(ctx, name):
    '''
    Export a dataset to a vtk file of same name
    '''
    arr = ctx.obj.get(name)
    scalars = {name: arr}
    pochoir.vtkexport.image3d(name, **scalars)


@cli.command()
@click.option("-d", "--domain", type=str, 
              help="Use named dataset for the domain, (def: indices)")
@click.option("-i","--initial", type=str,
              help="Name initial value array")
@click.option("-b","--boundary", type=str,
              help="Name the boundary array")
@click.argument("name")
@click.pass_context
def example(ctx, domain, initial, boundary, name):
    '''
    Generate a boundary and initial array example (try "list")
    '''
    if name == "list":
        for one in dir(pochoir.examples):
            if one.startswith("ex_"):
                print(one[3:])
        return

    meth = getattr(pochoir.examples, "ex_" + name)

    dom = None
    if domain:
        dom = ctx.obj.get_domain(domain)

    iarr, barr = meth(dom)
    ctx.obj.put(initial, iarr)
    ctx.obj.put(boundary, barr)
    

@cli.command()
@click.argument("things", nargs=-1)
@click.pass_context
def ls(ctx, things):
    '''
    List the store store
    '''
    if not things:
        things=["/"]

    for thing in things:
        got = ctx.obj.get(thing, True)
        if isinstance(got, tuple) and len(got) == 3:  # group
            dirs, arrs, mds = got
            print(f'store {ctx.obj.instore_name}: group {thing}:')
            for dirname in dirs:
                print(f'{dirname}/')
            for arrname in arrs:
                if thing == "/":
                    lookfor = arrname
                else:
                    lookfor = thing + "/" + arrname
                arr = ctx.obj.get(lookfor)
                print (f'{lookfor} {arr.dtype} {arr.shape}')
            return

        arr,md = got
        print(f'{thing}:')
        if arr is not None:
            print (f'\t{arr.shape} {arr.dtype}')
        if md is not None:
            print(f'\t{md}')

def _coarse_interpolator(carr, cdom):
    '''
    Build a multi-linear interpolator for a coarse field over its own
    domain coordinates, plus the list of per-axis coarse grid points
    (used to clamp query points to the coarse bounds).

    Returns (interp, cpoints).  Query with `_interp_clamped`.
    '''
    import numpy
    from pochoir.arrays import rgi
    cpoints = [numpy.asarray(ls, dtype=float) for ls in cdom.linspaces]
    interp = rgi(cpoints, numpy.asarray(carr, dtype=float))
    return interp, cpoints


def _interp_clamped(interp, cpoints, mesh, out_shape):
    '''
    Evaluate `interp` at the grid points given by `mesh` (list of
    meshgrid arrays), clamping each axis to the coarse bounds so points
    outside the coarse domain take the nearest edge value (no
    extrapolation).  Returns an array of shape `out_shape`.
    '''
    import numpy
    pts = numpy.stack([m.ravel() for m in mesh], axis=-1)
    for a in range(pts.shape[1]):
        pts[:, a] = numpy.clip(pts[:, a], cpoints[a][0], cpoints[a][-1])
    return interp(pts).reshape(out_shape)


@cli.command()
@click.option("-c", "--coarse", type=str, required=True,
              help="Input coarse potential array to upsample")
@click.option("-i", "--initial", type=str, required=True,
              help="Input fine initial value array (holds exact fine boundary values)")
@click.option("-b", "--boundary", type=str, required=True,
              help="Input fine boundary (bool) array")
@click.option("-I", "--output", type=str, required=True,
              help="Output refined fine initial value array")
@click.pass_context
def refine(ctx, coarse, initial, boundary, output):
    '''
    Grid-refine a coarse potential onto a fine grid for use as an FDM
    initial guess.

    The coarse potential is linearly (multi-linearly) interpolated from
    its own domain coordinates onto the fine domain coordinates.  The
    exact fine boundary values are then merged in at the boundary cells
    so the result satisfies the fine boundary conditions exactly.  The
    output is an "initial" array suitable to pass to `fdm --initial`.
    '''
    import numpy

    carr, cmd = ctx.obj.get(coarse, True)
    iarr, _   = ctx.obj.get(initial, True)
    barr, bmd = ctx.obj.get(boundary, True)

    if cmd is None or "domain" not in cmd:
        click.echo(f'failed to get domain for coarse potential {coarse}')
        info_msg(f'failed to get domain for coarse potential {coarse}')
        sys.exit(-1)
    if bmd is None or "domain" not in bmd:
        click.echo(f'failed to get domain for fine boundary {boundary}')
        info_msg(f'failed to get domain for fine boundary {boundary}')
        sys.exit(-1)

    cdom = ctx.obj.get_domain(cmd["domain"])
    fdom = ctx.obj.get_domain(bmd["domain"])

    fi   = numpy.asarray(iarr, dtype=float)
    bmask = numpy.asarray(barr).astype(bool)

    info_msg(f'refine: coarse {numpy.asarray(carr).shape} @ {cdom.spacing} -> '
             f'fine {fi.shape} @ {fdom.spacing}')

    # Multi-linear interpolation using the actual grid coordinates so the
    # refinement is correct even when the two domains do not share extent.
    interp, cpoints = _coarse_interpolator(carr, cdom)
    refined = _interp_clamped(interp, cpoints, fdom.meshgrid, fi.shape)

    # Merge exact fine boundary values at boundary (immutable) cells.
    refined[bmask] = fi[bmask]

    params = dict(operation="refine", domain=bmd["domain"],
                  coarse=coarse, initial=initial, boundary=boundary,
                  command="refine")
    ctx.obj.put(output, refined, taxon="initial", **params)


@cli.command("near-bc")
@click.option("-i", "--initial", type=str, required=True,
              help="Input near-field initial value array (e.g. refined coarse)")
@click.option("-b", "--boundary", type=str, required=True,
              help="Input near-field boundary (bool) array")
@click.option("-c", "--coarse", type=str, required=True,
              help="Input coarse potential supplying the interface values")
@click.option("-I", "--initial-out", type=str, required=True,
              help="Output near-field initial array with pinned interface plane")
@click.option("-B", "--boundary-out", type=str, required=True,
              help="Output near-field boundary array with the interface plane fixed")
@click.option("--axis", type=int, default=2,
              help="Axis normal to the interface plane (def: 2, i.e. z)")
@click.pass_context
def near_bc(ctx, initial, boundary, coarse, initial_out, boundary_out, axis):
    '''
    Add a Dirichlet interface plane to a near-field problem.

    For a near-field domain that covers only the first part of the drift
    region (e.g. z=0..20mm), the far interface plane (the last index
    along `--axis`) is pinned to the coarse bulk potential interpolated
    there.  This enforces continuity with the coarse far-field solve as
    a fixed-potential (Dirichlet) condition, matching the LArPix v2b
    .sif fixed-potential interface.

    Produces a new boundary array (with the interface plane marked
    immutable) and a new initial array (with the interface plane set to
    the coarse potential there).
    '''
    import numpy

    iarr, imd = ctx.obj.get(initial, True)
    barr, bmd = ctx.obj.get(boundary, True)
    carr, cmd = ctx.obj.get(coarse, True)

    if cmd is None or "domain" not in cmd:
        click.echo(f'failed to get domain for coarse potential {coarse}')
        info_msg(f'failed to get domain for coarse potential {coarse}')
        sys.exit(-1)
    if bmd is None or "domain" not in bmd:
        click.echo(f'failed to get domain for near-field boundary {boundary}')
        info_msg(f'failed to get domain for near-field boundary {boundary}')
        sys.exit(-1)

    cdom = ctx.obj.get_domain(cmd["domain"])
    ndom = ctx.obj.get_domain(bmd["domain"])

    fi    = numpy.asarray(iarr, dtype=float).copy()
    bmask = numpy.asarray(barr).astype(bool).copy()

    # The interface plane is the last grid index along `axis`.
    top = ndom.shape[axis] - 1
    plane = [slice(None)] * fi.ndim
    plane[axis] = top
    plane = tuple(plane)

    # Interpolate the coarse potential onto the interface plane points.
    interp, cpoints = _coarse_interpolator(carr, cdom)
    plane_mesh = [m[plane] for m in ndom.meshgrid]
    out_shape = numpy.asarray(ndom.shape)[
        [a for a in range(fi.ndim) if a != axis]]
    iface = _interp_clamped(interp, cpoints, plane_mesh, tuple(out_shape))

    z_iface = ndom.point([0] * fi.ndim)[axis] + top * ndom.spacing[axis]
    info_msg(f'near-bc: pinning interface plane axis={axis} index={top} '
             f'(coord={z_iface}) to coarse potential')

    fi[plane] = iface
    bmask[plane] = True

    iparams = dict(operation="near-bc", domain=bmd["domain"],
                   initial=initial, boundary=boundary, coarse=coarse,
                   interface_axis=axis, interface_index=int(top),
                   command="near-bc")
    bparams = dict(iparams)
    ctx.obj.put(initial_out, fi, taxon="initial", **iparams)
    ctx.obj.put(boundary_out, bmask, taxon="boundary", **bparams)


@cli.command("coarsen")
@click.option("-i", "--input", "input_", type=str, required=True,
              help="Input fine potential array to downsample")
@click.option("-d", "--domain", type=str, required=True,
              help="Target coarser domain key (already in store)")
@click.option("-o", "--output", type=str, required=True,
              help="Output coarsened potential")
@click.option("--average", is_flag=True, default=False,
              help="Block-average stride-sized blocks instead of stride-sampling")
@click.pass_context
def coarsen(ctx, input_, domain, output, average):
    '''
    Downsample a fine potential onto a coarser domain.

    Pixel sizes that are odd multiples of the coarse spacing (e.g. 3.9mm
    at 0.1mm) produce asymmetric FDM pixel tiles.  The near-field FDM
    solve is therefore run at half spacing (e.g. 0.05mm, where the tile
    is symmetric), then this command strides the result back to the
    coarse spacing before stitching.

    The per-axis stride is inferred from the spacing ratio
    `stride[i] = round(coarse.spacing[i] / fine.spacing[i])`.

    Without --average (default): stride-sample as `arr[::sx, ::sy, ::sz]`.
    With --average: block-average each stride-sized block.  Axes whose
    length is not an exact multiple of the stride are edge-padded so that
    the last partial block averages to the boundary value exactly.
    The resulting shape must match the target domain.
    '''
    import numpy

    iarr, imd = ctx.obj.get(input_, True)

    if iarr is None:
        click.echo(f'failed to load input potential {input_}')
        info_msg(f'failed to load input potential {input_}')
        sys.exit(-1)
    if imd is None or "domain" not in imd:
        click.echo(f'failed to get domain for input potential {input_}')
        info_msg(f'failed to get domain for input potential {input_}')
        sys.exit(-1)

    fdom = ctx.obj.get_domain(imd["domain"])
    cdom = ctx.obj.get_domain(domain)

    arr = numpy.asarray(iarr, dtype=float)

    # Infer the per-axis stride from the spacing ratio.
    stride = [int(round(cdom.spacing[a] / fdom.spacing[a]))
              for a in range(arr.ndim)]
    if any(s < 1 for s in stride):
        click.echo(f'coarsen: target spacing {tuple(cdom.spacing)} is finer '
                   f'than input {tuple(fdom.spacing)} on some axis')
        sys.exit(-1)

    if average:
        # Pad each axis to the nearest multiple of its stride using edge
        # replication, then reshape and mean.  Edge padding means a partial
        # last block (odd-length axis) averages to the exact boundary value.
        padded = arr
        for a in range(arr.ndim):
            s = stride[a]
            rem = padded.shape[a] % s
            if rem != 0:
                pad_width = [(0, 0)] * arr.ndim
                pad_width[a] = (0, s - rem)
                padded = numpy.pad(padded, pad_width, mode='edge')
        # Reshape to (..., n0, s0, n1, s1, ...) then mean over stride axes.
        new_shape = []
        for a in range(arr.ndim):
            new_shape += [padded.shape[a] // stride[a], stride[a]]
        mean_axes = tuple(range(1, 2 * arr.ndim, 2))
        out = padded.reshape(new_shape).mean(axis=mean_axes)
    else:
        sel = tuple(slice(None, None, s) for s in stride)
        out = arr[sel]

    cshape = tuple(int(s) for s in cdom.shape)
    if out.shape != cshape:
        mode = "averaged" if average else "strided"
        click.echo(f'coarsen: {mode} shape {out.shape} (stride {tuple(stride)}) '
                   f'does not match target domain shape {cshape}')
        info_msg(f'coarsen: {mode} shape {out.shape} != domain {cshape}')
        sys.exit(-1)

    mode = "average" if average else "stride"
    info_msg(f'coarsen ({mode}): {arr.shape} @ {tuple(fdom.spacing)} -> '
             f'{out.shape} @ {tuple(cdom.spacing)} (stride {tuple(stride)})')

    params = dict(operation="coarsen", domain=domain,
                  input=input_, stride=[int(s) for s in stride],
                  average=average, command="coarsen")
    ctx.obj.put(output, out, taxon="potential", **params)


@cli.command("stitch-near")
@click.option("-n", "--near", type=str, required=True,
              help="Input near-field fine potential (covers z=0..interface)")
@click.option("-c", "--coarse", type=str, required=True,
              help="Input coarse potential supplying the far-field bulk")
@click.option("-d", "--domain", type=str, required=True,
              help="Full fine target domain to stitch onto")
@click.option("-I", "--output", type=str, required=True,
              help="Output stitched full fine potential")
@click.option("--axis", type=int, default=2,
              help="Stitch axis (def: 2, i.e. z)")
@click.pass_context
def stitch_near(ctx, near, coarse, domain, output, axis):
    '''
    Stitch a near-field fine solve onto an upsampled coarse far field.

    The coarse potential is multi-linearly upsampled onto the full fine
    target domain, then the near-field fine solution overwrites the near
    region (the first `near.shape[axis]` planes along `--axis`).  Because
    `near-bc` pinned the near-field interface plane to the coarse value
    there, the result is continuous across the interface.
    '''
    import numpy

    narr, nmd = ctx.obj.get(near, True)
    carr, cmd = ctx.obj.get(coarse, True)

    if cmd is None or "domain" not in cmd:
        click.echo(f'failed to get domain for coarse potential {coarse}')
        info_msg(f'failed to get domain for coarse potential {coarse}')
        sys.exit(-1)

    if narr is None:
        click.echo(f'failed to load near-field potential {near}')
        info_msg(f'failed to load near-field potential {near}')
        sys.exit(-1)

    cdom = ctx.obj.get_domain(cmd["domain"])
    fdom = ctx.obj.get_domain(domain)

    near_pot = numpy.asarray(narr, dtype=float)
    fshape = tuple(int(s) for s in fdom.shape)

    # Upsample coarse onto the full fine grid.
    interp, cpoints = _coarse_interpolator(carr, cdom)
    full = _interp_clamped(interp, cpoints, fdom.meshgrid, fshape)

    # The near region occupies the first n planes along `axis`; the other
    # axes must share shape with the fine domain.
    for a in range(near_pot.ndim):
        if a == axis:
            if near_pot.shape[a] > fshape[a]:
                click.echo('near-field extent exceeds fine domain along '
                           f'stitch axis {axis}')
                sys.exit(-1)
        elif near_pot.shape[a] != fshape[a]:
            click.echo(f'near-field shape {near_pot.shape} incompatible with '
                       f'fine domain {fshape} on axis {a}')
            sys.exit(-1)

    nnear = near_pot.shape[axis]
    sel = [slice(None)] * full.ndim
    sel[axis] = slice(0, nnear)
    full[tuple(sel)] = near_pot

    info_msg(f'stitch-near: near {near_pot.shape} over first {nnear} planes '
             f'of fine {fshape} along axis {axis}')

    params = dict(operation="stitch-near", domain=domain,
                  near=near, coarse=coarse, stitch_axis=axis,
                  command="stitch-near")
    ctx.obj.put(output, full, taxon="potential", **params)


@cli.command("near-far-solve")
@click.option("-C", "--coarse-potential", type=str, required=True,
              help="Coarse full-domain solved potential (initial far field)")
@click.option("--coarse-initial", type=str, required=True,
              help="Coarse full-domain initial value array")
@click.option("--coarse-boundary", type=str, required=True,
              help="Coarse full-domain boundary (bool) array")
@click.option("--near-initial", type=str, required=True,
              help="Near-field initial value array (electrode values)")
@click.option("--near-boundary", type=str, required=True,
              help="Near-field boundary (bool) array (electrodes only)")
@click.option("--near-potential", type=str, default=None,
              help="Optional already-solved near potential to start from "
                   "(e.g. the discrete refine->near-bc->fdm sweep-0), so its "
                   "intermediate store files are preserved")
@click.option("--interface", type=str, required=True,
              help="Interface coordinate along --axis (e.g. '20*mm')")
@click.option("--axis", type=int, default=2,
              help="Stitch axis normal to the interface (def: 2, i.e. z)")
@click.option("-e", "--edges", default="periodic,periodic,fixed",
              help="Comma list of fixed/periodic per axis (matches fdm)")
@click.option("--engine", default="torch",
              help="FDM solver engine (numpy/numba/torch/cupy/cumba)")
@click.option("--epoch", default=1000000, type=int,
              help="FDM iterations per precision check")
@click.option("-n", "--nepochs", default=10, type=int,
              help="FDM max number of epochs")
@click.option("--near-precision", default=2e-11, type=float,
              help="Convergence precision for the fine near solve")
@click.option("--far-precision", default=2e-7, type=float,
              help="Convergence precision for the coarse far solve")
@click.option("--tol", type=str, default="1.0",
              help="Schwarz tolerance: max near-solution change per sweep "
                   "(may use units, e.g. '1*V')")
@click.option("--max-iters", default=6, type=int,
              help="Maximum number of near<->far Schwarz sweeps")
@click.option("--overlap", default=1, type=int,
              help="Schwarz overlap width in coarse cells (>=1, def 1). Wider "
                   "overlap converges faster and gives a C1 (gradient-"
                   "continuous) near/far interface. The far is pinned to the "
                   "near solution this many coarse cells below the interface.")
@click.option("--near-out", type=str, required=True,
              help="Output final near-field potential")
@click.option("--far-out", type=str, required=True,
              help="Output final far-field (coarse) potential")
@click.option('--insulator', type=str, default=None,
              help="Near-field no-flux insulator mask; applied in the NEAR "
                   "Schwarz re-solves so they preserve the no-flux FR4 field "
                   "instead of discarding it (NO epsilon). The coarse/far "
                   "re-solves never see it (FR4 unresolved at coarse pitch).")
@click.pass_context
def near_far_solve(ctx, coarse_potential, coarse_initial, coarse_boundary,
                   near_initial, near_boundary, near_potential, interface, axis,
                   edges, engine, epoch, nepochs, near_precision, far_precision,
                   tol, max_iters, near_out, far_out, insulator, overlap):
    '''
    Overlapping-Schwarz near/far solve for a continuous stitched potential.

    Alternately solves the fine near domain (interface plane pinned to the
    current far solution) and the full coarse far domain (an interior plane
    one coarse cell below the interface pinned to the near solution), sharing
    a 1-cell overlap.  Iterating to convergence makes the stitched field
    continuous in value *and* gradient across the interface, unlike the
    single-shot near-bc pin.  Writes the final near and far potentials, which
    are then coarsened + stitched (`coarsen`, `stitch-near`) as usual.
    '''
    import numpy
    import pochoir.fdm
    from pochoir import nearfar

    try:
        solve = getattr(pochoir.fdm, f'solve_{engine}')
    except AttributeError:
        click.echo(f'no fdm solver engine {engine}')
        sys.exit(-1)

    cpot, cmd = ctx.obj.get(coarse_potential, True)
    cinit = ctx.obj.get(coarse_initial)
    cbnd = ctx.obj.get(coarse_boundary)
    ninit, nmd = ctx.obj.get(near_initial, True)
    nbnd, nbmd = ctx.obj.get(near_boundary, True)
    near_start = ctx.obj.get(near_potential) if near_potential else None
    # The near-field FR4 no-flux mask (if any) is applied only to the NEAR
    # re-solves; the coarse/far domain does not resolve the FR4.
    near_insulator = ctx.obj.get(insulator) if insulator else None

    if cmd is None or "domain" not in cmd:
        click.echo(f'failed to get domain for coarse potential {coarse_potential}')
        sys.exit(-1)
    near_dom_key = nbmd["domain"] if (nbmd and "domain" in nbmd) else nmd["domain"]
    coarse_dom = ctx.obj.get_domain(cmd["domain"])
    near_dom = ctx.obj.get_domain(near_dom_key)

    bool_edges = [e.startswith("per") for e in edges.split(",")]
    if len(bool_edges) != numpy.asarray(cpot).ndim:
        raise ValueError("number of edge conditions does not match dimensions")

    interface_z = float(pochoir.arrays.fromstr1(interface)[0])
    # potentials are stored in raw volts (pot is not scaled by units.V), so
    # express the Schwarz tolerance in volts too: '1*V' -> 1.0, '0.05*V' -> 0.05.
    tol_v = float(pochoir.arrays.fromstr1(tol)[0]) / units.V

    def _make_solver(prec, insulator=None):
        def _solve(iarr, barr):
            if engine == "torch":
                # Pass the insulator mask only when present so the no-mask path
                # (and the far solve) stay byte-identical to before.
                extra = {} if insulator is None else {'insulator': insulator}
                arr, _err = solve(
                    numpy.asarray(iarr, dtype=float),
                    numpy.asarray(barr).astype(bool),
                    bool_edges, prec, epoch, nepochs,
                    info_msg=info_msg, _dtype=torch.float64,
                    ctx=ctx, potential=near_out, increment=near_out + "/inc",
                    params=dict(command="near-far-solve"), epsilon=None, **extra)
            else:
                arr, _err = solve(
                    numpy.asarray(iarr, dtype=float),
                    numpy.asarray(barr).astype(bool),
                    bool_edges, prec, epoch, nepochs)
            return numpy.asarray(arr)
        return _solve

    near_pot, far_pot, n_iters, delta = nearfar.schwarz_solve(
        cpot, cinit, cbnd, coarse_dom,
        ninit, nbnd, near_dom,
        _make_solver(near_precision, insulator=near_insulator),
        _make_solver(far_precision),
        axis=axis, interface_z=interface_z,
        tol=tol_v, max_iters=max_iters, log=info_msg,
        near_start=near_start, overlap=overlap)

    info_msg(f'near-far-solve: {n_iters} sweeps, final near delta={delta}')
    print(f'near-far-solve: {n_iters} sweeps, final near delta={delta}')

    nparams = dict(operation="near-far-solve", domain=near_dom_key,
                   coarse=coarse_potential, interface=interface,
                   interface_axis=axis, sweeps=int(n_iters),
                   command="near-far-solve")
    fparams = dict(operation="near-far-solve", domain=cmd["domain"],
                   near=near_initial, interface=interface,
                   interface_axis=axis, sweeps=int(n_iters),
                   command="near-far-solve")
    ctx.obj.put(near_out, near_pot, taxon="potential", **nparams)
    ctx.obj.put(far_out, far_pot, taxon="potential", **fparams)


@cli.command("hybrid-iterate")
@click.option("--coarse-config", type=click.Path(exists=True), required=True,
              help="JSON config for the 0.4mm coarse grid transcription")
@click.option("--fine-config", type=click.Path(exists=True), required=True,
              help="JSON config for the 0.05mm near and 0.1mm final grids")
@click.option("--interface", type=str, default='20*mm',
              help="Near/far interface coordinate on axis 2 (def: '20*mm')")
@click.option("--tol", type=float, default=2e-8,
              help="Convergence tolerance on max|phi_k - phi_(k-1)| in volts")
@click.option("--max-iters", type=int, default=20,
              help="Maximum outer iterations before reporting the achieved delta")
@click.pass_context
def hybrid_iterate(ctx, coarse_config, fine_config, interface, tol, max_iters):
    '''
    Task13 iterative hybrid near/far drift-field solve (drift field + paths).

    Alternates a 0.05mm near solve with a full-volume 0.4mm re-solve in which
    the near region FLOATS (stitched values are initial values only), then
    refines the converged field onto the 0.1mm full grid and runs
    velo/starts/drift on it.  Unlike `near-far-solve` the near region is never
    pinned inside the volume; only the z=interface plane is.
    '''
    import pochoir.hybrid_iterate
    pochoir.hybrid_iterate.hybrid_iterate(
        ctx, coarse_config, fine_config,
        interface=interface, tol=tol, max_iters=max_iters)


def main():
    cli(obj=None)


if '__main__' == __name__:
    main()
