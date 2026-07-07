# The method of `run-full-3d-pixel.sh`

`test/run-full-3d-pixel.sh` is the end-to-end driver that computes the field
response of a **pixelated LArTPC readout**. This document explains *what* it
computes and *how*, first at a big-picture level and then in detail — with
particular attention to the multi-resolution **coarse → fine → coarse** solve
that is the heart of the method.

---

## 1. Big picture

The script produces the electrical response of a pixel plane immersed in liquid
argon, in three parts:

- **PART A — drift field.** The *real* electrostatic potential φ_drift of the
  detector: the cathode at −15400 V, the PCB shield/grid plane at −1000 V, and
  the pixel pads at 0 V. From φ_drift the script derives the electric field
  E = −∇φ, then the liquid-argon drift velocity, then the actual trajectories
  of ionization electrons.

- **PART B — weighting field.** A *unit, source-free* (Laplace) potential
  φ_weight in which the single **central pixel pad is held at 1 V** and every
  other conductor — the other pads, the grid, and the cathode — is held at 0 V.
  This is a purely geometric quantity; it carries no real voltages.

- **PART C — induced current.** The two fields are combined via **Ramo's
  theorem**. For a charge q moving along the drift trajectory **r**(t), the
  charge induced on the pixel is Q(t) = q · φ_weight(**r**(t)), and the induced
  current is I = dQ/dt.

### Why two separate solves plus Ramo?

The drift field tells you **how** electrons move; the weighting field tells you
**how much** signal a moving electron induces on a given pixel. Ramo's theorem
lets these be computed independently and combined afterwards. Doing so is much
cheaper and more accurate than trying to read an induced current directly out of
a single drift solve, which is why the workflow deliberately keeps them apart
and multiplies them at the end.

### The central numerical trick: near-field refinement

The drift region is ~310 mm deep, but the physics that shapes the signal — the
pads, the gaps between them, the shield grid — all live within the first ~20 mm
above the pixel plane. Solving the whole 310 mm depth at the resolution needed
to resolve that geometry (0.05–0.1 mm) is infeasible in 3D (it runs out of
memory). So the script:

1. solves the **entire depth coarsely** (0.4 mm),
2. solves **only the near-field region** (z = 0…20 mm) **finely** (0.05 mm),
3. **stitches** the fine near-field onto the linearly-upsampled coarse
   far-field to build the full 0.05 mm potential.

The subtlety — and the part that took the most engineering — is making the join
between the fine and coarse regions physically seamless. Section 2.3 covers this
in full.

### Resumability: the `want` wrapper

Every step is wrapped in a `want` function that runs the step only if its output
store keys are missing, and verifies they were produced afterward. The script
overrides the single-key `want` from `helpers.sh` with a multi-key version, so a
re-run reuses existing results and an interrupted step can be resumed. This is
why the script can be run repeatedly without redoing expensive solves.

---

## 2. The method in detail

The script chains together pochoir CLI subcommands (defined in
`pochoir/__main__.py`). Each numbered step below maps a block of the script to
its command and its numerics. Concrete grid numbers are given for PART A
(drift); PART B (weighting) mirrors the same shapes at higher transverse
resolution.

### 2.1 Discretization and geometry (`domain`, `gen`)

- **`domain`** defines a uniform rectilinear grid from `--shape` (grid-point
  counts per axis), `--spacing` (grid step, mm), and `--origin`. It is pure
  bookkeeping — it introduces no physics, only the discretization other commands
  reference.

- **`gen`** runs a named geometry generator against a domain and a JSON config,
  rasterizing the electrode/PCB geometry into two arrays:
  - an **initial** array holding electrode potentials / seed values, and
  - a **boundary** boolean mask marking fixed (Dirichlet) cells.

  Two generators are used, and the *difference between them is exactly the set of
  boundary conditions* — which is why two separate solves are required:

  | Generator | Used for | Cathode | Shield/grid | Central pad | Other pads |
  |---|---|---|---|---|---|
  | `pcb_drift_pixel_with_grid` | drift (PART A) | −15400 V | −1000 V | 0 V | 0 V |
  | `pcb_pixel_with_grid` | weighting (PART B) | 0 V (grounded) | 0 V | **1 V** | 0 V |

  The weighting generator additionally grounds the cathode plane and the domain
  end planes to 0 V — required by Ramo's theorem, so the injected weighting flux
  has somewhere to terminate. It reads no real voltages (hence the weighting JSON
  omits `GridPotential`/`CathodePotential`).

- **Geometry constants** (from `example_gen_pcb_drift_pixel_with_grid.json` and
  `example_gen_pixel_with_grid.json`): square pads 3.5 mm on a side with a 0.9 mm
  gap → **4.4 mm pitch**, in a 5×5 array, corners rounded with a 0.7 mm chamfer;
  a 1.6 mm FR4 shield layer; the pixel plane at z ≈ 2 mm; the cathode at
  z = 308 mm (`driftZDepth`); LAr ε_r = 1.5, FR4 ε_r = 4.5.

### 2.2 Coarse solve (`fdm`)

- **`fdm`** solves the Laplace/Poisson equation on the grid by iterative
  finite-difference relaxation (a neighbor-averaging stencil), holding the
  boundary-mask cells fixed and updating the rest until convergence.
  - `--edges` sets the domain-edge behavior per axis: `per,per,fix` for drift
    (periodic transverse, fixed along z), `fix,fix,fix` for weighting.
  - `--engine torch` selects the float64, GPU-capable PyTorch backend.
  - `--precision` is the convergence threshold (stop when the max cell change
    falls below it); a run does up to `--nepochs × --epoch` iterations.

  The coarse solve runs on an **11×11×775 grid at 0.4 mm** (z = 0…309.6 mm),
  producing `potential/coarse` — the bulk far-field, correct in the bulk but too
  coarse to resolve the pad/grid geometry.

### 2.3 The coarse → fine → coarse solve

This is the core of the method. It has three resolutions along the z (drift)
axis and two possible ways to *couple* the fine near-field to the coarse
far-field.

#### The three grids

| Stage | Grid (drift) | Spacing | z extent | Store key |
|---|---|---|---|---|
| **Coarse** (full depth) | 11×11×775 | 0.4 mm | 0 … 309.6 mm | `potential/coarse` |
| **Fine / near** (near only) | 88×88×401 | 0.05 mm | 0 … 20 mm | `potential/near` |
| **Stitched full** | 88×88×6200 | 0.05 mm | 0 … 309.95 mm | `potential/drift3d` |

A full-depth *fine solve* would be 88×88×6200 and exhausts memory, so the fine
**FDM** runs only where the geometry is strong (z ≤ 20 mm). The far bulk is never
solved finely — it is the coarse 0.4 mm solve, linearly upsampled at stitch time
— so the full 0.05 mm grid only ever holds an interpolated (cheap) far field
plus the genuinely fine near field.

#### Why 0.05 mm all the way through

The pixel pitch (4.4 mm) is an **odd** multiple of 0.1 mm, which produces an
asymmetric FDM pixel tile at 0.1 mm. Solving at **0.05 mm** makes the tile
symmetric. The near solve keeps its native **0.05 mm** and is **not** coarsened;
instead the coarse far field is linearly upsampled 0.4 mm → 0.05 mm at stitch
time, so the whole stitched drift potential (`potential/drift3d`) is 0.05 mm
(88×88×6200). The full grid is ~48 M cells; stitch/velo/drift are numpy, so this
runs as a background job (see the risk note in the driver's Step 5).

#### Coupling flavor 1 — single-shot Dirichlet pin (C0 only)

Steps `refine` → `near-bc` → `fdm`:

- **`refine`** multi-linearly upsamples `potential/coarse` onto the 0.05 mm near
  grid to make an FDM *warm-start* (`initial/near_refined`), re-imposing the
  exact electrode values at boundary cells.
- **`near-bc`** takes the near domain's **top plane** (last z index, at
  z = 20 mm) and **pins it Dirichlet** to the coarse potential interpolated
  there. This single plane is the only far → near information transfer.
- **`fdm`** solves the fine near domain with that top plane and the electrodes
  held fixed, producing `potential/near`.

This enforces **value (C0) continuity only** at the seam. The near side is an
independent fine Poisson solve and the far side is an independent coarse
upsample, so E_z = −dφ/dz is generally **discontinuous** across z = 20 mm — a
kink in the field.

#### Coupling flavor 2 — overlapping-Schwarz (C0 *and* C1)

Command `near-far-solve` (core in `pochoir/nearfar.py::schwarz_solve`) replaces
the one-shot pin with an **alternating iteration on a 1-cell (2-plane)
overlap**. The relevant plane indices (resolved by `_interface_indices`):

- coarse `ci = 50` (z = 20 mm), inner `ci_in = 49` (z = **19.6 mm**)
- near `nt = 400` (z = 20 mm), inner `nt_in = 399` (z = **19.95 mm**)
- shared overlap interval: **[19.6, 20.0] mm**

```
             z = 19.6      19.95   20.0 mm
                |            |       |
  COARSE(0.4):  ci_in ======(overlap)====== ci        interface
                [PIN to near]           [solve free]
                                        |
  NEAR(0.05):              nt_in ========|=== nt
                       [solve free]        [PIN to far]

  far pins one cell BELOW the seam and solves the seam freely;
  near pins the seam and solves one cell below freely.
```

- **Sweep 0** is the flavor-1 near solve, handed in via `--near-potential` so
  those intermediate store files are preserved for resume/tests.
- **Each sweep** then does:
  1. **FAR solve** on the full coarse domain — warm-started from the previous
     far with electrode Dirichlet re-imposed, plane `ci_in` (19.6 mm) **pinned**
     to the current near, plane `ci` (20 mm) **seeded/free** — and re-solves.
  2. **NEAR solve** on the fine domain — plane `nt` (20 mm) **pinned** to the
     just-updated far, plane `nt_in` (19.95 mm) **seeded/free** — and re-solves.
  3. Convergence test: `delta = max|near_new − near_prev|`; stop when
     `delta < --tol` (default `1*V`) or after `--max-iters` (default 6).

Because the overlap is **co-owned** — the far solve pins one cell below the seam
and solves the seam freely, while the near solve pins the seam and solves one
cell below freely — at convergence both subdomains agree over the whole overlap.
The result is a single, globally consistent harmonic solution that is continuous
in **value and gradient** across the seam, not two independently truncated
pieces. Each subdomain's FDM is injected as a closure (`_make_solver`) bound to
its own precision (near `2e-11`, far `2e-7`). Ending every sweep on the near
solve leaves the near field exactly C0-pinned to the far field. Outputs: an
updated `potential/near` **and** a consistent far field `potential/far`.

#### The final "→ stitch" (no coarsen)

- **`stitch-near`** multi-linearly upsamples the far field (`potential/far`)
  onto the full **0.05 mm** grid (88×88×6200), then overwrites the **first 401
  planes** (z = 0…20 mm) with the native 0.05 mm near solve (`potential/near`).
  Transverse shapes already match (88×88), so no `coarsen` step is needed. The
  seam is continuous because it was pinned (and, for drift, Schwarz-refined).

#### The asymmetry between the two PARTs

This is important and easy to miss:

- **PART A (drift)** runs the **Schwarz** `near-far-solve` and stitches onto
  **`potential/far`** (the Schwarz-updated far field) → gradient-continuous.
  Drift-path integration needs a smooth E-field, so this matters.
- **PART B (weighting)** **skips** Schwarz. It uses only the **single-shot C0
  pin** and stitches onto the raw **`potential/weight_coarse`**.

The final stitched potentials are stored as **`potential/drift3d`** (drift) and
**`potential/weight3d`** (weighting, consumed by `induce-pixel`). Store keys are
non-overlapping between the two parts (drift uses `coarse/near/fine/full`;
weighting uses `weight_*`).

### 2.4 Velocity, paths, and induced current (PART C)

- **`velo`** computes E = −∇φ from `potential/drift3d` (finite-difference
  gradient), zeros E at electrode and domain-edge cells, then derives the
  liquid-argon drift velocity v = μ(|E|, T)·E using the field- and
  temperature-dependent LAr mobility at `--temperature 87*K`. Output:
  `velocity/drift3d`.

- **`starts`** builds the drift start points: a 10×10 grid (0.44 mm spacing,
  100 points total) launched from the cathode plane at z = 308 mm.

- **`drift`** integrates a trajectory for each start point through the velocity
  field over the requested time range (`0*us, 210*us, 0.05*us`). It uses
  `--interp-order linear`: cubic interpolation overshoots and over-focuses the
  paths near the strong pixel-plane geometry, whereas linear is monotone-safe.
  (Evidence: `test/interp_overshoot_test.py` and
  `test/debug/cubic_interpolation_overshoot.md`.) Output: `paths/drift3d_tight`.

- **`induce-pixel`** applies Ramo's theorem: it samples the weighting potential
  `potential/weight3d` along each drift path (replicating paths across a pixel
  grid for `--npixels 2`) to get Q(t) = q · φ_weight(**r**(t)), then differences
  it to get I = dQ/dt. Output: `current/induced_current`.

---

## 3. Data flow at a glance

The `want` chain, step → store keys produced:

| Step | Command | Produces (drift / weighting) |
|---|---|---|
| grids | `domain` | `domain/coarse`, `domain/near`, `domain/fine` (+ `weight_*`) |
| geometry | `gen` | `initial/coarse`,`boundary/coarse` (+ `near`, `fine`, `weight_*`) |
| coarse solve | `fdm` | `potential/coarse` / `potential/weight_coarse` |
| warm start | `refine` | `initial/near_refined` / `initial/weight_near_refined` |
| C0 pin | `near-bc` | `initial/near_bc`,`boundary/near_bc` (+ `weight_*`) |
| near solve | `fdm` | `potential/near` / `potential/weight_near` |
| Schwarz (drift only) | `near-far-solve` | `potential/near` (updated), `potential/far` |
| coarsen (weighting only) | `coarsen` | `potential/weight_near_01` |
| stitch | `stitch-near` | **`potential/drift3d`** / **`potential/weight3d`** |
| velocity | `velo` | `velocity/drift3d` |
| starts | `starts` | `starts/drift3d` |
| paths | `drift` | `paths/drift3d_tight` |
| current | `induce-pixel` | `current/induced_current` |
