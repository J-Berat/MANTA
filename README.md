# MANTA

Interactive FITS file viewer in Julia, built with Makie/GLMakie.

`MANTA` automatically opens several kinds of FITS data:

- standard 2D FITS images, with contrast controls and no slice slider;
- standard 3D FITS cubes, with slice navigation and a spectrum at the selected voxel;
- HEALPix FITS maps, displayed in Mollweide projection;
- 2D HEALPix-PPV cubes, with one map per channel and an associated spectrum.

Main features:

- navigate cube axes and slice indices;
- select a pixel or voxel with the mouse;
- inspect the spectrum at the selected position or the mean spectrum in a box/circle region;
- switch between `lin`, `log10`, and `ln` scales;
- invert the colormap;
- overlay automatic or manual contours;
- display simple FITS WCS coordinates when `CTYPE/CRVAL/CRPIX/CDELT` headers are present;
- tune contrast from a visible-slice histogram with percentile presets;
- apply optional Gaussian smoothing to 3D cubes and projected HEALPix maps;
- zoom interactively with right-click drag;
- export images, spectra, and GIFs;
- save and reload viewer settings.

## Requirements

- Julia `1.9`, `1.10`, `1.11`, or `1.12`
- a local graphical session with OpenGL support for `GLMakie`
- `git`, if you install the project from GitHub

Quick check:

```bash
julia --version
git --version
```

On a remote machine or a headless server, the GLMakie interface will not open an
interactive window unless a suitable graphical environment is configured, such
as X11 forwarding, VNC, or a desktop session.

## Installation

Clone the repository and enter the project directory:

```bash
git clone git@github.com:J-Berat/MANTA.jl.git
cd MANTA.jl
```

If you prefer HTTPS:

```bash
git clone https://github.com/J-Berat/MANTA.jl.git
cd MANTA.jl
```

Install and precompile the Julia dependencies:

```bash
julia --project=. -e 'import Pkg; Pkg.instantiate(); Pkg.precompile()'
```

The repository also provides a setup script. It activates the project, installs
any missing runtime dependencies, precompiles the environment, and prints the
project status:

```bash
julia --project=. scripts/setup.jl
```

The `manta` launcher is already executable in the repository. If needed:

```bash
chmod +x manta
```

## Quick Start

From the repository root, run the demo:

```bash
./manta
```

This creates a synthetic FITS cube at:

```text
demo/output/synthetic_cube.fits
```

and opens the interactive viewer.

Run the demo with custom dimensions:

```bash
NX=96 NY=72 NZ=48 ./manta
```

Set initial color limits:

```bash
VMIN=5 VMAX=1500 ./manta
```

You can combine both:

```bash
NX=96 NY=72 NZ=48 VMIN=5 VMAX=1500 ./manta
```

## Open a FITS File

Open an existing FITS cube:

```bash
./manta path/to/cube.fits
```

Run setup before opening a file:

```bash
./manta --setup path/to/cube.fits
```

Run the Julia demo directly, without the `manta` launcher:

```bash
julia --project=. demo/run_demo.jl
```

Pass cube dimensions as arguments:

```bash
julia --project=. demo/run_demo.jl 80 60 40
```

## Julia Usage

Start Julia in the project environment:

```bash
julia --project=.
```

Then load the local module:

```julia
push!(LOAD_PATH, "src")
using MANTA

fig = MANTA.manta("path/to/cube.fits"; cmap=:magma)
display(fig)
```

Example with options:

```julia
fig = MANTA.manta(
    "path/to/cube.fits";
    cmap=:viridis,
    vmin=0,
    vmax=100,
    invert=false,
    figsize=(1400, 900),
    save_dir="outputs",
    settings_path="viewer_settings.toml",
)
display(fig)
```

Useful options:

- `cmap`: Makie colormap, for example `:viridis`, `:magma`, or `:inferno`
- `vmin`, `vmax`: manual colorbar limits
- `invert`: invert the colormap
- `figsize`: window size, for example `(1400, 900)`
- `save_dir`: export directory
- `settings_path`: TOML file used to save and reload viewer state
- `activate_gl=false`: useful for tests without an OpenGL context
- `display_fig=false`: skip automatic display in tests

## RGB Images and Panels

`MANTA` can display RGB/RGBA arrays directly. Pass either a
`Matrix{RGB/RGBA}` or a numeric stack with 3 or 4 channels in the first or last
dimension:

```julia
rgb = MANTA.rgb_image(U, V, W; normalize=:symmetric)
fig = MANTA.manta(rgb; title="velocity RGB")
display(fig)
```

For RGB data stored in a FITS primary HDU as a 3/4-channel stack, opt in with
`rgb=true`:

```julia
fig = MANTA.manta("path/to/rgb_stack.fits"; rgb=true)
```

For 3D scalar cubes, the main viewer can be switched to a synchronized dual
view after launch: click `Add dual` in the `Output` panel, paste the second
FITS cube path into the revealed field, then click `Dual`. The second cube must
have the same `(nx, ny, nz)` size as the primary cube; slice axis, slice index,
colormap, contrast, contours, regions, and zoom stay synchronized.

Mixed scalar/RGB side-by-side panels are available through `manta_panels`:

```julia
fig = MANTA.manta_panels(
    rgb,
    speed;
    titles=("RGB composite", "speed"),
    cmaps=(:viridis, :magma),
)
display(fig)
```

## HEALPix

`MANTA.manta(...)` automatically detects HEALPix files when the header
contains `PIXTYPE = HEALPIX`, or when a 2D array matches the expected shape of a
HEALPix-PPV cube.

HEALPix map:

```julia
fig = MANTA.manta(
    "path/to/healpix_map.fits";
    column=1,
    scale=:lin,
    nx=1400,
    ny=700,
    cmap=:inferno,
)
display(fig)
```

HEALPix maps and HEALPix-PPV channel maps include a Gaussian smoothing control.
It filters the Mollweide-projected map with a NaN-aware normalization, so the
outside of the projection does not bleed into the data.

HEALPix-PPV cube:

```julia
fig = MANTA.manta(
    "path/to/healpix_ppv_cube.fits";
    v0=0.0,
    dv=1.0,
    vunit="km/s",
    scale=:lin,
)
display(fig)
```

HEALPix RGB/RGBA pixels can also be passed directly as a vector of length
`12*nside^2`, or as `npix×3`, `npix×4`, `3×npix`, or `4×npix` numeric arrays:

```julia
fig = MANTA.manta_healpix(rgb_pixels; nx=1400, ny=700)
display(fig)

fig = MANTA.manta_healpix_panels(
    rgb_pixels,
    scalar_pixels;
    titles=("RGB", "scalar"),
)
display(fig)
```

Exported HEALPix helpers:

- `manta_healpix`
- `manta_healpix_panels`
- `manta_healpix_cube`
- `is_healpix_fits`
- `read_healpix_map`
- `mollweide_grid`
- `mollweide_color_grid`
- `valid_healpix_npix`

## Development Commands

Install or update dependencies:

```bash
julia --project=. scripts/setup.jl
```

Run tests:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Check that the project loads:

```bash
julia --project=. -e 'push!(LOAD_PATH, "src"); using MANTA; println("MANTA OK")'
```

Show the Julia project status:

```bash
julia --project=. -e 'using Pkg; Pkg.status()'
```

## Interface Controls

3D FITS cube:

- `Slice axis`: choose the active slicing axis
- `Index`: change the current slice index
- left click: select a voxel
- `Region spectrum`: choose `point`, `box`, or `circle`; in box/circle mode, left-drag on the image to average the spectrum over that region
- arrow keys: move the cursor in the current slice
- right-click drag: zoom into a region
- `Reset zoom`: restore the full view
- `Image scale` and `Spectrum scale`: choose `lin`, `log10`, or `ln`
- `Invert colormap`: reverse the colormap
- `Gaussian filter` and `Gaussian sigma`: smooth the image
- `Colorbar limits`: apply manual color limits, automatic limits, or percentile contrast presets
- `Contours`: overlay automatic contours or provide manual contour levels; use `1:red, 2:#00ffaa, 3:blue` for per-level colors
- `Add dual`: load a second cube and compare `A`, `B`, `A - B`, `A / B`, or normalized residuals
- `Moment Maps`: compute and display/export moment 0, moment 1, or moment 2 along the active axis
- `Play`: animate channels interactively with FPS, loop, and ping-pong controls
- `Save image`: export the current slice
- `Save spectrum`: export the current spectrum
- `FITS Products`: export the current slice, averaged region spectrum, moment map, or Gaussian-filtered cube as FITS
- `Export GIF`: animate slices
- `Save settings` and `Load settings`: persist or restore the viewer state

HEALPix map:

- right-click drag: zoom in the Mollweide projection
- `Region`: choose `point`, `box`, or `circle`; in box/circle mode, left-drag on the map to compute a mean value over HEALPix pixels in that region
- `Reset zoom`: restore the global view
- `Scale`: choose `lin`, `log10`, or `ln`
- `Invert`: reverse the colormap
- `Colorbar` then `Apply`: apply manual color limits; `Auto`, `p1-p99`, and `p5-p95` adjust contrast from the current projected map values
- `Contours`: overlay automatic contours or provide manual contour levels, optionally with colors like `1:red, 2:#00ffaa`
- `Graticule`: show or hide the coordinate grid
- `Save PNG`: export the figure

HEALPix-PPV cube:

- `Channel`: change the spectral channel
- left click: select a HEALPix pixel and update the spectrum
- `Region`: choose `point`, `box`, or `circle`; in box/circle mode, left-drag on the map to show the mean spectrum of the selected HEALPix pixels
- `Scale`, `Invert`, `Colorbar`, `Graticule`, `Reset zoom`, `Save PNG`:
  same behavior as for a HEALPix map
- `Contours`: overlay automatic contours or provide manual contour levels, optionally with colors like `1:red, 2:#00ffaa`

## Exports

By default, exports are saved to the Desktop if that directory exists, otherwise
to the current working directory. To choose a directory explicitly:

```julia
fig = MANTA.manta("path/to/cube.fits"; save_dir="outputs")
```

Viewer settings can be persisted with:

```julia
fig = MANTA.manta(
    "path/to/cube.fits";
    settings_path="viewer_settings.toml",
)
```

## Troubleshooting

If no window opens:

```bash
julia --project=. -e 'using GLMakie; display(GLMakie.Figure()); sleep(2)'
```

If this command fails, the issue is probably with the graphical environment or
OpenGL, not with the FITS file itself.

If a package is missing or loading fails:

```bash
julia --project=. scripts/setup.jl
```

If `log10` or `ln` scales produce empty regions, check that the data is strictly
positive. Values `<= 0` are converted to `NaN` for these scales.

If GIF export fails, run the viewer from an active graphical session:

```bash
./manta path/to/cube.fits
```

## Repository Layout

```text
.
|-- Project.toml
|-- README.md
|-- manta
|-- demo/
|   `-- run_demo.jl
|-- scripts/
|   `-- setup.jl
|-- src/
|   |-- MANTA.jl
|   |-- MANTAHealpix.jl
|   `-- helpers/
|       `-- Helpers.jl
`-- test/
    `-- runtests.jl
```
