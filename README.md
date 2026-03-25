# CartaViewer

Interactive 3D FITS cube viewer in Julia with Makie/GLMakie.

Main features:
- navigate slices along the 3 axes;
- inspect the spectrum of a selected voxel;
- switch between `lin`, `log10`, and `ln` scales;
- invert the colormap;
- apply optional Gaussian smoothing;
- export images, spectra, and GIFs.

## Requirements

- Julia `1.9` to `1.12`
- a local graphical session with OpenGL support for `GLMakie`

## Installation

From the project root:

```bash
julia --project -e 'import Pkg; Pkg.instantiate()'
julia --project scripts/setup.jl
```

## Quick Start

Run the demo:

```bash
./carta
```

This generates a synthetic FITS cube in `demo/output/synthetic_cube.fits` and opens the viewer.

Open an existing FITS cube:

```bash
./carta path/to/cube.fits
```

First-time setup and launch:

```bash
./carta --setup
```

Optional demo parameters:

```bash
NX=96 NY=72 NZ=48 ./carta
VMIN=5 VMAX=1500 ./carta
```

## Julia Usage

```julia
julia --project
julia> push!(LOAD_PATH, "src")
julia> using CartaViewer
julia> fig = CartaViewer.carta("path/to/cube.fits"; figsize=(1400, 900))
```

Useful keyword arguments:
- `cmap`
- `vmin`, `vmax`
- `invert`
- `figsize`
- `save_dir`
- `settings_path`

## Controls

- `Slice axis`: choose the active axis
- `Index`: move through slices
- left click: select a voxel
- arrow keys: move the cursor in the current slice
- `Image scale` / `Spectrum scale`: switch scale mode
- `Invert colormap`: reverse the colormap
- `Gaussian filter`: smooth the image
- `Colorbar limits`: set manual bounds
- `Save image` / `Save spectrum`: export PNG or PDF
- `Export GIF`: animate along the active axis
- `Save settings` / `Load settings`: persist UI state

## Tests

```bash
julia --project -e 'using Pkg; Pkg.test()'
```

## Troubleshooting

- If no window appears, check that you are not running headless and that OpenGL is available.
- For `log10` and `ln`, values `<= 0` are converted to `NaN`.
- GIF export requires an active graphical session.
