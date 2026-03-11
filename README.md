# CartaViewer

Interactive 3D FITS cube viewer in Julia (Makie/GLMakie), with:
- slice navigation across all 3 axes;
- per-voxel spectrum inspection;
- scale control (`lin`, `log10`, `ln`);
- colormap inversion;
- optional Gaussian smoothing;
- image/spectrum export (PNG/PDF) and GIF export.

## Prerequisites
- Julia `1.9` to `1.12` (see `Project.toml`).
- A working OpenGL context for `GLMakie` (local graphical session recommended).

## Installation
From the project root:

```bash
julia --project -e 'import Pkg; Pkg.instantiate()'
julia --project scripts/setup.jl
```

The `scripts/setup.jl` script activates the project, adds missing runtime dependencies, and precompiles.

## Quick Start (Demo)

```bash
./carta
```

The demo:
- generates a synthetic cube;
- writes `demo/output/synthetic_cube.fits`;
- opens the UI on that cube.

## CLI Launcher

`./carta` is the recommended entrypoint:

```bash
# Run demo
./carta

# Open an existing FITS cube
./carta /absolute/or/relative/path/to/cube.fits

# First-time setup + run
./carta --setup
```

Notes:
- The launcher accepts a single FITS path argument.
- `--setup` runs `scripts/setup.jl` before launching.

You can override cube size:

```bash
NX=96 NY=72 NZ=48 ./carta
```

Or via CLI arguments on the demo script:

```bash
julia --project demo/run_demo.jl 96 72 48
```

You can also set initial color limits:

```bash
VMIN=5 VMAX=1500 ./carta
```

## REPL Usage

```julia
julia --project
julia> push!(LOAD_PATH, "src")
julia> using CartaViewer
julia> fig = CartaViewer.carta("path/to/cube.fits"; figsize=(1400, 900))
```

## Main API

```julia
CartaViewer.carta(filepath::String;
    cmap::Symbol = :viridis,
    vmin = nothing,
    vmax = nothing,
    invert::Bool = false,
    figsize::Union{Nothing,Tuple{Int,Int}} = nothing,
    save_dir::Union{Nothing,AbstractString} = nothing,
    activate_gl::Bool = true,
    display_fig::Bool = true,
    settings_path::Union{Nothing,AbstractString} = nothing
)
```

### Parameters
- `filepath`: path to a 3D FITS file (`ndims == 3` is enforced).
- `cmap`: initial Makie colormap (`:viridis`, `:magma`, etc.).
- `vmin`, `vmax`: initial manual colorbar bounds (enabled only if both are provided).
- `invert`: initial colormap inversion.
- `figsize`: window size `(width, height)` in pixels. Internal default: `(1800, 900)`.
- `save_dir`: export directory. If `nothing`, exports go to `~/Desktop` if available, otherwise current directory.
- `activate_gl`: if `false`, use `CairoMakie` backend (useful for headless tests/CI).
- `display_fig`: if `false`, skip `display(fig)` (useful for smoke tests).
- `settings_path`: optional TOML path for `Save settings` / `Load settings`.

### Return Value
- Returns a Makie `Figure` (and displays it).

## UI Controls

### Navigation
- `Slice axis` menu: choose the slicing axis (1/2/3).
- `Index` slider: slice index on the active axis.
- Left click in the image: select a voxel.
- Arrow keys: move the cursor within the current slice.

### Visualization
- `Image scale`: `lin`, `log10`, `ln`.
- `Spectrum scale`: `lin`, `log10`, `ln`.
- `i` key or `Invert colormap` checkbox: invert colormap.
- `Gaussian filter` + sigma slider: smooth the image.
- `Colorbar limits` (`min`, `max`, `Apply`): manual bounds. Empty fields switch back to auto mode.
- `Reset zoom`: resets image zoom.

### Export
- `Save image`: exports the current slice (with colorbar/crosshair) as `png` or `pdf`.
- `Save spectrum`: exports the spectrum plot as `png` or `pdf`.
- `Export GIF`: animates over the active axis using `start/stop/step/fps`, with optional ping-pong mode (`Back-and-forth mode`).
- `Save settings` / `Load settings`: persist and restore UI state (colormap, clims, axis, index, scales).

Export filename behavior:
- Image: `<fitsname>_idx<idx>_axis<axis>.<ext>`
- Spectrum: `<fitsname>_spectrum_i<i>_j<j>_axis<axis>.<ext>`
- GIF: `<fitsname>.gif`
- Output directory: `save_dir` if provided; otherwise `~/Desktop` (if present), else current working directory.

## Exported Utility Functions

The module also exports helper utilities tested in `test/runtests.jl`:
- `apply_scale`, `clamped_extrema`
- `ijk_to_uv`, `uv_to_ijk`, `get_slice`
- `make_info_tex`, `latex_safe`, `make_main_title`, `make_slice_title`, `make_spec_title`
- `to_cmap`, `get_box_str`, `_pick_fig_size`

## Tests

```bash
julia --project -e 'using Pkg; Pkg.test()'
```

Tests mainly cover helper functions (scaling, mapping, LaTeX helpers, I/O helper, default UI size).

## Project Structure

```text
.
|-- src/
|   |-- CartaViewer.jl
|   `-- helpers/Helpers.jl
|-- demo/run_demo.jl
|-- scripts/setup.jl
`-- test/runtests.jl
```

## Troubleshooting
- No window appears: make sure you are not running headless without a GPU/OpenGL display.
- Log scales (`log10`, `ln`): values `<= 0` are converted to `NaN` to avoid infinities.
- GIF export: requires an active graphical context.
