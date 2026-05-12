# path: src/MANTA.jl
module MANTA

const _KEEP_ALIVE = Any[]
keepalive!(x) = (push!(_KEEP_ALIVE, x); x)
forget!(x) = (filter!(y -> y !== x, _KEEP_ALIVE); nothing)

using GLMakie
using CairoMakie
using Makie
using Observables
using ImageFiltering
using LaTeXStrings
using FITSIO
using ColorTypes
using FFTW

# ---- helpers ----
include("helpers/Helpers.jl")

# ---- HEALPix viewer ----
import Statistics: quantile
include("MANTAHealpix.jl")
export manta_healpix, manta_healpix_cube, is_healpix_fits,
       read_healpix_map, mollweide_grid, mollweide_color_grid,
       valid_healpix_npix, manta_healpix_panels

# ---- datasets + loaders ----
include("datasets/Datasets.jl")
include("loaders/FITSLoader.jl")
include("loaders/HDF5Loader.jl")
include("loaders/InMemoryLoader.jl")
include("datasets/LoadDataset.jl")

# ---- views ----
include("views/CubeView.jl")

export load_dataset
export AbstractMANTADataset, AbstractCartaDataset
export VectorDataset, ImageDataset, CubeDataset
export MultiChannelDataset, HealpixMapDataset, HealpixCubeDataset
export get_slice_view, get_slice_copy, as_float32, parse_path_spec
export stable_source_id
export view_cube

spawn_safely(f::Function) = @async try f() catch e
    @error "Background task failed" exception=(e, catch_backtrace())
end

export manta, manta_panels

"""
    manta(filepath::String; kwargs...)

Interactive FITS viewer. **Dispatches automatically** based on file content :

- **3D cube** → slice + per-voxel spectrum viewer (default behavior).
- **HEALPix map** (header has `PIXTYPE = 'HEALPIX'`) → Mollweide
  projection with right-drag zoom (delegates to `manta_healpix`).

Common kwargs:
- `cmap`, `vmin`, `vmax`, `invert`, `figsize`, `save_dir`,
  `activate_gl`, `display_fig`.

Cube-only kwargs:
- `settings_path`.

HEALPix-only kwargs:
- `column` : column index in the BinTable (default 1).
- `nx`, `ny` : Mollweide grid resolution (default 1400×700).
- `scale` : `:lin | :log10 | :ln` (default `:lin`).

Notes:
- Manual color limits when `vmin` & `vmax` set (also sync spectrum Y for
  cubes).
- Window sized by explicit `figsize=(w,h)` or a fallback default.
- Export directory configurable via `save_dir`; defaults to your Desktop
  if it exists, otherwise the current working directory.
- `activate_gl=false` allows smoke tests without requiring an OpenGL
  context.
- `display_fig=false` skips window display (useful for automated tests).
"""
function manta(
    filepath::String;
    cmap::Symbol = :viridis,
    vmin = nothing,
    vmax = nothing,
    invert::Bool = false,
    figsize::Union{Nothing,Tuple{Int,Int}} = nothing,
    save_dir::Union{Nothing,AbstractString} = nothing,
    activate_gl::Bool = true,
    display_fig::Bool = true,
    settings_path::Union{Nothing,AbstractString} = nothing,
    rgb::Bool = false,
    # HEALPix-specific options (ignorés pour les cubes 3D)
    column::Int = 1,
    nx::Int = 1400,
    ny::Int = 700,
    scale::Symbol = :lin,
    # HEALPix PPV cube (npix×nv) — axe vitesse pour le spectre
    v0::Real = 0.0,
    dv::Real = 1.0,
    vunit::AbstractString = "km/s",
    )

    # ---------- Load ----------
    if !isfile(filepath)
        throw(ArgumentError("FITS file not found: $(abspath(filepath))"))
    end

    # On lit l'image primaire UNE fois : sert à la fois pour détecter un
    # cube HEALPix-PPV 2D et pour les cubes 3D classiques. La détection
    # HEALPix-image-1D (BinTable) reste basée sur les headers.
    header = nothing
    cube = try
        FITS(filepath) do f
            header = try
                read_header(f[1])
            catch
                nothing
            end
            read(f[1])
        end
    catch e
        nothing  # primary HDU peut être vide pour HEALPix BinTable → on tolère
    end

    # ---- Dispatch HEALPix PPV cube (2D, une dim = 12·nside²) ----
    # Important : tester AVANT `is_healpix_fits` car nos cubes embarquent
    # `PIXTYPE=HEALPIX` dans le header primaire pour transporter le WCS,
    # ce qui ferait sinon basculer sur le viewer carte 1D.
    if cube !== nothing && ndims(cube) == 2
        if rgb
            @info "Detected explicit RGB HEALPix pixels → using manta_healpix"
            return manta_healpix(cube;
                title=String(replace(basename(filepath), r"\.fits(\.gz)?$" => "")),
                nx=nx, ny=ny, figsize=figsize,
                activate_gl=activate_gl, display_fig=display_fig)
        end
        s = size(cube)
        if valid_healpix_npix(s[1]) > 0 || valid_healpix_npix(s[2]) > 0
            @info "Detected HEALPix PPV cube → using manta_healpix_cube"
            return manta_healpix_cube(filepath;
                cmap=(cmap === :viridis ? :inferno : cmap),
                vmin=vmin, vmax=vmax, invert=invert, scale=scale,
                v0=v0, dv=dv, vunit=vunit,
                nx=nx, ny=ny,
                figsize=figsize, save_dir=save_dir,
                activate_gl=activate_gl, display_fig=display_fig)
        end
    end

    # ---- Dispatch carte HEALPix 1D (BinTable) ----
    if is_healpix_fits(filepath)
        @info "Detected HEALPix map → using manta_healpix"
        return manta_healpix(filepath;
            cmap=(cmap === :viridis ? :inferno : cmap),
            vmin=vmin, vmax=vmax, invert=invert,
            scale=scale, column=column, nx=nx, ny=ny,
            figsize=figsize, save_dir=save_dir,
            activate_gl=activate_gl, display_fig=display_fig)
    end

    if cube === nothing
        throw(ArgumentError("Failed to read primary HDU of $(abspath(filepath))."))
    end

    if rgb
        return manta(
            as_rgb_image(cube);
            title=String(replace(basename(filepath), r"\.fits(\.gz)?$" => "")),
            figsize=figsize,
            activate_gl=activate_gl,
            display_fig=display_fig,
        )
    end

    if ndims(cube) == 2
        return manta(
            Float32.(cube);
            title=String(replace(basename(filepath), r"\.fits(\.gz)?$" => "")),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            invert=invert,
            scale=scale,
            figsize=figsize,
            save_dir=save_dir,
            activate_gl=activate_gl,
            display_fig=display_fig,
            unit_label=data_unit_label(header; fallback = "value"),
        )
    end

    if ndims(cube) != 3
        throw(ArgumentError("Expected a 3D FITS cube, got ndims=$(ndims(cube)) and size=$(size(cube))."))
    end

    # 3D FITS cube → build a CubeDataset and hand off to the cube viewer.
    # The header and original FITS path are stashed in `metadata` so the
    # viewer can reuse them for comparison-FITS path resolution, FITS
    # product export, and default save filenames.
    cube_data = as_float32(cube)
    wcs       = header === nothing ? SimpleWCSAxis[] : read_simple_wcs(header, 3)
    unit_lbl  = data_unit_label(header; fallback = "value")
    axis_lbls = String[wcs_axis_label(wcs, d; fallback = "axis $d") for d in 1:3]
    src_id    = String(replace(basename(filepath), r"\.fits(\.gz)?$"i => ""))
    metadata  = Dict{Symbol,Any}(:fits_path => abspath(filepath))
    if header !== nothing
        metadata[:fits_header] = header
    end
    ds = CubeDataset(
        cube_data;
        axis_labels = axis_lbls,
        wcs         = wcs,
        unit_label  = unit_lbl,
        source_id   = src_id,
        metadata    = metadata,
    )
    return _view_cube(
        ds;
        cmap = cmap,
        vmin = vmin,
        vmax = vmax,
        invert = invert,
        figsize = figsize,
        save_dir = save_dir,
        activate_gl = activate_gl,
        display_fig = display_fig,
        settings_path = settings_path,
    )
end

function manta(
    img::AbstractMatrix{<:Real};
    title::AbstractString = "2D image",
    cmap::Symbol = :viridis,
    vmin = nothing,
    vmax = nothing,
    invert::Bool = false,
    scale::Symbol = :lin,
    figsize::Union{Nothing,Tuple{Int,Int}} = nothing,
    save_dir::Union{Nothing,AbstractString} = nothing,
    activate_gl::Bool = true,
    display_fig::Bool = true,
    unit_label::AbstractString = "value",
)
    data2d = Float32.(img)
    unit_label_tex = latexstring("\\text{", latex_safe(unit_label), "}")

    cmap_name = Observable(cmap)
    invert_cmap = Observable(invert)
    cm_obs = lift(cmap_name, invert_cmap) do name, inv
        base = to_cmap(name); inv ? reverse(base) : base
    end
    scale_mode = Observable(scale)
    img_disp = lift(scale_mode) do m
        A = apply_scale(data2d, m)
        out = similar(A, Float32)
        @inbounds for i in eachindex(A)
            x = A[i]
            out[i] = isfinite(x) ? Float32(x) : 0f0
        end
        out
    end
    clims_auto = lift(img_disp) do im
        clamped_extrema(im)
    end
    clims_manual = Observable((0f0, 1f0))
    use_manual = Observable(false)
    if vmin !== nothing && vmax !== nothing
        a, b = Float32(vmin), Float32(vmax)
        a == b && (a = prevfloat(a); b = nextfloat(b))
        clims_manual[] = (a, b)
        use_manual[] = true
    end
    clims_obs = lift(use_manual, clims_auto, clims_manual) do um, ca, cm
        um ? cm : ca
    end
    clims_safe = lift(clims_obs) do (lo, hi)
        (isfinite(lo) && isfinite(hi) && lo != hi) ? (lo, hi) : (0f0, 1f0)
    end
    hist_pair_obs = lift(img_disp, clims_safe) do im, lim
        histogram_counts(im; bins = 64, limits = lim)
    end
    hist_x_obs = lift(p -> p[1], hist_pair_obs)
    hist_y_obs = lift(p -> p[2], hist_pair_obs)

    fig_bg_panels = RGBf(0.97, 0.975, 0.985)
    activate_gl ? GLMakie.activate!() : CairoMakie.activate!()
    fig = Figure(size = _pick_fig_size(figsize), backgroundcolor = fig_bg_panels)
    grid = fig[1, 1] = GridLayout()
    colgap!(grid, 16); rowgap!(grid, 14)
    img_grid = grid[1, 1] = GridLayout()
    colgap!(img_grid, -8)
    ax = Axis(
        img_grid[1, 1];
        title = make_main_title(title),
        xlabel = L"\text{pixel x}",
        ylabel = L"\text{pixel y}",
        aspect = DataAspect(),
    )
    hm = heatmap!(ax, img_disp; colormap = cm_obs, colorrange = clims_safe)
    Colorbar(
        img_grid[1, 2],
        hm;
        label = unit_label_tex,
        width = 20,
        height = _axis_render_height(ax),
        tellheight = false,
        valign = :center,
    )

    # Aligned with the modern indigo palette of `manta`
    ui_accent         = RGBf(0.36, 0.39, 0.92)
    ui_accent_dim     = RGBf(0.62, 0.64, 0.96)
    ui_accent_strong  = RGBf(0.28, 0.31, 0.82)
    ui_border         = RGBf(0.78, 0.81, 0.88)
    ui_surface        = RGBf(0.985, 0.988, 0.996)
    ui_surface_hover  = RGBf(0.94, 0.95, 0.99)
    ui_surface_active = RGBf(0.90, 0.92, 0.98)
    ui_text           = RGBf(0.10, 0.12, 0.20)
    ui_text_muted     = RGBf(0.42, 0.46, 0.56)

    ax_hist = Axis(
        grid[2, 1];
        title = L"\text{Image histogram}",
        xlabel = unit_label_tex,
        ylabel = L"\text{count}",
        height = 130,
    )
    lines!(ax_hist, hist_x_obs, hist_y_obs; color = ui_accent, linewidth = 1.6)
    vlines!(ax_hist, lift(lim -> [first(lim), last(lim)], clims_safe); color = (ui_text_muted, 0.65), linewidth = 1.1, linestyle = :dash)

    ctrl = grid[3, 1] = GridLayout(; alignmode = Outside())
    Label(ctrl[1, 1], text = "Image", halign = :left, tellwidth = false, fontsize = 14, color = ui_text_muted)
    scale_menu = Menu(ctrl[1, 2]; options = ["lin", "log10", "ln"], prompt = String(scale), width = 96)
    invert_chk = Checkbox(ctrl[1, 3])
    Label(ctrl[1, 4], text = "Invert", halign = :left, tellwidth = false, fontsize = 14, color = ui_text)
    clim_min_box = Textbox(ctrl[1, 5]; placeholder = "min", width = 110, height = 32)
    clim_max_box = Textbox(ctrl[1, 6]; placeholder = "max", width = 110, height = 32)
    apply_btn = Button(ctrl[1, 7]; label = "Apply", width = 82, height = 32)
    auto_btn = Button(ctrl[1, 8]; label = "Auto", width = 78, height = 32)
    p1_btn = Button(ctrl[1, 9]; label = "p1-p99", width = 92, height = 32)
    p5_btn = Button(ctrl[1, 10]; label = "p5-p95", width = 92, height = 32)
    save_btn = Button(ctrl[1, 11]; label = "Save PNG", width = 108, height = 32)
    ui_status = Observable(" ")
    grid[4, 1] = Label(grid[4, 1]; text = ui_status, halign = :left, tellwidth = false)

    style_button_local!(btn) = begin
        btn.height[] = 34
        btn.cornerradius[] = 8
        btn.strokewidth[] = 1.0
        btn.strokecolor[] = ui_border
        btn.buttoncolor[] = ui_surface
        btn.buttoncolor_hover[] = ui_surface_hover
        btn.buttoncolor_active[] = ui_surface_active
        btn.labelcolor[] = ui_text
        btn.labelcolor_hover[] = ui_accent_strong
        btn.labelcolor_active[] = ui_accent_strong
        btn.fontsize[] = 14
        btn.padding[] = (12, 12, 7, 7)
        btn
    end
    foreach(style_button_local!, (apply_btn, auto_btn, p1_btn, p5_btn, save_btn))
    invert_chk.checked[] = invert

    set_status!(msg::AbstractString) = (ui_status[] = String(msg); nothing)
    set_box_text!(tb, s::AbstractString) = begin
        str = String(s)
        tb.displayed_string[] = str
        tb.stored_string[] = str
        nothing
    end
    function apply_percentile_clims!(lo::Real, hi::Real)
        parsed = percentile_clims(img_disp[], lo, hi)
        clims_manual[] = parsed
        use_manual[] = true
        set_box_text!(clim_min_box, string(first(parsed)))
        set_box_text!(clim_max_box, string(last(parsed)))
        set_status!("Colorbar limits set to p$(lo)-p$(hi).")
    end

    on(scale_menu.selection) do sel
        sel === nothing && return
        scale_mode[] = Symbol(sel)
    end
    on(invert_chk.checked) do v
        invert_cmap[] = v
    end
    on(apply_btn.clicks) do _
        ok, manual, parsed, msg = parse_manual_clims(
            get_box_str(clim_min_box),
            get_box_str(clim_max_box);
            fallback = clims_manual[],
        )
        set_status!(msg)
        ok || return
        if manual
            clims_manual[] = parsed
            use_manual[] = true
            set_box_text!(clim_min_box, string(first(parsed)))
            set_box_text!(clim_max_box, string(last(parsed)))
        else
            use_manual[] = false
        end
    end
    on(auto_btn.clicks) do _
        use_manual[] = false
        set_box_text!(clim_min_box, "")
        set_box_text!(clim_max_box, "")
        set_status!("Automatic color limits enabled.")
    end
    on(p1_btn.clicks) do _; apply_percentile_clims!(1, 99); end
    on(p5_btn.clicks) do _; apply_percentile_clims!(5, 95); end

    save_root = if save_dir === nothing
        d = joinpath(homedir(), "Desktop")
        isdir(d) ? d : pwd()
    else
        path = String(save_dir)
        isdir(path) || mkpath(path)
        path
    end
    on(save_btn.clicks) do _
        out = joinpath(save_root, "$(title)_image2d.png")
        try
            CairoMakie.save(String(out), fig; backend = CairoMakie)
            set_status!("Saved image to $(out).")
        catch e
            msg = "Failed to save image: $(sprint(showerror, e))"
            set_status!(msg)
            @error msg exception=(e, catch_backtrace())
        end
    end

    keepalive!(fig)
    on(fig.scene.events.window_open) do is_open
        is_open || forget!(fig)
    end
    display_fig && display(fig)
    return fig
end

function manta(
    img::AbstractArray;
    title::AbstractString = "RGB image",
    xlabel = nothing,
    ylabel = nothing,
    figsize::Union{Nothing,Tuple{Int,Int}} = nothing,
    activate_gl::Bool = true,
    display_fig::Bool = true,
)
    rgb_img = as_rgb_image(img)
    activate_gl ? GLMakie.activate!() : CairoMakie.activate!()
    fig = Figure(size = _pick_fig_size(figsize))
    ax = Axis(
        fig[1, 1];
        title = make_main_title(title),
        xlabel = xlabel === nothing ? L"\text{pixel x}" : xlabel,
        ylabel = ylabel === nothing ? L"\text{pixel y}" : ylabel,
        aspect = DataAspect(),
    )
    rows, cols = size(rgb_img)
    image!(ax, (1, cols), (1, rows), permutedims(rgb_img))
    keepalive!(fig)
    on(fig.scene.events.window_open) do is_open
        is_open || forget!(fig)
    end
    display_fig && display(fig)
    return fig
end

function manta_panels(
    panels::Vararg{Any,N};
    titles = nothing,
    cmaps = nothing,
    clims = nothing,
    figsize::Union{Nothing,Tuple{Int,Int}} = nothing,
    activate_gl::Bool = true,
    display_fig::Bool = true,
) where {N}
    N >= 1 || throw(ArgumentError("Provide at least one panel."))
    activate_gl ? GLMakie.activate!() : CairoMakie.activate!()
    fig = Figure(size = _pick_fig_size(figsize))
    title_at(i) = titles === nothing ? "panel $(i)" : String(titles[i])
    cmap_at(i) = cmaps === nothing ? :viridis : cmaps[i]
    clim_at(i, vals) = clims === nothing ? clamped_extrema(vals) : clims[i]
    for (i, panel) in enumerate(panels)
        panel_grid = fig[1, i] = GridLayout()
        colgap!(panel_grid, -8)
        ax = Axis(
            panel_grid[1, 1];
            title = make_main_title(title_at(i)),
            aspect = DataAspect(),
        )
        if is_rgb_like(panel)
            img = as_rgb_image(panel)
            rows, cols = size(img)
            image!(ax, (1, cols), (1, rows), permutedims(img))
        else
            vals = Float32.(panel)
            hm = heatmap!(ax, vals; colormap = cmap_at(i), colorrange = clim_at(i, vals))
            Colorbar(
                panel_grid[1, 2],
                hm;
                width = 16,
                height = _axis_render_height(ax),
                tellheight = false,
                valign = :center,
            )
        end
    end
    keepalive!(fig)
    on(fig.scene.events.window_open) do is_open
        is_open || forget!(fig)
    end
    display_fig && display(fig)
    return fig
end

# ----------------------------------------------------------------------------
# Dataset-aware dispatch.
#
# These methods turn `manta(x)` into "load_dataset(x) → view it" for any input
# that isn't already a path, a 2D array or an RGB array (those keep their
# original methods above). `CubeDataset` is handled by `_view_cube` (alias
# `view_cube`) from `views/CubeView.jl`; HEALPix datasets currently still
# round-trip via their FITS path when one is available in metadata.
# ----------------------------------------------------------------------------

"""
    manta(ds::AbstractMANTADataset; kwargs...)

Dispatch a pre-built MANTA dataset to the matching viewer.
"""
function manta(ds::ImageDataset; kwargs...)
    return manta(ds.data;
        title = ds.source_id,
        unit_label = ds.unit_label,
        kwargs...)
end

function manta(ds::HealpixMapDataset; kwargs...)
    fits_path = get(ds.metadata, :fits_path, nothing)
    if fits_path isa AbstractString && isfile(fits_path)
        return manta_healpix(String(fits_path); kwargs...)
    end
    throw(ErrorException(
        "MANTA: viewing an in-memory HealpixMapDataset is not yet wired up. " *
        "Load via `manta(\"map.fits\")` so the file is available to the viewer, " *
        "or wait for Phase 2 which adds `view_healpix_map(::HealpixMapDataset)`."))
end

function manta(ds::HealpixCubeDataset; kwargs...)
    fits_path = get(ds.metadata, :fits_path, nothing)
    if fits_path isa AbstractString && isfile(fits_path)
        return manta_healpix_cube(String(fits_path);
            v0 = ds.v0, dv = ds.dv, vunit = ds.vunit, kwargs...)
    end
    throw(ErrorException(
        "MANTA: viewing an in-memory HealpixCubeDataset is not yet wired up. " *
        "Load via `manta(\"cube.fits\")` so the file is available to the viewer."))
end

function manta(ds::CubeDataset; kwargs...)
    return _view_cube(ds; kwargs...)
end

function manta(ds::VectorDataset; kwargs...)
    throw(ErrorException(
        "MANTA: viewing a VectorDataset is not implemented yet. " *
        "Plot the vector externally with `lines(ds.data)` for now."))
end

function manta(ds::MultiChannelDataset; kwargs...)
    if ds.kind === :image
        panels = [ds.channels[k].data for k in sort!(collect(keys(ds.channels)))]
        titles = [String(k) for k in sort!(collect(keys(ds.channels)))]
        return manta_panels(panels...; titles = titles, kwargs...)
    end
    throw(ErrorException(
        "MANTA: viewing MultiChannelDataset of kind $(ds.kind) is not implemented yet."))
end

# Bridges for inputs that are NOT already paths / numeric arrays. These are
# safe to add: existing `manta(::AbstractMatrix{<:Real})`, `manta(::AbstractArray)`
# (RGB) and `manta(filepath::String)` remain the most-specific matches for
# their respective inputs.
manta(x::NamedTuple; kwargs...) = manta(load_dataset(x); kwargs...)
manta(x::AbstractDict; kwargs...) = manta(load_dataset(x); kwargs...)
manta(x::Healpix.HealpixMap; kwargs...) = manta(load_dataset(x); kwargs...)

# 3D numeric arrays: route through load_dataset → CubeDataset → _view_cube,
# so an in-memory cube gets the full interactive viewer (slice navigation,
# spectra, moments, comparison, power spectrum, exports) without ever
# touching disk.
manta(x::AbstractArray{<:Real,3}; kwargs...) = manta(load_dataset(x); kwargs...)

end # module
