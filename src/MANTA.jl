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

    data = Float32.(cube)
    siz  = size(data)  # (nx, ny, nz)
    wcs = header === nothing ? SimpleWCSAxis[] : read_simple_wcs(header, 3)
    unit_label = data_unit_label(header; fallback = "value")
    unit_label_tex = latexstring("\\text{", latex_safe(unit_label), "}")
    
    slice_dims(axis::Integer) = if axis == 1
        (siz[2], siz[3])  # (y, z)
    elseif axis == 2
        (siz[1], siz[3])  # (x, z)
    else
        (siz[1], siz[2])  # (x, y)
    end

    slice_axis_dims(axis::Integer) = if axis == 1
        (2, 3)  # u=y, v=z
    elseif axis == 2
        (1, 3)  # u=x, v=z
    else
        (1, 2)  # u=x, v=y
    end

    pixel_axis_name(dim::Integer) = dim == 1 ? "pixel x" : dim == 2 ? "pixel y" : "pixel z"

    # Heatmap x-axis is v (columns) and y-axis is u (rows), with (u,v) from ijk_to_uv.
    slice_axis_labels(axis::Integer) = begin
        u_dim, v_dim = slice_axis_dims(axis)
        (
            wcs_axis_label(wcs, v_dim; fallback = pixel_axis_name(v_dim)),
            wcs_axis_label(wcs, u_dim; fallback = pixel_axis_name(u_dim)),
        )
    end

    pixel_world_tick_formatter(dim::Integer) = vals -> [
        has_wcs(wcs, dim) ? latex_tick(world_coord(wcs, dim, v)) : latex_tick(v)
        for v in vals
    ]

    spectral_coords(dim::Integer) = Float32[
        has_wcs(wcs, dim) ? Float32(world_coord(wcs, dim, chan)) : Float32(chan)
        for chan in 1:siz[dim]
    ]

    fname_full = basename(filepath)
    fname = String(replace(fname_full, r"\.fits$" => ""))

    @info "FITS ready" path=abspath(filepath) size=siz

    # ---------- State ----------
    axis   = Observable(3)          # 1/2/3
    idx    = Observable(1)          # slice index

    i_idx  = Observable(1)          # voxel indices
    j_idx  = Observable(1)
    k_idx  = Observable(1)

    u_idx  = Observable(1)          # row
    v_idx  = Observable(1)          # col

    cmap_name   = Observable(cmap)
    invert_cmap = Observable(invert)
    cm_obs = lift(cmap_name, invert_cmap) do name, inv
        base = to_cmap(name); inv ? reverse(base) : base
    end

    img_scale_mode  = Observable(:lin)
    spec_scale_mode = Observable(:lin)
    compare_data = Observable{Any}(nothing)
    compare_visible = Observable(false)
    compare_name = Observable("")
    compare_mode = Observable(:B)
    view_product = Observable(:slice)
    moment_order = Observable(0)
    anim_playing = Observable(false)

    slice_raw = lift(axis, idx) do a, id
        get_slice(data, a, clamp(id, 1, siz[a]))
    end

    compare_slice_raw = lift(compare_data, axis, idx) do cmp, a, id
        cmp === nothing && return fill(NaN32, slice_dims(a))
        get_slice(cmp, a, clamp(id, 1, siz[a]))
    end

    gauss_on = Observable(false)
    sigma    = Observable(1.5f0)
    show_crosshair = Observable(true)
    show_marker    = Observable(true)
    show_grid      = Observable(false)
    show_contours  = Observable(false)
    contour_use_manual = Observable(false)
    contour_manual_levels = Observable(Float32[])
    contour_manual_colors = Observable(String[])
    selection_mode = Observable(:point)
    region_shape = Observable(:box)
    region_uvs = Observable(Tuple{Int,Int}[])
    region_start = Observable(Point2f(NaN32, NaN32))
    region_end = Observable(Point2f(NaN32, NaN32))
    region_drag_active = Observable(false)
    zoom_drag_active = Observable(false)
    zoom_drag_start  = Observable(Point2f(NaN32, NaN32))
    zoom_drag_end    = Observable(Point2f(NaN32, NaN32))

    # Modern UI palette — indigo accent on cool neutral surfaces
    ui_accent         = RGBf(0.36, 0.39, 0.92)    # indigo-500
    ui_accent_dim     = RGBf(0.62, 0.64, 0.96)    # indigo-300
    ui_accent_strong  = RGBf(0.28, 0.31, 0.82)    # indigo-700 (hover/active)
    ui_track          = RGBf(0.88, 0.90, 0.95)    # slate-200
    ui_surface        = RGBf(0.985, 0.988, 0.996) # near-white card
    ui_surface_hover  = RGBf(0.94, 0.95, 0.99)
    ui_surface_active = RGBf(0.90, 0.92, 0.98)
    ui_panel          = RGBf(0.965, 0.970, 0.985) # softer slate panel
    ui_panel_header   = RGBf(0.93, 0.94, 0.97)
    ui_border         = RGBf(0.78, 0.81, 0.88)
    ui_border_strong  = RGBf(0.62, 0.66, 0.76)
    ui_text           = RGBf(0.10, 0.12, 0.20)
    ui_text_muted     = RGBf(0.42, 0.46, 0.56)
    fig_bg            = RGBf(0.97, 0.975, 0.985)

    style_checkbox!(chk) = begin
        chk.size[] = 22
        chk.checkmarksize[] = 0.62
        chk.roundness[] = 0.5
        chk.checkboxstrokewidth[] = 1.4
        chk.checkboxcolor_checked[] = ui_accent
        chk.checkboxcolor_unchecked[] = RGBf(0.96, 0.965, 0.985)
        chk.checkboxstrokecolor_checked[] = ui_accent_strong
        chk.checkboxstrokecolor_unchecked[] = ui_border
        chk.checkmarkcolor_checked[] = :white
        chk.checkmarkcolor_unchecked[] = RGBf(0.65, 0.70, 0.78)
        chk
    end

    # Sliders : track plus large, accent indigo, contraste fort
    style_slider!(sl) = begin
        sl.height[] = 26
        sl.linewidth[] = 10
        sl.color_active[] = ui_accent
        sl.color_active_dimmed[] = ui_accent_dim
        sl.color_inactive[] = ui_track
        sl
    end

    style_button!(btn) = begin
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

    style_menu!(menu) = begin
        menu.height[] = 34
        menu.width[] = max(menu.width[], 96)
        menu.textcolor[] = ui_text
        menu.fontsize[] = 14
        menu.dropdown_arrow_color[] = ui_accent
        menu.dropdown_arrow_size[] = 11
        menu.textpadding[] = (10, 10, 7, 7)
        menu.cell_color_inactive_even[] = ui_surface
        menu.cell_color_inactive_odd[] = ui_surface
        menu.selection_cell_color_inactive[] = ui_surface
        menu.cell_color_hover[] = ui_surface_hover
        menu.cell_color_active[] = ui_surface_active
        menu
    end

    style_textbox!(tb) = begin
        tb.height[] = 34
        tb.fontsize[] = 14
        tb.textcolor[] = ui_text
        tb.textcolor_placeholder[] = ui_text_muted
        tb.boxcolor[] = ui_surface
        tb.boxcolor_hover[] = ui_surface_hover
        tb.boxcolor_focused[] = RGBf(1.0, 1.0, 1.0)
        tb.bordercolor[] = ui_border
        tb.bordercolor_hover[] = ui_accent_dim
        tb.bordercolor_focused[] = ui_accent
        tb.borderwidth[] = 1.4
        tb.cornerradius[] = 8
        tb.textpadding[] = (10, 10, 7, 7)
        tb
    end

    latex_tick(v::Real) = begin
        x = abs(Float64(v)) < 1e-10 ? 0.0 : Float64(v)
        r = round(x)
        s = if abs(x - r) < 1e-8
            string(Int(r))
        else
            string(round(x; digits = 2))
        end
        latexstring("\\mathrm{", latex_safe(s), "}")
    end
    latex_tick_formatter(vals) = [latex_tick(v) for v in vals]

    base_slice_proc = lift(slice_raw, gauss_on, sigma) do s, on, σ
        if on && σ > 0
            k = ImageFiltering.Kernel.gaussian((σ, σ))
            imfilter(Float32.(s), k)
        else
            s
        end
    end

    moment_raw = lift(axis, moment_order) do a, ord
        moment_map(data, a, ord; coords = spectral_coords(a))
    end

    view_raw = lift(slice_raw, moment_raw, view_product) do s, m, product
        product === :moment ? m : s
    end

    slice_proc = lift(view_raw, gauss_on, sigma, view_product) do s, on, σ, product
        if product === :slice && on && σ > 0
            k = ImageFiltering.Kernel.gaussian((σ, σ))
            imfilter(Float32.(s), k)
        else
            s
        end
    end

    compare_slice_proc = lift(compare_slice_raw, gauss_on, sigma) do s, on, σ
        if on && σ > 0
            k = ImageFiltering.Kernel.gaussian((σ, σ))
            imfilter(Float32.(s), k)
        else
            s
        end
    end

    compare_product_proc = lift(base_slice_proc, compare_slice_proc, compare_mode, compare_data) do a, b, mode, cmp
        cmp === nothing && return fill(NaN32, size(a))
        dual_view_product(a, b, mode)
    end

    # ---- display array: protect against NaN/Inf after log/ln
    slice_disp = lift(slice_proc, img_scale_mode) do s, m
        A = apply_scale(s, m)
        out = similar(A, Float32)
        @inbounds for i in eachindex(A)
            x = A[i]
            out[i] = isfinite(x) ? Float32(x) : 0f0
        end
        out
    end

    compare_slice_disp = lift(compare_product_proc, img_scale_mode) do s, m
        A = apply_scale(s, m)
        out = similar(A, Float32)
        @inbounds for i in eachindex(A)
            x = A[i]
            out[i] = isfinite(x) ? Float32(x) : 0f0
        end
        out
    end

    clims_auto = lift(slice_disp) do s
        clamped_extrema(s)
    end

    contour_auto_levels = lift(slice_disp) do s
        automatic_contour_levels(s; n = 7)
    end

    contour_levels_obs = lift(contour_use_manual, contour_manual_levels, contour_auto_levels) do use_man, manual, auto
        use_man && !isempty(manual) ? manual : auto
    end
    contour_default_color = RGBAf(1, 1, 1, 0.72)
    contour_colors_obs = lift(contour_levels_obs, contour_use_manual, contour_manual_colors) do levels, use_man, colors
        contour_color_values(use_man ? colors : String[], length(levels), contour_default_color)
    end

    hist_pair_obs = lift(slice_disp, clims_auto) do s, lim
        histogram_counts(s; bins = 64, limits = lim)
    end
    hist_x_obs = lift(p -> p[1], hist_pair_obs)
    hist_y_obs = lift(p -> p[2], hist_pair_obs)

    clims_manual = Observable((0f0, 1f0))
    use_manual   = Observable(false)

    if vmin !== nothing && vmax !== nothing
        vmin_f, vmax_f = Float32(vmin), Float32(vmax)
        if vmin_f == vmax_f
            vmin_f = prevfloat(vmin_f); vmax_f = nextfloat(vmax_f)  # avoid zero-width
        end
        clims_manual[] = (vmin_f, vmax_f)
        use_manual[]   = true
    end

    clims_obs = lift(use_manual, clims_auto, clims_manual) do um, ca, cm
        um ? cm : ca
    end

    # safe clims for plotting/layout
    clims_safe = lift(clims_obs) do (cmin, cmax)
        if !(isfinite(cmin) && isfinite(cmax)) || isnan(cmin) || isnan(cmax) || cmin == cmax
            (0f0, 1f0)
        else
            (cmin, cmax)
        end
    end

    compare_clims_safe = lift(compare_slice_disp, compare_mode, clims_safe) do s, mode, lim
        if mode in (:A, :B)
            lim
        else
            clamped_extrema(s)
        end
    end

    spec_x_axes = (collect(0:(siz[1] - 1)), collect(0:(siz[2] - 1)), collect(0:(siz[3] - 1)))
    spec_y_buf  = Vector{Float32}(undef, siz[3])
    @views copyto!(spec_y_buf, data[1, 1, :])
    spec_x_raw  = Observable(spec_x_axes[3])
    spec_y_raw  = Observable(spec_y_buf)
    spec_y_disp = lift(spec_y_raw, spec_scale_mode) do y, m
        apply_scale(y, m)
    end

    # ---------- Figure & layout ----------
    if activate_gl
        GLMakie.activate!()
    else
        CairoMakie.activate!()
    end
    fig = Figure(size = _pick_fig_size(figsize), backgroundcolor = fig_bg)

    main_grid = fig[1, 1] = GridLayout()
    colgap!(main_grid, 18)
    rowgap!(main_grid, 14)
    # Image + colorbar
    img_grid  = main_grid[1, 1] = GridLayout()
    colgap!(img_grid, 8)

    xlab0, ylab0 = slice_axis_labels(axis[])
    main_title_obs = lift(view_product, moment_order) do product, order
        product === :moment ?
            latexstring("\\text{", latex_safe(fname), " ", latex_safe(order == 0 ? "moment 0" : order == 1 ? "moment 1" : "moment 2"), "}") :
            make_main_title(fname)
    end
    display_unit_label = lift(view_product, moment_order) do product, order
        product === :moment ?
            latexstring("\\text{", latex_safe(order == 0 ? "integrated intensity" : order == 1 ? "mean velocity" : "dispersion"), "}") :
            unit_label_tex
    end
    ax_img = Axis(
        img_grid[1, 1];
        title     = main_title_obs,
        xlabel    = xlab0,
        ylabel    = ylab0,
        aspect    = DataAspect(),
        xtickformat = pixel_world_tick_formatter(slice_axis_dims(axis[])[2]),
        ytickformat = pixel_world_tick_formatter(slice_axis_dims(axis[])[1]),
    )
    compare_mode_label(mode::Symbol) = mode === :A ? "A" :
        mode === :B ? "B" :
        mode === :diff ? "A - B" :
        mode === :ratio ? "A / B" :
        "normalized residuals"

    compare_title_obs = lift(compare_name, compare_mode) do name, mode
        label = compare_mode_label(mode)
        isempty(name) ? latexstring("\\text{", latex_safe(label), "}") :
            latexstring("\\text{", latex_safe(label), ": ", latex_safe(name), "}")
    end
    ax_cmp = Axis(
        img_grid[1, 2];
        title     = compare_title_obs,
        xlabel    = xlab0,
        ylabel    = ylab0,
        aspect    = DataAspect(),
        xtickformat = pixel_world_tick_formatter(slice_axis_dims(axis[])[2]),
        ytickformat = pixel_world_tick_formatter(slice_axis_dims(axis[])[1]),
    )
    colsize!(img_grid, 2, Fixed(0))

    uv_point = Observable(Point2f(1, 1))
    hm = heatmap!(ax_img, slice_disp; colormap = cm_obs, colorrange = clims_safe)
    heatmap!(ax_cmp, compare_slice_disp; colormap = cm_obs, colorrange = compare_clims_safe, visible = compare_visible)
    contour!(ax_img, slice_disp; levels = contour_levels_obs, color = contour_colors_obs, linewidth = 1.2, visible = show_contours)
    compare_contours_visible = lift(show_contours, compare_visible) do contours, visible
        contours && visible
    end
    contour!(ax_cmp, compare_slice_disp; levels = contour_levels_obs, color = contour_colors_obs, linewidth = 1.2, visible = compare_contours_visible)
    crosshair_segments = lift(axis, u_idx, v_idx, show_crosshair) do a, u, v, enabled
        enabled || return Point2f[]
        u_max, v_max = slice_dims(a)
        Point2f[
            Point2f(1, u), Point2f(v_max, u),
            Point2f(v, 1), Point2f(v, u_max),
        ]
    end
    zoom_box_segments = lift(zoom_drag_active, zoom_drag_start, zoom_drag_end) do active, p0, p1
        active || return Point2f[]
        if !(isfinite(p0[1]) && isfinite(p0[2]) && isfinite(p1[1]) && isfinite(p1[2]))
            return Point2f[]
        end
        x0, y0 = p0
        x1, y1 = p1
        Point2f[
            Point2f(x0, y0), Point2f(x1, y0),
            Point2f(x1, y0), Point2f(x1, y1),
            Point2f(x1, y1), Point2f(x0, y1),
            Point2f(x0, y1), Point2f(x0, y0),
        ]
    end
    region_segments_from_points(p0, p1, shape::Symbol) = begin
        if !(isfinite(p0[1]) && isfinite(p0[2]) && isfinite(p1[1]) && isfinite(p1[2]))
            return Point2f[]
        end
        x0, y0 = p0
        x1, y1 = p1
        if shape === :circle
            r = hypot(x1 - x0, y1 - y0)
            r < 0.5 && return Point2f[]
            pts = Point2f[]
            for t in LinRange(0, 2π, 97)
                push!(pts, Point2f(x0 + r * cos(t), y0 + r * sin(t)))
            end
            return pts
        else
            return Point2f[
                Point2f(x0, y0), Point2f(x1, y0),
                Point2f(x1, y1), Point2f(x0, y1),
                Point2f(x0, y0),
            ]
        end
    end
    region_segments = lift(region_start, region_end, region_shape, region_uvs, region_drag_active) do p0, p1, shape, uv, dragging
        (dragging || !isempty(uv)) ? region_segments_from_points(p0, p1, shape) : Point2f[]
    end
    linesegments!(ax_img, crosshair_segments; color = (:white, 0.9), linewidth = 1.6, linestyle = :dot)
    linesegments!(ax_cmp, crosshair_segments; color = (:white, 0.9), linewidth = 1.6, linestyle = :dot, visible = compare_visible)
    linesegments!(ax_img, zoom_box_segments; color = (ui_accent, 0.95), linewidth = 2.0, linestyle = :dash)
    linesegments!(ax_cmp, zoom_box_segments; color = (ui_accent, 0.95), linewidth = 2.0, linestyle = :dash, visible = compare_visible)
    lines!(ax_img, region_segments; color = (RGBf(1.0, 0.78, 0.18), 0.98), linewidth = 2.4)
    lines!(ax_cmp, region_segments; color = (RGBf(1.0, 0.78, 0.18), 0.98), linewidth = 2.4, visible = compare_visible)
    marker_points = lift(uv_point, show_marker) do p, enabled
        enabled ? Point2f[p] : Point2f[]
    end
    scatter!(ax_img, marker_points; markersize = 10)
    scatter!(ax_cmp, marker_points; markersize = 10, visible = compare_visible)

    # Colorbar linked to plot; tellheight=false avoids layout feedback loops
    Colorbar(img_grid[1, 3], hm; label = display_unit_label, width = 20, tellheight = false)

    # Info + spectrum
    spec_grid = main_grid[1, 2] = GridLayout()
    info_panel = spec_grid[1, 1] = GridLayout(; alignmode = Outside())
    Box(
        info_panel[1, 1];
        color = ui_surface,
        strokecolor = ui_border,
        strokewidth = 1.0,
        cornerradius = 12,
        z = -5,
    )
    lab_info = Label(
        info_panel[1, 1];
        text      = make_info_tex(1, 1, 1, 1, 1, 0f0),
        halign    = :left,
        valign    = :center,
        fontsize  = 16,
        color     = ui_text,
        padding   = (16, 16, 12, 12),
        lineheight = 1.2,
        tellwidth = false,
    )

    ax_spec = Axis(
        spec_grid[2, 1];
        title  = L"\text{Spectrum at selected pixel}",
        xlabel = L"\text{index along slice axis}",
        ylabel = unit_label_tex,
        width  = 600,
        height = 320,
        xtickformat = latex_tick_formatter,
        ytickformat = latex_tick_formatter,
    )
    lines!(ax_spec, spec_x_raw, spec_y_disp)
    ax_img.xgridvisible[] = show_grid[]
    ax_img.ygridvisible[] = show_grid[]
    ax_cmp.xgridvisible[] = show_grid[]
    ax_cmp.ygridvisible[] = show_grid[]
    ax_spec.xgridvisible[] = show_grid[]
    ax_spec.ygridvisible[] = show_grid[]

    ax_hist = Axis(
        spec_grid[3, 1];
        title = L"\text{Visible slice histogram}",
        xlabel = unit_label_tex,
        ylabel = L"\text{count}",
        height = 105,
        xtickformat = latex_tick_formatter,
        ytickformat = latex_tick_formatter,
    )
    lines!(ax_hist, hist_x_obs, hist_y_obs; color = ui_accent, linewidth = 1.6)
    vlines!(ax_hist, lift(lim -> [first(lim), last(lim)], clims_safe); color = (ui_text_muted, 0.65), linewidth = 1.1, linestyle = :dash)

    # Controls
    controls_grid = main_grid[2, 1:2] = GridLayout(; alignmode = Outside())
    colgap!(controls_grid, 16)
    rowgap!(controls_grid, 16)
    rowsize!(main_grid, 2, Fixed(488))

    function control_card!(parent, row, col, title::AbstractString; rows::Int = 4, cols::Int = 4)
        card = parent[row, col] = GridLayout(; alignmode = Outside(12))
        # Card body
        Box(card[1:rows, 1:cols];
            color = ui_panel, strokecolor = ui_border,
            strokewidth = 1.0, cornerradius = 12, z = -6)
        # Header band (visually distinct title row)
        Box(card[1, 1:cols];
            color = ui_panel_header, strokecolor = (:transparent, 0.0),
            strokewidth = 0.0, cornerradius = 10, z = -5)
        Label(card[1, 1:cols];
            text = uppercase(title),
            halign = :left, tellwidth = false,
            fontsize = 13, font = :bold,
            color = ui_accent_strong,
            padding = (10, 10, 6, 6))
        rowgap!(card, 10)
        colgap!(card, 10)
        return card
    end
    control_label!(layout, pos, txt) = Label(layout[pos...]; text = txt, halign = :left, tellwidth = false, fontsize = 13, color = ui_text_muted)

    view_card = control_card!(controls_grid, 1, 1, "View"; rows = 5, cols = 4)
    control_label!(view_card, (2, 1), "Image")
    img_scale_menu = Menu(view_card[2, 2]; options = ["lin", "log10", "ln"], prompt = "lin", width = 96)
    control_label!(view_card, (3, 1), "Spectrum")
    spec_scale_menu = Menu(view_card[3, 2]; options = ["lin", "log10", "ln"], prompt = "lin", width = 96)
    reset_zoom_btn = Button(view_card[2, 3:4]; label = "Reset zoom", width = 132, height = 32)
    ps_btn = Button(view_card[4, 1:4]; label = "Power spectrum…", width = 240, height = 32)
    foreach(c -> colsize!(view_card, c, Auto()), 1:4)

    slice_card = control_card!(controls_grid, 1, 2, "Slice"; rows = 4, cols = 5)
    axes_labels = ["dim1 (x)", "dim2 (y)", "dim3 (z)"]
    control_label!(slice_card, (2, 1), "Axis")
    axis_menu = Menu(slice_card[2, 2]; options = axes_labels, prompt = "dim3 (z)", width = 128)
    status_label = Label(slice_card[2, 3:5]; text = latexstring("\\text{axis } 3,\\, \\text{index } 1"), fontsize = 14, halign = :left, tellwidth = false, color = ui_text)
    control_label!(slice_card, (3, 1), "Index")
    slice_slider = Slider(slice_card[3, 2:5]; range = 1:siz[3], startvalue = 1, width = 320, height = 26)
    control_label!(slice_card, (4, 1), "Gaussian")
    sigma_label = Label(slice_card[4, 2]; text = latexstring("\\sigma = 1.5\\,\\text{px}"), fontsize = 14, halign = :left, tellwidth = false, color = ui_text)
    sigma_slider = Slider(slice_card[4, 3:5]; range = LinRange(0, 10, 101), startvalue = 1.5, width = 230, height = 26)
    foreach(c -> colsize!(slice_card, c, Auto()), 1:5)

    contrast_card = control_card!(controls_grid, 1, 3, "Contrast"; rows = 4, cols = 5)
    clim_min_box   = Textbox(contrast_card[2, 1]; placeholder = "min", width = 120, height = 32)
    clim_max_box   = Textbox(contrast_card[2, 2]; placeholder = "max", width = 120, height = 32)
    clim_apply_btn = Button(contrast_card[2, 3]; label = "Apply", width = 86, height = 32)
    clim_auto_btn  = Button(contrast_card[2, 4]; label = "Auto", width = 78, height = 32)
    clim_p1_btn    = Button(contrast_card[3, 1]; label = "p1-p99", width = 92, height = 32)
    clim_p5_btn    = Button(contrast_card[3, 2]; label = "p5-p95", width = 92, height = 32)
    foreach(c -> colsize!(contrast_card, c, Auto()), 1:5)

    output_card = control_card!(controls_grid, 2, 1, "Output"; rows = 5, cols = 5)
    fmt_menu  = Menu(output_card[2, 1]; options = ["png", "pdf"], prompt = "png", width = 90)
    fname_box = Textbox(output_card[2, 2:4]; placeholder = "filename base", width = 220, height = 32)
    btn_save_img  = Button(output_card[3, 1]; label = "Image", width = 88, height = 32)
    btn_save_spec = Button(output_card[3, 2]; label = "Spectrum", width = 108, height = 32)
    btn_save_state = Button(output_card[3, 3]; label = "Save state", width = 112, height = 32)
    btn_load_state = Button(output_card[3, 4]; label = "Load state", width = 112, height = 32)
    btn_show_compare = Button(output_card[4, 1]; label = "Add dual", width = 112, height = 32)
    compare_path_box = Textbox(output_card[4, 2:4]; placeholder = "", width = 0, height = 32)
    btn_load_compare = Button(output_card[4, 5]; label = "", width = 0, height = 32)
    compare_mode_menu = Menu(output_card[4, 2:3]; options = ["A", "B", "A - B", "A / B", "resid z"], prompt = "B", width = 0)
    foreach(c -> colsize!(output_card, c, Auto()), 1:5)

    region_card = control_card!(controls_grid, 2, 2, "Region Spectrum"; rows = 3, cols = 4)
    region_mode_menu = Menu(region_card[2, 1]; options = ["point", "box", "circle"], prompt = "point", width = 112)
    region_clear_btn = Button(region_card[2, 2]; label = "Clear", width = 92, height = 32)
    region_count_label = Label(region_card[2, 3:4]; text = "0 px", halign = :left, tellwidth = false, fontsize = 14, color = ui_text_muted)
    foreach(c -> colsize!(region_card, c, Auto()), 1:4)

    contour_card = control_card!(controls_grid, 3, 1, "Contours"; rows = 3, cols = 5)
    contour_chk = Checkbox(contour_card[2, 1])
    Label(contour_card[2, 2]; text = "Show", halign = :left, tellwidth = false, fontsize = 14, color = ui_text)
    contour_levels_box = Textbox(contour_card[2, 3:4]; placeholder = "auto or 1:red, 2:#00ffaa", width = 190, height = 32)
    contour_apply_btn = Button(contour_card[2, 5]; label = "Apply", width = 82, height = 32)
    foreach(c -> colsize!(contour_card, c, Auto()), 1:5)

    anim_card = control_card!(controls_grid, 3, 2, "Animation"; rows = 4, cols = 5)
    start_box = Textbox(anim_card[2, 1]; placeholder = "start", width = 72, height = 32)
    stop_box  = Textbox(anim_card[2, 2]; placeholder = "stop",  width = 72, height = 32)
    step_box  = Textbox(anim_card[2, 3]; placeholder = "step",  width = 72, height = 32)
    fps_box   = Textbox(anim_card[2, 4]; placeholder = "fps",   width = 72, height = 32)
    play_btn = Button(anim_card[3, 1]; label = "Play", width = 78, height = 32)
    anim_btn = Button(anim_card[3, 2:3]; label = "Export GIF", width = 132, height = 32)
    loop_chk = Checkbox(anim_card[3, 4]); Label(anim_card[3, 5], text = "Loop", halign = :left, tellwidth = false, fontsize = 14, color = ui_text)
    foreach(c -> colsize!(anim_card, c, Auto()), 1:5)

    display_card = control_card!(controls_grid, 2, 3, "Display"; rows = 5, cols = 4)
    invert_chk = Checkbox(display_card[2, 1]); Label(display_card[2, 2], text = "Invert", halign = :left, tellwidth = false, fontsize = 14, color = ui_text)
    gauss_chk = Checkbox(display_card[2, 3]); Label(display_card[2, 4], text = "Gaussian", halign = :left, tellwidth = false, fontsize = 14, color = ui_text)
    crosshair_chk = Checkbox(display_card[3, 1]); Label(display_card[3, 2], text = "Crosshair", halign = :left, tellwidth = false, fontsize = 14, color = ui_text)
    marker_chk = Checkbox(display_card[3, 3]); Label(display_card[3, 4], text = "Point", halign = :left, tellwidth = false, fontsize = 14, color = ui_text)
    grid_chk = Checkbox(display_card[4, 1]); Label(display_card[4, 2], text = "Grid", halign = :left, tellwidth = false, fontsize = 14, color = ui_text)
    pingpong_chk = Checkbox(display_card[4, 3]); Label(display_card[4, 4], text = "Ping-pong", halign = :left, tellwidth = false, fontsize = 14, color = ui_text)
    foreach(c -> colsize!(display_card, c, Auto()), 1:4)

    moment_card = control_card!(controls_grid, 3, 3, "Products"; rows = 4, cols = 5)
    moment_menu = Menu(moment_card[2, 1]; options = ["M0 integrated", "M1 mean", "M2 dispersion"], prompt = "M0 integrated", width = 138)
    btn_show_moment = Button(moment_card[2, 2]; label = "Show", width = 82, height = 32)
    btn_show_slice = Button(moment_card[2, 3]; label = "Slice", width = 82, height = 32)
    btn_moment_png = Button(moment_card[2, 4]; label = "PNG", width = 74, height = 32)
    btn_moment_fits = Button(moment_card[2, 5]; label = "FITS", width = 74, height = 32)
    fits_product_menu = Menu(moment_card[3, 1:2]; options = ["slice", "region", "moment", "filtered cube"], prompt = "slice", width = 150)
    btn_save_fits = Button(moment_card[3, 3]; label = "Export FITS", width = 118, height = 32)
    foreach(c -> colsize!(moment_card, c, Auto()), 1:5)

    foreach(c -> colsize!(controls_grid, c, Relative(1 / 3)), 1:3)
    rowsize!(controls_grid, 1, Fixed(184))
    rowsize!(controls_grid, 2, Fixed(148))
    rowsize!(controls_grid, 3, Fixed(148))

    style_checkbox!(pingpong_chk)
    style_checkbox!(loop_chk)
    style_checkbox!(invert_chk)
    style_checkbox!(gauss_chk)
    style_checkbox!(crosshair_chk)
    style_checkbox!(marker_chk)
    style_checkbox!(grid_chk)
    style_menu!(img_scale_menu)
    style_menu!(spec_scale_menu)
    style_menu!(fmt_menu)
    style_menu!(compare_mode_menu)
    style_menu!(axis_menu)
    style_menu!(moment_menu)
    style_menu!(fits_product_menu)
    style_textbox!(fname_box)
    style_textbox!(compare_path_box)
    style_textbox!(start_box)
    style_textbox!(stop_box)
    style_textbox!(step_box)
    style_textbox!(fps_box)
    style_textbox!(clim_min_box)
    style_textbox!(clim_max_box)
    style_button!(reset_zoom_btn)
    style_button!(ps_btn)
    style_button!(btn_save_img)
    style_button!(btn_save_spec)
    style_button!(btn_save_state)
    style_button!(btn_load_state)
    style_button!(btn_show_compare)
    style_button!(btn_load_compare)
    style_button!(play_btn)
    style_button!(anim_btn)
    style_button!(clim_apply_btn)
    style_button!(clim_auto_btn)
    style_button!(clim_p1_btn)
    style_button!(clim_p5_btn)
    style_menu!(region_mode_menu)
    style_button!(region_clear_btn)
    style_checkbox!(contour_chk)
    style_textbox!(contour_levels_box)
    style_button!(contour_apply_btn)
    style_button!(btn_show_moment)
    style_button!(btn_show_slice)
    style_button!(btn_moment_png)
    style_button!(btn_moment_fits)
    style_button!(btn_save_fits)
    style_slider!(slice_slider)
    style_slider!(sigma_slider)

    invert_chk.checked[] = invert_cmap[]
    gauss_chk.checked[] = gauss_on[]
    crosshair_chk.checked[] = show_crosshair[]
    marker_chk.checked[] = show_marker[]
    grid_chk.checked[] = show_grid[]
    contour_chk.checked[] = show_contours[]
    loop_chk.checked[] = true
    main_grid[3, 2] = Label(
        main_grid[3, 2];
        text      = "arrows: move crosshair    left-click: pick / draw region    right-drag: zoom    i: invert colormap",
        halign    = :right,
        fontsize  = 13,
        color     = ui_text_muted,
        tellwidth = false,
    )
    ui_status = Observable(" ")
    main_grid[4, 1:2] = Label(
        main_grid[4, 1:2];
        text = ui_status,
        halign = :left,
        tellwidth = false,
    )

    # ---------- Helpers ----------
    set_status!(msg::AbstractString) = (ui_status[] = String(msg); nothing)
    set_box_text!(tb, s::AbstractString) = begin
        str = String(s)
        tb.displayed_string[] = str
        tb.stored_string[] = str
        nothing
    end

    function show_compare_loader!()
        btn_show_compare.label[] = ""
        btn_show_compare.width[] = 0
        compare_mode_menu.width[] = 0
        compare_path_box.placeholder[] = "second cube FITS path"
        compare_path_box.width[] = 310
        btn_load_compare.label[] = "Dual"
        btn_load_compare.width[] = 82
        set_status!("Enter the second cube FITS path, then click Dual.")
        nothing
    end

    function hide_compare_loader!()
        btn_show_compare.label[] = ""
        btn_show_compare.width[] = 0
        compare_path_box.placeholder[] = ""
        compare_path_box.width[] = 0
        btn_load_compare.label[] = ""
        btn_load_compare.width[] = 0
        compare_mode_menu.width[] = compare_visible[] ? 150 : 0
        nothing
    end

    function resolve_compare_path(path_txt::AbstractString)
        p = strip(String(path_txt))
        isempty(p) && return ""
        isfile(p) && return p
        beside_primary = joinpath(dirname(abspath(filepath)), p)
        isfile(beside_primary) && return beside_primary
        return p
    end

    function load_compare_cube!(path_txt::AbstractString)
        cmp_path = resolve_compare_path(path_txt)
        if isempty(cmp_path)
            set_status!("Provide a second cube FITS path before enabling dual view.")
            return false
        end
        if !isfile(cmp_path)
            set_status!("Second cube not found: $(cmp_path)")
            return false
        end
        raw_cmp = try
            FITS(cmp_path) do f
                read(f[1])
            end
        catch e
            msg = "Failed to read second cube: $(sprint(showerror, e))"
            set_status!(msg)
            @error msg exception=(e, catch_backtrace())
            return false
        end
        if ndims(raw_cmp) != 3
            set_status!("Second cube must be 3D, got ndims=$(ndims(raw_cmp)).")
            return false
        end
        if size(raw_cmp) != siz
            set_status!("Second cube size $(size(raw_cmp)) does not match primary cube size $(siz).")
            return false
        end
        compare_data[] = Float32.(raw_cmp)
        compare_name[] = String(replace(basename(cmp_path), r"\.fits(\.gz)?$" => ""))
        compare_visible[] = true
        colsize!(img_grid, 2, Auto())
        ax_cmp.xgridvisible[] = show_grid[]
        ax_cmp.ygridvisible[] = show_grid[]
        autolimits!(ax_cmp)
        hide_compare_loader!()
        set_status!("Dual view enabled with $(cmp_path).")
        return true
    end

    function world_info_string()
        any(has_wcs(wcs, dim) for dim in 1:3) || return ""
        coords = (
            format_world_coord(wcs, 1, i_idx[]),
            format_world_coord(wcs, 2, j_idx[]),
            format_world_coord(wcs, 3, k_idx[]),
        )
        return join(coords, ", ")
    end

    function selection_info_tex()
        if isempty(region_uvs[])
            val = data[i_idx[], j_idx[], k_idx[]]
            base = String(make_info_tex(i_idx[], j_idx[], k_idx[], u_idx[], v_idx[], val))
            winfo = world_info_string()
            isempty(winfo) && return latexstring(base)
            return latexstring(base, "\\quad\\mathbf{WCS}\\,(", latex_safe(winfo), ")")
        else
            npx = length(region_uvs[])
            y = mean_region_spectrum(data, axis[], region_uvs[])
            chan = clamp(idx[], 1, length(y))
            val = y[chan]
            shape = region_shape[] === :circle ? "circle" : "box"
            return latexstring(
                "\\mathbf{region}\\,\\text{", shape, "}\\quad\\mathbf{pixels}=",
                npx,
                "\\quad\\mathbf{slice\\ mean}= ",
                isnan(val) ? "NaN" : string(round(Float32(val); digits = 4)),
            )
        end
    end

    function clear_region!()
        region_uvs[] = Tuple{Int,Int}[]
        region_start[] = Point2f(NaN32, NaN32)
        region_end[] = Point2f(NaN32, NaN32)
        region_drag_active[] = false
        region_count_label.text[] = "0 px"
        nothing
    end

    function update_region_from_drag!(p0::Point2f, p1::Point2f)
        u_max, v_max = slice_dims(axis[])
        uv = region_uv_indices(u_max, v_max, p0[1], p0[2], p1[1], p1[2], region_shape[])
        region_uvs[] = uv
        region_count_label.text[] = "$(length(uv)) px"
        if isempty(uv)
            set_status!("Region canceled: draw a larger $(String(region_shape[])).")
        else
            set_status!("Region spectrum averaged over $(length(uv)) pixels.")
        end
        nothing
    end

    function apply_percentile_clims!(lo::Real, hi::Real)
        parsed_clims = percentile_clims(slice_disp[], lo, hi)
        clims_manual[] = parsed_clims
        use_manual[] = true
        set_box_text!(clim_min_box, string(first(parsed_clims)))
        set_box_text!(clim_max_box, string(last(parsed_clims)))
        limits!(ax_spec, nothing, nothing, first(parsed_clims), last(parsed_clims))
        xlims!(ax_spec, 0f0, Float32(max(0, length(spec_y_buf) - 1)))
        set_status!("Colorbar limits set to p$(lo)-p$(hi).")
        nothing
    end

    function refresh_uv!()
        a = axis[]
        u_max, v_max = slice_dims(a)
        u, v = ijk_to_uv(i_idx[], j_idx[], k_idx[], a)
        u = clamp(u, 1, u_max)
        v = clamp(v, 1, v_max)
        u_idx[] = u; v_idx[] = v
        uv_point[] = Point2f(v, u)
    end

    function refresh_labels!()
        lab_info.text[] = selection_info_tex()
        status_label.text[] = latexstring("\\text{axis } $(axis[]),\\, \\text{index } $(idx[])")
    end

    function refresh_spectrum!()
        if !isempty(region_uvs[])
            spec_x_raw[] = spec_x_axes[axis[]]
            y = mean_region_spectrum(data, axis[], region_uvs[])
            resize!(spec_y_buf, length(y))
            copyto!(spec_y_buf, y)
            ax_spec.title[] = latexstring("\\text{Mean spectrum in selected region}")
        elseif axis[] == 1
            spec_x_raw[] = spec_x_axes[1]
            resize!(spec_y_buf, siz[1])
            @views copyto!(spec_y_buf, data[:, j_idx[], k_idx[]])
            ax_spec.title[] = L"\text{Spectrum at selected pixel}"
        elseif axis[] == 2
            spec_x_raw[] = spec_x_axes[2]
            resize!(spec_y_buf, siz[2])
            @views copyto!(spec_y_buf, data[i_idx[], :, k_idx[]])
            ax_spec.title[] = L"\text{Spectrum at selected pixel}"
        else
            spec_x_raw[] = spec_x_axes[3]
            resize!(spec_y_buf, siz[3])
            @views copyto!(spec_y_buf, data[i_idx[], j_idx[], :])
            ax_spec.title[] = L"\text{Spectrum at selected pixel}"
        end
        spec_y_raw[] = spec_y_buf
        x_max = Float32(max(0, length(spec_y_buf) - 1))
        if use_manual[]
            vmin_, vmax_ = clims_manual[]; limits!(ax_spec, nothing, nothing, vmin_, vmax_)
            xlims!(ax_spec, 0f0, x_max)
        else
            autolimits!(ax_spec)
            xlims!(ax_spec, 0f0, x_max)
        end
    end

    function refresh_axis_labels!()
        xlab, ylab = slice_axis_labels(axis[])
        ax_img.xlabel[] = xlab
        ax_img.ylabel[] = ylab
        ax_cmp.xlabel[] = xlab
        ax_cmp.ylabel[] = ylab
        u_dim, v_dim = slice_axis_dims(axis[])
        ax_img.xtickformat[] = pixel_world_tick_formatter(v_dim)
        ax_img.ytickformat[] = pixel_world_tick_formatter(u_dim)
        ax_cmp.xtickformat[] = pixel_world_tick_formatter(v_dim)
        ax_cmp.ytickformat[] = pixel_world_tick_formatter(u_dim)
    end

    refresh_all!() = (refresh_axis_labels!(); refresh_uv!(); refresh_labels!(); refresh_spectrum!())

    # ---------- Reactivity ----------
    on(clims_obs) do (cmin, cmax)
        if use_manual[]
            limits!(ax_spec, nothing, nothing, cmin, cmax)
        end
    end

    on(spec_scale_mode) do _
        x_max = Float32(max(0, length(spec_y_buf) - 1))
        if use_manual[]
            vmin_, vmax_ = clims_manual[]; limits!(ax_spec, nothing, nothing, vmin_, vmax_)
            xlims!(ax_spec, 0f0, x_max)
        else
            autolimits!(ax_spec)
            xlims!(ax_spec, 0f0, x_max)
        end
    end

    on(reset_zoom_btn.clicks) do _
        autolimits!(ax_img)
        compare_visible[] && autolimits!(ax_cmp)
    end

    # ---------- UI callbacks ----------
        # Keep the slice slider synced to the active axis (range + knob position)
    on(axis_menu.selection) do sel
        sel === nothing && return
        new_axis = findfirst(==(String(sel)), axes_labels)
        new_axis === nothing && return
        axis[] = new_axis
        new_range = 1:siz[new_axis]
        slice_slider.range[] = new_range
        idx[] = clamp(idx[], first(new_range), last(new_range))
        slice_slider.value[] = idx[]  # move the thumb if the old value was out of bounds
        ii, jj, kk = uv_to_ijk(u_idx[], v_idx[], axis[], idx[])
        i_idx[] = clamp(ii, 1, siz[1]); j_idx[] = clamp(jj, 1, siz[2]); k_idx[] = clamp(kk, 1, siz[3])
        clear_region!()
        refresh_all!()
    end

    on(slice_slider.value) do v
        idx[] = Int(round(v))
        ii, jj, kk = uv_to_ijk(u_idx[], v_idx[], axis[], idx[])
        i_idx[] = clamp(ii, 1, siz[1]); j_idx[] = clamp(jj, 1, siz[2]); k_idx[] = clamp(kk, 1, siz[3])
        refresh_labels!(); refresh_spectrum!()
    end

    on(img_scale_menu.selection) do sel
        sel === nothing && return
        img_scale_mode[] = Symbol(sel)
    end

    on(spec_scale_menu.selection) do sel
        sel === nothing && return
        spec_scale_mode[] = Symbol(sel)
    end

    on(invert_chk.checked) do v
        invert_cmap[] = v
    end

    on(gauss_chk.checked) do v
        gauss_on[] = v
        refresh_spectrum!()
    end

    on(crosshair_chk.checked) do v
        show_crosshair[] = v
    end

    on(marker_chk.checked) do v
        show_marker[] = v
    end

    on(grid_chk.checked) do v
        show_grid[] = v
        ax_img.xgridvisible[] = v
        ax_img.ygridvisible[] = v
        ax_cmp.xgridvisible[] = v
        ax_cmp.ygridvisible[] = v
        ax_spec.xgridvisible[] = v
        ax_spec.ygridvisible[] = v
    end

    on(btn_show_compare.clicks) do _
        show_compare_loader!()
    end

    on(btn_load_compare.clicks) do _
        btn_load_compare.width[] <= 0 && return
        load_compare_cube!(get_box_str(compare_path_box))
    end

    on(compare_mode_menu.selection) do sel
        sel === nothing && return
        label = String(sel)
        compare_mode[] = label == "A" ? :A :
            label == "B" ? :B :
            label == "A - B" ? :diff :
            label == "A / B" ? :ratio :
            :residuals
        set_status!("Dual product set to $(compare_mode_label(compare_mode[])).")
    end

    on(sigma_slider.value) do v
        sigma[] = Float32(v)
        sigma_label.text[] = latexstring("\\sigma = $(round(v; digits = 2))\\,\\text{px}")
    end

    on(clim_apply_btn.clicks) do _
        txtmin = get_box_str(clim_min_box)
        txtmax = get_box_str(clim_max_box)
        ok, new_manual, parsed_clims, msg = parse_manual_clims(txtmin, txtmax; fallback = clims_manual[])
        set_status!(msg)
        if !ok
            @warn "Could not apply colorbar limits" txtmin txtmax msg
            return
        end
        if new_manual
            clims_manual[] = parsed_clims
            use_manual[] = true
            limits!(ax_spec, nothing, nothing, first(parsed_clims), last(parsed_clims))
            xlims!(ax_spec, 0f0, Float32(max(0, length(spec_y_buf) - 1)))
            set_box_text!(clim_min_box, string(first(parsed_clims)))
            set_box_text!(clim_max_box, string(last(parsed_clims)))
        else
            use_manual[] = false
            autolimits!(ax_spec)
            xlims!(ax_spec, 0f0, Float32(max(0, length(spec_y_buf) - 1)))
        end
    end

    on(clim_auto_btn.clicks) do _
        use_manual[] = false
        set_box_text!(clim_min_box, "")
        set_box_text!(clim_max_box, "")
        autolimits!(ax_spec)
        xlims!(ax_spec, 0f0, Float32(max(0, length(spec_y_buf) - 1)))
        set_status!("Automatic color limits enabled.")
    end

    on(clim_p1_btn.clicks) do _
        apply_percentile_clims!(1, 99)
    end

    on(clim_p5_btn.clicks) do _
        apply_percentile_clims!(5, 95)
    end

    on(region_mode_menu.selection) do sel
        sel === nothing && return
        mode = Symbol(String(sel))
        if mode in (:point, :box, :circle)
            selection_mode[] = mode
            region_shape[] = mode === :circle ? :circle : :box
            if mode === :point
                clear_region!()
                refresh_labels!()
                refresh_spectrum!()
            else
                set_status!("Draw a $(String(mode)) with left drag on the image.")
            end
        end
    end

    on(region_clear_btn.clicks) do _
        clear_region!()
        refresh_labels!()
        refresh_spectrum!()
        set_status!("Region cleared; spectrum follows the selected pixel.")
    end

    on(contour_chk.checked) do v
        show_contours[] = v
        set_status!(v ? "Contours enabled." : "Contours hidden.")
    end

    on(contour_apply_btn.clicks) do _
        ok, use_man, levels, colors, msg = parse_contour_specs(
            get_box_str(contour_levels_box);
            fallback_levels = contour_manual_levels[],
            fallback_colors = contour_manual_colors[],
        )
        set_status!(msg)
        if !ok
            @warn "Could not apply contour levels" msg
            return
        end
        contour_use_manual[] = use_man
        contour_manual_levels[] = levels
        contour_manual_colors[] = colors
        if use_man
            set_box_text!(contour_levels_box, format_contour_specs(levels, colors))
        else
            set_box_text!(contour_levels_box, "")
        end
        show_contours[] = true
        contour_chk.checked[] = true
    end

    on(moment_menu.selection) do sel
        sel === nothing && return
        label = String(sel)
        moment_order[] = startswith(label, "M1") ? 1 : startswith(label, "M2") ? 2 : 0
        view_product[] === :moment && set_status!("Showing $(label) along axis $(axis[]).")
    end

    on(btn_show_moment.clicks) do _
        view_product[] = :moment
        use_manual[] = false
        autolimits!(ax_spec)
        xlims!(ax_spec, 0f0, Float32(max(0, length(spec_y_buf) - 1)))
        autolimits!(ax_img)
        set_status!("Moment map displayed along axis $(axis[]).")
    end

    on(btn_show_slice.clicks) do _
        view_product[] = :slice
        use_manual[] = false
        autolimits!(ax_img)
        set_status!("Slice view restored.")
    end

    # Keyboard navigation (+ invert)
    on(events(fig).keyboardbutton) do ev
        ev.action == Keyboard.press || return
        u_max, v_max = slice_dims(axis[])
        if ev.key == Keyboard.i
            invert_cmap[] = !invert_cmap[]
        elseif ev.key == Keyboard.left
            v_idx[] = max(1, v_idx[] - 1)
        elseif ev.key == Keyboard.right
            v_idx[] = min(v_max, v_idx[] + 1)
        elseif ev.key == Keyboard.up
            u_idx[] = min(u_max, u_idx[] + 1)
        elseif ev.key == Keyboard.down
            u_idx[] = max(1, u_idx[] - 1)
        else
            return
        end
        ii, jj, kk = uv_to_ijk(u_idx[], v_idx[], axis[], idx[])
        i_idx[] = clamp(ii, 1, siz[1]); j_idx[] = clamp(jj, 1, siz[2]); k_idx[] = clamp(kk, 1, siz[3])
        refresh_labels!(); refresh_spectrum!()
        uv_point[] = Point2f(v_idx[], u_idx[])
    end

    # Mouse pick
    on(events(ax_img).mousebutton) do ev
        if ev.button == Mouse.right && ev.action == Mouse.press
            p = mouseposition(ax_img)
            if any(isnan, p)
                return
            end
            zoom_drag_start[] = Point2f(p[1], p[2])
            zoom_drag_end[] = Point2f(p[1], p[2])
            zoom_drag_active[] = true
            set_status!("Zoom box: right-drag and release to apply.")
            return
        elseif ev.button == Mouse.right && ev.action == Mouse.release
            if !zoom_drag_active[]
                return
            end
            p = mouseposition(ax_img)
            if !any(isnan, p)
                zoom_drag_end[] = Point2f(p[1], p[2])
            end
            p0 = zoom_drag_start[]
            p1 = zoom_drag_end[]
            zoom_drag_active[] = false
            zoom_drag_start[] = Point2f(NaN32, NaN32)
            zoom_drag_end[] = Point2f(NaN32, NaN32)
            if !(isfinite(p0[1]) && isfinite(p0[2]) && isfinite(p1[1]) && isfinite(p1[2]))
                return
            end
            x0, y0 = p0
            x1, y1 = p1
            xmin, xmax = minmax(x0, x1)
            ymin, ymax = minmax(y0, y1)
            if abs(xmax - xmin) < 1e-3 || abs(ymax - ymin) < 1e-3
                set_status!("Zoom canceled: draw a larger rectangle.")
                return
            end
            limits!(ax_img, xmin, xmax, ymin, ymax)
            compare_visible[] && limits!(ax_cmp, xmin, xmax, ymin, ymax)
            set_status!("Zoom applied.")
            return
        elseif ev.button == Mouse.left && ev.action == Mouse.press && selection_mode[] != :point
            p = mouseposition(ax_img)
            if any(isnan, p)
                return
            end
            u_max, v_max = slice_dims(axis[])
            p0 = Point2f(clamp(p[1], 1, v_max), clamp(p[2], 1, u_max))
            region_start[] = p0
            region_end[] = p0
            region_drag_active[] = true
            region_uvs[] = Tuple{Int,Int}[]
            region_count_label.text[] = "0 px"
            set_status!("Drawing $(String(selection_mode[])) region.")
            return
        elseif ev.button == Mouse.left && ev.action == Mouse.release && region_drag_active[]
            p = mouseposition(ax_img)
            u_max, v_max = slice_dims(axis[])
            if !any(isnan, p)
                region_end[] = Point2f(clamp(p[1], 1, v_max), clamp(p[2], 1, u_max))
            end
            p0 = region_start[]
            p1 = region_end[]
            region_drag_active[] = false
            if !(isfinite(p0[1]) && isfinite(p0[2]) && isfinite(p1[1]) && isfinite(p1[2]))
                clear_region!()
                return
            end
            update_region_from_drag!(p0, p1)
            refresh_labels!()
            refresh_spectrum!()
            return
        elseif ev.button == Mouse.left && ev.action == Mouse.press
            p = mouseposition(ax_img)
            if any(isnan, p)
                return
            end
            clear_region!()
            u_max, v_max = slice_dims(axis[])
            u = Int(round(clamp(p[2], 1, u_max)))
            v = Int(round(clamp(p[1], 1, v_max)))
            u_idx[] = u; v_idx[] = v
            ii, jj, kk = uv_to_ijk(u, v, axis[], idx[])
            i_idx[] = clamp(ii, 1, siz[1]); j_idx[] = clamp(jj, 1, siz[2]); k_idx[] = clamp(kk, 1, siz[3])
            refresh_labels!(); refresh_spectrum!()
            uv_point[] = Point2f(v, u)
        end
    end

    on(events(ax_img).mouseposition) do p
        if zoom_drag_active[] && !any(isnan, p)
            zoom_drag_end[] = Point2f(p[1], p[2])
        elseif region_drag_active[] && !any(isnan, p)
            u_max, v_max = slice_dims(axis[])
            region_end[] = Point2f(clamp(p[1], 1, v_max), clamp(p[2], 1, u_max))
        end
    end

    # ---------- Saving ----------
    default_desktop = joinpath(homedir(), "Desktop")
    save_root = if save_dir === nothing
        isdir(default_desktop) ? default_desktop : pwd()
    else
        path = String(save_dir)
        if !isdir(path)
            mkpath(path)
        end
        path
    end
    resolved_settings_path = settings_path === nothing ?
        joinpath(save_root, "$(fname)_viewer_settings.toml") :
        abspath(String(settings_path))

    current_settings() = Dict{String,Any}(
        "axis" => axis[],
        "index" => idx[],
        "img_scale" => String(img_scale_mode[]),
        "spec_scale" => String(spec_scale_mode[]),
        "colormap" => String(cmap_name[]),
        "invert_colormap" => invert_cmap[],
        "show_crosshair" => show_crosshair[],
        "show_marker" => show_marker[],
        "show_grid" => show_grid[],
        "show_contours" => show_contours[],
        "contour_use_manual" => contour_use_manual[],
        "contour_levels" => collect(contour_manual_levels[]),
        "contour_colors" => collect(contour_manual_colors[]),
        "use_manual_clims" => use_manual[],
        "clim_min" => use_manual[] ? first(clims_manual[]) : first(clims_auto[]),
        "clim_max" => use_manual[] ? last(clims_manual[]) : last(clims_auto[]),
    )

    make_name = function (base::AbstractString, ext::AbstractString)
        b = isempty(base) ? fname : base
        return "$(b)_axis$(axis[])_idx$(idx[])_i$(i_idx[])_j$(j_idx[])_k$(k_idx[])_img$(String(img_scale_mode[]))_spec$(String(spec_scale_mode[])).$(ext)"
    end

    # Unified export (let Makie/Cairo choose the backend)
    save_with_format(path::AbstractString, fig) = CairoMakie.save(String(path), fig)

    moment_label() = moment_order[] == 0 ? "moment0" : moment_order[] == 1 ? "moment1" : "moment2"

    function write_fits_array(path::AbstractString, arr)
        FITS(String(path), "w") do f
            write(f, Float32.(arr))
        end
        nothing
    end

    function save_moment_png!(out::AbstractString)
        f_mom = CairoMakie.Figure(size = (700, 560))
        xlab_s, ylab_s = slice_axis_labels(axis[])
        axM = CairoMakie.Axis(
            f_mom[1, 1];
            title = latexstring("\\text{", latex_safe(fname), " ", latex_safe(moment_label()), " axis $(axis[])}"),
            xlabel = xlab_s,
            ylabel = ylab_s,
            aspect = CairoMakie.DataAspect(),
            yreversed = true,
            xtickformat = latex_tick_formatter,
            ytickformat = latex_tick_formatter,
        )
        lim = clamped_extrema(moment_raw[])
        hmM = CairoMakie.heatmap!(axM, moment_raw[]; colormap = cm_obs[], colorrange = lim)
        CairoMakie.Colorbar(f_mom[1, 2], hmM; label = moment_label(), width = 20)
        CairoMakie.save(String(out), f_mom; backend = CairoMakie)
        nothing
    end

    function export_fits_product!(product::AbstractString, out::AbstractString)
        if product == "slice"
            write_fits_array(out, get_slice(data, axis[], idx[]))
        elseif product == "region"
            if isempty(region_uvs[])
                throw(ArgumentError("select a box or circle region before exporting the averaged region FITS"))
            end
            write_fits_array(out, mean_region_spectrum(data, axis[], region_uvs[]))
        elseif product == "moment"
            write_fits_array(out, moment_raw[])
        elseif product == "filtered cube"
            write_fits_array(out, filtered_cube_by_slice(data, axis[], sigma[]))
        else
            throw(ArgumentError("unknown FITS product: $(product)"))
        end
        nothing
    end

    on(btn_moment_png.clicks) do _
        spawn_safely() do
            base = get_box_str(fname_box)
            base = isempty(base) ? "$(fname)_$(moment_label())" : base
            out = joinpath(save_root, make_name(base, "png"))
            try
                save_moment_png!(out)
                set_status!("Saved moment PNG to $(out).")
            catch e
                msg = "Failed to save moment PNG $(out): $(sprint(showerror, e))"
                set_status!(msg)
                @error msg exception=(e, catch_backtrace())
            end
        end
    end

    on(btn_moment_fits.clicks) do _
        spawn_safely() do
            base = get_box_str(fname_box)
            base = isempty(base) ? "$(fname)_$(moment_label())" : base
            out = joinpath(save_root, make_name(base, "fits"))
            try
                write_fits_array(out, moment_raw[])
                set_status!("Saved moment FITS to $(out).")
            catch e
                msg = "Failed to save moment FITS $(out): $(sprint(showerror, e))"
                set_status!(msg)
                @error msg exception=(e, catch_backtrace())
            end
        end
    end

    on(btn_save_fits.clicks) do _
        spawn_safely() do
            product = String(something(fits_product_menu.selection[], "slice"))
            clean_product = replace(product, " " => "_")
            base = get_box_str(fname_box)
            base = isempty(base) ? "$(fname)_$(clean_product)" : base
            out = joinpath(save_root, make_name(base, "fits"))
            try
                export_fits_product!(product, out)
                set_status!("Saved $(product) FITS to $(out).")
            catch e
                msg = "Failed to save $(product) FITS $(out): $(sprint(showerror, e))"
                set_status!(msg)
                @error msg exception=(e, catch_backtrace())
            end
        end
    end

    on(btn_save_state.clicks) do _
        try
            save_viewer_settings(resolved_settings_path, current_settings())
            set_status!("Saved settings to $(resolved_settings_path).")
        catch e
            msg = "Failed to save settings: $(sprint(showerror, e))"
            set_status!(msg)
            @error msg exception=(e, catch_backtrace())
        end
    end

    on(btn_load_state.clicks) do _
        if !isfile(resolved_settings_path)
            set_status!("Settings file not found: $(resolved_settings_path)")
            return
        end
        try
            st = load_viewer_settings(resolved_settings_path)

            axis_val = clamp(Int(get(st, "axis", axis[])), 1, 3)
            axis_menu.selection[] = axes_labels[axis_val]

            idx_val = clamp(Int(get(st, "index", idx[])), 1, siz[axis_val])
            slice_slider.value[] = idx_val

            img_scale_val = String(get(st, "img_scale", String(img_scale_mode[])))
            if img_scale_val in ("lin", "log10", "ln")
                img_scale_menu.selection[] = img_scale_val
            end

            spec_scale_val = String(get(st, "spec_scale", String(spec_scale_mode[])))
            if spec_scale_val in ("lin", "log10", "ln")
                spec_scale_menu.selection[] = spec_scale_val
            end

            cmap_val = Symbol(String(get(st, "colormap", String(cmap_name[]))))
            try
                to_cmap(cmap_val)
                cmap_name[] = cmap_val
            catch
                @warn "Ignoring invalid colormap in settings" colormap=cmap_val
            end

            invert_val = Bool(get(st, "invert_colormap", invert_cmap[]))
            invert_chk.checked[] = invert_val

            crosshair_chk.checked[] = Bool(get(st, "show_crosshair", show_crosshair[]))
            marker_chk.checked[] = Bool(get(st, "show_marker", show_marker[]))
            grid_chk.checked[] = Bool(get(st, "show_grid", show_grid[]))
            contour_chk.checked[] = Bool(get(st, "show_contours", show_contours[]))
            contour_use_manual[] = Bool(get(st, "contour_use_manual", contour_use_manual[]))
            raw_levels = get(st, "contour_levels", contour_manual_levels[])
            raw_colors = get(st, "contour_colors", contour_manual_colors[])
            contour_manual_levels[] = Float32.(raw_levels)
            contour_manual_colors[] = String.(raw_colors)
            if contour_use_manual[] && !isempty(contour_manual_levels[])
                set_box_text!(contour_levels_box, format_contour_specs(contour_manual_levels[], contour_manual_colors[]))
            else
                set_box_text!(contour_levels_box, "")
            end

            use_manual_val = Bool(get(st, "use_manual_clims", use_manual[]))
            if use_manual_val
                cmin = Float32(get(st, "clim_min", first(clims_manual[])))
                cmax = Float32(get(st, "clim_max", last(clims_manual[])))
                ok, new_manual, parsed_clims, msg = parse_manual_clims(string(cmin), string(cmax); fallback = clims_manual[])
                if ok && new_manual
                    clims_manual[] = parsed_clims
                    use_manual[] = true
                    set_box_text!(clim_min_box, string(first(parsed_clims)))
                    set_box_text!(clim_max_box, string(last(parsed_clims)))
                    limits!(ax_spec, nothing, nothing, first(parsed_clims), last(parsed_clims))
                    set_status!(msg)
                end
                else
                    use_manual[] = false
                    set_box_text!(clim_min_box, "")
                    set_box_text!(clim_max_box, "")
                    autolimits!(ax_spec)
                    xlims!(ax_spec, 0f0, Float32(max(0, length(spec_y_buf) - 1)))
                end
            set_status!("Loaded settings from $(resolved_settings_path).")
        catch e
            msg = "Failed to load settings: $(sprint(showerror, e))"
            set_status!(msg)
            @error msg exception=(e, catch_backtrace())
        end
    end

    # ---------- Save image (slice + colorbar + crosshair) ----------
    on(btn_save_img.clicks) do _
        spawn_safely() do
            ext  = String(something(fmt_menu.selection[], "png"))
            out  = joinpath(save_root, make_name(get_box_str(fname_box), ext))
            try
                f_slice = CairoMakie.Figure(size = (700, 560))
                xlab_s, ylab_s = slice_axis_labels(axis[])
                axS = CairoMakie.Axis(
                    f_slice[1, 1];
                    title     = make_slice_title(fname, axis[], idx[]),
                    xlabel    = xlab_s,
                    ylabel    = ylab_s,
                    aspect    = CairoMakie.DataAspect(),
                    yreversed = true,
                    xtickformat = latex_tick_formatter,
                    ytickformat = latex_tick_formatter,
                )
                hmS = CairoMakie.heatmap!(axS, slice_disp[]; colormap = cm_obs[], colorrange = clims_obs[])
                if show_contours[] && !isempty(contour_levels_obs[])
                    CairoMakie.contour!(axS, slice_disp[]; levels = contour_levels_obs[], color = contour_colors_obs[], linewidth = 1.2)
                end
                axS.xgridvisible[] = show_grid[]
                axS.ygridvisible[] = show_grid[]
                if show_crosshair[]
                    u_max, v_max = slice_dims(axis[])
                    u, v = u_idx[], v_idx[]
                    CairoMakie.linesegments!(
                        axS,
                        Point2f[
                            Point2f(1, u), Point2f(v_max, u),
                            Point2f(v, 1), Point2f(v, u_max),
                        ];
                        color = (:white, 0.9),
                        linewidth = 1.6,
                        linestyle = :dot,
                    )
                end
                if show_marker[]
                    CairoMakie.scatter!(axS, [Point2f(uv_point[]...)], markersize = 10)
                end
                if !isempty(region_uvs[])
                    CairoMakie.lines!(
                        axS,
                        region_segments_from_points(region_start[], region_end[], region_shape[]);
                        color = (RGBf(1.0, 0.78, 0.18), 0.98),
                        linewidth = 2.4,
                    )
                end
                CairoMakie.Colorbar(f_slice[1, 2], hmS; label = unit_label, width = 20)

                CairoMakie.save(String(out), f_slice; backend = CairoMakie)
                @info "Saved image" out
                set_status!("Saved image to $(out).")
            catch e
                msg = "Failed to save image $(out): $(sprint(showerror, e))"
                set_status!(msg)
                @error msg exception=(e, catch_backtrace())
            end
        end
    end

    # ---------- Save spectrum (lines plot) ----------
    on(btn_save_spec.clicks) do _
        spawn_safely() do
            ext = String(something(fmt_menu.selection[], "png"))
            base = get_box_str(fname_box)
            base = isempty(base) ? "$(fname)_spectrum" : base
            out = joinpath(save_root, make_name(base, ext))

            try
                f_spec = CairoMakie.Figure(size = (600, 400))
                axP = CairoMakie.Axis(
                    f_spec[1, 1];
                    title  = isempty(region_uvs[]) ? make_spec_title(i_idx[], j_idx[], k_idx[]) : L"\text{Mean spectrum in selected region}",
                    xlabel = L"\text{index along slice axis}",
                    ylabel = unit_label_tex,
                    xtickformat = latex_tick_formatter,
                    ytickformat = latex_tick_formatter,
                )
                CairoMakie.lines!(axP, spec_x_raw[], spec_y_disp[])
                CairoMakie.xlims!(axP, 0f0, Float32(max(0, length(spec_x_raw[]) - 1)))

                CairoMakie.save(String(out), f_spec; backend = CairoMakie)
                @info "Saved spectrum" out
                set_status!("Saved spectrum to $(out).")
            catch e
                msg = "Failed to save spectrum $(out): $(sprint(showerror, e))"
                set_status!(msg)
                @error msg exception=(e, catch_backtrace())
            end
        end
    end

    function current_animation_request()
        a = axis[]; amax = siz[a]
        parse_gif_request(
            get_box_str(start_box),
            get_box_str(stop_box),
            get_box_str(step_box),
            get_box_str(fps_box),
            amax;
            pingpong = pingpong_chk.checked[]
        )
    end

    function start_channel_animation!(frames::Vector{Int}, fps::Int)
        anim_playing[] = true
        play_btn.label[] = "Pause"
        spawn_safely() do
            delay = 1 / max(1, fps)
            try
                while anim_playing[]
                    for fidx in frames
                        anim_playing[] || break
                        slice_slider.value[] = fidx
                        sleep(delay)
                    end
                    loop_chk.checked[] || break
                end
            finally
                anim_playing[] = false
                play_btn.label[] = "Play"
            end
        end
        nothing
    end

    on(play_btn.clicks) do _
        if anim_playing[]
            anim_playing[] = false
            play_btn.label[] = "Play"
            set_status!("Animation paused.")
            return
        end
        ok, frames, fps, msg = current_animation_request()
        set_status!(ok ? "Interactive animation playing at $(fps) FPS." : msg)
        ok || return
        start_channel_animation!(frames, fps)
    end


    # ---------- GIF export  ----------
    on(anim_btn.clicks) do _
        a = axis[]; amax = siz[a]
        ok, frames, fps, msg = current_animation_request()
        set_status!(msg)
        if !ok
            @warn "Invalid GIF parameters" msg axis=a amax=amax
            return
        end

        # strict name: <fits_name>.gif (e.g., synthetic_cube.gif)
        outfile = joinpath(save_root, "$(fname).gif")
        ny, nx = slice_dims(axis[])
        w_img = 640
        h_img = Int(round(w_img * ny / nx))
        extra_for_cb = 80 # colorbar space
        fig_gif = CairoMakie.Figure(size = (w_img + extra_for_cb, h_img))
        axG = CairoMakie.Axis(fig_gif[1, 1]; aspect = DataAspect(), yreversed = true)
        Makie.hidedecorations!(axG, grid = false)

        hmG = CairoMakie.heatmap!(axG, slice_disp; colormap = cm_obs, colorrange = clims_obs)
        CairoMakie.Colorbar(fig_gif[1, 2], hmG; label = "intensity", width = 20)

        try
            record(fig_gif, outfile, frames; framerate = fps) do fidx
                idx[] = fidx
            end
            @info "Animation saved: $outfile"
            set_status!("GIF exported to $(outfile).")
        catch e
            msg2 = "Failed to export animation $(outfile): $(sprint(showerror, e))"
            set_status!(msg2)
            @error msg2 exception=(e, catch_backtrace())
        end
    end


    # ---------- Power spectrum window ----------
    ps_fig_ref = Ref{Any}(nothing)
    ps_alive_ref = Ref(false)

    function open_power_spectrum_window!()
        if ps_alive_ref[] && ps_fig_ref[] !== nothing
            try
                display(ps_fig_ref[])
                return ps_fig_ref[]
            catch
                ps_alive_ref[] = false
                ps_fig_ref[] = nothing
            end
        end

        ps_mode    = Observable(:two_d)   # :two_d | :one_d
        ps_src     = Observable(:zoom)    # :zoom  | :full
        ps_window  = Observable(:hann)    # :hann | :hamming | :none
        ps_pad     = Observable(false)
        ps_nanapo  = Observable(false)
        ps_units   = Observable(:pixel)   # :pixel | :physical
        ps_fit_on  = Observable(false)

        fig_ps = Figure(size = (1000, 760))
        header = fig_ps[1, 1] = GridLayout(; alignmode = Outside(8))

        # Row 1: mode/source/window/units/refresh
        Label(header[1, 1]; text = "Mode", halign = :right, fontsize = 13)
        mode_menu  = Menu(header[1, 2]; options = ["2D", "1D"], prompt = "2D", width = 80)
        Label(header[1, 3]; text = "Source", halign = :right, fontsize = 13)
        src_menu   = Menu(header[1, 4]; options = ["zoom", "full"], prompt = "zoom", width = 80)
        Label(header[1, 5]; text = "Window", halign = :right, fontsize = 13)
        win_menu   = Menu(header[1, 6]; options = ["Hann", "Hamming", "None"], prompt = "Hann", width = 96)
        Label(header[1, 7]; text = "Units", halign = :right, fontsize = 13)
        unit_menu  = Menu(header[1, 8]; options = ["pixel", "physical"], prompt = "pixel", width = 96)
        refresh_btn = Button(header[1, 9]; label = "Refresh", width = 88, height = 30)

        # Row 2: toggles + fit band
        pad_chk    = Checkbox(header[2, 1]); Label(header[2, 2]; text = "Pad pow2", halign = :left, fontsize = 13, color = ui_text)
        nanapo_chk = Checkbox(header[2, 3]); Label(header[2, 4]; text = "NaN apod.", halign = :left, fontsize = 13, color = ui_text)
        fit_chk    = Checkbox(header[2, 5]); Label(header[2, 6]; text = "Fit slope", halign = :left, fontsize = 13, color = ui_text)
        Label(header[2, 7]; text = "k range", halign = :right, fontsize = 13)
        kmin_box   = Textbox(header[2, 8]; placeholder = "k_min", width = 84, height = 28)
        kmax_box   = Textbox(header[2, 9]; placeholder = "k_max", width = 84, height = 28)

        # Row 3: save buttons
        save_png_btn = Button(header[3, 1:2]; label = "Save PNG", width = 120, height = 28)
        save_pdf_btn = Button(header[3, 3:4]; label = "Save PDF", width = 120, height = 28)
        save_csv_btn = Button(header[3, 5:6]; label = "Save CSV (1D)", width = 140, height = 28)

        ps_status = Observable(" ")
        Label(header[3, 7:9]; text = ps_status, halign = :left, fontsize = 12, color = ui_text_muted, tellwidth = false)

        style_menu!(mode_menu); style_menu!(src_menu); style_menu!(win_menu); style_menu!(unit_menu)
        style_button!(refresh_btn); style_button!(save_png_btn); style_button!(save_pdf_btn); style_button!(save_csv_btn)
        style_checkbox!(pad_chk); style_checkbox!(nanapo_chk); style_checkbox!(fit_chk)
        style_textbox!(kmin_box); style_textbox!(kmax_box)

        plot_grid = fig_ps[2, 1] = GridLayout()
        rowsize!(fig_ps.layout, 1, Fixed(120))

        # WCS-derived physical pixel scale (for 1D + 2D physical units).
        u_dim_now() = slice_axis_dims(axis[])[1]
        v_dim_now() = slice_axis_dims(axis[])[2]
        physical_available() = has_wcs(wcs, u_dim_now()) && has_wcs(wcs, v_dim_now())
        pixel_scales() = begin
            if physical_available()
                dy = abs(wcs[u_dim_now()].cdelt)
                dx = abs(wcs[v_dim_now()].cdelt)
                (dx, dy)
            else
                (1.0, 1.0)
            end
        end
        physical_unit_label() = begin
            if physical_available()
                u = wcs[v_dim_now()].cunit
                isempty(u) ? "1" : u
            else
                ""
            end
        end

        # Cache of the latest 1D points, used by Save CSV and Fit.
        last_1d_k     = Float32[]
        last_1d_p     = Float32[]
        last_1d_units = "cycles/pixel"
        last_meta     = (; ny_in = 0, nx_in = 0, ny_eff = 0, nx_eff = 0,
                          padded = false, window = :none, apodized = false,
                          f_sky = 1.0, w_norm = 0.0, k_phys = false, src = "zoom")

        ps_blocks = Any[]
        clear_plot!() = begin
            for b in ps_blocks
                try
                    Makie.delete!(b)
                catch
                end
            end
            empty!(ps_blocks)
        end

        function ps_subimage()
            M = slice_proc[]
            if ps_src[] === :full
                return M
            end
            fl = ax_img.finallimits[]
            x0 = Float64(fl.origin[1])
            y0 = Float64(fl.origin[2])
            x1 = x0 + Float64(fl.widths[1])
            y1 = y0 + Float64(fl.widths[2])
            i_lo = clamp(Int(floor(min(x0, x1))), 1, size(M, 1))
            i_hi = clamp(Int(ceil(max(x0, x1))),  1, size(M, 1))
            j_lo = clamp(Int(floor(min(y0, y1))), 1, size(M, 2))
            j_hi = clamp(Int(ceil(max(y0, y1))),  1, size(M, 2))
            (i_hi <= i_lo || j_hi <= j_lo) && return M
            return M[i_lo:i_hi, j_lo:j_hi]
        end

        function format_status(meta)
            io = IOBuffer()
            print(io, "size $(meta.ny_in)×$(meta.nx_in)")
            if meta.padded
                print(io, " (pad→$(meta.ny_eff)×$(meta.nx_eff))")
            end
            print(io, " • $(meta.src) • ")
            print(io, meta.window === :none ? "none" : titlecase(String(meta.window)))
            if meta.apodized
                print(io, " • NaN apod")
            end
            if meta.f_sky < 1.0
                print(io, " • f_sky=$(round(meta.f_sky; digits = 3))")
            end
            print(io, " • k=", meta.k_phys ? "1/$(physical_unit_label())" : "cycles/pixel")
            return String(take!(io))
        end

        function ps_render!()
            clear_plot!()
            sub = ps_subimage()
            ny0, nx0 = size(sub)
            if ny0 < 4 || nx0 < 4
                lab = Label(plot_grid[1, 1]; text = "Region too small for FFT (need ≥ 4×4).", fontsize = 14)
                push!(ps_blocks, lab)
                ps_status[] = " "
                empty!(last_1d_k); empty!(last_1d_p)
                return
            end
            res = power_spectrum_2d(sub;
                                    window      = ps_window[],
                                    pad_pow2    = ps_pad[],
                                    apodize_nan = ps_nanapo[])
            P2d = res.P2d
            ny, nx = res.ny_eff, res.nx_eff
            src_label = ps_src[] === :full ? "full" : "zoom"
            use_phys = ps_units[] === :physical && physical_available()
            dx, dy = pixel_scales()
            k_unit_lbl = use_phys ? "1/$(physical_unit_label())" : "cycles/pixel"

            meta = (; ny_in = res.ny_in, nx_in = res.nx_in,
                      ny_eff = ny, nx_eff = nx,
                      padded = res.padded, window = res.window,
                      apodized = res.apodized, f_sky = res.f_sky,
                      w_norm = res.w_norm,
                      k_phys = use_phys, src = src_label)
            last_meta = meta

            if ps_mode[] === :two_d
                empty!(last_1d_k); empty!(last_1d_p)
                last_1d_units = k_unit_lbl
                # log10 with floor relative to the spectrum max so DC bin etc.
                # never blow the colormap downward.
                pmax = maximum(P2d)
                floor_val = max(eps(Float64), pmax * 1e-12)
                vis = log10.(max.(P2d, floor_val))
                ax = Axis(
                    plot_grid[1, 1];
                    title  = latexstring("\\text{2D power spectrum (log10) — ", latex_safe(src_label), "}"),
                    xlabel = use_phys ?
                        latexstring("k_x\\;(", latex_safe(k_unit_lbl), ")") :
                        L"k_x\;\text{(cycles/pixel)}",
                    ylabel = use_phys ?
                        latexstring("k_y\\;(", latex_safe(k_unit_lbl), ")") :
                        L"k_y\;\text{(cycles/pixel)}",
                    aspect = DataAspect(),
                    xtickformat = latex_tick_formatter,
                    ytickformat = latex_tick_formatter,
                )
                kx = collect(Float32, (-nx / 2):(nx / 2 - 1)) ./ Float32(nx)
                ky = collect(Float32, (-ny / 2):(ny / 2 - 1)) ./ Float32(ny)
                if use_phys
                    kx ./= Float32(dx)
                    ky ./= Float32(dy)
                end
                hm = heatmap!(ax, kx, ky, vis; colormap = cm_obs[])
                cb = Colorbar(plot_grid[1, 2], hm; label = L"\log_{10}|F|^2", width = 18)
                push!(ps_blocks, ax); push!(ps_blocks, cb)
            else
                radii, prof = power_spectrum_1d_radial(P2d)
                k_cyc = radii ./ Float32(min(ny, nx))      # cycles/pixel
                k = if use_phys
                    Float32.(k_cyc ./ Float32(sqrt(dx * dy)))
                else
                    k_cyc
                end
                pmax = isempty(prof) ? 1.0f0 : maximum(prof)
                floor_val = Float32(max(eps(Float32), pmax * 1f-12))
                p_floored = max.(prof, floor_val)
                resize!(last_1d_k, length(k));  copyto!(last_1d_k, k)
                resize!(last_1d_p, length(prof)); copyto!(last_1d_p, prof)
                last_1d_units = k_unit_lbl

                ax = Axis(
                    plot_grid[1, 1];
                    title  = latexstring("\\text{1D radial power spectrum — ", latex_safe(src_label), "}"),
                    xlabel = use_phys ?
                        latexstring("k\\;(", latex_safe(k_unit_lbl), ")") :
                        L"k\;\text{(cycles/pixel)}",
                    ylabel = L"\langle|F|^2\rangle",
                    yscale = log10,
                    xtickformat = latex_tick_formatter,
                )
                if !isempty(k)
                    lines!(ax, k, p_floored; color = ui_accent, linewidth = 1.8)
                end
                push!(ps_blocks, ax)

                if ps_fit_on[] && length(k) >= 3
                    kmin_txt = get_box_str(kmin_box)
                    kmax_txt = get_box_str(kmax_box)
                    valid_k = filter(>(0), k)
                    auto_lo = isempty(valid_k) ? 0.0 : Float64(first(valid_k))
                    auto_hi = isempty(k) ? Inf : Float64(last(k))
                    kmin_v = isempty(kmin_txt) ? auto_lo : something(tryparse(Float64, kmin_txt), auto_lo)
                    kmax_v = isempty(kmax_txt) ? auto_hi : something(tryparse(Float64, kmax_txt), auto_hi)
                    slope, intercept, n_used = fit_loglog_slope(k, prof; kmin = kmin_v, kmax = kmax_v)
                    if isfinite(slope) && n_used >= 2
                        kfit = filter(ki -> ki > 0 && ki >= kmin_v && ki <= kmax_v, k)
                        if !isempty(kfit)
                            yfit = Float32.(10 .^ (slope .* log10.(Float64.(kfit)) .+ intercept))
                            lines!(ax, kfit, yfit; color = :red, linestyle = :dash, linewidth = 1.5)
                            slope_str = "slope=$(round(slope; digits = 3)) [n=$(n_used)]"
                            ps_status[] = format_status(meta) * " • " * slope_str
                            return
                        end
                    end
                end
            end
            ps_status[] = format_status(meta)
        end

        on(mode_menu.selection)  do sel; sel === nothing || (ps_mode[] = sel == "1D" ? :one_d : :two_d; ps_render!()); end
        on(src_menu.selection)   do sel; sel === nothing || (ps_src[] = sel == "full" ? :full : :zoom; ps_render!()); end
        on(win_menu.selection)   do sel
            sel === nothing && return
            ps_window[] = sel == "Hamming" ? :hamming : sel == "None" ? :none : :hann
            ps_render!()
        end
        on(unit_menu.selection)  do sel
            sel === nothing && return
            ps_units[] = sel == "physical" ? :physical : :pixel
            ps_render!()
        end
        on(pad_chk.checked)    do v; ps_pad[]    = v; ps_render!(); end
        on(nanapo_chk.checked) do v; ps_nanapo[] = v; ps_render!(); end
        on(fit_chk.checked)    do v; ps_fit_on[] = v; ps_render!(); end
        on(kmin_box.stored_string) do _; ps_fit_on[] && ps_render!(); end
        on(kmax_box.stored_string) do _; ps_fit_on[] && ps_render!(); end
        on(refresh_btn.clicks) do _; ps_render!(); end

        ps_save_path(ext) = joinpath(save_root, make_name(get_box_str(fname_box), "powerspec.$(ext)"))
        on(save_png_btn.clicks) do _
            try
                out = ps_save_path("png")
                CairoMakie.save(String(out), fig_ps; backend = CairoMakie)
                set_status!("Saved power spectrum to $(out).")
            catch e
                set_status!("Failed to save PNG: $(sprint(showerror, e))")
            end
        end
        on(save_pdf_btn.clicks) do _
            try
                out = ps_save_path("pdf")
                CairoMakie.save(String(out), fig_ps; backend = CairoMakie)
                set_status!("Saved power spectrum to $(out).")
            catch e
                set_status!("Failed to save PDF: $(sprint(showerror, e))")
            end
        end
        on(save_csv_btn.clicks) do _
            if isempty(last_1d_k)
                set_status!("No 1D points to save (switch to 1D mode first).")
                return
            end
            try
                out = ps_save_path("csv")
                open(String(out), "w") do io
                    println(io, "# window=$(last_meta.window) pad=$(last_meta.padded) nan_apod=$(last_meta.apodized) f_sky=$(last_meta.f_sky) src=$(last_meta.src)")
                    println(io, "k_$(replace(last_1d_units, ' ' => '_')),power")
                    for i in eachindex(last_1d_k)
                        println(io, last_1d_k[i], ",", last_1d_p[i])
                    end
                end
                set_status!("Saved 1D PS CSV to $(out).")
            catch e
                set_status!("Failed to save CSV: $(sprint(showerror, e))")
            end
        end

        ps_render!()
        keepalive!(fig_ps)
        ps_fig_ref[]   = fig_ps
        ps_alive_ref[] = true
        on(fig_ps.scene.events.window_open) do is_open
            if !is_open
                ps_alive_ref[] = false
                ps_fig_ref[]   = nothing
                forget!(fig_ps)
            end
        end
        display(fig_ps)
        return fig_ps
    end

    on(ps_btn.clicks) do _
        try
            open_power_spectrum_window!()
        catch e
            msg = "Failed to open power spectrum: $(sprint(showerror, e))"
            set_status!(msg)
            @error msg exception=(e, catch_backtrace())
        end
    end

    # ---------- Init ----------
    refresh_all!()
    keepalive!(fig)

    on(fig.scene.events.window_open) do is_open
        if !is_open
            forget!(fig)
        end
    end
    if display_fig
        display(fig)
    end
    return fig
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
    hist_pair_obs = lift(img_disp, clims_auto) do im, lim
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
    ax = Axis(
        img_grid[1, 1];
        title = make_main_title(title),
        xlabel = L"\text{pixel x}",
        ylabel = L"\text{pixel y}",
        aspect = DataAspect(),
    )
    hm = heatmap!(ax, img_disp; colormap = cm_obs, colorrange = clims_safe)
    Colorbar(img_grid[1, 2], hm; label = unit_label_tex, width = 20, tellheight = false)

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
        ax = Axis(
            fig[1, i];
            title = make_main_title(title_at(i)),
            aspect = DataAspect(),
            xgridvisible = false,
            ygridvisible = false,
        )
        if is_rgb_like(panel)
            img = as_rgb_image(panel)
            rows, cols = size(img)
            image!(ax, (1, cols), (1, rows), permutedims(img))
        else
            vals = Float32.(panel)
            hm = heatmap!(ax, vals; colormap = cmap_at(i), colorrange = clim_at(i, vals))
            Colorbar(fig[1, N + i], hm; width = 16)
        end
    end
    keepalive!(fig)
    on(fig.scene.events.window_open) do is_open
        is_open || forget!(fig)
    end
    display_fig && display(fig)
    return fig
end

end # module
