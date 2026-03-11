# path: src/CartaViewer.jl
module CartaViewer

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

# ---- helpers ----
include("helpers/Helpers.jl")

spawn_safely(f::Function) = @async try f() catch e
    @error "Background task failed" exception=(e, catch_backtrace())
end

export carta

"""
    carta(filepath::String;
          cmap::Symbol = :viridis,
          vmin = nothing, vmax = nothing,
          invert::Bool = false,
          figsize::Union{Nothing,Tuple{Int,Int}} = nothing,
          save_dir::Union{Nothing,AbstractString} = nothing,
          activate_gl::Bool = true,
          display_fig::Bool = true,
          settings_path::Union{Nothing,AbstractString} = nothing)

Interactive 3D FITS viewer (slice + per-voxel spectrum).
- Manual color limits when `vmin` & `vmax` set (also sync spectrum Y).
- Window sized by explicit `figsize=(w,h)` or a fallback default.
- Export directory configurable via `save_dir`; defaults to your Desktop if
  it exists, otherwise the current working directory.
- `activate_gl=false` allows smoke tests without requiring an OpenGL context.
- `display_fig=false` skips window display (useful for automated tests).
"""
function carta(
    filepath::String;
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

    # ---------- Load ----------
    if !isfile(filepath)
        throw(ArgumentError("FITS file not found: $(abspath(filepath))"))
    end

    cube = try
        FITS(filepath) do f
            read(f[1])
        end
    catch e
        throw(ArgumentError("Failed to read FITS file $(abspath(filepath)): $(sprint(showerror, e))"))
    end

    if ndims(cube) != 3
        throw(ArgumentError("Expected a 3D FITS cube, got ndims=$(ndims(cube)) and size=$(size(cube))."))
    end

    data = Float32.(cube)
    siz  = size(data)  # (nx, ny, nz)
    
    slice_dims(axis::Integer) = if axis == 1
        (siz[2], siz[3])  # (y, z)
    elseif axis == 2
        (siz[1], siz[3])  # (x, z)
    else
        (siz[1], siz[2])  # (x, y)
    end

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

    slice_raw = lift(axis, idx) do a, id
        get_slice(data, a, clamp(id, 1, siz[a]))
    end

    gauss_on = Observable(false)
    sigma    = Observable(1.5f0)

    slice_proc = lift(slice_raw, gauss_on, sigma) do s, on, σ
        if on && σ > 0
            k = ImageFiltering.Kernel.gaussian((σ, σ))
            imfilter(Float32.(s), k)
        else
            s
        end
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

    clims_auto = lift(slice_disp) do s
        clamped_extrema(s)
    end

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

    spec_x_axes = (collect(1:siz[1]), collect(1:siz[2]), collect(1:siz[3]))
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
    fig = Figure(size = _pick_fig_size(figsize))

    main_grid = fig[1, 1] = GridLayout()
    # Image + colorbar
    img_grid  = main_grid[1, 1] = GridLayout()

    ax_img = Axis(
        img_grid[1, 1];
        title     = make_main_title(fname),
        xlabel    = L"\text{pixel } x",
        ylabel    = L"\text{pixel } y",
        aspect    = DataAspect(),
    )

    uv_point = Observable(Point2f(1, 1))
    hm = heatmap!(ax_img, slice_disp; colormap = cm_obs, colorrange = clims_safe)
    scatter!(ax_img, lift(p -> [p], uv_point); markersize = 10)

    # Colorbar linked to plot; tellheight=false avoids layout feedback loops
    Colorbar(img_grid[1, 2], hm; label = L"\text{intensity}", width = 20, tellheight = false)

    # Info + spectrum
    spec_grid = main_grid[1, 2] = GridLayout()
    lab_info = Label(
        spec_grid[1, 1];
        text      = make_info_tex(1, 1, 1, 1, 1, 0f0),
        halign    = :left,
        valign    = :top,
        fontsize  = 14,
        tellwidth = false,
    )

    ax_spec = Axis(
        spec_grid[2, 1];
        title  = L"\text{Spectrum at selected pixel}",
        xlabel = L"\text{index along slice axis}",
        ylabel = L"\text{intensity}",
        width  = 600,
        height = 400,
    )
    lines!(ax_spec, spec_x_raw, spec_y_disp)

    # Controls
    img_ctrl_grid  = main_grid[2, 1] = GridLayout(; alignmode = Outside(), tellwidth = false, tellheight = false)
    spec_ctrl_grid = main_grid[2, 2] = GridLayout(; alignmode = Outside(), tellwidth = false, tellheight = false)

    colgap!(img_ctrl_grid, 6); rowgap!(img_ctrl_grid, 6)
    colgap!(spec_ctrl_grid, 6); rowgap!(spec_ctrl_grid, 6)

    # Image controls (row1)
    im_row1 = img_ctrl_grid[1, 1:2] = GridLayout(; alignmode = Outside())
    colgap!(im_row1, 6)

    Label(im_row1[1, 1], text = L"\text{Image scale}", halign = :left, tellwidth = false)
    img_scale_menu = Menu(im_row1[1, 2]; options = ["lin", "log10", "ln"], prompt = "lin", width = 60)

    Label(im_row1[1, 3], text = L"\text{Spectrum scale}", halign = :left, tellwidth = false)
    spec_scale_menu = Menu(im_row1[1, 4]; options = ["lin", "log10", "ln"], prompt = "lin", width = 60)

    invert_chk = Checkbox(im_row1[1, 5])
    Label(im_row1[1, 6], text = L"\text{Invert colormap}", halign = :left, tellwidth = false)

    reset_zoom_btn = Button(im_row1[1, 7]; label = "Reset zoom", width = 96, height = 26)

    foreach(c -> colsize!(im_row1, c, Auto()), 1:7)

    # Image controls (row2)
    im_row2 = img_ctrl_grid[2, 1:2] = GridLayout(; alignmode = Outside())
    colgap!(im_row2, 4)

    Label(im_row2[1, 1], text = L"\text{Save}", halign = :left, tellwidth = false)
    fmt_menu  = Menu(im_row2[1, 2]; options = ["png", "pdf"], prompt = "png", width = 60)
    fname_box = Textbox(im_row2[1, 3]; placeholder = "filename base", width = 150, height = 24)
    btn_save_img  = Button(im_row2[1, 4]; label = "Save image", width = 96, height = 26)
    btn_save_spec = Button(im_row2[1, 5]; label = "Save spectrum", width = 118, height = 26)
    btn_save_state = Button(im_row2[1, 6]; label = "Save settings", width = 112, height = 26)
    btn_load_state = Button(im_row2[1, 7]; label = "Load settings", width = 112, height = 26)

    foreach(c -> colsize!(im_row2, c, Auto()), 1:7)

    # Image controls (row3)
    im_row3 = img_ctrl_grid[3, 1:2] = GridLayout(; alignmode = Outside())
    colgap!(im_row3, 6)

    Label(im_row3[1, 1], text = L"\text{GIF indices}", halign = :left, tellwidth = false)
    start_box = Textbox(im_row3[1, 2]; placeholder = "start", width = 70, height = 26)
    stop_box  = Textbox(im_row3[1, 3]; placeholder = "stop",  width = 70, height = 26)
    step_box  = Textbox(im_row3[1, 4]; placeholder = "step",  width = 70, height = 26)
    fps_box   = Textbox(im_row3[1, 5]; placeholder = "fps",   width = 70, height = 26)

    pingpong_chk = Checkbox(im_row3[1, 6])
    Label(im_row3[1, 7], text = L"\text{Back-and-forth mode}", halign = :left, tellwidth = false)
    anim_btn = Button(im_row3[1, 8], label = "Export GIF")

    foreach(c -> colsize!(im_row3, c, Auto()), 1:8)

    # Image controls (row4)
    im_row4 = img_ctrl_grid[4, 1:2] = GridLayout(; alignmode = Outside())
    colgap!(im_row4, 6)

    Label(im_row4[1, 1], text = L"\text{Colorbar limits}", halign = :left, tellwidth = false)
    clim_min_box   = Textbox(im_row4[1, 2]; placeholder = "min", width = 100, height = 24)
    clim_max_box   = Textbox(im_row4[1, 3]; placeholder = "max", width = 100, height = 24)
    clim_apply_btn = Button(im_row4[1, 4], label = "Apply")

    foreach(c -> colsize!(im_row4, c, Auto()), 1:4)

    if use_manual[]
        mn, mx = clims_manual[]
        s_mn, s_mx = string(mn), string(mx)
        clim_min_box.displayed_string[] = s_mn; clim_min_box.stored_string[] = s_mn
        clim_max_box.displayed_string[] = s_mx; clim_max_box.stored_string[] = s_mx
    end

    # Spectrum controls moved under the image controls for easier access
    im_row5 = img_ctrl_grid[5, 1:2] = GridLayout(; alignmode = Outside())
    colgap!(im_row5, 6)

    Label(im_row5[1, 1], text = L"\text{Slice axis}", halign = :left, tellwidth = false)
    axes_labels = ["dim1 (x)", "dim2 (y)", "dim3 (z)"]
    axis_menu = Menu(im_row5[1, 2]; options = axes_labels, prompt = "dim3 (z)", width = 90)
    status_label = Label(
        im_row5[1, 3];
        text      = latexstring("\\text{axis } 3,\\, \\text{index } 1"),
        fontsize  = 12,
        halign    = :left,
        tellwidth = false,
    )
    Label(im_row5[1, 4], text = L"\text{Index}", halign = :left, tellwidth = false)
    slice_slider = Slider(im_row5[1, 5]; range = 1:siz[3], startvalue = 1, width = 200, height = 10)

    foreach(c -> colsize!(im_row5, c, Auto()), 1:5)

    im_row6 = img_ctrl_grid[6, 1:2] = GridLayout(; alignmode = Outside())
    colgap!(im_row6, 6)

    Label(im_row6[1, 1], text = L"\text{Gaussian filter}", halign = :left, tellwidth = false)
    gauss_chk   = Checkbox(im_row6[1, 2])
    sigma_label = Label(
        im_row6[1, 3];
        text      = latexstring("\\sigma = 1.5\\,\\text{px}"),
        fontsize  = 12,
        halign    = :left,
        tellwidth = false,
    )

    sigma_slider = Slider(im_row6[1, 4]; range = LinRange(0, 10, 101), startvalue = 1.5, width = 200, height = 10)
    foreach(c -> colsize!(im_row6, c, Auto()), 1:4)
    main_grid[3, 1:2] = Label(
        main_grid[3, 1:2];
        text      = "Shortcuts: arrow keys move the crosshair, mouse click picks a voxel, press 'i' to invert the colormap.",
        halign    = :left,
        tellwidth = false,
    )
    ui_status = Observable("Ready.")
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
        val = data[i_idx[], j_idx[], k_idx[]]
        lab_info.text[] = make_info_tex(i_idx[], j_idx[], k_idx[], u_idx[], v_idx[], val)
        status_label.text[] = latexstring("\\text{axis } $(axis[]),\\, \\text{index } $(idx[])")
    end

    function refresh_spectrum!()
        if axis[] == 1
            spec_x_raw[] = spec_x_axes[1]
            resize!(spec_y_buf, siz[1])
            @views copyto!(spec_y_buf, data[:, j_idx[], k_idx[]])
        elseif axis[] == 2
            spec_x_raw[] = spec_x_axes[2]
            resize!(spec_y_buf, siz[2])
            @views copyto!(spec_y_buf, data[i_idx[], :, k_idx[]])
        else
            spec_x_raw[] = spec_x_axes[3]
            resize!(spec_y_buf, siz[3])
            @views copyto!(spec_y_buf, data[i_idx[], j_idx[], :])
        end
        spec_y_raw[] = spec_y_buf
        if use_manual[]
            vmin_, vmax_ = clims_manual[]; limits!(ax_spec, nothing, nothing, vmin_, vmax_)
        else
            autolimits!(ax_spec)
        end
    end

    refresh_all!() = (refresh_uv!(); refresh_labels!(); refresh_spectrum!())

    # ---------- Reactivity ----------
    on(clims_obs) do (cmin, cmax)
        if use_manual[]
            limits!(ax_spec, nothing, nothing, cmin, cmax)
        end
    end

    on(spec_scale_mode) do _
        if use_manual[]
            vmin_, vmax_ = clims_manual[]; limits!(ax_spec, nothing, nothing, vmin_, vmax_)
        else
            autolimits!(ax_spec)
        end
    end

    on(reset_zoom_btn.clicks) do _
        autolimits!(ax_img)
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
            set_box_text!(clim_min_box, string(first(parsed_clims)))
            set_box_text!(clim_max_box, string(last(parsed_clims)))
        else
            use_manual[] = false
            autolimits!(ax_spec)
        end
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
        if ev.button == Mouse.left && ev.action == Mouse.press
            p = mouseposition(ax_img)
            if any(isnan, p)
                return
            end
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
                axS = CairoMakie.Axis(
                    f_slice[1, 1];
                    title     = make_slice_title(fname, axis[], idx[]),
                    xlabel    = L"\text{pixel } x",
                    ylabel    = L"\text{pixel } y",
                    aspect    = CairoMakie.DataAspect(),
                    yreversed = true,
                )
                hmS = CairoMakie.heatmap!(axS, slice_disp[]; colormap = cm_obs[], colorrange = clims_obs[])
                CairoMakie.scatter!(axS, [Point2f(uv_point[]...)], markersize = 10)
                CairoMakie.Colorbar(f_slice[1, 2], hmS; label = "intensity", width = 20)

                CairoMakie.save(String(out), f_slice)
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
                    title  = make_spec_title(i_idx[], j_idx[], k_idx[]),
                    xlabel = L"\text{index along slice axis}",
                    ylabel = L"\text{intensity}",
                )
                CairoMakie.lines!(axP, spec_x_raw[], spec_y_disp[])

                CairoMakie.save(String(out), f_spec)
                @info "Saved spectrum" out
                set_status!("Saved spectrum to $(out).")
            catch e
                msg = "Failed to save spectrum $(out): $(sprint(showerror, e))"
                set_status!(msg)
                @error msg exception=(e, catch_backtrace())
            end
        end
    end


    # ---------- GIF export  ----------
    on(anim_btn.clicks) do _
        a = axis[]; amax = siz[a]
        ok, frames, fps, msg = parse_gif_request(
            get_box_str(start_box),
            get_box_str(stop_box),
            get_box_str(step_box),
            get_box_str(fps_box),
            amax;
            pingpong = pingpong_chk.checked[]
        )
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

end # module
