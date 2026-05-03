# HEALPix Mollweide viewer with interactive zoom.
# API publique : `is_healpix_fits`, `read_healpix_map`, `mollweide_grid`,
# `carta_healpix(filepath; ...)`.
#
# Compatible avec les conventions de `carta(...)` (zoom right-drag, reset,
# colormap, vlims, save image, échelles lin/log10/ln).

using GLMakie, CairoMakie, Makie, Observables, FITSIO, LaTeXStrings
using Healpix

############################
# Détection / Lecture
############################

"""
    is_healpix_fits(path) -> Bool

Heuristique : un fichier HEALPix expose `PIXTYPE = 'HEALPIX'` dans le
header d'une extension BinTable. On lit les headers sans charger les
données.
"""
function is_healpix_fits(path::AbstractString)
    isfile(path) || return false
    try
        FITS(path) do f
            for hdu in f
                hdr = read_header(hdu)
                if haskey(hdr, "PIXTYPE")
                    val = uppercase(strip(string(hdr["PIXTYPE"])))
                    val == "HEALPIX" && return true
                end
            end
            return false
        end
    catch
        return false
    end
end

"""
    read_healpix_map(path; column=1) -> (HealpixMap, header_dict)

Lit la carte HEALPix (RING ou NESTED auto-détecté). `column` est le
numéro de colonne dans la BinTable (1 pour I_STOKES, etc.).
Retourne aussi le header de l'extension lue, utile pour récupérer
unités et noms.
"""
function read_healpix_map(path::AbstractString; column::Int=1)
    m = Healpix.readMapFromFITS(String(path), column, Float64)
    hdr = FITS(path) do f
        # Le header de la HDU 2 (BinTable HEALPix) contient les infos utiles.
        h = length(f) >= 2 ? read_header(f[2]) : read_header(f[1])
        Dict{String,Any}(string(k) => h[k] for k in keys(h))
    end
    return m, hdr
end

############################
# Projection Mollweide
############################

"""
    mollweide_grid(map; nx=1200, ny=600) -> Matrix{Float32}

Reprojette `map::HealpixMap` sur une grille `(ny, nx)` Mollweide centrée
sur `(l, b) = (0, 0)`. Pixels hors ellipse → `NaN32`. Convention :
- `x ∈ [-2, 2]`, `y ∈ [-1, 1]`
- longitude croît vers la gauche dans la convention astro ; on garde la
  même convention que `Healpix.jl` (lon = φ HEALPix).
"""
function mollweide_grid(m::Healpix.HealpixMap; nx::Int=1200, ny::Int=600)
    img = fill(NaN32, ny, nx)
    res = m.resolution
    @inbounds for j in 1:ny, i in 1:nx
        x = 2 * (2 * (i - 0.5) / nx - 1)        # x ∈ [-2, 2]
        y = 2 * (j - 0.5) / ny - 1              # y ∈ [-1, 1]
        (x^2 / 4 + y^2 > 1) && continue
        θaux = asin(y)
        sinφ = (2θaux + sin(2θaux)) / π
        abs(sinφ) > 1 && continue
        lat = asin(sinφ)
        lon = π * x / (2 * cos(θaux))
        abs(lon) > π && continue
        θhp = π/2 - lat
        φhp = lon < 0 ? lon + 2π : lon
        ipix = Healpix.ang2pixRing(res, θhp, φhp)
        v = m.pixels[ipix]
        img[j, i] = (isfinite(v) && v != Healpix.UNSEEN) ? Float32(v) : NaN32
    end
    img
end

"""
    mollweide_xy_to_lonlat(x, y) -> (lon_deg, lat_deg) | nothing

Inverse de la projection Mollweide. Retourne `nothing` si (x,y) est hors
ellipse. Longitude ∈ (-180°, 180°], latitude ∈ [-90°, 90°].
"""
@inline function mollweide_xy_to_lonlat(x::Real, y::Real)
    (x^2 / 4 + y^2 > 1) && return nothing
    θaux = asin(y)
    sinφ = (2θaux + sin(2θaux)) / π
    abs(sinφ) > 1 && return nothing
    lat = asin(sinφ)
    lon = π * x / (2 * cos(θaux))
    abs(lon) > π && return nothing
    return (rad2deg(lon), rad2deg(lat))
end

############################
# Viewer interactif
############################

"""
    carta_healpix(filepath::String;
                  cmap=:inferno, vmin=nothing, vmax=nothing,
                  invert=false, scale=:lin, column=1,
                  nx=1400, ny=700,
                  figsize=nothing, save_dir=nothing,
                  activate_gl=true, display_fig=true)

Visualiseur interactif HEALPix en projection Mollweide.

- **Zoom** : maintenir clic-droit et glisser pour dessiner un rectangle ;
  le bouton "Reset zoom" rétablit la vue complète.
- **Hover/clic gauche** : affiche `(l, b)` galactiques et la valeur du
  pixel.
- **Échelle** : `:lin`, `:log10`, `:ln` (sélectionnable au runtime).
- **Colorbar** : auto (quantiles 2/98 % en lin, 5/98 % en log) ou
  `vmin`/`vmax` manuels.

Retourne la `Figure` GLMakie.
"""
function carta_healpix(
    filepath::String;
    cmap::Symbol = :inferno,
    vmin = nothing,
    vmax = nothing,
    invert::Bool = false,
    scale::Symbol = :lin,
    column::Int = 1,
    nx::Int = 1400,
    ny::Int = 700,
    figsize::Union{Nothing,Tuple{Int,Int}} = nothing,
    save_dir::Union{Nothing,AbstractString} = nothing,
    activate_gl::Bool = true,
    display_fig::Bool = true,
)
    isfile(filepath) || throw(ArgumentError("HEALPix FITS not found: $(abspath(filepath))"))

    m, hdr = read_healpix_map(filepath; column=column)
    @info "HEALPix map" path=abspath(filepath) nside=m.resolution.nside npix=length(m)
    fname_full = basename(filepath)
    fname = String(replace(fname_full, r"\.fits(\.gz)?$" => ""))

    unit_str = String(get(hdr, "TUNIT$column", get(hdr, "BUNIT", "")))

    # ---------- Reprojection (une seule fois, conservée en mémoire) ----------
    img_raw = mollweide_grid(m; nx=nx, ny=ny)

    # ---------- État ----------
    cmap_name   = Observable(cmap)
    invert_cmap = Observable(invert)
    cm_obs = lift(cmap_name, invert_cmap) do name, inv
        base = to_cmap(name); inv ? reverse(base) : base
    end

    scale_mode = Observable(scale)
    img_disp = lift(scale_mode) do m_
        out = apply_scale(img_raw, m_)
        # protect: NaN/Inf already turned to NaN by apply_scale in log modes
        out2 = similar(out, Float32)
        @inbounds for k in eachindex(out)
            x = out[k]
            out2[k] = isfinite(x) ? Float32(x) : Float32(NaN32)
        end
        out2
    end

    use_manual = Observable(false)
    clims_manual = Observable((0f0, 1f0))
    clims_auto = lift(img_disp) do im
        fin = filter(isfinite, im)
        isempty(fin) && return (0f0, 1f0)
        qlo = scale_mode[] === :lin ? 0.02 : 0.05
        (Float32(quantile(fin, qlo)), Float32(quantile(fin, 0.98)))
    end
    if vmin !== nothing && vmax !== nothing
        a, b = Float32(vmin), Float32(vmax)
        a == b && (a = prevfloat(a); b = nextfloat(b))
        clims_manual[] = (a, b); use_manual[] = true
    end
    clims_obs = lift(use_manual, clims_auto, clims_manual) do um, ca, cm
        um ? cm : ca
    end
    clims_safe = lift(clims_obs) do (lo, hi)
        if !(isfinite(lo) && isfinite(hi)) || lo == hi
            (0f0, 1f0)
        else
            (lo, hi)
        end
    end

    zoom_drag_active = Observable(false)
    zoom_drag_start  = Observable(Point2f(NaN32, NaN32))
    zoom_drag_end    = Observable(Point2f(NaN32, NaN32))

    ui_accent = RGBf(0.12, 0.45, 0.82)

    # ---------- Figure ----------
    activate_gl ? GLMakie.activate!() : CairoMakie.activate!()
    fig = Figure(size = _pick_fig_size(figsize))

    main_grid = fig[1, 1] = GridLayout()

    ax_img = Axis(
        main_grid[1, 1];
        title  = make_main_title(fname),
        aspect = DataAspect(),
        xgridvisible = false, ygridvisible = false,
        xticksvisible = false, yticksvisible = false,
        xticklabelsvisible = false, yticklabelsvisible = false,
        bottomspinevisible = false, topspinevisible = false,
        leftspinevisible   = false, rightspinevisible = false,
    )
    xs = LinRange(-2f0, 2f0, nx)
    ys = LinRange(-1f0, 1f0, ny)
    img_for_plot = lift(img_disp) do im
        permutedims(im)  # (nx, ny) layout pour heatmap(xs, ys, A)
    end
    hm = heatmap!(ax_img, xs, ys, img_for_plot;
                  colormap=cm_obs, colorrange=clims_safe, nan_color=:white)
    limits!(ax_img, -2.05, 2.05, -1.05, 1.05)

    # cadre ellipse Mollweide (purement esthétique)
    ell_x = [2cos(t) for t in LinRange(0, 2π, 200)]
    ell_y = [sin(t)  for t in LinRange(0, 2π, 200)]
    lines!(ax_img, ell_x, ell_y; color=:black, linewidth=0.8)

    # rectangle de zoom
    zoom_box_segments = lift(zoom_drag_active, zoom_drag_start, zoom_drag_end) do active, p0, p1
        active || return Point2f[]
        if !(isfinite(p0[1]) && isfinite(p0[2]) && isfinite(p1[1]) && isfinite(p1[2]))
            return Point2f[]
        end
        x0, y0 = p0; x1, y1 = p1
        Point2f[
            Point2f(x0,y0), Point2f(x1,y0),
            Point2f(x1,y0), Point2f(x1,y1),
            Point2f(x1,y1), Point2f(x0,y1),
            Point2f(x0,y1), Point2f(x0,y0),
        ]
    end
    linesegments!(ax_img, zoom_box_segments; color=(ui_accent, 0.95),
                  linewidth=2.0, linestyle=:dash)

    Colorbar(main_grid[1, 2], hm;
             label = isempty(unit_str) ? L"\text{value}" : latexstring("\\text{", latex_safe(unit_str), "}"),
             width = 18)

    # Bandeau info
    info_obs = Observable(latexstring("\\text{move cursor over the map}"))
    Label(main_grid[2, 1:2], info_obs; halign=:left, fontsize=15)

    # Contrôles
    ctrl = main_grid[3, 1:2] = GridLayout(; alignmode=Outside())
    Label(ctrl[1,1], text=L"\text{Scale}", halign=:left, tellwidth=false, fontsize=15)
    scale_menu = Menu(ctrl[1,2]; options=["lin","log10","ln"],
                     prompt = String(scale), width=92)
    invert_chk = Checkbox(ctrl[1,3])
    Label(ctrl[1,4], text="Invert colormap", halign=:left, tellwidth=false, fontsize=15)
    invert_chk.checked[] = invert_cmap[]

    Label(ctrl[1,5], text=L"\text{Colorbar}", halign=:left, tellwidth=false, fontsize=15)
    clim_min_box = Textbox(ctrl[1,6]; placeholder="min", width=110, height=30)
    clim_max_box = Textbox(ctrl[1,7]; placeholder="max", width=110, height=30)
    apply_btn    = Button(ctrl[1,8]; label="Apply", width=80, height=30)
    reset_zoom_btn = Button(ctrl[1,9]; label="Reset zoom", width=120, height=30)
    save_btn       = Button(ctrl[1,10]; label="Save PNG", width=120, height=30)

    if use_manual[]
        a, b = clims_manual[]
        s_a = string(a); s_b = string(b)
        clim_min_box.displayed_string[] = s_a; clim_min_box.stored_string[] = s_a
        clim_max_box.displayed_string[] = s_b; clim_max_box.stored_string[] = s_b
    end

    # ---------- Reactivity ----------
    on(scale_menu.selection) do sel
        sel === nothing && return
        scale_mode[] = Symbol(sel)
    end
    on(invert_chk.checked) do v; invert_cmap[] = v; end
    on(reset_zoom_btn.clicks) do _
        limits!(ax_img, -2.05, 2.05, -1.05, 1.05)
    end
    on(apply_btn.clicks) do _
        ok, manual, clims, _msg = parse_manual_clims(
            get_box_str(clim_min_box), get_box_str(clim_max_box);
            fallback = clims_manual[])
        ok || return
        if manual
            clims_manual[] = clims; use_manual[] = true
        else
            use_manual[] = false
        end
    end

    # zoom right-drag, identique à `carta`
    on(events(ax_img).mousebutton) do ev
        if ev.button == Mouse.right && ev.action == Mouse.press
            p = mouseposition(ax_img); any(isnan, p) && return
            zoom_drag_start[] = Point2f(p[1], p[2])
            zoom_drag_end[]   = Point2f(p[1], p[2])
            zoom_drag_active[] = true
        elseif ev.button == Mouse.right && ev.action == Mouse.release
            zoom_drag_active[] || return
            p = mouseposition(ax_img)
            !any(isnan, p) && (zoom_drag_end[] = Point2f(p[1], p[2]))
            p0 = zoom_drag_start[]; p1 = zoom_drag_end[]
            zoom_drag_active[] = false
            zoom_drag_start[] = Point2f(NaN32, NaN32)
            zoom_drag_end[]   = Point2f(NaN32, NaN32)
            (isfinite(p0[1]) && isfinite(p1[1])) || return
            xmin, xmax = minmax(p0[1], p1[1])
            ymin, ymax = minmax(p0[2], p1[2])
            (abs(xmax-xmin) < 1e-3 || abs(ymax-ymin) < 1e-3) && return
            limits!(ax_img, xmin, xmax, ymin, ymax)
        end
    end
    on(events(ax_img).mouseposition) do p
        if zoom_drag_active[] && !any(isnan, p)
            zoom_drag_end[] = Point2f(p[1], p[2])
        end
        # info hover (l, b)
        ll = mollweide_xy_to_lonlat(p[1], p[2])
        if ll === nothing
            info_obs[] = latexstring("\\text{outside Mollweide ellipse}")
        else
            l_deg, b_deg = ll
            # on remappe l ∈ (-180, 180] → l ∈ [0, 360) pour conv. astro
            l_disp = mod(l_deg, 360)
            θhp = deg2rad(90 - b_deg)
            φhp = deg2rad(mod(l_deg, 360))
            ipix = Healpix.ang2pixRing(m.resolution, θhp, φhp)
            val  = m.pixels[ipix]
            valstr = (isfinite(val) && val != Healpix.UNSEEN) ?
                     string(round(Float32(val); digits=4)) : "NaN"
            info_obs[] = latexstring(
                "(l, b) = (",
                string(round(l_disp; digits=2)), "^\\circ, ",
                string(round(b_deg; digits=2)), "^\\circ),\\;",
                "\\text{value} = ", valstr,
                isempty(unit_str) ? "" : ("\\;\\mathrm{", latex_safe(unit_str), "}")
            )
        end
    end

    # save image
    save_root = save_dir === nothing ? begin
        d = joinpath(homedir(), "Desktop"); isdir(d) ? d : pwd()
    end : (isdir(save_dir) ? String(save_dir) : (mkpath(save_dir); String(save_dir)))

    on(save_btn.clicks) do _
        out = joinpath(save_root, "$(fname)_mollweide.png")
        try
            CairoMakie.save(String(out), fig; backend=CairoMakie)
            @info "Saved" out
        catch e
            @error "Failed to save" exception=(e, catch_backtrace())
        end
    end

    keepalive!(fig)
    on(fig.scene.events.window_open) do is_open
        is_open || forget!(fig)
    end
    display_fig && display(fig)
    return fig
end
