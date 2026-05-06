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
    mollweide_pixel_index(res, nx, ny) -> Matrix{Int32}

Précalcule l'index HEALPix (RING) à chaque point de la grille Mollweide.
0 = pixel hors ellipse. Permet de regénérer une nouvelle frame en O(npx)
sans recalculer la projection.
"""
function mollweide_pixel_index(res::Healpix.Resolution, nx::Int, ny::Int)
    idx = zeros(Int32, ny, nx)
    @inbounds for j in 1:ny, i in 1:nx
        x = 2 * (2 * (i - 0.5) / nx - 1)
        y = 2 * (j - 0.5) / ny - 1
        (x^2 / 4 + y^2 > 1) && continue
        θaux = asin(y)
        sinφ = (2θaux + sin(2θaux)) / π
        abs(sinφ) > 1 && continue
        lat = asin(sinφ)
        lon = π * x / (2 * cos(θaux))
        abs(lon) > π && continue
        θhp = π/2 - lat
        φhp = lon < 0 ? lon + 2π : lon
        idx[j, i] = Int32(Healpix.ang2pixRing(res, θhp, φhp))
    end
    idx
end

@inline function mollweide_lonlat_to_xy(lon_deg::Real, lat_deg::Real)
    lat = deg2rad(clamp(Float64(lat_deg), -90.0, 90.0))
    lon = deg2rad(clamp(Float64(lon_deg), -180.0, 180.0))
    target = π * sin(lat)

    θaux = lat
    for _ in 1:14
        f = 2θaux + sin(2θaux) - target
        fp = 2 + 2cos(2θaux)
        abs(fp) < 1e-12 && break
        θaux = clamp(θaux - f / fp, -π/2, π/2)
    end

    x = 2 * lon * cos(θaux) / π
    y = sin(θaux)
    (x^2 / 4 + y^2 > 1 + 1e-5) && return nothing
    return Point2f(Float32(x), Float32(y))
end

_angle_label(v::Real) = begin
    r = round(Int, v)
    r == 0 ? "0°" : "$(r)°"
end

function draw_mollweide_graticule!(
    ax;
    lon_values = -150:30:150,
    lat_values = -60:30:60,
    line_color = RGBAf(1, 1, 1, 0.30),
    label_color = RGBAf(1, 1, 1, 0.62),
    linewidth::Real = 0.9,
    fontsize::Real = 12,
)
    line_plots = Any[]
    for lon in lon_values
        pts = Point2f[]
        for lat in LinRange(-89.0, 89.0, 240)
            p = mollweide_lonlat_to_xy(lon, lat)
            p === nothing || push!(pts, p)
        end
        length(pts) > 1 && push!(
            line_plots,
            lines!(ax, pts; color=line_color, linewidth=linewidth, linestyle=:dot),
        )
    end

    for lat in lat_values
        pts = Point2f[]
        for lon in LinRange(-180.0, 180.0, 360)
            p = mollweide_lonlat_to_xy(lon, lat)
            p === nothing || push!(pts, p)
        end
        length(pts) > 1 && push!(
            line_plots,
            lines!(ax, pts; color=line_color, linewidth=linewidth, linestyle=:dot),
        )
    end

    lon_bottom_pos = Observable(Point2f[])
    lon_bottom_txt = Observable(String[])
    lon_top_pos = Observable(Point2f[])
    lon_top_txt = Observable(String[])
    lat_left_pos = Observable(Point2f[])
    lat_left_txt = Observable(String[])
    lat_right_pos = Observable(Point2f[])
    lat_right_txt = Observable(String[])

    label_plots = Any[
        text!(ax, lon_bottom_pos; text=lon_bottom_txt, color=label_color,
              fontsize=fontsize, align=(:center, :top)),
        text!(ax, lon_top_pos; text=lon_top_txt, color=label_color,
              fontsize=fontsize, align=(:center, :bottom)),
        text!(ax, lat_left_pos; text=lat_left_txt, color=label_color,
              fontsize=fontsize, align=(:right, :center)),
        text!(ax, lat_right_pos; text=lat_right_txt, color=label_color,
              fontsize=fontsize, align=(:left, :center)),
    ]

    graticule = (
        lines = line_plots,
        labels = label_plots,
        lon_values = collect(lon_values),
        lat_values = collect(lat_values),
        lon_bottom_pos = lon_bottom_pos,
        lon_bottom_txt = lon_bottom_txt,
        lon_top_pos = lon_top_pos,
        lon_top_txt = lon_top_txt,
        lat_left_pos = lat_left_pos,
        lat_left_txt = lat_left_txt,
        lat_right_pos = lat_right_pos,
        lat_right_txt = lat_right_txt,
    )
    refresh_graticule_labels!(graticule, ax)
    return graticule
end

function _expanded_graticule_bounds(xlo, xhi, ylo, yhi)
    dx = max(Float64(xhi) - Float64(xlo), 1e-6)
    dy = max(Float64(yhi) - Float64(ylo), 1e-6)
    xpad = 0.075 * dx
    ypad = 0.080 * dy
    return (Float64(xlo) - xpad, Float64(xhi) + xpad,
            Float64(ylo) - ypad, Float64(yhi) + ypad)
end

function set_mollweide_view!(ax, xlo, xhi, ylo, yhi)
    xmin, xmax, ymin, ymax = _expanded_graticule_bounds(xlo, xhi, ylo, yhi)
    limits!(ax, xmin, xmax, ymin, ymax)
end

function _axis_bounds(ax)
    r = ax.finallimits[]
    xlo = Float64(r.origin[1])
    ylo = Float64(r.origin[2])
    return (xlo, xlo + Float64(r.widths[1]), ylo, ylo + Float64(r.widths[2]))
end

function _visible_meridian_points(lon, xlo, xhi, ylo, yhi)
    pts = Point2f[]
    for lat in LinRange(-89.0, 89.0, 360)
        p = mollweide_lonlat_to_xy(lon, lat)
        if p !== nothing && xlo ≤ p[1] ≤ xhi && ylo ≤ p[2] ≤ yhi
            push!(pts, p)
        end
    end
    return pts
end

function _visible_parallel_points(lat, xlo, xhi, ylo, yhi)
    pts = Point2f[]
    for lon in LinRange(-180.0, 180.0, 720)
        p = mollweide_lonlat_to_xy(lon, lat)
        if p !== nothing && xlo ≤ p[1] ≤ xhi && ylo ≤ p[2] ≤ yhi
            push!(pts, p)
        end
    end
    return pts
end

function refresh_graticule_labels!(graticule, ax; bounds = nothing)
    xlo, xhi, ylo, yhi = bounds === nothing ? _axis_bounds(ax) : bounds
    dx = max(xhi - xlo, 1e-6)
    dy = max(yhi - ylo, 1e-6)
    xoff = Float32(0.025 * dx)
    yoff = Float32(0.030 * dy)

    lon_bottom_pos = Point2f[]
    lon_bottom_txt = String[]
    lon_top_pos = Point2f[]
    lon_top_txt = String[]
    for lon in graticule.lon_values
        pts = _visible_meridian_points(lon, xlo, xhi, ylo, yhi)
        isempty(pts) && continue
        pbot = pts[argmin(getindex.(pts, 2))]
        ptop = pts[argmax(getindex.(pts, 2))]
        label = _angle_label(lon)
        push!(lon_bottom_pos, Point2f(pbot[1], pbot[2] - yoff))
        push!(lon_bottom_txt, label)
        push!(lon_top_pos, Point2f(ptop[1], ptop[2] + yoff))
        push!(lon_top_txt, label)
    end

    lat_left_pos = Point2f[]
    lat_left_txt = String[]
    lat_right_pos = Point2f[]
    lat_right_txt = String[]
    for lat in graticule.lat_values
        pts = _visible_parallel_points(lat, xlo, xhi, ylo, yhi)
        isempty(pts) && continue
        pleft = pts[argmin(getindex.(pts, 1))]
        pright = pts[argmax(getindex.(pts, 1))]
        label = _angle_label(lat)
        push!(lat_left_pos, Point2f(pleft[1] - xoff, pleft[2]))
        push!(lat_left_txt, label)
        push!(lat_right_pos, Point2f(pright[1] + xoff, pright[2]))
        push!(lat_right_txt, label)
    end

    graticule.lon_bottom_pos[] = lon_bottom_pos
    graticule.lon_bottom_txt[] = lon_bottom_txt
    graticule.lon_top_pos[] = lon_top_pos
    graticule.lon_top_txt[] = lon_top_txt
    graticule.lat_left_pos[] = lat_left_pos
    graticule.lat_left_txt[] = lat_left_txt
    graticule.lat_right_pos[] = lat_right_pos
    graticule.lat_right_txt[] = lat_right_txt
    return graticule
end

function set_graticule_visible!(graticule, visible::Bool)
    foreach(p -> p.visible[] = visible, graticule.lines)
    foreach(p -> p.visible[] = visible, graticule.labels)
    return graticule
end

"""
    detect_velocity_axis(filepath, ndim) -> (axis, v0, dv, vunit) | nothing

Scan les `CTYPE{i}` (i=1..ndim) de la HDU primaire pour identifier l'axe
vitesse/fréquence. Reconnaît `VRAD`, `VOPT`, `VELO`, `VELOCITY`, `FREQ`,
`FELO`. Si trouvé, lit `CRVAL/CDELT/CRPIX/CUNIT` du même axe et calcule
`v0 = CRVAL - (CRPIX - 1) * CDELT`, `dv = CDELT`. Conversion `m/s → km/s`.

Retourne `nothing` si aucun CTYPE vitesse n'est trouvé. La dim non
détectée est alors l'axe HEALPix.
"""
function detect_velocity_axis(filepath::AbstractString, ndim::Int)
    try
        FITS(String(filepath)) do f
            h = read_header(f[1])
            v_axis = 0
            ctype_found = ""
            for i in 1:ndim
                k = "CTYPE$(i)"
                haskey(h, k) || continue
                ct = uppercase(strip(String(h[k])))
                # On accepte les CTYPE typiques d'un axe spectral : vitesse
                # radio/optique, fréquence, longueur d'onde. On veut juste
                # identifier l'axe non-spatial du cube.
                if startswith(ct, "VRAD") || startswith(ct, "VOPT") ||
                   startswith(ct, "VELO") || startswith(ct, "FREQ") ||
                   startswith(ct, "FELO") || startswith(ct, "WAVE") ||
                   startswith(ct, "AWAV") || ct == "VELOCITY"
                    v_axis = i; ctype_found = ct; break
                end
            end
            v_axis == 0 && return nothing
            kCRVAL = "CRVAL$(v_axis)"
            kCDELT = "CDELT$(v_axis)"
            (haskey(h, kCRVAL) && haskey(h, kCDELT)) || return nothing
            crval = Float64(h[kCRVAL])
            cdelt = Float64(h[kCDELT])
            crpix = haskey(h, "CRPIX$(v_axis)") ? Float64(h["CRPIX$(v_axis)"]) : 1.0
            unit_raw = haskey(h, "CUNIT$(v_axis)") ?
                lowercase(strip(String(h["CUNIT$(v_axis)"]))) : ""
            v0 = crval - (crpix - 1) * cdelt
            dv = cdelt
            unit_norm = unit_raw
            if unit_raw in ("m/s", "m s-1", "m.s-1")
                v0 *= 1e-3; dv *= 1e-3; unit_norm = "km/s"
            elseif unit_raw in ("hz",)
                unit_norm = "Hz"
            elseif unit_raw in ("khz", "mhz", "ghz")
                unit_norm = unit_raw
            elseif isempty(unit_raw)
                # Heuristique : si CTYPE est une vitesse, on suppose km/s ;
                # si c'est une fréquence, on suppose Hz.
                unit_norm = startswith(ctype_found, "F") ? "Hz" : "km/s"
            end
            return (v_axis, v0, dv, unit_norm)
        end
    catch
        return nothing
    end
end

"""
    valid_healpix_npix(n) -> Int

Retourne `nside` si `n = 12·nside²`, sinon 0. Sert à détecter si une
dimension d'un tableau 2D est un nombre HEALPix valide.
"""
function valid_healpix_npix(n::Integer)
    n <= 0 && return 0
    if n % 12 == 0
        s2 = n ÷ 12
        s = isqrt(s2)
        s*s == s2 && (s & (s-1)) == 0 && return s   # nside puissance de 2
    end
    return 0
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
    show_graticule   = Observable(true)

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
    full_map_bounds = (-2.0, 2.0, -1.0, 1.0)
    set_mollweide_view!(ax_img, full_map_bounds...)
    graticule = draw_mollweide_graticule!(ax_img)
    refresh_graticule_labels!(graticule, ax_img; bounds=full_map_bounds)

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
    graticule_chk = Checkbox(ctrl[1,9])
    Label(ctrl[1,10], text="Graticule", halign=:left, tellwidth=false, fontsize=15)
    graticule_chk.checked[] = show_graticule[]
    reset_zoom_btn = Button(ctrl[1,11]; label="Reset zoom", width=120, height=30)
    save_btn       = Button(ctrl[1,12]; label="Save PNG", width=120, height=30)

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
    on(graticule_chk.checked) do v
        show_graticule[] = v
        set_graticule_visible!(graticule, v)
    end
    on(reset_zoom_btn.clicks) do _
        set_mollweide_view!(ax_img, full_map_bounds...)
        refresh_graticule_labels!(graticule, ax_img; bounds=full_map_bounds)
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
            zoom_bounds = (Float64(xmin), Float64(xmax), Float64(ymin), Float64(ymax))
            set_mollweide_view!(ax_img, zoom_bounds...)
            refresh_graticule_labels!(graticule, ax_img; bounds=zoom_bounds)
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

############################
# HEALPix PPV cube viewer
############################

"""
    carta_healpix_cube(filepath::String;
                       cmap=:inferno, vmin=nothing, vmax=nothing,
                       invert=false, scale=:lin,
                       v0=0.0, dv=1.0, vunit="km/s",
                       nx=1200, ny=600,
                       figsize=nothing, save_dir=nothing,
                       activate_gl=true, display_fig=true)

Visualiseur interactif d'un **cube HEALPix-PPV** stocké comme un tableau
2D `(npix, nv)` ou `(nv, npix)` dans un FITS classique. Affiche :

- en haut, la **carte Mollweide** du canal courant ;
- en bas, le **spectre** au pixel cliqué.

Contrôles :
- slider "Channel" → change de canal (réutilise l'index Mollweide
  précalculé, pas de recalcul de projection).
- right-drag → zoom rectangulaire sur la Mollweide.
- left-click → sélectionne un pixel HEALPix, met à jour le spectre.
- échelle, colorbar manuelle, invert colormap, save PNG.

`v0`, `dv`, `vunit` : axe vitesse `v(j) = v0 + (j-1)*dv` pour le spectre.
"""
function carta_healpix_cube(
    filepath::String;
    cmap::Symbol = :inferno,
    vmin = nothing,
    vmax = nothing,
    invert::Bool = false,
    scale::Symbol = :lin,
    v0::Real = 0.0,
    dv::Real = 1.0,
    vunit::AbstractString = "km/s",
    nx::Int = 1200,
    ny::Int = 600,
    figsize::Union{Nothing,Tuple{Int,Int}} = nothing,
    save_dir::Union{Nothing,AbstractString} = nothing,
    activate_gl::Bool = true,
    display_fig::Bool = true,
)
    isfile(filepath) || throw(ArgumentError("FITS file not found: $(abspath(filepath))"))

    raw = FITS(filepath) do f; read(f[1]); end
    ndims(raw) == 2 || throw(ArgumentError("Expected 2D array (npix×nv), got ndims=$(ndims(raw))"))

    s = size(raw)
    nside1 = valid_healpix_npix(s[1])
    nside2 = valid_healpix_npix(s[2])

    # Détection prioritaire : on lit le header pour savoir QUEL axe FITS est
    # spectral. La dim non-spectrale doit alors être un npix HEALPix valide.
    # Sans header → on retombe sur l'heuristique "la dim qui ne l'est pas".
    user_set_wcs = !(v0 == 0.0 && dv == 1.0)
    wcs = user_set_wcs ? nothing : detect_velocity_axis(filepath, 2)

    nside, npix, nv, vaxis, v0_eff, dv_eff, vunit_eff = if wcs !== nothing
        (vax, v0_h, dv_h, unit_h) = wcs
        # vax (FITS axis, 1-based) est l'axe vitesse → l'autre est HEALPix.
        hpix_dim = vax == 1 ? 2 : 1
        nside_h = valid_healpix_npix(s[hpix_dim])
        nside_h == 0 && throw(ArgumentError(
            "Header indique CTYPE$(vax) spectral mais NAXIS$(hpix_dim)=$(s[hpix_dim]) " *
            "n'est pas un npix HEALPix valide (12·nside²)."))
        # vaxis = :last  → cube (npix, nv) (l'axe vitesse est NAXIS2)
        # vaxis = :first → cube (nv, npix) (l'axe vitesse est NAXIS1)
        vax_sym = vax == 2 ? :last : :first
        @info "Velocity axis from FITS header" fits_axis=vax CRVAL=v0_h CDELT=dv_h CUNIT=unit_h
        (nside_h, s[hpix_dim], s[vax], vax_sym, Float64(v0_h), Float64(dv_h), String(unit_h))
    elseif nside1 > 0
        (nside1, s[1], s[2], :last,  Float64(v0), Float64(dv), String(vunit))
    elseif nside2 > 0
        (nside2, s[2], s[1], :first, Float64(v0), Float64(dv), String(vunit))
    else
        throw(ArgumentError("Neither dimension of $(s) is a valid HEALPix npix=12·nside² and no spectral CTYPE in header."))
    end

    # Si AUCUN header WCS et AUCUN kwarg utilisateur → on tombe sur
    # v0=0, dv=1, vunit="km/s" (les defaults de la signature). Mais
    # sémantiquement c'est juste un numéro de canal : on le signale via
    # l'unité et le label x.
    no_wcs = (wcs === nothing) && !user_set_wcs
    if no_wcs
        vunit_eff = "channel"
        v0_eff = 1.0   # canal 1 = v=1
        dv_eff = 1.0
    end
    @info "HEALPix PPV cube" path=abspath(filepath) nside npix nv vaxis v0=v0_eff dv=dv_eff unit=vunit_eff

    # Cube transposé pour avoir un layout (npix, nv) (col-major friendly)
    cube = vaxis === :last ? Float32.(raw) : Float32.(permutedims(raw))
    fname = String(replace(basename(filepath), r"\.fits(\.gz)?$" => ""))

    # ---------- Précalcul de l'index Mollweide (une fois) ----------
    res = Healpix.Resolution(nside)
    ipix_grid = mollweide_pixel_index(res, nx, ny)   # 0 = hors ellipse

    function frame_image(j::Int)
        out = fill(NaN32, ny, nx)
        @inbounds for q in eachindex(ipix_grid)
            ip = ipix_grid[q]
            ip == 0 && continue
            v = cube[ip, j]
            out[q] = (isfinite(v) && v != Float32(Healpix.UNSEEN)) ? v : NaN32
        end
        out
    end

    # ---------- État ----------
    cmap_name   = Observable(cmap)
    invert_cmap = Observable(invert)
    cm_obs = lift(cmap_name, invert_cmap) do name, inv
        base = to_cmap(name); inv ? reverse(base) : base
    end
    scale_mode = Observable(scale)
    chan_idx   = Observable(max(1, nv ÷ 2))

    img_raw = lift(chan_idx) do j
        frame_image(j)
    end
    img_disp = lift(img_raw, scale_mode) do im, m_
        out = apply_scale(im, m_)
        out2 = similar(out, Float32)
        @inbounds for k in eachindex(out)
            x = out[k]; out2[k] = isfinite(x) ? Float32(x) : NaN32
        end
        out2
    end

    # Échelle de couleur globale, calculée dans l'espace transformé (cohérent
    # entre frames). On évalue les quantiles sur tout le cube pour le mode
    # actif. Hypothèse : les `clims_manual` sont dans le même espace que
    # l'image affichée (i.e. l'utilisateur tape les valeurs après log).
    use_manual = Observable(false)
    clims_manual = Observable((0f0, 1f0))
    function _global_clims(mode::Symbol)
        if mode === :lin
            fin = Float32[]
            @inbounds for v in cube
                (isfinite(v) && v != Float32(Healpix.UNSEEN)) && push!(fin, v)
            end
            isempty(fin) && return (0f0, 1f0)
            return (Float32(quantile(fin, 0.01)), Float32(quantile(fin, 0.995)))
        else
            f = mode === :log10 ? log10 : log
            fin = Float32[]
            @inbounds for v in cube
                (isfinite(v) && v != Float32(Healpix.UNSEEN) && v > 0) && push!(fin, Float32(f(v)))
            end
            isempty(fin) && return (-1f0, 1f0)
            return (Float32(quantile(fin, 0.05)), Float32(quantile(fin, 0.995)))
        end
    end
    clims_auto = lift(scale_mode) do m_; _global_clims(m_); end

    if vmin !== nothing && vmax !== nothing
        a, b = Float32(vmin), Float32(vmax)
        a == b && (a = prevfloat(a); b = nextfloat(b))
        clims_manual[] = (a, b); use_manual[] = true
    end
    clims_obs = lift(use_manual, clims_auto, clims_manual) do um, ca, cm
        um ? cm : ca
    end
    clims_safe = lift(clims_obs) do (lo, hi)
        (isfinite(lo) && isfinite(hi) && lo != hi) ? (lo, hi) : (0f0, 1f0)
    end

    zoom_drag_active = Observable(false)
    zoom_drag_start  = Observable(Point2f(NaN32, NaN32))
    zoom_drag_end    = Observable(Point2f(NaN32, NaN32))
    show_graticule   = Observable(true)

    # Pixel sélectionné (initial : centre)
    sel_ipix  = Observable(0)
    sel_xy    = Observable(Point2f(NaN32, NaN32))
    sel_label = Observable(latexstring("\\text{click on map to select a pixel}"))

    spec_x = Float32.(v0_eff .+ (0:nv-1) .* dv_eff)
    spec_y_obs = Observable(zeros(Float32, nv))
    function update_spectrum!(ip::Int)
        if 1 ≤ ip ≤ npix
            sel_ipix[] = ip
            spec_y_obs[] = Float32.(@view cube[ip, :])
            θ, φ = Healpix.pix2angRing(res, ip)
            l_deg = rad2deg(φ); b_deg = 90 - rad2deg(θ)
            sel_label[] = latexstring(
                "\\text{pixel ", ip, "}\\;(l, b) = (",
                string(round(mod(l_deg, 360); digits=2)), "^\\circ, ",
                string(round(b_deg; digits=2)), "^\\circ)")
        end
    end

    # ---------- Figure ----------
    activate_gl ? GLMakie.activate!() : CairoMakie.activate!()
    fig = Figure(size = _pick_fig_size(figsize))
    main_grid = fig[1, 1] = GridLayout()

    # Carte
    map_grid = main_grid[1, 1] = GridLayout()
    is_channel_axis = (vunit_eff == "channel")
    title_obs = lift(chan_idx) do j
        v = v0_eff + (j-1)*dv_eff
        if is_channel_axis
            latexstring("\\text{", latex_safe(fname), "}\\;\\text{ch}=", j)
        else
            latexstring("\\text{", latex_safe(fname), "}\\;\\text{ch}=", j,
                        ",\\;v=", string(round(v; digits=2)), "\\,\\mathrm{",
                        latex_safe(vunit_eff), "}")
        end
    end
    ax_img = Axis(map_grid[1, 1];
        title = title_obs,
        aspect = DataAspect(),
        xgridvisible = false, ygridvisible = false,
        xticksvisible = false, yticksvisible = false,
        xticklabelsvisible = false, yticklabelsvisible = false,
        bottomspinevisible = false, topspinevisible = false,
        leftspinevisible   = false, rightspinevisible = false)

    xs = LinRange(-2f0, 2f0, nx)
    ys = LinRange(-1f0, 1f0, ny)
    img_for_plot = lift(img_disp) do im; permutedims(im); end
    hm = heatmap!(ax_img, xs, ys, img_for_plot;
                  colormap=cm_obs, colorrange=clims_safe, nan_color=:white)
    full_map_bounds = (-2.0, 2.0, -1.0, 1.0)
    set_mollweide_view!(ax_img, full_map_bounds...)
    graticule = draw_mollweide_graticule!(ax_img)
    refresh_graticule_labels!(graticule, ax_img; bounds=full_map_bounds)

    # ellipse + zoom box + marker
    ell_x = [2cos(t) for t in LinRange(0, 2π, 200)]
    ell_y = [sin(t)  for t in LinRange(0, 2π, 200)]
    lines!(ax_img, ell_x, ell_y; color=:black, linewidth=0.8)

    ui_accent = RGBf(0.12, 0.45, 0.82)
    zoom_box_segments = lift(zoom_drag_active, zoom_drag_start, zoom_drag_end) do active, p0, p1
        active || return Point2f[]
        (isfinite(p0[1]) && isfinite(p1[1])) || return Point2f[]
        x0,y0 = p0; x1,y1 = p1
        Point2f[Point2f(x0,y0),Point2f(x1,y0),Point2f(x1,y0),Point2f(x1,y1),
                Point2f(x1,y1),Point2f(x0,y1),Point2f(x0,y1),Point2f(x0,y0)]
    end
    linesegments!(ax_img, zoom_box_segments; color=(ui_accent,0.95),
                  linewidth=2.0, linestyle=:dash)
    marker_pts = lift(sel_xy) do p
        (isfinite(p[1]) && isfinite(p[2])) ? Point2f[p] : Point2f[]
    end
    scatter!(ax_img, marker_pts; color=ui_accent, markersize=12, marker=:cross)

    Colorbar(map_grid[1, 2], hm; label=L"\text{value}", width=18)

    # Spectre
    # Affiché dans le même espace que la carte (lin/log10/ln) → cohérence
    # avec la colorbar : le spectre est mis à l'échelle, et les bornes
    # `clims_manual` (entrées par l'utilisateur dans le même espace
    # transformé) lui sont appliquées en y-limits.
    spec_y_disp = lift(spec_y_obs, scale_mode) do y, m_
        out = apply_scale(y, m_)
        out2 = similar(out, Float32)
        @inbounds for k in eachindex(out)
            x = out[k]; out2[k] = isfinite(x) ? Float32(x) : NaN32
        end
        out2
    end
    ax_spec = Axis(main_grid[2, 1];
        title  = sel_label,
        xlabel = is_channel_axis ?
            L"\text{channel}" :
            latexstring("v\\;[\\mathrm{", latex_safe(vunit_eff), "}]"),
        ylabel = lift(m_ -> m_ === :lin   ? L"T_b" :
                            m_ === :log10 ? L"\log_{10}\,T_b" :
                                            L"\ln\,T_b", scale_mode),
        xgridvisible = false, ygridvisible = false)
    lines!(ax_spec, spec_x, spec_y_disp; color=:black, linewidth=1.5)
    # ligne verticale à v(chan_idx)
    chan_v = lift(chan_idx) do j; Float32(v0_eff + (j-1)*dv_eff); end
    vlines!(ax_spec, lift(v -> [v], chan_v); color=ui_accent, linewidth=1.2, linestyle=:dash)

    # ylimits du spectre : suit clims si manuel, sinon auto sur le spectre courant
    function _refresh_spec_ylim!()
        if use_manual[]
            lo, hi = clims_manual[]
            ylims!(ax_spec, Float32(lo), Float32(hi))
        else
            ys = spec_y_disp[]
            fin = filter(isfinite, ys)
            if isempty(fin)
                autolimits!(ax_spec)
            else
                lo = Float32(minimum(fin)); hi = Float32(maximum(fin))
                lo == hi && (lo = prevfloat(lo); hi = nextfloat(hi))
                ylims!(ax_spec, lo, hi)
            end
        end
        xlims!(ax_spec, Float32(spec_x[1]), Float32(spec_x[end]))
    end

    # Contrôles
    ctrl = main_grid[3, 1] = GridLayout(; alignmode=Outside())
    Label(ctrl[1,1], text=L"\text{Channel}", halign=:left, tellwidth=false, fontsize=15)
    chan_slider = Slider(ctrl[1,2]; range=1:nv, startvalue=chan_idx[],
                         width=320, height=14)
    chan_label  = Label(ctrl[1,3];
        text=lift(j -> is_channel_axis ?
                latexstring("j=", j) :
                latexstring("j=", j, ",\\;v=", string(round(v0_eff+(j-1)*dv_eff; digits=2)),
                            "\\,\\mathrm{", latex_safe(vunit_eff), "}"), chan_idx),
        fontsize=15, halign=:left, tellwidth=true, width=160)

    Label(ctrl[1,4], text=L"\text{Scale}", halign=:left, tellwidth=false, fontsize=15)
    scale_menu = Menu(ctrl[1,5]; options=["lin","log10","ln"], prompt=String(scale), width=92)
    invert_chk = Checkbox(ctrl[1,6]); Label(ctrl[1,7], text="Invert", halign=:left, tellwidth=false, fontsize=15)
    invert_chk.checked[] = invert_cmap[]

    Label(ctrl[1,8], text=L"\text{Colorbar}", halign=:left, tellwidth=false, fontsize=15)
    clim_min_box = Textbox(ctrl[1,9];  placeholder="min", width=100, height=30)
    clim_max_box = Textbox(ctrl[1,10]; placeholder="max", width=100, height=30)
    apply_btn    = Button(ctrl[1,11]; label="Apply",      width=80,  height=30)
    graticule_chk = Checkbox(ctrl[1,12])
    Label(ctrl[1,13], text="Graticule", halign=:left, tellwidth=false, fontsize=15)
    graticule_chk.checked[] = show_graticule[]
    reset_btn    = Button(ctrl[1,14]; label="Reset zoom", width=120, height=30)
    save_btn     = Button(ctrl[1,15]; label="Save PNG",   width=110, height=30)

    if use_manual[]
        a, b = clims_manual[]
        sa, sb = string(a), string(b)
        clim_min_box.displayed_string[] = sa; clim_min_box.stored_string[] = sa
        clim_max_box.displayed_string[] = sb; clim_max_box.stored_string[] = sb
    end

    # ---------- Reactivity ----------
    on(chan_slider.value) do v
        chan_idx[] = Int(round(v))
    end
    on(scale_menu.selection) do sel
        sel === nothing && return
        new_mode = Symbol(sel)
        new_mode === scale_mode[] && return
        # Les clims_manual étaient exprimées dans l'ancien espace (lin/log10/ln).
        # Les invalider et vider les textboxes pour repartir en auto dans le
        # nouvel espace — sinon le spectre et la colorbar restent bloqués sur
        # des bornes incohérentes.
        if use_manual[]
            use_manual[] = false
        end
        clim_min_box.displayed_string[] = ""; clim_min_box.stored_string[] = ""
        clim_max_box.displayed_string[] = ""; clim_max_box.stored_string[] = ""
        scale_mode[] = new_mode
    end
    on(invert_chk.checked) do v; invert_cmap[] = v; end
    on(graticule_chk.checked) do v
        show_graticule[] = v
        set_graticule_visible!(graticule, v)
    end
    on(reset_btn.clicks) do _
        set_mollweide_view!(ax_img, full_map_bounds...)
        refresh_graticule_labels!(graticule, ax_img; bounds=full_map_bounds)
    end
    on(apply_btn.clicks) do _
        ok, manual, clims, _msg = parse_manual_clims(
            get_box_str(clim_min_box), get_box_str(clim_max_box);
            fallback = clims_manual[])
        ok || return
        if manual
            clims_manual[] = clims
            use_manual[]   = true
        else
            use_manual[]   = false
        end
        _refresh_spec_ylim!()                # propage au spectre
    end
    on(scale_mode)        do _; _refresh_spec_ylim!(); end
    on(spec_y_disp)       do _; _refresh_spec_ylim!(); end
    on(use_manual)        do _; _refresh_spec_ylim!(); end
    on(clims_manual)      do _; _refresh_spec_ylim!(); end

    # zoom right-drag + click left → select pixel
    on(events(ax_img).mousebutton) do ev
        if ev.button == Mouse.right && ev.action == Mouse.press
            p = mouseposition(ax_img); any(isnan, p) && return
            zoom_drag_start[] = Point2f(p[1], p[2])
            zoom_drag_end[]   = Point2f(p[1], p[2])
            zoom_drag_active[] = true
        elseif ev.button == Mouse.right && ev.action == Mouse.release
            zoom_drag_active[] || return
            p = mouseposition(ax_img); !any(isnan, p) && (zoom_drag_end[] = Point2f(p[1], p[2]))
            p0 = zoom_drag_start[]; p1 = zoom_drag_end[]
            zoom_drag_active[] = false
            zoom_drag_start[] = Point2f(NaN32, NaN32); zoom_drag_end[] = Point2f(NaN32, NaN32)
            (isfinite(p0[1]) && isfinite(p1[1])) || return
            xmin,xmax = minmax(p0[1], p1[1]); ymin,ymax = minmax(p0[2], p1[2])
            (abs(xmax-xmin) < 1e-3 || abs(ymax-ymin) < 1e-3) && return
            zoom_bounds = (Float64(xmin), Float64(xmax), Float64(ymin), Float64(ymax))
            set_mollweide_view!(ax_img, zoom_bounds...)
            refresh_graticule_labels!(graticule, ax_img; bounds=zoom_bounds)
        elseif ev.button == Mouse.left && ev.action == Mouse.press
            p = mouseposition(ax_img); any(isnan, p) && return
            ll = mollweide_xy_to_lonlat(p[1], p[2]); ll === nothing && return
            l_deg, b_deg = ll
            θhp = deg2rad(90 - b_deg); φhp = deg2rad(mod(l_deg, 360))
            ip = Healpix.ang2pixRing(res, θhp, φhp)
            sel_xy[] = Point2f(p[1], p[2])
            update_spectrum!(ip)
        end
    end
    on(events(ax_img).mouseposition) do p
        if zoom_drag_active[] && !any(isnan, p)
            zoom_drag_end[] = Point2f(p[1], p[2])
        end
    end

    # save PNG
    save_root = save_dir === nothing ? begin
        d = joinpath(homedir(), "Desktop"); isdir(d) ? d : pwd()
    end : (isdir(save_dir) ? String(save_dir) : (mkpath(save_dir); String(save_dir)))
    on(save_btn.clicks) do _
        out = joinpath(save_root, "$(fname)_ch$(chan_idx[]).png")
        try CairoMakie.save(String(out), fig; backend=CairoMakie); @info "Saved" out
        catch e; @error "Failed to save" exception=(e, catch_backtrace()) end
    end

    # init
    update_spectrum!(max(1, npix ÷ 2))     # spectre par défaut au pixel central
    _refresh_spec_ylim!()

    # Espacement vertical : éloigne la ligne de contrôles des xticks du
    # spectre pour éviter le chevauchement (ex: "j=41, v=80km/s" qui se
    # superposait au tick "80").
    try
        rowgap!(main_grid, 2, 22)
        rowgap!(main_grid, 1, 6)
    catch
        # rowgap! échoue si l'index est hors limites — silencieux.
    end
    try
        colgap!(ctrl, 10)
    catch
    end

    keepalive!(fig)
    on(fig.scene.events.window_open) do is_open
        is_open || forget!(fig)
    end
    display_fig && display(fig)
    return fig
end
