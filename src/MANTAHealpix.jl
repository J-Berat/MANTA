# HEALPix Mollweide viewer with interactive zoom.
# API publique : `is_healpix_fits`, `read_healpix_map`, `mollweide_grid`,
# `manta_healpix(filepath; ...)`.
#
# Compatible avec les conventions de `manta(...)` (zoom right-drag, reset,
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

# ---- HEALPix projection / graticule helpers ----
include("views/HealpixProjection.jl")

function manta_healpix(
    pixels::AbstractArray;
    title::AbstractString = "RGB HEALPix",
    nx::Int = 1400,
    ny::Int = 700,
    figsize::Union{Nothing,Tuple{Int,Int}} = nothing,
    activate_gl::Bool = true,
    display_fig::Bool = true,
    show_graticule::Bool = true,
)
    rgb_pixels = as_rgb_pixels(pixels)
    img = mollweide_color_grid(rgb_pixels; nx=nx, ny=ny)
    activate_gl ? GLMakie.activate!() : CairoMakie.activate!()
    fig = Figure(size = _pick_fig_size(figsize))
    ax = Axis(
        fig[1, 1];
        title = make_main_title(title),
        aspect = DataAspect(),
        xticksvisible = false,
        yticksvisible = false,
        xticklabelsvisible = false,
        yticklabelsvisible = false,
        bottomspinevisible = false,
        topspinevisible = false,
        leftspinevisible = false,
        rightspinevisible = false,
    )
    image!(ax, (-2f0, 2f0), (-1f0, 1f0), permutedims(img))
    set_mollweide_view!(ax, -2.0, 2.0, -1.0, 1.0)
    graticule = draw_mollweide_graticule!(ax)
    set_graticule_visible!(graticule, show_graticule)
    ell_x = [2cos(t) for t in LinRange(0, 2π, 200)]
    ell_y = [sin(t) for t in LinRange(0, 2π, 200)]
    lines!(ax, ell_x, ell_y; color=:black, linewidth=0.8)
    keepalive!(fig)
    on(fig.scene.events.window_open) do is_open
        is_open || forget!(fig)
    end
    display_fig && display(fig)
    return fig
end

function manta_healpix_panels(
    panels::Vararg{Any,N};
    titles = nothing,
    cmaps = nothing,
    clims = nothing,
    nx::Int = 1400,
    ny::Int = 700,
    figsize::Union{Nothing,Tuple{Int,Int}} = nothing,
    activate_gl::Bool = true,
    display_fig::Bool = true,
    show_graticule::Bool = true,
) where {N}
    N >= 1 || throw(ArgumentError("Provide at least one HEALPix panel."))
    activate_gl ? GLMakie.activate!() : CairoMakie.activate!()
    fig = Figure(size = _pick_fig_size(figsize))
    rowgap!(fig.layout, -8)
    title_at(i) = titles === nothing ? "panel $(i)" : String(titles[i])
    cmap_at(i) = cmaps === nothing ? :inferno : cmaps[i]
    clim_at(i, vals) = clims === nothing ? clamped_extrema(vals) : clims[i]
    for (i, panel) in enumerate(panels)
        ax = Axis(
            fig[1, i];
            title = make_main_title(title_at(i)),
            aspect = DataAspect(),
            xticksvisible = false,
            yticksvisible = false,
            xticklabelsvisible = false,
            yticklabelsvisible = false,
            bottomspinevisible = false,
            topspinevisible = false,
            leftspinevisible = false,
            rightspinevisible = false,
        )
        if is_rgb_like(panel)
            img = mollweide_color_grid(as_rgb_pixels(panel); nx=nx, ny=ny)
            image!(ax, (-2f0, 2f0), (-1f0, 1f0), permutedims(img))
        else
            vals = _mollweide_scalar_grid(panel; nx=nx, ny=ny)
            plot_vals = permutedims(vals)
            hm = heatmap!(
                ax,
                LinRange(-2f0, 2f0, nx),
                LinRange(-1f0, 1f0, ny),
                plot_vals;
                colormap=cmap_at(i),
                colorrange=clim_at(i, vals),
                nan_color=:white,
            )
            Colorbar(
                fig[2, i],
                hm;
                vertical=false,
                height=16,
                tellwidth=false,
                halign=:center,
            )
            rowsize!(fig.layout, 1, Relative(1))
            rowsize!(fig.layout, 2, Fixed(44))
        end
        set_mollweide_view!(ax, -2.0, 2.0, -1.0, 1.0)
        graticule = draw_mollweide_graticule!(ax)
        set_graticule_visible!(graticule, show_graticule)
        ell_x = [2cos(t) for t in LinRange(0, 2π, 200)]
        ell_y = [sin(t) for t in LinRange(0, 2π, 200)]
        lines!(ax, ell_x, ell_y; color=:black, linewidth=0.8)
    end
    keepalive!(fig)
    on(fig.scene.events.window_open) do is_open
        is_open || forget!(fig)
    end
    display_fig && display(fig)
    return fig
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
    manta_healpix(filepath::String;
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
- **Contrast** : auto (quantiles 2/98 % en lin, 5/98 % en log) ou
  `vmin`/`vmax` manuels.

Retourne la `Figure` GLMakie.
"""
function manta_healpix(
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
    hist_mode::Symbol = :bars,
    hist_bins::Int = 64,
    hist_xlimits::Union{Nothing,Tuple{<:Real,<:Real}} = nothing,
    hist_ylimits::Union{Nothing,Tuple{<:Real,<:Real}} = nothing,
)
    ds = load_dataset(filepath; column = column)
    ds isa HealpixMapDataset || throw(ArgumentError(
        "MANTA: expected a HEALPix map in $(abspath(filepath)), got $(typeof(ds))."))
    return _view_healpix_map(ds;
        cmap = cmap, vmin = vmin, vmax = vmax, invert = invert,
        scale = scale, nx = nx, ny = ny, figsize = figsize,
        save_dir = save_dir, activate_gl = activate_gl,
        display_fig = display_fig,
        hist_mode = hist_mode, hist_bins = hist_bins,
        hist_xlimits = hist_xlimits, hist_ylimits = hist_ylimits)
end

"""
    manta_healpix_cube(filepath::String;
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
- échelle, contraste manuel, colormap, invert colormap, save PNG.

`v0`, `dv`, `vunit` : axe vitesse `v(j) = v0 + (j-1)*dv` pour le spectre.
"""
function manta_healpix_cube(
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
    hist_mode::Symbol = :bars,
    hist_bins::Int = 64,
    hist_xlimits::Union{Nothing,Tuple{<:Real,<:Real}} = nothing,
    hist_ylimits::Union{Nothing,Tuple{<:Real,<:Real}} = nothing,
    spec_ylimits::Union{Nothing,Tuple{<:Real,<:Real}} = nothing,
    moment_threshold::Real = 0.0,
    moment_nsigma::Union{Nothing,Real} = nothing,
    moment_channels::Union{Nothing,AbstractVector{<:Integer}} = nothing,
)
    ds = load_dataset(filepath; v0 = v0, dv = dv, vunit = vunit)
    ds isa HealpixCubeDataset || throw(ArgumentError(
        "MANTA: expected a HEALPix PPV cube in $(abspath(filepath)), got $(typeof(ds))."))
    return _view_healpix_cube(ds;
        cmap = cmap, vmin = vmin, vmax = vmax, invert = invert,
        scale = scale, nx = nx, ny = ny, figsize = figsize,
        save_dir = save_dir, activate_gl = activate_gl,
        display_fig = display_fig,
        hist_mode = hist_mode, hist_bins = hist_bins,
        hist_xlimits = hist_xlimits, hist_ylimits = hist_ylimits,
        spec_ylimits = spec_ylimits,
        moment_threshold = moment_threshold,
        moment_nsigma = moment_nsigma,
        moment_channels = moment_channels)
end

"""
    _vunit_quantity_word(vunit) -> String

Classify a spectral CUNIT-style string into the matching quantity word
("velocity", "frequency", "wavelength"). HEALPix-PPV datasets don't carry
a CTYPE through the viewer pipeline, so we infer from the unit instead.
Defaults to "velocity" because the historical default was km/s.
"""
function _vunit_quantity_word(vunit::AbstractString)
    u = lowercase(strip(String(vunit)))
    if u in ("hz", "khz", "mhz", "ghz", "thz")
        return "frequency"
    elseif u in ("m", "nm", "um", "µm", "mm", "cm", "angstrom", "å", "a")
        return "wavelength"
    elseif u == "channel"
        return "value"
    else
        return "velocity"  # km/s, m/s and bare blank fall here
    end
end

function _view_healpix_cube(
    ds::HealpixCubeDataset;
    cmap::Symbol = :inferno,
    vmin = nothing,
    vmax = nothing,
    invert::Bool = false,
    scale::Symbol = :lin,
    nx::Int = 1200,
    ny::Int = 600,
    figsize::Union{Nothing,Tuple{Int,Int}} = nothing,
    save_dir::Union{Nothing,AbstractString} = nothing,
    activate_gl::Bool = true,
    display_fig::Bool = true,
    hist_mode::Symbol = :bars,
    hist_bins::Int = 64,
    hist_xlimits::Union{Nothing,Tuple{<:Real,<:Real}} = nothing,
    hist_ylimits::Union{Nothing,Tuple{<:Real,<:Real}} = nothing,
    spec_ylimits::Union{Nothing,Tuple{<:Real,<:Real}} = nothing,
    moment_threshold::Real = 0.0,
    moment_nsigma::Union{Nothing,Real} = nothing,
    moment_channels::Union{Nothing,AbstractVector{<:Integer}} = nothing,
)
    cube = as_float32(ds.data)
    nside = ds.nside
    npix, nv = size(cube)
    v0_eff = ds.v0
    dv_eff = ds.dv
    vunit_eff = ds.vunit
    data_unit = ds.unit_label
    data_unit_tex = latexstring("\\text{", latex_safe(data_unit), "}")
    fname = ds.source_id
    spec_x = Float32.(v0_eff .+ (0:nv-1) .* dv_eff)
    # Pass the actual Δx through so M0 is "K·(spectral unit)" (e.g. K·km/s,
    # K·Hz) rather than the historical "K·channel" summation. Threshold and
    # channel window are forwarded from the public viewer kwargs so noisy or
    # continuum-subtracted cubes can be cleaned up without a code edit.
    moment_dx_unit = vunit_eff == "channel" ? 1.0 : abs(Float64(dv_eff))
    moment_vecs = moment_vectors(cube, spec_x;
                                 threshold = moment_threshold,
                                 nsigma = moment_nsigma,
                                 channels = moment_channels === nothing ? (1:nv) : moment_channels,
                                 dx = moment_dx_unit)
    # Choose a label that follows the spectral axis: velocity / frequency /
    # wavelength (heuristic on `vunit`, since HealpixCubeDataset only stores
    # the unit string, not a CTYPE).
    spec_word = _vunit_quantity_word(vunit_eff)
    moment_caption(order::Integer) = order == 0 ? "moment 0 [$(data_unit) " * vunit_eff * "]" :
                                     order == 1 ? "mean " * spec_word * " [" * vunit_eff * "]" :
                                                  spec_word * " dispersion [" * vunit_eff * "]"

    # ---------- Précalcul de l'index Mollweide (une fois) ----------
    res = Healpix.Resolution(nside)
    ipix_grid = mollweide_pixel_index(res, nx, ny)   # 0 = hors ellipse

    function projected_vector_image(vals)
        out = fill(NaN32, ny, nx)
        @inbounds for q in eachindex(ipix_grid)
            ip = ipix_grid[q]
            ip == 0 && continue
            v = vals[ip]
            out[q] = (isfinite(v) && v != Float32(Healpix.UNSEEN)) ? v : NaN32
        end
        out
    end

    frame_image(j::Int) = projected_vector_image(@view(cube[:, j]))
    moment_vector(order::Integer) = order == 0 ? moment_vecs[1] : order == 1 ? moment_vecs[2] : moment_vecs[3]
    moment_label(order::Integer) = order == 0 ? "moment 0" : order == 1 ? "moment 1" : "moment 2"
    # Public caption shown in titles: encodes the spectral quantity and unit.
    moment_long_label(order::Integer) = moment_caption(order)

    # ---------- État ----------
    cmap_name   = Observable(cmap)
    invert_cmap = Observable(invert)
    cm_obs = lift(cmap_name, invert_cmap) do name, inv
        base = to_cmap(name); inv ? reverse(base) : base
    end
    scale_mode = Observable(scale)
    chan_idx   = Observable(max(1, nv ÷ 2))
    show_moment = Observable(false)
    moment_order = Observable(0)
    gauss_on = Observable(false)
    sigma = Observable(1.5f0)

    img_raw = lift(chan_idx, show_moment, moment_order) do j, show_mom, ord
        show_mom ? projected_vector_image(moment_vector(ord)) : frame_image(j)
    end
    img_proc = lift(img_raw, gauss_on, sigma) do im, on, σ
        on ? nan_gaussian_filter(im, σ) : im
    end
    img_disp = lift(img_proc, scale_mode) do im, m_
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
    function _vector_clims(vals, mode::Symbol)
        fin = Float32[]
        if mode === :lin
            @inbounds for v in vals
                (isfinite(v) && v != Float32(Healpix.UNSEEN)) && push!(fin, Float32(v))
            end
        else
            f = mode === :log10 ? log10 : log
            @inbounds for v in vals
                (isfinite(v) && v != Float32(Healpix.UNSEEN) && v > 0) && push!(fin, Float32(f(v)))
            end
        end
        isempty(fin) && return mode === :lin ? (0f0, 1f0) : (-1f0, 1f0)
        lo = Float32(quantile(fin, mode === :lin ? 0.01 : 0.05))
        hi = Float32(quantile(fin, 0.995))
        lo == hi && (lo = prevfloat(lo); hi = nextfloat(hi))
        return (lo, hi)
    end

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
    clims_auto = lift(scale_mode, show_moment, moment_order) do m_, show_mom, ord
        show_mom ? _vector_clims(moment_vector(ord), m_) : _global_clims(m_)
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
        (isfinite(lo) && isfinite(hi) && lo != hi) ? (lo, hi) : (0f0, 1f0)
    end

    contour_auto_levels = lift(img_disp) do im
        automatic_contour_levels(im; n = 7)
    end
    contour_use_manual = Observable(false)
    contour_manual_levels = Observable(Float32[])
    contour_manual_colors = Observable(String[])
    contour_levels_obs = lift(contour_use_manual, contour_manual_levels, contour_auto_levels) do use_man, manual, auto
        use_man && !isempty(manual) ? manual : auto
    end
    contour_default_color = RGBAf(0, 0, 0, 0.62)
    contour_colors_obs = lift(contour_levels_obs, contour_use_manual, contour_manual_colors) do levels, use_man, colors
        contour_color_values(use_man ? colors : String[], length(levels), contour_default_color)
    end
    show_contours = Observable(false)

    hist_mode_obs = Observable(normalize_histogram_mode(hist_mode))
    hist_bins_obs = Observable(clamp(hist_bins, 4, 512))
    hist_xlimits_manual = Observable(hist_xlimits !== nothing)
    hist_xlimits_manual_value = Observable(hist_xlimits === nothing ?
        (0f0, 1f0) :
        parse_histogram_xlimits(string(first(hist_xlimits)), string(last(hist_xlimits)))[3])
    hist_ylimits_manual = Observable(hist_ylimits !== nothing)
    hist_ylimits_manual_value = Observable(hist_ylimits === nothing ?
        (0f0, 1f0) :
        parse_histogram_ylimits(string(first(hist_ylimits)), string(last(hist_ylimits)))[3])
    hist_limits_obs = lift(hist_xlimits_manual, hist_xlimits_manual_value, clims_safe) do manual, xlim, clim
        manual ? xlim : clim
    end
    hist_pair_obs = lift(img_disp, hist_limits_obs, hist_bins_obs, hist_mode_obs) do im, lim, bins, mode
        histogram_profile(im; bins = bins, limits = lim, mode = mode)
    end
    hist_x_obs = lift(p -> p.x, hist_pair_obs)
    hist_y_obs = lift(p -> p.y, hist_pair_obs)
    hist_width_obs = lift(p -> p.width, hist_pair_obs)
    hist_bars_visible = lift(m -> m === :bars, hist_mode_obs)
    hist_kde_visible = lift(m -> m === :kde, hist_mode_obs)
    hist_ylabel_obs = lift(histogram_ylabel, hist_mode_obs)

    zoom_drag_active = Observable(false)
    zoom_drag_start  = Observable(Point2f(NaN32, NaN32))
    zoom_drag_end    = Observable(Point2f(NaN32, NaN32))
    show_graticule   = Observable(true)
    selection_mode = Observable(:point)
    region_shape = Observable(:box)
    region_drag_active = Observable(false)
    region_start = Observable(Point2f(NaN32, NaN32))
    region_end = Observable(Point2f(NaN32, NaN32))
    region_ipix = Observable(Int[])

    # Pixel sélectionné (initial : centre)
    sel_ipix  = Observable(0)
    sel_xy    = Observable(Point2f(NaN32, NaN32))
    sel_label = Observable(latexstring("\\text{click on map to select a pixel}"))

    spec_y_obs = Observable(zeros(Float32, nv))
    spec_ylimits_value = Observable(spec_ylimits === nothing ?
        (use_manual[] ? clims_manual[] : (0f0, 1f0)) :
        parse_spectrum_ylimits(string(first(spec_ylimits)), string(last(spec_ylimits)))[3])
    spec_ylimits_source = Observable(spec_ylimits === nothing ? (use_manual[] ? :contrast : :auto) : :manual)
    function update_spectrum!(ip::Int)
        if 1 ≤ ip ≤ npix
            region_ipix[] = Int[]
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

    function update_region_spectrum!(ipixels)
        ips = Int.(ipixels)
        region_ipix[] = ips
        spec_y_obs[] = healpix_region_mean_spectrum(cube, ips, nv)
        shape = region_shape[] === :circle ? "circle" : "box"
        j = clamp(chan_idx[], 1, nv)
        mean_val = healpix_region_mean(@view(cube[:, j]), ips)
        valstr = isfinite(mean_val) ? string(round(mean_val; digits=4)) : "NaN"
        sel_label[] = latexstring(
            "\\text{mean spectrum in ", shape, " region}\\;N=", length(ips),
            "\\;\\text{channel mean}=", valstr,
            "\\;\\mathrm{", latex_safe(data_unit), "}"
        )
    end

    ui_theme = default_ui_theme()
    ui_accent = ui_theme.accent
    ui_selection = ui_theme.selection
    ui_text_muted = ui_theme.text_muted

    # ---------- Figure ----------
    activate_gl ? GLMakie.activate!() : CairoMakie.activate!()
    fig = Figure(size = _pick_fig_size(figsize), backgroundcolor = ui_theme.background)
    main_grid = fig[1, 1] = GridLayout()

    # Carte
    map_grid = main_grid[1, 1] = GridLayout()
    colgap!(map_grid, -8)
    rowgap!(map_grid, -8)
    is_channel_axis = (vunit_eff == "channel")
    title_obs = lift(chan_idx, show_moment, moment_order) do j, show_mom, ord
        if show_mom
            return latexstring("\\text{", latex_safe(fname), "}\\;\\text{", latex_safe(moment_long_label(ord)), "}")
        end
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
        xticksvisible = false, yticksvisible = false,
        xticklabelsvisible = false, yticklabelsvisible = false,
        bottomspinevisible = false, topspinevisible = false,
        leftspinevisible   = false, rightspinevisible = false)

    xs = LinRange(-2f0, 2f0, nx)
    ys = LinRange(-1f0, 1f0, ny)
    img_for_plot = lift(img_disp) do im; permutedims(im); end
    hm = heatmap!(ax_img, xs, ys, img_for_plot;
                  colormap=cm_obs, colorrange=clims_safe, nan_color=:white)
    contour!(ax_img, xs, ys, img_for_plot;
             levels=contour_levels_obs, color=contour_colors_obs, linewidth=1.1,
             visible=show_contours)
    full_map_bounds = (-2.0, 2.0, -1.0, 1.0)
    set_mollweide_view!(ax_img, full_map_bounds...)
    graticule = draw_mollweide_graticule!(ax_img)
    refresh_graticule_labels!(graticule, ax_img; bounds=full_map_bounds)

    # ellipse + zoom box + marker
    ell_x = [2cos(t) for t in LinRange(0, 2π, 200)]
    ell_y = [sin(t)  for t in LinRange(0, 2π, 200)]
    lines!(ax_img, ell_x, ell_y; color=:black, linewidth=0.8)

    zoom_box_segments = lift(zoom_drag_active, zoom_drag_start, zoom_drag_end) do active, p0, p1
        active || return Point2f[]
        (isfinite(p0[1]) && isfinite(p1[1])) || return Point2f[]
        x0,y0 = p0; x1,y1 = p1
        Point2f[Point2f(x0,y0),Point2f(x1,y0),Point2f(x1,y0),Point2f(x1,y1),
                Point2f(x1,y1),Point2f(x0,y1),Point2f(x0,y1),Point2f(x0,y0)]
    end
    linesegments!(ax_img, zoom_box_segments; color=(ui_selection,0.95),
                  linewidth=2.0, linestyle=:dash)
    region_segments = lift(region_start, region_end, region_shape, region_ipix, region_drag_active) do p0, p1, shape, ipixs, dragging
        (dragging || !isempty(ipixs)) ? projected_region_segments(p0, p1, shape) : Point2f[]
    end
    lines!(ax_img, region_segments; color=(ui_selection, 0.98), linewidth=2.3)
    marker_pts = lift(sel_xy) do p
        (isfinite(p[1]) && isfinite(p[2])) ? Point2f[p] : Point2f[]
    end
    scatter!(ax_img, marker_pts; color=ui_accent, markersize=12, marker=:cross)

    map_unit_label = lift(show_moment, moment_order) do show_mom, ord
        show_mom ? latexstring("\\text{", latex_safe(moment_label(ord)), "}") : data_unit_tex
    end
    Colorbar(
        map_grid[2, 1],
        hm;
        label=map_unit_label,
        vertical=false,
        height=18,
        tellwidth=false,
        halign=:center,
    )
    rowsize!(map_grid, 1, Relative(1))
    rowsize!(map_grid, 2, Fixed(52))

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
        ylabel = lift(m_ -> m_ === :lin   ? data_unit_tex :
                            m_ === :log10 ? latexstring("\\log_{10}\\,\\text{", latex_safe(data_unit), "}") :
                                            latexstring("\\ln\\,\\text{", latex_safe(data_unit), "}"), scale_mode))
    lines!(ax_spec, spec_x, spec_y_disp; color=:black, linewidth=1.5)
    # ligne verticale à v(chan_idx)
    chan_v = lift(chan_idx) do j; Float32(v0_eff + (j-1)*dv_eff); end
    vlines!(ax_spec, lift(v -> [v], chan_v); color=ui_accent, linewidth=1.2, linestyle=:dash)

    # ylimits du spectre : manuel, hérité du contraste initial, ou auto.
    function _refresh_spec_ylim!()
        if spec_ylimits_source[] === :manual || spec_ylimits_source[] === :contrast
            lo, hi = spec_ylimits_value[]
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

    function _refresh_hist_axes!()
        xlo, xhi = hist_limits_obs[]
        if hist_ylimits_manual[]
            ylo, yhi = hist_ylimits_manual_value[]
            limits!(ax_hist, Float32(xlo), Float32(xhi), Float32(ylo), Float32(yhi))
        else
            autolimits!(ax_hist)
            xlims!(ax_hist, Float32(xlo), Float32(xhi))
        end
    end

    ax_hist = Axis(
        main_grid[3, 1];
        title = L"\text{Visible channel histogram}",
        xlabel = data_unit_tex,
        ylabel = hist_ylabel_obs,
        # height is governed by `rowsize!(main_grid, 3, ...)` below — no
        # hard-coded value here (cf. CLAUDE.md / anti-patterns).
        xtickformat = _latex_tick_formatter,
        ytickformat = _latex_tick_formatter,
    )
    barplot!(ax_hist, hist_x_obs, hist_y_obs; width=hist_width_obs, color=(ui_accent, 0.44), strokecolor=ui_accent, strokewidth=0.3, visible=hist_bars_visible)
    lines!(ax_hist, hist_x_obs, hist_y_obs; color=ui_accent, linewidth=1.8, visible=hist_kde_visible)
    vlines!(ax_hist, lift(lim -> [first(lim), last(lim)], clims_safe);
            color=(ui_text_muted, 0.65), linewidth=1.0, linestyle=:dash)

    # Contrôles
    ctrl = main_grid[4, 1] = GridLayout(; alignmode=Outside())
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
    Label(ctrl[1,6], text=L"\text{Colormap}", halign=:left, tellwidth=false, fontsize=15)
    cmap_menu = Menu(ctrl[1,7]; options=ui_colormap_options(), prompt=String(cmap), width=112)
    invert_chk = Checkbox(ctrl[1,8]); Label(ctrl[1,9], text="Invert", halign=:left, tellwidth=false, fontsize=15)
    invert_chk.checked[] = invert_cmap[]

    Label(ctrl[2,1], text=L"\text{Contrast}", halign=:left, tellwidth=false, fontsize=15)
    clim_min_box = Textbox(ctrl[2,2];  placeholder="min", width=100, height=30)
    clim_max_box = Textbox(ctrl[2,3]; placeholder="max", width=100, height=30)
    apply_btn    = Button(ctrl[2,4]; label="Apply",      width=80,  height=30)
    auto_btn     = Button(ctrl[2,5]; label="Auto",       width=76,  height=30)
    p1_btn       = Button(ctrl[2,6]; label="p1-p99",     width=88,  height=30)
    p5_btn       = Button(ctrl[2,7]; label="p5-p95",     width=88,  height=30)
    graticule_chk = Checkbox(ctrl[2,8])
    Label(ctrl[2,9], text="Graticule", halign=:left, tellwidth=false, fontsize=15)
    graticule_chk.checked[] = show_graticule[]
    reset_btn    = Button(ctrl[2,10]; label="Reset zoom", width=120, height=30)
    save_btn     = Button(ctrl[2,11]; label="Save PNG",   width=110, height=30)

    gauss_chk = Checkbox(ctrl[3,1])
    Label(ctrl[3,2], text="Smoothing", halign=:left, tellwidth=false, fontsize=15)
    sigma_label = Label(ctrl[3,3], text=latexstring("\\sigma = 1.5\\,\\text{px}"), fontsize=15, halign=:left, tellwidth=false)
    sigma_slider = Slider(ctrl[3,4:6]; range=LinRange(0, 10, 101), startvalue=1.5, width=220, height=14)

    Label(ctrl[3,7], text=L"\text{Selection}", halign=:left, tellwidth=false, fontsize=15)
    region_mode_menu = Menu(ctrl[3,8]; options=["point", "box", "circle"], prompt="point", width=108)
    region_clear_btn = Button(ctrl[3,9]; label="Clear selection", width=138, height=30)
    region_count_label = Label(ctrl[3,10]; text="0 pix", halign=:left, tellwidth=false, fontsize=15)
    Label(ctrl[4,1], text=L"\text{Contours}", halign=:left, tellwidth=false, fontsize=15)
    contour_chk = Checkbox(ctrl[4,2])
    Label(ctrl[4,3], text="Show", halign=:left, tellwidth=false, fontsize=15)
    contour_levels_box = Textbox(ctrl[4,4:6]; placeholder="auto or 1:red, 2:#00ffaa", width=250, height=30)
    contour_apply_btn = Button(ctrl[4,7]; label="Apply", width=80, height=30)
    contour_chk.checked[] = show_contours[]

    Label(ctrl[5,1], text=L"\text{Moment}", halign=:left, tellwidth=false, fontsize=15)
    moment_menu = Menu(ctrl[5,2]; options=["M0 integrated", "M1 mean", "M2 dispersion"], prompt="M0 integrated", width=138)
    show_moment_btn = Button(ctrl[5,3]; label="Show", width=80, height=30)
    show_channel_btn = Button(ctrl[5,4]; label="Channel", width=92, height=30)
    save_moment_fits_btn = Button(ctrl[5,5]; label="Save moment FITS", width=150, height=30)
    Label(ctrl[6,1], text=L"\text{Histogram}", halign=:left, tellwidth=false, fontsize=15)
    hist_mode_menu = Menu(ctrl[6,2]; options=["bars", "kde"], prompt=String(hist_mode_obs[]), width=92)
    hist_bins_box = Textbox(ctrl[6,3]; placeholder="bins", width=80, height=30)
    hist_xmin_box = Textbox(ctrl[6,4]; placeholder="x min", width=100, height=30)
    hist_xmax_box = Textbox(ctrl[6,5]; placeholder="x max", width=100, height=30)
    hist_apply_btn = Button(ctrl[6,6]; label="Apply x", width=88, height=30)
    hist_auto_btn = Button(ctrl[6,7]; label="Auto x", width=82, height=30)
    hist_ymin_box = Textbox(ctrl[6,8]; placeholder="y min", width=100, height=30)
    hist_ymax_box = Textbox(ctrl[6,9]; placeholder="y max", width=100, height=30)
    hist_y_apply_btn = Button(ctrl[6,10]; label="Apply y", width=88, height=30)
    hist_y_auto_btn = Button(ctrl[6,11]; label="Auto y", width=82, height=30)
    Label(ctrl[7,1], text=L"\text{Spectrum y}", halign=:left, tellwidth=false, fontsize=15)
    spec_ymin_box = Textbox(ctrl[7,2]; placeholder="y min", width=100, height=30)
    spec_ymax_box = Textbox(ctrl[7,3]; placeholder="y max", width=100, height=30)
    spec_y_apply_btn = Button(ctrl[7,4]; label="Apply y", width=88, height=30)
    spec_y_auto_btn = Button(ctrl[7,5]; label="Auto y", width=82, height=30)
    foreach(w -> manta_style_menu!(w, ui_theme), (scale_menu, cmap_menu, region_mode_menu, moment_menu, hist_mode_menu))
    foreach(w -> manta_style_button!(w, ui_theme), (apply_btn, auto_btn, p1_btn, p5_btn, reset_btn, save_btn, region_clear_btn, contour_apply_btn, show_moment_btn, show_channel_btn, save_moment_fits_btn, hist_apply_btn, hist_auto_btn, hist_y_apply_btn, hist_y_auto_btn, spec_y_apply_btn, spec_y_auto_btn))
    foreach(w -> manta_style_checkbox!(w, ui_theme), (invert_chk, graticule_chk, gauss_chk, contour_chk))
    foreach(w -> manta_style_textbox!(w, ui_theme), (clim_min_box, clim_max_box, contour_levels_box, hist_bins_box, hist_xmin_box, hist_xmax_box, hist_ymin_box, hist_ymax_box, spec_ymin_box, spec_ymax_box))
    foreach(w -> manta_style_slider!(w, ui_theme), (chan_slider, sigma_slider))
    rowsize!(main_grid, 1, Relative(1))
    rowsize!(main_grid, 2, Fixed(165))
    rowsize!(main_grid, 3, Fixed(100))
    rowsize!(main_grid, 4, Fixed(270))

    if use_manual[]
        a, b = clims_manual[]
        sa, sb = string(a), string(b)
        clim_min_box.displayed_string[] = sa; clim_min_box.stored_string[] = sa
        clim_max_box.displayed_string[] = sb; clim_max_box.stored_string[] = sb
    end

    set_box_text!(tb, s::AbstractString) = begin
        str = String(s)
        tb.displayed_string[] = str
        tb.stored_string[] = str
        nothing
    end
    set_box_text!(hist_bins_box, string(hist_bins_obs[]))
    if hist_xlimits_manual[]
        lo, hi = hist_xlimits_manual_value[]
        set_box_text!(hist_xmin_box, string(lo))
        set_box_text!(hist_xmax_box, string(hi))
    end
    if hist_ylimits_manual[]
        lo, hi = hist_ylimits_manual_value[]
        set_box_text!(hist_ymin_box, string(lo))
        set_box_text!(hist_ymax_box, string(hi))
    end
    if spec_ylimits_source[] !== :auto
        lo, hi = spec_ylimits_value[]
        set_box_text!(spec_ymin_box, string(lo))
        set_box_text!(spec_ymax_box, string(hi))
    end
    function clear_region!()
        region_ipix[] = Int[]
        region_start[] = Point2f(NaN32, NaN32)
        region_end[] = Point2f(NaN32, NaN32)
        region_drag_active[] = false
        region_count_label.text[] = "0 pix"
        nothing
    end
    function apply_region!(p0::Point2f, p1::Point2f)
        ips = projected_region_ipix(ipix_grid, p0[1], p0[2], p1[1], p1[2], region_shape[])
        region_count_label.text[] = "$(length(ips)) pix"
        update_region_spectrum!(ips)
        _refresh_spec_ylim!()
        nothing
    end
    function apply_percentile_clims!(lo::Real, hi::Real)
        clims = percentile_clims(img_disp[], lo, hi)
        clims_manual[] = clims
        use_manual[] = true
        set_box_text!(clim_min_box, string(first(clims)))
        set_box_text!(clim_max_box, string(last(clims)))
        if spec_ylimits_source[] === :contrast
            spec_ylimits_value[] = clims
            set_box_text!(spec_ymin_box, string(first(clims)))
            set_box_text!(spec_ymax_box, string(last(clims)))
            _refresh_spec_ylim!()
        end
        nothing
    end

    # ---------- Reactivity ----------
    on(chan_slider.value) do v
        chan_idx[] = Int(round(v))
        if !isempty(region_ipix[])
            update_region_spectrum!(region_ipix[])
            _refresh_spec_ylim!()
        end
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
    on(cmap_menu.selection) do sel
        sel === nothing && return
        cmap_name[] = Symbol(sel)
    end
    on(invert_chk.checked) do v; invert_cmap[] = v; end
    on(gauss_chk.checked) do v
        gauss_on[] = v
    end
    on(sigma_slider.value) do v
        sigma[] = Float32(v)
        sigma_label.text[] = latexstring("\\sigma = $(round(v; digits=2))\\,\\text{px}")
    end
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
            if spec_ylimits_source[] === :contrast
                spec_ylimits_value[] = clims
                set_box_text!(spec_ymin_box, string(first(clims)))
                set_box_text!(spec_ymax_box, string(last(clims)))
            end
        else
            use_manual[]   = false
            if spec_ylimits_source[] === :contrast
                spec_ylimits_source[] = :auto
                set_box_text!(spec_ymin_box, "")
                set_box_text!(spec_ymax_box, "")
            end
        end
        _refresh_spec_ylim!()                # propage au spectre
    end
    on(auto_btn.clicks) do _
        use_manual[] = false
        set_box_text!(clim_min_box, "")
        set_box_text!(clim_max_box, "")
        if spec_ylimits_source[] === :contrast
            spec_ylimits_source[] = :auto
            set_box_text!(spec_ymin_box, "")
            set_box_text!(spec_ymax_box, "")
        end
        _refresh_spec_ylim!()
    end
    on(p1_btn.clicks) do _; apply_percentile_clims!(1, 99); end
    on(p5_btn.clicks) do _; apply_percentile_clims!(5, 95); end
    on(hist_mode_menu.selection) do sel
        sel === nothing && return
        hist_mode_obs[] = normalize_histogram_mode(sel)
    end
    on(hist_apply_btn.clicks) do _
        ok_bins, bins, _bins_msg = parse_histogram_bins(get_box_str(hist_bins_box); fallback = hist_bins_obs[])
        ok_x, manual_x, xlim, _x_msg = parse_histogram_xlimits(
            get_box_str(hist_xmin_box),
            get_box_str(hist_xmax_box);
            fallback = hist_xlimits_manual_value[],
        )
        ok_bins && ok_x || return
        hist_bins_obs[] = bins
        hist_xlimits_manual_value[] = xlim
        hist_xlimits_manual[] = manual_x
        set_box_text!(hist_bins_box, string(bins))
        set_box_text!(hist_xmin_box, manual_x ? string(first(xlim)) : "")
        set_box_text!(hist_xmax_box, manual_x ? string(last(xlim)) : "")
        _refresh_hist_axes!()
    end
    on(hist_auto_btn.clicks) do _
        hist_xlimits_manual[] = false
        set_box_text!(hist_xmin_box, "")
        set_box_text!(hist_xmax_box, "")
        _refresh_hist_axes!()
    end
    on(hist_y_auto_btn.clicks) do _
        hist_ylimits_manual[] = false
        set_box_text!(hist_ymin_box, "")
        set_box_text!(hist_ymax_box, "")
        _refresh_hist_axes!()
    end
    on(hist_y_apply_btn.clicks) do _
        ok_y, manual_y, ylim, _msg = parse_histogram_ylimits(
            get_box_str(hist_ymin_box),
            get_box_str(hist_ymax_box);
            fallback = hist_ylimits_manual_value[],
        )
        ok_y || return
        hist_ylimits_manual_value[] = ylim
        hist_ylimits_manual[] = manual_y
        set_box_text!(hist_ymin_box, manual_y ? string(first(ylim)) : "")
        set_box_text!(hist_ymax_box, manual_y ? string(last(ylim)) : "")
        _refresh_hist_axes!()
    end
    on(spec_y_apply_btn.clicks) do _
        ok, manual, ylim, _msg = parse_spectrum_ylimits(
            get_box_str(spec_ymin_box),
            get_box_str(spec_ymax_box);
            fallback = spec_ylimits_value[],
        )
        ok || return
        if manual
            spec_ylimits_value[] = ylim
            spec_ylimits_source[] = :manual
            set_box_text!(spec_ymin_box, string(first(ylim)))
            set_box_text!(spec_ymax_box, string(last(ylim)))
        else
            spec_ylimits_source[] = :auto
            set_box_text!(spec_ymin_box, "")
            set_box_text!(spec_ymax_box, "")
        end
        _refresh_spec_ylim!()
    end
    on(spec_y_auto_btn.clicks) do _
        spec_ylimits_source[] = :auto
        set_box_text!(spec_ymin_box, "")
        set_box_text!(spec_ymax_box, "")
        _refresh_spec_ylim!()
    end
    on(hist_limits_obs) do _
        _refresh_hist_axes!()
    end
    on(hist_y_obs) do _
        _refresh_hist_axes!()
    end
    on(region_mode_menu.selection) do sel
        sel === nothing && return
        mode = Symbol(String(sel))
        mode in (:point, :box, :circle) || return
        selection_mode[] = mode
        region_shape[] = mode === :circle ? :circle : :box
        if mode === :point
            clear_region!()
            sel_ipix[] > 0 && update_spectrum!(sel_ipix[])
            _refresh_spec_ylim!()
        end
    end
    on(region_clear_btn.clicks) do _
        clear_region!()
        sel_ipix[] > 0 && update_spectrum!(sel_ipix[])
        _refresh_spec_ylim!()
    end
    on(contour_chk.checked) do v
        show_contours[] = v
    end
    on(contour_apply_btn.clicks) do _
        ok, use_man, levels, colors, _msg = parse_contour_specs(
            get_box_str(contour_levels_box);
            fallback_levels=contour_manual_levels[],
            fallback_colors=contour_manual_colors[],
        )
        ok || return
        contour_use_manual[] = use_man
        contour_manual_levels[] = levels
        contour_manual_colors[] = colors
        set_box_text!(contour_levels_box, use_man ? format_contour_specs(levels, colors) : "")
        show_contours[] = true
        contour_chk.checked[] = true
    end
    on(moment_menu.selection) do sel
        sel === nothing && return
        label = String(sel)
        moment_order[] = startswith(label, "M1") ? 1 : startswith(label, "M2") ? 2 : 0
    end
    on(show_moment_btn.clicks) do _
        show_moment[] = true
        use_manual[] = false
        set_box_text!(clim_min_box, "")
        set_box_text!(clim_max_box, "")
    end
    on(show_channel_btn.clicks) do _
        show_moment[] = false
        use_manual[] = false
        set_box_text!(clim_min_box, "")
        set_box_text!(clim_max_box, "")
    end
    on(scale_mode)        do _; _refresh_spec_ylim!(); end
    on(spec_y_disp)       do _; _refresh_spec_ylim!(); end
    on(use_manual)        do _; _refresh_spec_ylim!(); end
    on(clims_manual)      do clims
        if spec_ylimits_source[] === :contrast
            spec_ylimits_value[] = clims
        end
        _refresh_spec_ylim!()
    end

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
        elseif ev.button == Mouse.left && ev.action == Mouse.press && selection_mode[] != :point
            p = mouseposition(ax_img); any(isnan, p) && return
            mollweide_xy_to_lonlat(p[1], p[2]) === nothing && return
            region_start[] = Point2f(p[1], p[2])
            region_end[] = Point2f(p[1], p[2])
            region_drag_active[] = true
            region_ipix[] = Int[]
            region_count_label.text[] = "0 pix"
        elseif ev.button == Mouse.left && ev.action == Mouse.release && region_drag_active[]
            p = mouseposition(ax_img)
            !any(isnan, p) && (region_end[] = Point2f(p[1], p[2]))
            p0 = region_start[]; p1 = region_end[]
            region_drag_active[] = false
            if isfinite(p0[1]) && isfinite(p1[1])
                apply_region!(p0, p1)
            else
                clear_region!()
            end
        elseif ev.button == Mouse.left && ev.action == Mouse.press
            p = mouseposition(ax_img); any(isnan, p) && return
            ll = mollweide_xy_to_lonlat(p[1], p[2]); ll === nothing && return
            l_deg, b_deg = ll
            θhp = deg2rad(90 - b_deg); φhp = deg2rad(mod(l_deg, 360))
            ip = Healpix.ang2pixRing(res, θhp, φhp)
            sel_xy[] = Point2f(p[1], p[2])
            clear_region!()
            update_spectrum!(ip)
        end
    end
    on(events(ax_img).mouseposition) do p
        if zoom_drag_active[] && !any(isnan, p)
            zoom_drag_end[] = Point2f(p[1], p[2])
        elseif region_drag_active[] && !any(isnan, p)
            region_end[] = Point2f(p[1], p[2])
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
    on(save_moment_fits_btn.clicks) do _
        label = replace(moment_label(moment_order[]), " " => "")
        out = joinpath(save_root, "$(fname)_$(label)_healpix.fits")
        try
            FITS(String(out), "w") do f
                write(f, Float32.(moment_vector(moment_order[])))
            end
            @info "Saved moment FITS" out
        catch e
            @error "Failed to save moment FITS" exception=(e, catch_backtrace())
        end
    end

    # init
    update_spectrum!(max(1, npix ÷ 2))     # spectre par défaut au pixel central
    _refresh_spec_ylim!()
    _refresh_hist_axes!()

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
