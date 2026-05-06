#       API stable: apply_scale, clamped_extrema, percentile_clims, histogram_counts,
#                   automatic_contour_levels, parse_contour_levels,
#                   parse_contour_specs, format_contour_specs, contour_color_values,
#                   ijk_to_uv, uv_to_ijk, get_slice,
#                   region_uv_indices, mean_region_spectrum,
#                   make_info_tex, to_cmap, get_box_str, _pick_fig_size,
#                   latex_safe, make_main_title, make_slice_title, make_spec_title,
#                   parse_manual_clims, parse_gif_request,
#                   SimpleWCSAxis, read_simple_wcs, has_wcs, world_coord,
#                   wcs_axis_label, format_world_coord, data_unit_label,
#                   save_viewer_settings, load_viewer_settings

############################
# Exports
############################
export apply_scale, clamped_extrema, percentile_clims, histogram_counts
export automatic_contour_levels, parse_contour_levels
export parse_contour_specs, format_contour_specs, contour_color_values
export ijk_to_uv, uv_to_ijk, get_slice
export region_uv_indices, mean_region_spectrum
export make_info_tex
export to_cmap, get_box_str, _pick_fig_size
export latex_safe, make_main_title, make_slice_title, make_spec_title
export parse_manual_clims, parse_gif_request
export SimpleWCSAxis, read_simple_wcs, has_wcs, world_coord
export wcs_axis_label, format_world_coord, data_unit_label
export save_viewer_settings, load_viewer_settings

############################
# Deps
############################
using Makie
using LaTeXStrings
using TOML
using Statistics: quantile

############################
# Scaling / Extrema
############################

"""
    apply_scale(x, mode::Symbol) -> Array{Float32}

Display modes: :lin | :log10 | :ln.
In log mode, values ≤ 0 become NaN to avoid -Inf/+Inf.
"""
function apply_scale(x::AbstractArray, mode::Symbol)
    if mode === :lin
        return x isa AbstractArray{Float32} ? x : Float32.(x)
    elseif mode === :log10
        y = similar(x, Float32)
        @inbounds @fastmath for i in eachindex(x)
            xi = x[i]
            y[i] = xi > 0 ? Float32(log10(xi)) : Float32(NaN32)  # why: UI-safe
        end
        return y
    elseif mode === :ln
        y = similar(x, Float32)
        @inbounds @fastmath for i in eachindex(x)
            xi = x[i]
            y[i] = xi > 0 ? Float32(log(xi)) : Float32(NaN32)    # why: UI-safe
        end
        return y
    else
        return Float32.(x)
    end
end

"""
    clamped_extrema(vals) -> (Float32, Float32)

Ignore NaN, expand zero ranges, fallback to (0,1).
"""
function clamped_extrema(vals)::Tuple{Float32,Float32}
    found = false
    mn = 0f0
    mx = 0f0
    @inbounds for v in vals
        fv = Float32(v)
        if isfinite(fv)
            if !found
                mn = fv
                mx = fv
                found = true
            else
                mn = min(mn, fv)
                mx = max(mx, fv)
            end
        end
    end
    if !found
        return (0f0, 1f0)
    end
    if mn == mx
        return (prevfloat(mn), nextfloat(mx))
    end
    return (mn, mx)
end

finite_float_values(vals) = begin
    out = Float32[]
    for v in vals
        fv = Float32(v)
        isfinite(fv) && push!(out, fv)
    end
    out
end

"""
    percentile_clims(vals, lo_pct, hi_pct) -> (Float32, Float32)

Return finite-value percentile limits, expanding degenerate ranges.
Percentiles are in `[0, 100]`.
"""
function percentile_clims(vals, lo_pct::Real, hi_pct::Real)::Tuple{Float32,Float32}
    xs = finite_float_values(vals)
    isempty(xs) && return (0f0, 1f0)
    lo = clamp(Float64(lo_pct), 0.0, 100.0) / 100.0
    hi = clamp(Float64(hi_pct), 0.0, 100.0) / 100.0
    lo > hi && ((lo, hi) = (hi, lo))
    qlo = Float32(quantile(xs, lo))
    qhi = Float32(quantile(xs, hi))
    if qlo == qhi
        return (prevfloat(qlo), nextfloat(qhi))
    end
    return (qlo, qhi)
end

"""
    histogram_counts(vals; bins=48, limits=nothing) -> (centers, counts)

Small finite-value histogram for UI display.
"""
function histogram_counts(vals; bins::Int = 48, limits = nothing)
    nb = max(1, bins)
    xs = finite_float_values(vals)
    if isempty(xs)
        return (Float32[], Float32[])
    end
    lo, hi = limits === nothing ? clamped_extrema(xs) : (Float32(first(limits)), Float32(last(limits)))
    if !(isfinite(lo) && isfinite(hi)) || lo == hi
        lo, hi = clamped_extrema(xs)
    end
    lo > hi && ((lo, hi) = (hi, lo))
    width = (hi - lo) / nb
    if width <= 0
        return (Float32[lo], Float32[length(xs)])
    end
    counts = zeros(Float32, nb)
    @inbounds for x in xs
        if lo <= x <= hi
            b = clamp(Int(floor((x - lo) / width)) + 1, 1, nb)
            counts[b] += 1f0
        end
    end
    centers = Float32[lo + (i - 0.5f0) * width for i in 1:nb]
    return (centers, counts)
end

"""
    automatic_contour_levels(vals; n=7) -> Vector{Float32}

Robust automatic contour levels from the finite 5th to 95th percentiles.
"""
function automatic_contour_levels(vals; n::Int = 7)
    nlev = max(2, n)
    lo, hi = percentile_clims(vals, 5, 95)
    if !(isfinite(lo) && isfinite(hi)) || lo == hi
        lo, hi = clamped_extrema(vals)
    end
    lo == hi && return Float32[]
    return collect(Float32, LinRange(lo, hi, nlev))
end

"""
    parse_contour_levels(txt; fallback=Float32[]) -> (ok, use_manual, levels, message)

Parse comma/space/semicolon-separated contour levels. Empty text means auto.
"""
function parse_contour_levels(txt::AbstractString; fallback = Float32[])
    ok, use_manual, levels, _colors, message = parse_contour_specs(txt; fallback_levels = fallback)
    return (ok, use_manual, levels, message)
end

function _try_hex_contour_color(s::AbstractString)
    h = startswith(s, "#") ? s[2:end] : s
    if !(length(h) in (6, 8)) || any(c -> !(c in '0':'9' || c in 'a':'f' || c in 'A':'F'), h)
        return (false, RGBAf(0, 0, 0, 1))
    end
    r = parse(Int, h[1:2]; base = 16) / 255
    g = parse(Int, h[3:4]; base = 16) / 255
    b = parse(Int, h[5:6]; base = 16) / 255
    a = length(h) == 8 ? parse(Int, h[7:8]; base = 16) / 255 : 1.0
    return (true, RGBAf(r, g, b, a))
end

function _try_contour_color(token::AbstractString)
    s = strip(String(token))
    isempty(s) && return (true, nothing)
    if startswith(s, "#")
        return _try_hex_contour_color(s)
    end
    for key in (Symbol(s), Symbol(lowercase(s)))
        try
            return (true, Makie.to_color(key))
        catch
        end
    end
    return (false, nothing)
end

"""
    parse_contour_specs(txt; fallback_levels=Float32[], fallback_colors=String[])

Parse manual contours. Empty text means automatic contours. Each entry can be
just a level (`1, 2, 3`) or a level with a color (`1:red, 2:#00ffaa`).
"""
function parse_contour_specs(
    txt::AbstractString;
    fallback_levels = Float32[],
    fallback_colors = String[],
)
    s = strip(String(txt))
    isempty(s) && return (
        true,
        false,
        Float32.(fallback_levels),
        String.(fallback_colors),
        "Automatic contour levels enabled.",
    )
    tokens = if occursin(r"[:=]", s)
        filter(!isempty, strip.(split(s, r"[,;\n]+")))
    else
        filter(!isempty, split(s, r"[,;\s]+"))
    end
    isempty(tokens) && return (
        true,
        false,
        Float32.(fallback_levels),
        String.(fallback_colors),
        "Automatic contour levels enabled.",
    )
    vals = Float32[]
    colors = String[]
    for tok in tokens
        parts = split(tok, r"\s*[:=]\s*"; limit = 2)
        level_txt = strip(first(parts))
        color_txt = length(parts) == 2 ? strip(last(parts)) : ""
        v = tryparse(Float32, level_txt)
        v === nothing && return (
            false,
            true,
            Float32.(fallback_levels),
            String.(fallback_colors),
            "Contour levels must be valid numbers.",
        )
        isfinite(v) || return (
            false,
            true,
            Float32.(fallback_levels),
            String.(fallback_colors),
            "Contour levels must be finite.",
        )
        ok_color, _ = _try_contour_color(color_txt)
        ok_color || return (
            false,
            true,
            Float32.(fallback_levels),
            String.(fallback_colors),
            "Contour colors must be names like red/blue or hex codes like #00ffaa.",
        )
        push!(vals, v)
        push!(colors, color_txt)
    end
    order = sortperm(vals)
    vals = vals[order]
    colors = colors[order]
    keep = trues(length(vals))
    for i in 2:length(vals)
        if vals[i] == vals[i - 1]
            keep[i - 1] = false
        end
    end
    vals = vals[keep]
    colors = colors[keep]
    length(vals) < 1 && return (
        false,
        true,
        Float32.(fallback_levels),
        String.(fallback_colors),
        "Provide at least one contour level.",
    )
    colored = any(!isempty, colors)
    msg = colored ? "Manual contour levels and colors applied." : "Manual contour levels applied."
    return (true, true, vals, colors, msg)
end

function _format_level(x::Real)
    xf = Float64(x)
    r = round(xf)
    return abs(xf - r) < 1e-8 ? string(Int(r)) : string(x)
end

"""
    format_contour_specs(levels, colors) -> String

Format contour levels back into the UI textbox syntax.
"""
function format_contour_specs(levels, colors = String[])
    out = String[]
    for (i, level) in enumerate(levels)
        color = i <= length(colors) ? strip(String(colors[i])) : ""
        push!(out, isempty(color) ? _format_level(level) : string(_format_level(level), ":", color))
    end
    return join(out, ", ")
end

"""
    contour_color_values(colors, n, default_color) -> Vector

Return Makie-ready color values for `n` contour levels.
"""
function contour_color_values(colors, n::Integer, default_color)
    out = Any[]
    for i in 1:max(0, n)
        token = i <= length(colors) ? strip(String(colors[i])) : ""
        if isempty(token)
            push!(out, default_color)
        else
            ok, c = _try_contour_color(token)
            push!(out, ok && c !== nothing ? c : default_color)
        end
    end
    return out
end

############################
# Mapping / Slicing
############################

"""
    ijk_to_uv(i, j, k, axis) -> (u, v)

Map 3D voxel → 2D slice coords.
axis=1 ⇒ (u=j, v=k), axis=2 ⇒ (u=i, v=k), axis=3 ⇒ (u=i, v=j).
"""
@inline function ijk_to_uv(i::Int, j::Int, k::Int, axis::Int)
    axis == 1 && return (j, k)  # (y,z)
    axis == 2 && return (i, k)  # (x,z)
    return (i, j)               # (x,y)
end

"""
    uv_to_ijk(u, v, axis, idx) -> (i, j, k)

Inverse: 2D coords + slice index → 3D voxel.
"""
@inline function uv_to_ijk(u::Int, v::Int, axis::Int, idx::Int)
    axis == 1 && return (idx, u, v)
    axis == 2 && return (u, idx, v)
    return (u, v, idx)
end

"""
    get_slice(data::Array{T,3}, axis, idx) -> AbstractMatrix

Returns a 2D slice (view when possible), orientation consistent with `ijk_to_uv`.
"""
function get_slice(data::AbstractArray{T,3}, axis::Integer, idx::Integer) where {T}
    @assert 1 ≤ axis ≤ 3 "axis must be 1,2,3"
    if axis == 1
        @views return copy(data[idx, :, :])  # (y,z)
    elseif axis == 2
        @views return copy(data[:, idx, :])  # (x,z)
    else
        @views return copy(data[:, :, idx])  # (x,y)
    end
end

"""
    region_uv_indices(u_max, v_max, x0, y0, x1, y1, shape) -> Vector{Tuple{Int,Int}}

Return `(u, v)` pixels inside a drawn region. `x` maps to `v`, `y` maps
to `u`, matching the image axis convention.
"""
function region_uv_indices(
    u_max::Int,
    v_max::Int,
    x0::Real,
    y0::Real,
    x1::Real,
    y1::Real,
    shape::Symbol,
)
    if u_max < 1 || v_max < 1
        return Tuple{Int,Int}[]
    end
    if shape === :circle
        cx, cy = Float64(x0), Float64(y0)
        r = hypot(Float64(x1) - cx, Float64(y1) - cy)
        r < 0.5 && return Tuple{Int,Int}[]
        umin = clamp(Int(floor(cy - r)), 1, u_max)
        umax = clamp(Int(ceil(cy + r)), 1, u_max)
        vmin = clamp(Int(floor(cx - r)), 1, v_max)
        vmax = clamp(Int(ceil(cx + r)), 1, v_max)
        rr = r * r
        return [(u, v) for u in umin:umax for v in vmin:vmax if (v - cx)^2 + (u - cy)^2 <= rr]
    else
        xmin, xmax = minmax(Float64(x0), Float64(x1))
        ymin, ymax = minmax(Float64(y0), Float64(y1))
        if abs(xmax - xmin) < 0.5 || abs(ymax - ymin) < 0.5
            return Tuple{Int,Int}[]
        end
        umin = clamp(Int(round(ymin)), 1, u_max)
        umax = clamp(Int(round(ymax)), 1, u_max)
        vmin = clamp(Int(round(xmin)), 1, v_max)
        vmax = clamp(Int(round(xmax)), 1, v_max)
        return [(u, v) for u in umin:umax for v in vmin:vmax]
    end
end

"""
    mean_region_spectrum(data, axis, uv_indices) -> Vector{Float32}

Average the spectrum along `axis` over a set of `(u, v)` pixels in the
current slice plane. Non-finite voxels are ignored channel by channel.
"""
function mean_region_spectrum(data::AbstractArray{T,3}, axis::Integer, uv_indices) where {T}
    1 <= axis <= 3 || throw(ArgumentError("axis must be 1, 2, or 3"))
    n = size(data, axis)
    y = fill(Float32(NaN), n)
    isempty(uv_indices) && return y
    @inbounds for chan in 1:n
        acc = 0.0
        cnt = 0
        for (u, v) in uv_indices
            val = if axis == 1
                data[chan, u, v]
            elseif axis == 2
                data[u, chan, v]
            else
                data[u, v, chan]
            end
            fv = Float32(val)
            if isfinite(fv)
                acc += Float64(fv)
                cnt += 1
            end
        end
        y[chan] = cnt == 0 ? Float32(NaN) : Float32(acc / cnt)
    end
    return y
end

############################
# Simple FITS WCS
############################

struct SimpleWCSAxis
    ctype::String
    cunit::String
    crval::Float64
    crpix::Float64
    cdelt::Float64
    available::Bool
end

header_has(header, key::AbstractString) = try
    haskey(header, String(key))
catch
    false
end

header_get(header, key::AbstractString, default) = header_has(header, key) ? header[String(key)] : default

"""
    read_simple_wcs(header, naxes) -> Vector{SimpleWCSAxis}

Read a lightweight linear WCS from FITS header keywords. This intentionally
handles the common `CTYPE/CRVAL/CRPIX/CDELT/CUNIT` case without requiring a
full WCS dependency.
"""
function read_simple_wcs(header, naxes::Integer)
    axes = SimpleWCSAxis[]
    for dim in 1:naxes
        ctype = String(header_get(header, "CTYPE$(dim)", ""))
        cunit = String(header_get(header, "CUNIT$(dim)", ""))
        crval = Float64(header_get(header, "CRVAL$(dim)", 0.0))
        crpix = Float64(header_get(header, "CRPIX$(dim)", 1.0))
        cdelt = Float64(header_get(header, "CDELT$(dim)", 1.0))
        available = !isempty(ctype) || header_has(header, "CRVAL$(dim)") || header_has(header, "CDELT$(dim)")
        push!(axes, SimpleWCSAxis(ctype, cunit, crval, crpix, cdelt, available))
    end
    return axes
end

has_wcs(wcs, dim::Integer) = 1 <= dim <= length(wcs) && wcs[dim].available

world_coord(wcs, dim::Integer, pix::Real) =
    has_wcs(wcs, dim) ? wcs[dim].crval + (Float64(pix) - wcs[dim].crpix) * wcs[dim].cdelt : Float64(pix)

function wcs_axis_label(wcs, dim::Integer; fallback::AbstractString = "pixel")
    if !has_wcs(wcs, dim)
        return latexstring("\\text{", latex_safe(fallback), "}")
    end
    ax = wcs[dim]
    ctype = uppercase(ax.ctype)
    name = if occursin("RA", ctype)
        "RA"
    elseif occursin("DEC", ctype)
        "Dec"
    elseif occursin("GLON", ctype) || occursin("LON", ctype)
        "longitude"
    elseif occursin("GLAT", ctype) || occursin("LAT", ctype)
        "latitude"
    elseif isempty(ax.ctype)
        "world $(dim)"
    else
        ax.ctype
    end
    unit = isempty(ax.cunit) ? "" : " [$(ax.cunit)]"
    return latexstring("\\text{", latex_safe(name * unit), "}")
end

function format_world_coord(wcs, dim::Integer, pix::Real)
    if !has_wcs(wcs, dim)
        return "pix$(dim)=" * string(round(Float64(pix); digits = 2))
    end
    ax = wcs[dim]
    ctype = isempty(ax.ctype) ? "axis$(dim)" : ax.ctype
    val = world_coord(wcs, dim, pix)
    unit = isempty(ax.cunit) ? "" : " $(ax.cunit)"
    return "$(ctype)=" * string(round(val; digits = 5)) * unit
end

"""
    data_unit_label(header; fallback="value") -> String

Return the FITS image data unit from `BUNIT` when present, otherwise a
generic fallback label for scalar pixel values.
"""
function data_unit_label(header; fallback::AbstractString = "value")
    header === nothing && return String(fallback)
    unit = header_get(header, "BUNIT", "")
    s = strip(String(unit))
    return isempty(s) ? String(fallback) : s
end

############################
# LaTeX helpers (safe)
############################

"""
    latex_safe(s) -> String

Escape special LaTeX characters.
"""
function latex_safe(s::AbstractString)
    t = String(s)
    t = replace(t, "\\" => "\\textbackslash{}")
    t = replace(t, "_" => "\\_")
    t = replace(t, "%" => "\\%")
    t = replace(t, "&" => "\\&")
    t = replace(t, "#" => "\\#")
    t = replace(t, "\$" => "\\\$")
    t = replace(t, "{" => "\\{")
    t = replace(t, "}" => "\\}")
    t = replace(t, "^" => "\\^{}")
    t = replace(t, "~" => "\\~{}")
    return t
end

"""
    make_main_title(fname) -> LaTeXString
"""
make_main_title(fname::AbstractString) = latexstring("\\text{", latex_safe(fname), "}")

"""
    make_slice_title(fname, axis, idx) -> LaTeXString
"""
make_slice_title(fname::AbstractString, axis::Int, idx::Int) =
    latexstring("\\text{", latex_safe(fname), " — slice axis $(axis), index $(idx)}")

"""
    make_spec_title(i,j,k) -> LaTeXString
"""
make_spec_title(i::Int, j::Int, k::Int) =
    latexstring("\\text{Spectrum at pixel }(i,j,k) = ($i,$j,$k)")

"""
    make_info_tex(i,j,k,u,v,val) -> LaTeXString

Inline format; no line breaks to keep layout stable.
"""
make_info_tex(i::Int, j::Int, k::Int, u::Int, v::Int, val::Real) = latexstring(
    "\\mathbf{pixel}\\,(i,j,k)=($i,$j,$k)\\quad\\mathbf{slice}\\,(\\text{row},\\text{col})=($u,$v)\\quad\\mathbf{intensity}= ",
    isnan(val) ? "NaN" : string(round(Float32(val); digits=4))
)

############################
# IO / UI helpers
############################

"""
    to_cmap(name::Union{Symbol,String}) -> colormap

Resolve to a Makie colormap.
"""
to_cmap(name::Union{Symbol,String}) = Makie.to_colormap(Symbol(name))

"""
    get_box_str(textbox) -> String

Read the content of a Makie Textbox robustly.
"""
function get_box_str(tb)
    s = try
        tb.stored_string[]
    catch
        nothing
    end
    if s === nothing || (s isa AbstractString && isempty(s))
        s2 = try
            tb.displayed_string[]
        catch
            ""
        end
        return strip(String(s2))
    else
        return strip(String(s))
    end
end

############################
# Window size
############################

"""
    _pick_fig_size(sizeopt) -> (w::Int, h::Int)

Use an explicit size when provided; otherwise, return a fallback `(1800, 1000)`.
"""
@inline function _pick_fig_size(sizeopt)
    sizeopt !== nothing ? (Int(sizeopt[1]), Int(sizeopt[2])) : (1800, 1000)
end

############################
# Input validation
############################

"""
    parse_manual_clims(min_txt, max_txt; fallback=(0f0, 1f0))
      -> (ok, use_manual, clims, message)

Validate and normalize user-provided colorbar limits.
"""
function parse_manual_clims(
    min_txt::AbstractString,
    max_txt::AbstractString;
    fallback::Tuple{Float32,Float32} = (0f0, 1f0)
)
    smin = strip(String(min_txt))
    smax = strip(String(max_txt))
    if isempty(smin) && isempty(smax)
        return (true, false, fallback, "Automatic color limits enabled.")
    end
    if isempty(smin) ⊻ isempty(smax)
        return (false, false, fallback, "Fill both min and max, or clear both for auto mode.")
    end
    vmin = tryparse(Float32, smin)
    vmax = tryparse(Float32, smax)
    if vmin === nothing || vmax === nothing
        return (false, false, fallback, "Colorbar limits must be valid numbers.")
    end
    lo = Float32(vmin)
    hi = Float32(vmax)
    if lo > hi
        lo, hi = hi, lo
        return (true, true, (lo, hi), "Colorbar limits were swapped because min > max.")
    end
    if lo == hi
        lo = prevfloat(lo)
        hi = nextfloat(hi)
        return (true, true, (lo, hi), "Expanded equal min/max colorbar limits to avoid zero width.")
    end
    return (true, true, (lo, hi), "Manual color limits applied.")
end

"""
    parse_gif_request(start_txt, stop_txt, step_txt, fps_txt, amax; pingpong=false)
      -> (ok, frames, fps, message)

Validate and normalize GIF export parameters.
"""
function parse_gif_request(
    start_txt::AbstractString,
    stop_txt::AbstractString,
    step_txt::AbstractString,
    fps_txt::AbstractString,
    amax::Int;
    pingpong::Bool = false
)
    amax < 1 && return (false, Int[], 12, "Cannot export GIF: axis length must be >= 1.")

    parse_int_or_default(txt::AbstractString, default::Int) =
        isempty(strip(txt)) ? default : something(tryparse(Int, strip(txt)), typemin(Int))

    startv = parse_int_or_default(start_txt, 1)
    stopv  = parse_int_or_default(stop_txt, amax)
    stepv  = parse_int_or_default(step_txt, 1)
    fpsv   = parse_int_or_default(fps_txt, 12)

    if startv == typemin(Int) || stopv == typemin(Int) || stepv == typemin(Int) || fpsv == typemin(Int)
        return (false, Int[], 12, "GIF fields must be integers.")
    end
    if stepv <= 0
        return (false, Int[], 12, "GIF step must be >= 1.")
    end
    if fpsv <= 0
        return (false, Int[], 12, "GIF fps must be >= 1.")
    end

    swapped = false
    if startv > stopv
        startv, stopv = stopv, startv
        swapped = true
    end

    startv = clamp(startv, 1, amax)
    stopv  = clamp(stopv, 1, amax)
    frames = collect(startv:stepv:stopv)
    isempty(frames) && return (false, Int[], fpsv, "No GIF frame generated from the selected range.")

    if pingpong && length(frames) >= 2
        frames = vcat(frames, reverse(frames[2:end-1]))
    end

    if swapped
        return (true, frames, fpsv, "GIF start/stop were swapped because start > stop.")
    end
    return (true, frames, fpsv, "GIF settings applied.")
end

############################
# Settings I/O
############################

"""
    save_viewer_settings(path, settings)

Write a viewer settings dict to TOML.
"""
function save_viewer_settings(path::AbstractString, settings::AbstractDict{<:AbstractString,<:Any})
    open(path, "w") do io
        TOML.print(io, Dict{String,Any}(settings))
    end
    return nothing
end

"""
    load_viewer_settings(path) -> Dict{String, Any}

Read a viewer settings dict from TOML.
"""
function load_viewer_settings(path::AbstractString)::Dict{String,Any}
    return Dict{String,Any}(TOML.parsefile(path))
end
