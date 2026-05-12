#       API stable: apply_scale, clamped_extrema, percentile_clims, histogram_counts,
#                   histogram_profile, histogram_ylabel,
#                   is_rgb_like, as_rgb_image, as_rgb_pixels, rgb_image,
#                   nan_gaussian_filter,
#                   automatic_contour_levels, parse_contour_levels,
#                   parse_contour_specs, format_contour_specs, contour_color_values,
#                   ijk_to_uv, uv_to_ijk, get_slice,
#                   region_uv_indices, mean_region_spectrum,
#                   dual_view_product, moments, moments_map, moment_map,
#                   moment_vectors,
#                   filtered_cube_by_slice,
#                   make_info_tex, to_cmap, get_box_str, _pick_fig_size,
#                   _axis_render_height,
#                   latex_safe, make_main_title, make_slice_title, make_spec_title,
#                   parse_manual_clims, parse_gif_request,
#                   SimpleWCSAxis, read_simple_wcs, has_wcs, world_coord,
#                   wcs_axis_label, format_world_coord, data_unit_label,
#                   save_viewer_settings, load_viewer_settings

############################
# Exports
############################
export apply_scale, clamped_extrema, percentile_clims, histogram_counts
export histogram_profile, histogram_ylabel
export is_rgb_like, as_rgb_image, as_rgb_pixels, rgb_image
export nan_gaussian_filter
export automatic_contour_levels, parse_contour_levels
export parse_contour_specs, format_contour_specs, contour_color_values
export ijk_to_uv, uv_to_ijk, get_slice, get_slice_view, get_slice_copy
export as_float32, parse_path_spec
export region_uv_indices, mean_region_spectrum
export dual_view_product, moments, moments_map, moment_map, moment_vectors, filtered_cube_by_slice
export make_info_tex
export MANTA_COLORMAP_OPTIONS, ui_colormap_options
export to_cmap, get_box_str, _pick_fig_size, _axis_render_height
export latex_safe, make_main_title, make_slice_title, make_spec_title
export parse_manual_clims, parse_histogram_bins, parse_histogram_xlimits
export parse_histogram_ylimits, parse_spectrum_ylimits, parse_gif_request
export SimpleWCSAxis, read_simple_wcs, has_wcs, world_coord
export wcs_axis_label, format_world_coord, data_unit_label
export save_viewer_settings, load_viewer_settings
export power_spectrum_2d, power_spectrum_1d_radial, fit_loglog_slope

############################
# Deps
############################
using Makie
using LaTeXStrings
using TOML
using Statistics: quantile, mean
using ImageFiltering
using FFTW: fft, fftshift
import GLFW

############################
# Filtering
############################

"""
    nan_gaussian_filter(img, sigma) -> Matrix{Float32}

Gaussian smoothing for projected maps with NaNs. Finite values are filtered and
renormalized by a filtered validity mask, so invalid/outside pixels do not bleed
into the map. Invalid output pixels stay `NaN32`.
"""
function nan_gaussian_filter(img::AbstractMatrix, sigma::Real)
    σ = Float32(sigma)
    σ <= 0 && return Float32.(img)
    values = similar(img, Float32)
    weights = similar(img, Float32)
    @inbounds for i in eachindex(img)
        v = Float32(img[i])
        if isfinite(v)
            values[i] = v
            weights[i] = 1f0
        else
            values[i] = 0f0
            weights[i] = 0f0
        end
    end
    k = ImageFiltering.Kernel.gaussian((σ, σ))
    smooth_values = imfilter(values, k)
    smooth_weights = imfilter(weights, k)
    out = similar(values, Float32)
    @inbounds for i in eachindex(out)
        w = smooth_weights[i]
        out[i] = w > 1f-6 ? Float32(smooth_values[i] / w) : NaN32
    end
    return out
end

############################
# RGB helpers
############################

is_rgb_like(x) = x isa AbstractArray && (
    eltype(x) <: Colorant ||
    (ndims(x) == 3 && (size(x, 1) in (3, 4) || size(x, 3) in (3, 4))) ||
    (ndims(x) == 2 && (size(x, 1) in (3, 4) || size(x, 2) in (3, 4)) && valid_healpix_npix(maximum(size(x))) > 0)
)

_unit_channel(v) = begin
    x = Float32(v)
    isfinite(x) ? clamp(x, 0f0, 1f0) : 0f0
end

function _normalize_rgb_channel(ch, mode::Symbol)
    out = similar(ch, Float32)
    if mode === :none
        @inbounds for i in eachindex(ch)
            out[i] = _unit_channel(ch[i])
        end
        return out
    elseif mode === :symmetric
        m = 0f0
        @inbounds for v in ch
            fv = Float32(v)
            isfinite(fv) && (m = max(m, abs(fv)))
        end
        if m == 0f0
            fill!(out, 0.5f0)
        else
            @inbounds for i in eachindex(ch)
                fv = Float32(ch[i])
                out[i] = isfinite(fv) ? clamp(0.5f0 + 0.5f0 * fv / m, 0f0, 1f0) : 0f0
            end
        end
        return out
    elseif mode === :minmax
        lo, hi = clamped_extrema(ch)
        span = hi - lo
        if span == 0f0
            fill!(out, 0.5f0)
        else
            @inbounds for i in eachindex(ch)
                fv = Float32(ch[i])
                out[i] = isfinite(fv) ? clamp((fv - lo) / span, 0f0, 1f0) : 0f0
            end
        end
        return out
    else
        throw(ArgumentError("RGB normalization must be :none, :minmax, or :symmetric."))
    end
end

"""
    rgb_image(r, g, b; normalize=:symmetric) -> Matrix{RGBf}

Build a display-ready RGB image from three scalar channels of the same size.
`normalize` can be `:symmetric`, `:minmax`, or `:none`.
"""
function rgb_image(r::AbstractMatrix, g::AbstractMatrix, b::AbstractMatrix; normalize::Symbol = :symmetric)
    (size(r) == size(g) && size(r) == size(b)) || throw(ArgumentError("RGB channels must have identical sizes."))
    R = _normalize_rgb_channel(r, normalize)
    G = _normalize_rgb_channel(g, normalize)
    B = _normalize_rgb_channel(b, normalize)
    out = Matrix{RGBf}(undef, size(R))
    @inbounds for i in eachindex(R)
        out[i] = RGBf(R[i], G[i], B[i])
    end
    return out
end

"""
    as_rgb_image(img) -> AbstractMatrix{<:Colorant}

Accept either a 2D colorant matrix or a numeric 3/4-channel stack with channels
in the first or last dimension. Numeric channels are interpreted in `[0, 1]`.
"""
function as_rgb_image(img::AbstractMatrix{<:Colorant})
    return img
end

function as_rgb_image(img::AbstractArray)
    ndims(img) == 3 || throw(ArgumentError("RGB image must be a color matrix or a 3D stack with 3/4 channels."))
    if size(img, 1) in (3, 4)
        rows, cols = size(img, 2), size(img, 3)
        if size(img, 1) == 3
            return RGBf[
                RGBf(_unit_channel(img[1, i, j]), _unit_channel(img[2, i, j]), _unit_channel(img[3, i, j]))
                for i in 1:rows, j in 1:cols
            ]
        else
            return RGBAf[
                RGBAf(_unit_channel(img[1, i, j]), _unit_channel(img[2, i, j]), _unit_channel(img[3, i, j]), _unit_channel(img[4, i, j]))
                for i in 1:rows, j in 1:cols
            ]
        end
    elseif size(img, 3) in (3, 4)
        rows, cols = size(img, 1), size(img, 2)
        if size(img, 3) == 3
            return RGBf[
                RGBf(_unit_channel(img[i, j, 1]), _unit_channel(img[i, j, 2]), _unit_channel(img[i, j, 3]))
                for i in 1:rows, j in 1:cols
            ]
        else
            return RGBAf[
                RGBAf(_unit_channel(img[i, j, 1]), _unit_channel(img[i, j, 2]), _unit_channel(img[i, j, 3]), _unit_channel(img[i, j, 4]))
                for i in 1:rows, j in 1:cols
            ]
        end
    else
        throw(ArgumentError("RGB stack must have 3 or 4 channels in the first or last dimension."))
    end
end

"""
    as_rgb_pixels(pixels) -> Vector{<:Colorant}

Accept a HEALPix RGB vector or a numeric `npix×3`, `npix×4`, `3×npix`, or
`4×npix` array. Numeric channels are interpreted in `[0, 1]`.
"""
function as_rgb_pixels(pixels::AbstractVector{<:Colorant})
    valid_healpix_npix(length(pixels)) > 0 || throw(ArgumentError("RGB HEALPix vector length must be 12*nside^2."))
    return pixels
end

function as_rgb_pixels(pixels::AbstractMatrix)
    rows, cols = size(pixels)
    if cols in (3, 4) && valid_healpix_npix(rows) > 0
        if cols == 3
            return RGBf[RGBf(_unit_channel(pixels[i, 1]), _unit_channel(pixels[i, 2]), _unit_channel(pixels[i, 3])) for i in 1:rows]
        else
            return RGBAf[RGBAf(_unit_channel(pixels[i, 1]), _unit_channel(pixels[i, 2]), _unit_channel(pixels[i, 3]), _unit_channel(pixels[i, 4])) for i in 1:rows]
        end
    elseif rows in (3, 4) && valid_healpix_npix(cols) > 0
        if rows == 3
            return RGBf[RGBf(_unit_channel(pixels[1, i]), _unit_channel(pixels[2, i]), _unit_channel(pixels[3, i])) for i in 1:cols]
        else
            return RGBAf[RGBAf(_unit_channel(pixels[1, i]), _unit_channel(pixels[2, i]), _unit_channel(pixels[3, i]), _unit_channel(pixels[4, i])) for i in 1:cols]
        end
    else
        throw(ArgumentError("RGB HEALPix pixels must be a vector or a numeric npix×3/4 or 3/4×npix matrix."))
    end
end

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

normalize_histogram_mode(mode)::Symbol = begin
    m = Symbol(lowercase(String(mode)))
    if m in (:bar, :bars, :hist, :histogram)
        :bars
    elseif m in (:kde, :density)
        :kde
    else
        :bars
    end
end

histogram_ylabel(mode) = normalize_histogram_mode(mode) === :kde ? L"\text{density}" : L"\text{count}"

function _smooth_histogram_density(counts::Vector{Float32}, width::Float32)
    n = length(counts)
    n == 0 && return Float32[]
    total = sum(counts)
    if total <= 0f0 || width <= 0f0
        return zeros(Float32, n)
    end
    σ = Float32(max(1.0, n / 64))
    radius = max(1, ceil(Int, 3σ))
    offsets = -radius:radius
    kernel = Float32[exp(-0.5f0 * (Float32(k) / σ)^2) for k in offsets]
    kernel ./= sum(kernel)
    smoothed = zeros(Float32, n)
    @inbounds for i in 1:n
        acc = 0f0
        for (j, k) in enumerate(offsets)
            idx = i + k
            if 1 <= idx <= n
                acc += counts[idx] * kernel[j]
            end
        end
        smoothed[i] = acc / (total * width)
    end
    smoothed
end

"""
    histogram_profile(vals; bins=48, limits=nothing, mode=:bars)
        -> (x, y, width, mode)

Histogram data prepared for UI display. `mode=:bars` returns bin counts;
`mode=:kde` returns a binned Gaussian density estimate on the same x grid.
"""
function histogram_profile(vals; bins::Int = 48, limits = nothing, mode = :bars)
    nb = max(1, bins)
    centers, counts = histogram_counts(vals; bins = nb, limits = limits)
    if isempty(centers)
        return (x = Float32[], y = Float32[], width = 1f0, mode = normalize_histogram_mode(mode))
    end
    width = length(centers) > 1 ? Float32(centers[2] - centers[1]) : 1f0
    mode_sym = normalize_histogram_mode(mode)
    y = mode_sym === :kde ? _smooth_histogram_density(counts, width) : counts
    return (x = centers, y = y, width = width, mode = mode_sym)
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
    get_slice_view(data::AbstractArray{T,3}, axis, idx) -> SubArray

Return a non-allocating 2D view into `data` along `axis` at index `idx`.
Orientation is consistent with [`ijk_to_uv`](@ref): `axis==1` ⇒ (y,z),
`axis==2` ⇒ (x,z), `axis==3` ⇒ (x,y).

Mutating the returned view will mutate the underlying cube. Use
[`get_slice_copy`](@ref) when an independent buffer is required.
"""
function get_slice_view(data::AbstractArray{T,3}, axis::Integer, idx::Integer) where {T}
    1 <= axis <= 3 || throw(ArgumentError("MANTA: axis must be 1, 2 or 3, got $(axis)"))
    1 <= idx <= size(data, axis) || throw(BoundsError(data, (axis, idx)))
    if axis == 1
        return @view data[idx, :, :]   # (y, z)
    elseif axis == 2
        return @view data[:, idx, :]   # (x, z)
    else
        return @view data[:, :, idx]   # (x, y)
    end
end

"""
    get_slice_copy(data::AbstractArray{T,3}, axis, idx) -> Array

Return a freshly allocated 2D slice. Equivalent to `copy(get_slice_view(...))`.
Use this when the caller needs to mutate the slice without touching the cube.
"""
get_slice_copy(data::AbstractArray, axis::Integer, idx::Integer) =
    copy(get_slice_view(data, axis, idx))

"""
    get_slice(data, axis, idx) -> AbstractMatrix

Backwards-compatible slice accessor. Historically copying; kept as a thin
alias over [`get_slice_copy`](@ref) so existing callers in the cube viewer
keep their independent buffers. New code should pick `get_slice_view` when
no mutation is needed.
"""
get_slice(data::AbstractArray, axis::Integer, idx::Integer) =
    get_slice_copy(data, axis, idx)

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

function finite_mean_std(vals)
    acc = 0.0
    cnt = 0
    @inbounds for v in vals
        x = Float64(v)
        if isfinite(x)
            acc += x
            cnt += 1
        end
    end
    cnt == 0 && return (NaN, NaN)
    μ = acc / cnt
    acc2 = 0.0
    @inbounds for v in vals
        x = Float64(v)
        if isfinite(x)
            acc2 += (x - μ)^2
        end
    end
    σ = cnt <= 1 ? 0.0 : sqrt(acc2 / (cnt - 1))
    return (μ, σ)
end

"""
    dual_view_product(a, b, mode) -> Matrix{Float32}

Compute the right-hand dual-view product from primary slice `a` and secondary
slice `b`. `mode` accepts `:A`, `:B`, `:diff`, `:ratio`, or `:residuals`.
"""
function dual_view_product(a::AbstractMatrix, b::AbstractMatrix, mode::Symbol)
    size(a) == size(b) || throw(DimensionMismatch("dual slices must have the same size"))
    out = similar(Float32.(a))
    if mode === :A
        copyto!(out, Float32.(a))
    elseif mode === :B
        copyto!(out, Float32.(b))
    elseif mode === :diff
        @inbounds for i in eachindex(out, a, b)
            out[i] = Float32(a[i]) - Float32(b[i])
        end
    elseif mode === :ratio
        @inbounds for i in eachindex(out, a, b)
            den = Float32(b[i])
            num = Float32(a[i])
            out[i] = isfinite(num) && isfinite(den) && den != 0f0 ? num / den : NaN32
        end
    elseif mode === :residuals
        diff = similar(out)
        @inbounds for i in eachindex(diff, a, b)
            diff[i] = Float32(a[i]) - Float32(b[i])
        end
        μ, σ = finite_mean_std(diff)
        if !isfinite(σ) || σ <= 0
            fill!(out, NaN32)
        else
            @inbounds for i in eachindex(out, diff)
                x = Float64(diff[i])
                out[i] = isfinite(x) ? Float32((x - μ) / σ) : NaN32
            end
        end
    else
        throw(ArgumentError("unknown dual view mode: $(mode)"))
    end
    return out
end

function _channel_value(data, axis::Integer, u::Integer, v::Integer, chan::Integer)
    if axis == 1
        return data[chan, u, v]
    elseif axis == 2
        return data[u, chan, v]
    else
        return data[u, v, chan]
    end
end

"""
    moments(y; x=1:length(y), threshold=0.0) -> (m0, m1, m2)

Calculate zeroth, first, and second moments using only samples where
`y > threshold`, matching the project-local `Statistics/Moments.jl` routine.
"""
function moments(y; x = 1:length(y), threshold = 0.0)
    length(x) == length(y) || throw(DimensionMismatch("x and y must have the same length"))
    acc0 = 0.0
    acc1 = 0.0
    any_sample = false
    @inbounds for i in eachindex(y, x)
        yi = Float64(y[i])
        if isfinite(yi) && yi > threshold
            any_sample = true
            xi = Float64(x[i])
            acc0 += yi
            acc1 += yi * xi
        end
    end
    (!any_sample || acc0 == 0.0) && return (NaN, NaN, NaN)
    m0 = acc0
    m1 = acc1 / m0
    acc2 = 0.0
    @inbounds for i in eachindex(y, x)
        yi = Float64(y[i])
        if isfinite(yi) && yi > threshold
            xi = Float64(x[i])
            acc2 += yi * (xi - m1)^2
        end
    end
    m2_2 = acc2 / m0
    m2 = m2_2 >= 0 ? sqrt(m2_2) : NaN
    return (m0, m1, m2)
end

"""
    moments_map(data, array; threshold=0.0) -> (M0, M1, M2)

Calculate moment maps for a cube whose spectral axis is the third dimension,
matching `Statistics/Moments.jl`.
"""
function moments_map(data::AbstractArray{T,3}, array; threshold = 0.0) where {T}
    length(array) == size(data, 3) || throw(DimensionMismatch("array length must match size(data, 3)"))
    M0 = Matrix{Float32}(undef, size(data, 1), size(data, 2))
    M1 = similar(M0)
    M2 = similar(M0)
    @inbounds for i in 1:size(data, 1), j in 1:size(data, 2)
        m0, m1, m2 = moments(@view(data[i, j, :]); x = array, threshold = threshold)
        M0[i, j] = Float32(m0)
        M1[i, j] = Float32(m1)
        M2[i, j] = Float32(m2)
    end
    return M0, M1, M2
end

"""
    moment_map(data, axis, order; coords=1:size(data, axis), channels=1:size(data, axis), threshold=0.0)

Compute moment 0, 1, or 2 along `axis`, returning a 2D map in the same
orientation as `get_slice`. The moment definition is delegated to
`moments(y; x=coords, threshold=threshold)`.
"""
function moment_map(
    data::AbstractArray{T,3},
    axis::Integer,
    order::Integer;
    coords = collect(Float32, 1:size(data, axis)),
    channels = 1:size(data, axis),
    threshold = 0.0,
) where {T}
    1 <= axis <= 3 || throw(ArgumentError("axis must be 1, 2, or 3"))
    order in (0, 1, 2) || throw(ArgumentError("moment order must be 0, 1, or 2"))
    u_max, v_max = axis == 1 ? (size(data, 2), size(data, 3)) :
                   axis == 2 ? (size(data, 1), size(data, 3)) :
                               (size(data, 1), size(data, 2))
    out = fill(NaN32, u_max, v_max)
    chan_vec = [c for c in channels if 1 <= c <= size(data, axis)]
    isempty(chan_vec) && return out

    y = Vector{Float32}(undef, length(chan_vec))
    x = Vector{Float32}(undef, length(chan_vec))
    @inbounds for u in 1:u_max, v in 1:v_max
        for (n, c) in pairs(chan_vec)
            y[n] = Float32(_channel_value(data, axis, u, v, c))
            x[n] = Float32(coords[c])
        end
        m0, m1, m2 = moments(y; x = x, threshold = threshold)
        out[u, v] = order == 0 ? Float32(m0) : order == 1 ? Float32(m1) : Float32(m2)
    end
    return out
end

function moment_vectors(data::AbstractMatrix, x; threshold = 0.0)
    length(x) == size(data, 2) || throw(DimensionMismatch("x length must match size(data, 2)"))
    M0 = Vector{Float32}(undef, size(data, 1))
    M1 = similar(M0)
    M2 = similar(M0)
    @inbounds for i in 1:size(data, 1)
        m0, m1, m2 = moments(@view(data[i, :]); x = x, threshold = threshold)
        M0[i] = Float32(m0)
        M1[i] = Float32(m1)
        M2[i] = Float32(m2)
    end
    return M0, M1, M2
end

"""
    filtered_cube_by_slice(data, axis, sigma) -> Array{Float32,3}

Apply the viewer's 2D Gaussian filter independently to every slice along
`axis`.
"""
function filtered_cube_by_slice(data::AbstractArray{T,3}, axis::Integer, sigma::Real) where {T}
    1 <= axis <= 3 || throw(ArgumentError("axis must be 1, 2, or 3"))
    σ = Float32(sigma)
    σ <= 0 && return Float32.(data)
    out = similar(Float32.(data))
    for idx in 1:size(data, axis)
        s = nan_gaussian_filter(get_slice(data, axis, idx), σ)
        if axis == 1
            @views out[idx, :, :] .= s
        elseif axis == 2
            @views out[:, idx, :] .= s
        else
            @views out[:, :, idx] .= s
        end
    end
    return out
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
# Type conversion helpers
############################

"""
    as_float32(x) -> AbstractArray{Float32}

Return `x` unchanged if it is already a dense `Array{Float32}`, otherwise
allocate a fresh `Float32` copy. Centralizes the "make-it-display-ready"
conversion used by loaders and the cube viewer so that we avoid a redundant
allocation every time the data already has the right type.
"""
@inline as_float32(x::Array{Float32}) = x
@inline as_float32(x::AbstractArray) = eltype(x) === Float32 ? Array{Float32}(x) : Float32.(x)

############################
# Path spec parsing
############################

"""
    parse_path_spec(s) -> (kind, path[, address])

Inspect a path-like string and dispatch to the appropriate loader family:

- `"file.fits"`, `"file.fit"`, `"file.fits.gz"` → `(:fits, path)`
- `"file.h5"`, `"file.hdf5"`, `"file.he5"`     → `(:hdf5, path, "/")`
- `"file.h5:/group/dataset"`                    → `(:hdf5, "file.h5", "/group/dataset")`
- otherwise                                     → `(:unknown, path)`

The HDF5 `path:address` form splits on the LAST `:` only when the prefix has
an HDF5 extension and the suffix begins with `/`. This rejects Windows drive
letters (`C:/path/file.h5`) and plain FITS paths.
"""
function parse_path_spec(s::AbstractString)
    str = String(s)
    # Try HDF5 group-address form first.
    idx = findlast(==(':'), str)
    if idx !== nothing
        prefix = str[1:idx-1]
        suffix = str[idx+1:end]
        if !isempty(suffix) && startswith(suffix, "/") &&
           lowercase(splitext(prefix)[2]) ∈ (".h5", ".hdf5", ".he5")
            return (:hdf5, String(prefix), String(suffix))
        end
    end
    lower = lowercase(str)
    if endswith(lower, ".fits") || endswith(lower, ".fit") || endswith(lower, ".fits.gz")
        return (:fits, str)
    elseif endswith(lower, ".h5") || endswith(lower, ".hdf5") || endswith(lower, ".he5")
        return (:hdf5, str, "/")
    else
        return (:unknown, str)
    end
end

############################
# IO / UI helpers
############################

"""
    to_cmap(name::Union{Symbol,String}) -> colormap

Resolve to a Makie colormap.
"""
function to_cmap(name::Union{Symbol,String})
    cmap_name = Symbol(name)
    cmap_name = cmap_name in (:gray, :grey) ? :grayC : cmap_name
    return Makie.to_colormap(cmap_name)
end

const MANTA_COLORMAP_OPTIONS = ("viridis", "cividis", "magma", "inferno", "plasma", "gray")

ui_colormap_options() = collect(MANTA_COLORMAP_OPTIONS)

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

# Defaults used when no display is available (CI / headless / Docker).
const _DEFAULT_FIG_SIZE   = (1500, 900)
# Minimum we still consider "usable" so we never collapse below this.
const _MIN_FIG_SIZE       = (1100, 720)
# Maximum fraction of the screen we want a default figure to occupy. Tuned to
# leave room for the OS chrome (title bar, dock/taskbar) on small laptops.
const _FIG_SCREEN_FRAC_W  = 0.92
const _FIG_SCREEN_FRAC_H  = 0.88

# Cache so repeated calls (panels, dual view, …) don't hit GLFW each time and
# so headless runs (`activate_gl=false`) never even try to initialize it.
const _SCREEN_SIZE_CACHE = Ref{Union{Nothing,Tuple{Int,Int}}}(nothing)
const _SCREEN_SIZE_PROBED = Ref(false)

"""
    _detect_screen_size() -> Union{Nothing,Tuple{Int,Int}}

Best-effort query of the primary monitor work area (px). Resolution order:

  1. environment override `MANTA_SCREEN_W` / `MANTA_SCREEN_H` (useful for
     Docker / VNC where GLFW often misreports the workarea),
  2. early bail-out on Linux when neither `DISPLAY` nor `WAYLAND_DISPLAY` is
     set (avoids initializing GLFW in headless containers),
  3. `GLFW.GetMonitorWorkarea` on the primary monitor, falling back to the
     full video-mode size if the workarea API isn't available.

Returns `nothing` if no size could be obtained — callers must handle this
gracefully (see `_pick_fig_size` which falls back to `_DEFAULT_FIG_SIZE`).

Result is cached: the first call probes, subsequent calls reuse the value.
"""
function _detect_screen_size()::Union{Nothing,Tuple{Int,Int}}
    _SCREEN_SIZE_PROBED[] && return _SCREEN_SIZE_CACHE[]
    _SCREEN_SIZE_PROBED[] = true

    # Environment override (useful for Docker / VNC where GLFW reports wrong values).
    env_w = tryparse(Int, get(ENV, "MANTA_SCREEN_W", ""))
    env_h = tryparse(Int, get(ENV, "MANTA_SCREEN_H", ""))
    if env_w !== nothing && env_h !== nothing && env_w > 0 && env_h > 0
        _SCREEN_SIZE_CACHE[] = (env_w, env_h)
        return _SCREEN_SIZE_CACHE[]
    end

    # On Linux without DISPLAY there is no usable screen; don't probe GLFW
    # at all so we don't risk an Init() error in headless containers.
    if Sys.islinux() && isempty(get(ENV, "DISPLAY", "")) && isempty(get(ENV, "WAYLAND_DISPLAY", ""))
        _SCREEN_SIZE_CACHE[] = nothing
        return nothing
    end

    # GLFW probe. Wrapped in try/catch because:
    #   - GLFW may already be initialized by GLMakie (Init is idempotent),
    #   - the platform may not expose a primary monitor,
    #   - the workarea API may not be available on some drivers.
    val = try
        try; GLFW.Init(); catch; end
        mon = GLFW.GetPrimaryMonitor()
        # On some bindings the null monitor has a zero handle. Treat that as
        # "no monitor".
        if mon === nothing || (hasproperty(mon, :handle) && mon.handle == C_NULL)
            nothing
        else
            # GetMonitorWorkarea returns (x, y, w, h) of the usable area
            # (i.e. screen minus dock / taskbar / menu bar). Falls back to
            # the video mode if the workarea entry-point is missing.
            wa = try
                GLFW.GetMonitorWorkarea(mon)
            catch
                nothing
            end
            if wa !== nothing
                (Int(wa[3]), Int(wa[4]))
            else
                vm = GLFW.GetVideoMode(mon)
                (Int(vm.width), Int(vm.height))
            end
        end
    catch
        nothing
    end

    _SCREEN_SIZE_CACHE[] = val
    return val
end

"""
    _pick_fig_size(sizeopt) -> (w::Int, h::Int)

Resolve the figure size for a viewer:

  * if `sizeopt` is a `(w, h)` tuple it is used verbatim,
  * otherwise the primary monitor work area is queried and the result is
    capped to `_FIG_SCREEN_FRAC_W / _FIG_SCREEN_FRAC_H` of that area,
  * if no screen can be detected (headless / CI / Docker), the conservative
    `_DEFAULT_FIG_SIZE` fallback is returned.

In all cases the output is clamped above `_MIN_FIG_SIZE` so a tiny screen
doesn't yield an unusable layout.
"""
@inline function _pick_fig_size(sizeopt)
    if sizeopt !== nothing
        return (Int(sizeopt[1]), Int(sizeopt[2]))
    end
    scr = _detect_screen_size()
    if scr === nothing
        return _DEFAULT_FIG_SIZE
    end
    sw, sh = scr
    w = round(Int, sw * _FIG_SCREEN_FRAC_W)
    h = round(Int, sh * _FIG_SCREEN_FRAC_H)
    w = max(w, _MIN_FIG_SIZE[1])
    h = max(h, _MIN_FIG_SIZE[2])
    return (w, h)
end

"""
    _axis_render_height(axis)

Return an observable height matching the axis' rendered data viewport.
Useful for keeping adjacent colorbars the same height as `DataAspect()` images.
"""
_axis_render_height(axis) = lift(axis.scene.viewport) do rect
    max(1, rect.widths[2])
end

############################
# Input validation
############################

"""
    parse_manual_clims(min_txt, max_txt; fallback=(0f0, 1f0))
      -> (ok, use_manual, clims, message)

Validate and normalize user-provided contrast limits.
"""
function parse_manual_clims(
    min_txt::AbstractString,
    max_txt::AbstractString;
    fallback::Tuple{Float32,Float32} = (0f0, 1f0)
)
    smin = strip(String(min_txt))
    smax = strip(String(max_txt))
    if isempty(smin) && isempty(smax)
        return (true, false, fallback, "Automatic contrast enabled.")
    end
    if isempty(smin) ⊻ isempty(smax)
        return (false, false, fallback, "Fill both min and max, or clear both for auto mode.")
    end
    vmin = tryparse(Float32, smin)
    vmax = tryparse(Float32, smax)
    if vmin === nothing || vmax === nothing
        return (false, false, fallback, "Contrast limits must be valid numbers.")
    end
    lo = Float32(vmin)
    hi = Float32(vmax)
    if lo > hi
        lo, hi = hi, lo
        return (true, true, (lo, hi), "Contrast limits were swapped because min > max.")
    end
    if lo == hi
        lo = prevfloat(lo)
        hi = nextfloat(hi)
        return (true, true, (lo, hi), "Expanded equal min/max contrast limits to avoid zero width.")
    end
    return (true, true, (lo, hi), "Manual contrast applied.")
end

"""
    parse_histogram_bins(txt; fallback=64, min_bins=4, max_bins=512)
      -> (ok, bins, message)

Validate the histogram bin count entered in the UI.
"""
function parse_histogram_bins(
    txt::AbstractString;
    fallback::Int = 64,
    min_bins::Int = 4,
    max_bins::Int = 512,
)
    s = strip(String(txt))
    isempty(s) && return (true, clamp(fallback, min_bins, max_bins), "Histogram bin count unchanged.")
    parsed = tryparse(Int, s)
    if parsed === nothing
        return (false, fallback, "Histogram bins must be an integer.")
    end
    bins = clamp(parsed, min_bins, max_bins)
    if bins != parsed
        return (true, bins, "Histogram bins were clamped to $(bins).")
    end
    return (true, bins, "Histogram bins set to $(bins).")
end

"""
    parse_histogram_xlimits(min_txt, max_txt; fallback=(0f0, 1f0))
      -> (ok, use_manual, limits, message)

Validate user-provided histogram x-axis limits. Empty fields restore automatic
limits, which follow the current color scale limits.
"""
function parse_histogram_xlimits(
    min_txt::AbstractString,
    max_txt::AbstractString;
    fallback::Tuple{Float32,Float32} = (0f0, 1f0),
)
    smin = strip(String(min_txt))
    smax = strip(String(max_txt))
    if isempty(smin) && isempty(smax)
        return (true, false, fallback, "Automatic histogram x-axis enabled.")
    end
    if isempty(smin) ⊻ isempty(smax)
        return (false, false, fallback, "Fill both histogram x min and max, or clear both for auto mode.")
    end
    xmin = tryparse(Float32, smin)
    xmax = tryparse(Float32, smax)
    if xmin === nothing || xmax === nothing
        return (false, false, fallback, "Histogram x-axis limits must be valid numbers.")
    end
    lo = Float32(xmin)
    hi = Float32(xmax)
    if !(isfinite(lo) && isfinite(hi))
        return (false, false, fallback, "Histogram x-axis limits must be finite numbers.")
    end
    if lo > hi
        lo, hi = hi, lo
        return (true, true, (lo, hi), "Histogram x-axis limits were swapped because min > max.")
    end
    if lo == hi
        lo = prevfloat(lo)
        hi = nextfloat(hi)
        return (true, true, (lo, hi), "Expanded equal histogram x-axis limits to avoid zero width.")
    end
    return (true, true, (lo, hi), "Manual histogram x-axis applied.")
end

function _parse_axis_limits(
    min_txt::AbstractString,
    max_txt::AbstractString;
    fallback::Tuple{Float32,Float32} = (0f0, 1f0),
    axis_name::AbstractString = "axis",
)
    smin = strip(String(min_txt))
    smax = strip(String(max_txt))
    if isempty(smin) && isempty(smax)
        return (true, false, fallback, "Automatic $(axis_name) enabled.")
    end
    if isempty(smin) ⊻ isempty(smax)
        return (false, false, fallback, "Fill both $(axis_name) min and max, or clear both for auto mode.")
    end
    vmin = tryparse(Float32, smin)
    vmax = tryparse(Float32, smax)
    if vmin === nothing || vmax === nothing
        return (false, false, fallback, "$(axis_name) limits must be valid numbers.")
    end
    lo = Float32(vmin)
    hi = Float32(vmax)
    if !(isfinite(lo) && isfinite(hi))
        return (false, false, fallback, "$(axis_name) limits must be finite numbers.")
    end
    if lo > hi
        lo, hi = hi, lo
        return (true, true, (lo, hi), "$(axis_name) limits were swapped because min > max.")
    end
    if lo == hi
        lo = prevfloat(lo)
        hi = nextfloat(hi)
        return (true, true, (lo, hi), "Expanded equal $(axis_name) limits to avoid zero width.")
    end
    return (true, true, (lo, hi), "Manual $(axis_name) applied.")
end

"""
    parse_histogram_ylimits(min_txt, max_txt; fallback=(0f0, 1f0))
      -> (ok, use_manual, limits, message)

Validate user-provided histogram y-axis limits. Empty fields restore automatic
limits.
"""
parse_histogram_ylimits(min_txt::AbstractString, max_txt::AbstractString; fallback::Tuple{Float32,Float32} = (0f0, 1f0)) =
    _parse_axis_limits(min_txt, max_txt; fallback = fallback, axis_name = "histogram y-axis")

"""
    parse_spectrum_ylimits(min_txt, max_txt; fallback=(0f0, 1f0))
      -> (ok, use_manual, limits, message)

Validate user-provided spectrum y-axis limits. Empty fields restore automatic
limits.
"""
parse_spectrum_ylimits(min_txt::AbstractString, max_txt::AbstractString; fallback::Tuple{Float32,Float32} = (0f0, 1f0)) =
    _parse_axis_limits(min_txt, max_txt; fallback = fallback, axis_name = "spectrum y-axis")

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

############################
# Power spectrum
############################

"""
    _ps_window1d(kind, n) -> Vector{Float64}

1D apodization window of length `n`. `kind ∈ (:hann, :hamming, :none)`.
"""
function _ps_window1d(kind::Symbol, n::Integer)
    n <= 1 && return ones(Float64, max(n, 0))
    if kind === :hann
        return Float64[0.5 - 0.5 * cos(2π * (i - 1) / (n - 1)) for i in 1:n]
    elseif kind === :hamming
        return Float64[0.54 - 0.46 * cos(2π * (i - 1) / (n - 1)) for i in 1:n]
    else
        return ones(Float64, n)
    end
end

"""
    _ps_apodize_mask(mask, taper) -> Matrix{Float64}

Cosine taper of a binary validity mask. The taper width `taper` (in pixels)
determines how far inside the valid region the apodization runs to zero at
the boundary with invalid pixels. Uses an L∞ (Chebyshev) two-pass distance
transform — only the relative magnitude matters for the cosine taper, so the
cheap chamfer is sufficient.
"""
function _ps_apodize_mask(mask::AbstractMatrix{Bool}, taper::Integer)
    ny, nx = size(mask)
    big = float(ny + nx + 1)
    d = fill(big, ny, nx)
    @inbounds for i in eachindex(mask)
        if !mask[i]
            d[i] = 0.0
        end
    end
    @inbounds for j in 1:nx, i in 1:ny
        di = d[i, j]
        if i > 1
            di = min(di, d[i - 1, j] + 1.0)
            if j > 1;  di = min(di, d[i - 1, j - 1] + 1.0); end
            if j < nx; di = min(di, d[i - 1, j + 1] + 1.0); end
        end
        if j > 1; di = min(di, d[i, j - 1] + 1.0); end
        d[i, j] = di
    end
    @inbounds for j in nx:-1:1, i in ny:-1:1
        di = d[i, j]
        if i < ny
            di = min(di, d[i + 1, j] + 1.0)
            if j < nx; di = min(di, d[i + 1, j + 1] + 1.0); end
            if j > 1;  di = min(di, d[i + 1, j - 1] + 1.0); end
        end
        if j < nx; di = min(di, d[i, j + 1] + 1.0); end
        d[i, j] = di
    end
    out = Matrix{Float64}(undef, ny, nx)
    t = float(max(taper, 1))
    @inbounds for i in eachindex(mask)
        if !mask[i]
            out[i] = 0.0
        else
            di = d[i]
            out[i] = di >= t ? 1.0 : 0.5 * (1.0 - cos(π * di / t))
        end
    end
    return out
end

"""
    power_spectrum_2d(img;
                      window=:hann, demean=true,
                      pad_pow2=false,
                      apodize_nan=false, nan_taper=4)
        -> NamedTuple

Centered (`fftshift`) 2D power spectrum `|F(img)|² / ⟨W_eff²⟩`.

NamedTuple fields:
  - `P2d::Matrix{Float64}`           shifted power spectrum, size `(ny_eff, nx_eff)`
  - `ny_in, nx_in::Int`              size of the input image
  - `ny_eff, nx_eff::Int`            size after optional zero-padding
  - `padded::Bool`                   `true` iff `pad_pow2` actually grew the array
  - `window::Symbol`                 effective window kind
  - `apodized::Bool`                 `true` iff a NaN apodization mask was applied
  - `f_sky::Float64`                 fraction of finite input pixels
  - `w_norm::Float64`                `⟨(window × mask)²⟩`, the MASTER-light
                                     normalization that has already been
                                     divided out of `P2d`

NaN handling: non-finite pixels are first replaced with zero; the demean
operates over valid pixels only; with `apodize_nan=true` an L∞-distance
cosine taper is built around the NaN regions and combined with the spectral
window. The MASTER-light correction divides the raw `|F|²` by `⟨W_eff²⟩` so
that, for a stationary signal whose spectrum is locally flat over the window
support, the recovered amplitude is unbiased to first order.
"""
function power_spectrum_2d(img::AbstractMatrix;
                            window::Symbol = :hann,
                            demean::Bool = true,
                            pad_pow2::Bool = false,
                            apodize_nan::Bool = false,
                            nan_taper::Integer = 4)
    A = Float64.(img)
    ny0, nx0 = size(A)
    finite_mask = isfinite.(A)
    n_valid = count(finite_mask)
    f_sky = isempty(finite_mask) ? 0.0 : n_valid / length(finite_mask)
    @inbounds for i in eachindex(A)
        if !finite_mask[i]
            A[i] = 0.0
        end
    end
    if demean && n_valid > 0
        s = 0.0
        @inbounds for i in eachindex(A)
            if finite_mask[i]; s += A[i]; end
        end
        m = s / n_valid
        if isfinite(m) && m != 0.0
            @inbounds for i in eachindex(A)
                if finite_mask[i]; A[i] -= m; end
            end
        end
    end
    has_invalid = n_valid < length(finite_mask)
    M = if apodize_nan && has_invalid
        _ps_apodize_mask(finite_mask, max(Int(nan_taper), 1))
    elseif has_invalid
        Float64.(finite_mask)
    else
        ones(Float64, ny0, nx0)
    end
    wy = _ps_window1d(window, ny0)
    wx = _ps_window1d(window, nx0)
    Weff = Matrix{Float64}(undef, ny0, nx0)
    @inbounds for j in 1:nx0, i in 1:ny0
        Weff[i, j] = wy[i] * wx[j] * M[i, j]
        A[i, j] *= Weff[i, j]
    end
    ny_eff, nx_eff = ny0, nx0
    padded = false
    if pad_pow2
        ny_eff = nextpow(2, max(ny0, 1))
        nx_eff = nextpow(2, max(nx0, 1))
        if ny_eff != ny0 || nx_eff != nx0
            A_pad = zeros(Float64, ny_eff, nx_eff)
            @inbounds A_pad[1:ny0, 1:nx0] .= A
            A = A_pad
            padded = true
        end
    end
    F = fftshift(fft(A))
    P2d = abs2.(F)
    w_norm = isempty(Weff) ? 0.0 : mean(Weff .^ 2)
    if w_norm > 0
        P2d ./= w_norm
    end
    return (P2d = P2d,
            ny_in = ny0, nx_in = nx0,
            ny_eff = ny_eff, nx_eff = nx_eff,
            padded = padded,
            window = window,
            apodized = apodize_nan && has_invalid,
            f_sky = f_sky,
            w_norm = w_norm)
end

"""
    power_spectrum_1d_radial(P2d) -> (radii::Vector{Float32}, profile::Vector{Float32})

Radial average of a centered 2D power spectrum, binned by integer pixel
radius from the FFT-shifted DC center. The returned `radii` are pixel-radii
(0, 1, 2, …); convert to cycles/pixel by dividing by `min(ny, nx)`.
"""
function power_spectrum_1d_radial(P2d::AbstractMatrix)
    ny, nx = size(P2d)
    cy = (ny + 1) / 2
    cx = (nx + 1) / 2
    rmax = floor(Int, min(cy - 1, cx - 1))
    rmax < 1 && return (Float32[], Float32[])
    nb = rmax + 1
    counts = zeros(Float64, nb)
    sums = zeros(Float64, nb)
    @inbounds for j in 1:nx, i in 1:ny
        r = sqrt((i - cy)^2 + (j - cx)^2)
        b = round(Int, r) + 1
        if 1 <= b <= nb
            sums[b] += P2d[i, j]
            counts[b] += 1
        end
    end
    radii = Float32.(0:rmax)
    prof = Float32[counts[i] > 0 ? sums[i] / counts[i] : 0.0 for i in 1:nb]
    return radii, prof
end

"""
    fit_loglog_slope(k, p; kmin, kmax) -> (slope, intercept, n_used)

Least-squares fit of `log10(p) = slope·log10(k) + intercept` over the band
`kmin ≤ k ≤ kmax`. Non-positive `k` and `p` values are dropped. Returns
`(NaN, NaN, 0)` if fewer than 2 valid points fall in the band.
"""
function fit_loglog_slope(k::AbstractVector, p::AbstractVector;
                          kmin::Real = 0.0, kmax::Real = Inf)
    length(k) == length(p) || throw(ArgumentError("k and p must have the same length."))
    xs = Float64[]
    ys = Float64[]
    for i in eachindex(k)
        ki = Float64(k[i]); pi = Float64(p[i])
        if isfinite(ki) && isfinite(pi) && ki > 0 && pi > 0 && ki >= kmin && ki <= kmax
            push!(xs, log10(ki)); push!(ys, log10(pi))
        end
    end
    n = length(xs)
    n < 2 && return (NaN, NaN, n)
    mx = mean(xs); my = mean(ys)
    sxx = 0.0; sxy = 0.0
    @inbounds for i in 1:n
        dx = xs[i] - mx
        sxx += dx * dx
        sxy += dx * (ys[i] - my)
    end
    sxx == 0 && return (NaN, NaN, n)
    slope = sxy / sxx
    intercept = my - slope * mx
    return (slope, intercept, n)
end
