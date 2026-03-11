#       API stable: apply_scale, clamped_extrema, ijk_to_uv, uv_to_ijk, get_slice,
#                   make_info_tex, to_cmap, get_box_str, _pick_fig_size,
#                   latex_safe, make_main_title, make_slice_title, make_spec_title,
#                   parse_manual_clims, parse_gif_request, save_viewer_settings, load_viewer_settings

############################
# Exports
############################
export apply_scale, clamped_extrema
export ijk_to_uv, uv_to_ijk, get_slice
export make_info_tex
export to_cmap, get_box_str, _pick_fig_size
export latex_safe, make_main_title, make_slice_title, make_spec_title
export parse_manual_clims, parse_gif_request
export save_viewer_settings, load_viewer_settings

############################
# Deps
############################
using Makie
using LaTeXStrings
using TOML

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
        @views return data[idx, :, :]  # (y,z)
    elseif axis == 2
        @views return data[:, idx, :]  # (x,z)
    else
        @views return data[:, :, idx]  # (x,y)
    end
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
    "\\text{pixel }(i,j,k) = ($i,$j,$k)\\,\\text{ ; slice }(\\text{row},\\text{col}) = ($u,$v)\\,\\text{ ; value }= ",
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

Use an explicit size when provided; otherwise, return a fallback `(1800, 900)`.
"""
@inline function _pick_fig_size(sizeopt)
    sizeopt !== nothing ? (Int(sizeopt[1]), Int(sizeopt[2])) : (1800, 900)
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
