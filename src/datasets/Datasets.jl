# path: src/datasets/Datasets.jl
#
# Generic dataset abstraction for MANTA.
#
# A *dataset* wraps the data array(s) plus the metadata that the viewer cores
# need (axis labels, simple WCS, unit label, source identifier for settings
# files, free-form metadata). Loaders convert FITS / HDF5 / in-memory inputs
# into datasets; viewers consume datasets and never see file handles.
#
# Reuses `SimpleWCSAxis` from `helpers/Helpers.jl` (do not duplicate).

abstract type AbstractMANTADataset end

# Backwards-compatibility alias for any external code still using the
# previous name. New code should prefer `AbstractMANTADataset`.
const AbstractCartaDataset = AbstractMANTADataset

const _EMPTY_WCS = SimpleWCSAxis[]

# ---------- VectorDataset (1D line plot) ----------
struct VectorDataset{T<:Real} <: AbstractMANTADataset
    data::AbstractVector{T}
    axis_label::String
    wcs::Union{Nothing,SimpleWCSAxis}
    unit_label::String
    source_id::String
    metadata::Dict{Symbol,Any}
end

function VectorDataset(
    data::AbstractVector{T};
    axis_label::AbstractString = "index",
    wcs::Union{Nothing,SimpleWCSAxis} = nothing,
    unit_label::AbstractString = "value",
    source_id::AbstractString = "vector",
    metadata::AbstractDict = Dict{Symbol,Any}(),
) where {T<:Real}
    return VectorDataset{T}(
        data,
        String(axis_label),
        wcs,
        String(unit_label),
        String(source_id),
        Dict{Symbol,Any}(metadata),
    )
end

# ---------- ImageDataset (2D heatmap) ----------
struct ImageDataset{T<:Real} <: AbstractMANTADataset
    data::AbstractMatrix{T}
    axis_labels::Vector{String}
    wcs::Vector{SimpleWCSAxis}
    unit_label::String
    source_id::String
    metadata::Dict{Symbol,Any}
end

function ImageDataset(
    data::AbstractMatrix{T};
    axis_labels::AbstractVector{<:AbstractString} = ["axis1", "axis2"],
    wcs::AbstractVector{SimpleWCSAxis} = _EMPTY_WCS,
    unit_label::AbstractString = "value",
    source_id::AbstractString = "image",
    metadata::AbstractDict = Dict{Symbol,Any}(),
) where {T<:Real}
    length(axis_labels) == 2 || throw(ArgumentError(
        "ImageDataset needs exactly 2 axis_labels, got $(length(axis_labels))"))
    return ImageDataset{T}(
        data,
        String.(collect(axis_labels)),
        collect(SimpleWCSAxis, wcs),
        String(unit_label),
        String(source_id),
        Dict{Symbol,Any}(metadata),
    )
end

# ---------- CubeDataset (3D cube) ----------
struct CubeDataset{T<:Real} <: AbstractMANTADataset
    data::AbstractArray{T,3}
    axis_labels::Vector{String}
    wcs::Vector{SimpleWCSAxis}
    unit_label::String
    source_id::String
    metadata::Dict{Symbol,Any}
end

function CubeDataset(
    data::AbstractArray{T,3};
    axis_labels::AbstractVector{<:AbstractString} = ["axis1", "axis2", "axis3"],
    wcs::AbstractVector{SimpleWCSAxis} = _EMPTY_WCS,
    unit_label::AbstractString = "value",
    source_id::AbstractString = "cube",
    metadata::AbstractDict = Dict{Symbol,Any}(),
) where {T<:Real}
    length(axis_labels) == 3 || throw(ArgumentError(
        "CubeDataset needs exactly 3 axis_labels, got $(length(axis_labels))"))
    return CubeDataset{T}(
        data,
        String.(collect(axis_labels)),
        collect(SimpleWCSAxis, wcs),
        String(unit_label),
        String(source_id),
        Dict{Symbol,Any}(metadata),
    )
end

# ---------- MultiChannelDataset (NamedTuple/Dict of homogeneous datasets) ----------
struct MultiChannelDataset <: AbstractMANTADataset
    channels::Dict{Symbol,AbstractMANTADataset}
    kind::Symbol                  # :image | :cube | :vector
    default_channel::Symbol
    source_id::String
    metadata::Dict{Symbol,Any}
end

function MultiChannelDataset(
    channels::AbstractDict{Symbol,<:AbstractMANTADataset};
    default_channel::Union{Nothing,Symbol} = nothing,
    source_id::AbstractString = "multichannel",
    metadata::AbstractDict = Dict{Symbol,Any}(),
)
    isempty(channels) && throw(ArgumentError("MultiChannelDataset needs at least one channel"))
    keys_sorted = sort!(collect(keys(channels)))
    first_ds = channels[first(keys_sorted)]
    kind = if first_ds isa ImageDataset
        :image
    elseif first_ds isa CubeDataset
        :cube
    elseif first_ds isa VectorDataset
        :vector
    else
        throw(ArgumentError("MultiChannelDataset channel kind must be image/cube/vector, got $(typeof(first_ds))"))
    end
    ref_size = size(first_ds.data)
    for k in keys_sorted
        ds = channels[k]
        same_kind = (kind === :image && ds isa ImageDataset) ||
                    (kind === :cube  && ds isa CubeDataset)  ||
                    (kind === :vector && ds isa VectorDataset)
        same_kind || throw(ArgumentError(
            "MultiChannelDataset: channel $(k) has kind $(typeof(ds)) but expected $(kind)"))
        size(ds.data) == ref_size || throw(ArgumentError(
            "MultiChannelDataset: channel $(k) shape $(size(ds.data)) != reference $(ref_size)"))
    end
    chosen = default_channel === nothing ? first(keys_sorted) : default_channel
    haskey(channels, chosen) || throw(ArgumentError(
        "default_channel $(chosen) not in channels $(keys_sorted)"))
    return MultiChannelDataset(
        Dict{Symbol,AbstractMANTADataset}(channels),
        kind,
        chosen,
        String(source_id),
        Dict{Symbol,Any}(metadata),
    )
end

# ---------- HealpixMapDataset (1D HEALPix map) ----------
struct HealpixMapDataset <: AbstractMANTADataset
    map::Healpix.HealpixMap
    column::Int
    unit_label::String
    source_id::String
    metadata::Dict{Symbol,Any}
end

function HealpixMapDataset(
    map::Healpix.HealpixMap;
    column::Int = 1,
    unit_label::AbstractString = "value",
    source_id::AbstractString = "healpix_map",
    metadata::AbstractDict = Dict{Symbol,Any}(),
)
    return HealpixMapDataset(
        map,
        column,
        String(unit_label),
        String(source_id),
        Dict{Symbol,Any}(metadata),
    )
end

# Convenience alias
const HealpixDataset = HealpixMapDataset

# ---------- HealpixCubeDataset (npix x nv) ----------
struct HealpixCubeDataset{T<:Real} <: AbstractMANTADataset
    data::AbstractMatrix{T}        # already oriented (npix, nv)
    nside::Int
    nv::Int
    v0::Float64
    dv::Float64
    vunit::String
    unit_label::String
    source_id::String
    metadata::Dict{Symbol,Any}
end

function HealpixCubeDataset(
    data::AbstractMatrix{T};
    nside::Int,
    v0::Real = 0.0,
    dv::Real = 1.0,
    vunit::AbstractString = "km/s",
    unit_label::AbstractString = "value",
    source_id::AbstractString = "healpix_cube",
    metadata::AbstractDict = Dict{Symbol,Any}(),
) where {T<:Real}
    npix, nv = size(data)
    npix == 12 * nside^2 || throw(ArgumentError(
        "HealpixCubeDataset: npix=$(npix) != 12*nside^2=$(12*nside^2) for nside=$(nside)"))
    return HealpixCubeDataset{T}(
        data,
        nside,
        nv,
        Float64(v0),
        Float64(dv),
        String(vunit),
        String(unit_label),
        String(source_id),
        Dict{Symbol,Any}(metadata),
    )
end
