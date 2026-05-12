# path: src/loaders/InMemoryLoader.jl
#
# Convert in-memory inputs (`AbstractVector`, `AbstractMatrix`, 3D arrays,
# NamedTuple/Dict of arrays, `Healpix.HealpixMap`) into AbstractMANTADataset.

using SHA

"""
    stable_source_id(x) -> String

Deterministic, content-shape-aware identifier used as a fallback when the
caller does not provide one. Two calls with arrays of the same concrete type
and the same size will produce the same id, which is what the persistent
viewer-settings file expects.
"""
function stable_source_id(x)
    s = string(typeof(x), "_", size(x))
    return "inmem_" * bytes2hex(sha1(s))[1:12]
end

# Already-a-dataset → return as-is. Allows `manta(ds)` to work uniformly.
_dataset_from(ds::AbstractMANTADataset; _kw...) = ds

function _dataset_from(
    v::AbstractVector{T};
    axis_label::AbstractString = "index",
    wcs::Union{Nothing,SimpleWCSAxis} = nothing,
    unit_label::AbstractString = "value",
    source_id::Union{Nothing,AbstractString} = nothing,
    label::Union{Nothing,AbstractString} = nothing,
    metadata::AbstractDict = Dict{Symbol,Any}(),
) where {T<:Real}
    sid = source_id === nothing ? stable_source_id(v) : String(source_id)
    md = Dict{Symbol,Any}(metadata)
    label === nothing || (md[:label] = String(label))
    return VectorDataset(v;
        axis_label = axis_label, wcs = wcs, unit_label = unit_label,
        source_id = sid, metadata = md)
end

function _dataset_from(
    M::AbstractMatrix{T};
    axis_labels::AbstractVector{<:AbstractString} = ["axis1", "axis2"],
    wcs::AbstractVector{SimpleWCSAxis} = SimpleWCSAxis[],
    unit_label::AbstractString = "value",
    source_id::Union{Nothing,AbstractString} = nothing,
    label::Union{Nothing,AbstractString} = nothing,
    metadata::AbstractDict = Dict{Symbol,Any}(),
) where {T<:Real}
    sid = source_id === nothing ? stable_source_id(M) : String(source_id)
    md = Dict{Symbol,Any}(metadata)
    label === nothing || (md[:label] = String(label))
    return ImageDataset(M;
        axis_labels = axis_labels, wcs = wcs, unit_label = unit_label,
        source_id = sid, metadata = md)
end

function _dataset_from(
    A::AbstractArray{T,3};
    axis_labels::AbstractVector{<:AbstractString} = ["axis1", "axis2", "axis3"],
    wcs::AbstractVector{SimpleWCSAxis} = SimpleWCSAxis[],
    unit_label::AbstractString = "value",
    source_id::Union{Nothing,AbstractString} = nothing,
    label::Union{Nothing,AbstractString} = nothing,
    metadata::AbstractDict = Dict{Symbol,Any}(),
) where {T<:Real}
    sid = source_id === nothing ? stable_source_id(A) : String(source_id)
    md = Dict{Symbol,Any}(metadata)
    label === nothing || (md[:label] = String(label))
    return CubeDataset(A;
        axis_labels = axis_labels, wcs = wcs, unit_label = unit_label,
        source_id = sid, metadata = md)
end

# Catch-all for unsupported ndims so users get a clear message rather than
# a method-error trace.
function _dataset_from(A::AbstractArray; kwargs...)
    throw(ArgumentError(
        "MANTA: cannot wrap array with ndims=$(ndims(A)) and size=$(size(A)). " *
        "Supported: 1D vector, 2D image, 3D cube."))
end

# NamedTuple / Dict → MultiChannelDataset. Builds a per-channel inner dataset
# by recursing into _dataset_from(value).
function _dataset_from(
    nt::NamedTuple;
    source_id::Union{Nothing,AbstractString} = nothing,
    default_channel::Union{Nothing,Symbol} = nothing,
    metadata::AbstractDict = Dict{Symbol,Any}(),
    inner_kwargs...
)
    isempty(nt) && throw(ArgumentError("MANTA: empty NamedTuple has no channels"))
    return _build_multichannel(
        Dict{Symbol,Any}(k => v for (k, v) in pairs(nt));
        source_id = source_id, default_channel = default_channel,
        metadata = metadata, inner_kwargs...,
    )
end

function _dataset_from(
    d::AbstractDict;
    source_id::Union{Nothing,AbstractString} = nothing,
    default_channel::Union{Nothing,Symbol} = nothing,
    metadata::AbstractDict = Dict{Symbol,Any}(),
    inner_kwargs...
)
    isempty(d) && throw(ArgumentError("MANTA: empty Dict has no channels"))
    sym_dict = Dict{Symbol,Any}()
    for (k, v) in pairs(d)
        sym_dict[Symbol(k)] = v
    end
    return _build_multichannel(
        sym_dict;
        source_id = source_id, default_channel = default_channel,
        metadata = metadata, inner_kwargs...,
    )
end

function _build_multichannel(
    sym_dict::AbstractDict{Symbol,<:Any};
    source_id::Union{Nothing,AbstractString} = nothing,
    default_channel::Union{Nothing,Symbol} = nothing,
    metadata::AbstractDict = Dict{Symbol,Any}(),
    inner_kwargs...
)
    inner = Dict{Symbol,AbstractMANTADataset}()
    for (k, v) in sym_dict
        if v isa AbstractMANTADataset
            inner[k] = v
        else
            inner[k] = _dataset_from(v;
                source_id = "$(k)",
                inner_kwargs...)
        end
    end
    sid = if source_id !== nothing
        String(source_id)
    else
        keys_sorted = sort!(collect(keys(inner)))
        first_ds = inner[first(keys_sorted)]
        digest = bytes2hex(sha1(string(keys_sorted, "_", size(first_ds.data))))[1:12]
        "inmem_mc_" * digest
    end
    return MultiChannelDataset(inner;
        default_channel = default_channel,
        source_id = sid,
        metadata = metadata)
end

# HEALPix map wrapper: explicit only, never auto-detected from a Vector.
function _dataset_from(
    m::Healpix.HealpixMap;
    column::Int = 1,
    unit_label::AbstractString = "value",
    source_id::Union{Nothing,AbstractString} = nothing,
    label::Union{Nothing,AbstractString} = nothing,
    metadata::AbstractDict = Dict{Symbol,Any}(),
)
    sid = if source_id === nothing
        "inmem_healpix_" * bytes2hex(sha1(string(size(m.pixels))))[1:12]
    else
        String(source_id)
    end
    md = Dict{Symbol,Any}(metadata)
    label === nothing || (md[:label] = String(label))
    return HealpixMapDataset(m;
        column = column, unit_label = unit_label,
        source_id = sid, metadata = md)
end
