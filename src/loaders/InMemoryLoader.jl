# path: src/loaders/InMemoryLoader.jl
#
# Convert in-memory inputs (`AbstractVector`, `AbstractMatrix`, 3D arrays,
# NamedTuple/Dict of arrays, `Healpix.HealpixMap`) into AbstractCartaDataset.

# Stable per-session id derived from concrete type + shape (used for the
# settings TOML filename when the user provides no source_id).
function _inmem_source_id(x)
    h = hash((typeof(x), size(x)))
    return "inmem_" * string(h; base = 16)
end

# Already-a-dataset → return as-is. Allows `carta(ds)` to work uniformly.
_dataset_from(ds::AbstractCartaDataset; _kw...) = ds

function _dataset_from(
    v::AbstractVector{T};
    axis_label::AbstractString = "index",
    wcs::Union{Nothing,SimpleWCSAxis} = nothing,
    unit_label::AbstractString = "value",
    source_id::Union{Nothing,AbstractString} = nothing,
    metadata::AbstractDict = Dict{Symbol,Any}(),
) where {T<:Real}
    sid = source_id === nothing ? _inmem_source_id(v) : String(source_id)
    return VectorDataset(v;
        axis_label = axis_label, wcs = wcs, unit_label = unit_label,
        source_id = sid, metadata = metadata)
end

function _dataset_from(
    M::AbstractMatrix{T};
    axis_labels::AbstractVector{<:AbstractString} = ["axis1", "axis2"],
    wcs::AbstractVector{SimpleWCSAxis} = SimpleWCSAxis[],
    unit_label::AbstractString = "value",
    source_id::Union{Nothing,AbstractString} = nothing,
    metadata::AbstractDict = Dict{Symbol,Any}(),
) where {T<:Real}
    sid = source_id === nothing ? _inmem_source_id(M) : String(source_id)
    return ImageDataset(M;
        axis_labels = axis_labels, wcs = wcs, unit_label = unit_label,
        source_id = sid, metadata = metadata)
end

function _dataset_from(
    A::AbstractArray{T,3};
    axis_labels::AbstractVector{<:AbstractString} = ["axis1", "axis2", "axis3"],
    wcs::AbstractVector{SimpleWCSAxis} = SimpleWCSAxis[],
    unit_label::AbstractString = "value",
    source_id::Union{Nothing,AbstractString} = nothing,
    metadata::AbstractDict = Dict{Symbol,Any}(),
) where {T<:Real}
    sid = source_id === nothing ? _inmem_source_id(A) : String(source_id)
    return CubeDataset(A;
        axis_labels = axis_labels, wcs = wcs, unit_label = unit_label,
        source_id = sid, metadata = metadata)
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
    isempty(nt) && throw(ArgumentError("carta: empty NamedTuple has no channels"))
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
    isempty(d) && throw(ArgumentError("carta: empty Dict has no channels"))
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
    inner = Dict{Symbol,AbstractCartaDataset}()
    for (k, v) in sym_dict
        if v isa AbstractCartaDataset
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
        # Hash of channel keys + shape of the first dataset
        first_ds = inner[first(sort!(collect(keys(inner))))]
        "inmem_mc_" * string(hash((sort!(collect(keys(inner))), size(first_ds.data))); base = 16)
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
    metadata::AbstractDict = Dict{Symbol,Any}(),
)
    sid = source_id === nothing ? "inmem_healpix_" * string(hash(size(m.pixels)); base = 16) : String(source_id)
    return HealpixMapDataset(m;
        column = column, unit_label = unit_label,
        source_id = sid, metadata = metadata)
end
