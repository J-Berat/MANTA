# path: src/loaders/HDF5Loader.jl
#
# Convert an HDF5 path + internal address ("file.h5:/group/dataset") into an
# `AbstractCartaDataset`. Metadata is read from HDF5 attributes attached to
# the addressed dataset (or its parent group).
#
# Attribute conventions (case-insensitive lookup):
#   - "units" / "bunit"           → unit_label (fallback "value")
#   - "AXIS{i}NAME"               → axis_labels[i]
#   - "CTYPE{i}" / "CRVAL{i}" /
#     "CRPIX{i}" / "CDELT{i}" /
#     "CUNIT{i}"                  → SimpleWCSAxis (linear WCS)
#   - "PIXTYPE" == "HEALPIX" or
#     "healpix" truthy            → HEALPix routing
#   - "v0", "dv", "vunit"         → spectral axis for HEALPix-PPV cubes

using HDF5

"""
    parse_hdf5_spec(s) -> (path, address) | nothing

Splits "file.h5:/group/ds" on the LAST `:`, but only when the prefix has an
HDF5 extension (.h5 / .hdf5 / .he5) and the suffix begins with `/`.
This rejects Windows drive letters ("C:/path/file.h5") and plain FITS paths.
"""
function parse_hdf5_spec(s::AbstractString)
    idx = findlast(==(':'), s)
    idx === nothing && return nothing
    path = s[1:idx-1]
    addr = s[idx+1:end]
    (isempty(addr) || !startswith(addr, "/")) && return nothing
    lowercase(splitext(path)[2]) ∈ (".h5", ".hdf5", ".he5") || return nothing
    return (String(path), String(addr))
end

# Read one HDF5 attribute, fall back to the default if missing or unreadable.
function _h5_attr_get(attrs, key, default)
    try
        haskey(attrs, key) || return default
        v = read(attrs[key])
        return v
    catch
        return default
    end
end

_h5_attr_str(attrs, key, default = "") = begin
    v = _h5_attr_get(attrs, key, nothing)
    v === nothing ? String(default) : try
        String(strip(string(v)))
    catch
        String(default)
    end
end

_h5_attr_float(attrs, key, default) = begin
    v = _h5_attr_get(attrs, key, nothing)
    v === nothing && return Float64(default)
    try
        return Float64(v)
    catch
        return Float64(default)
    end
end

# Case-insensitive attribute fetch (HDF5 attribute names are case-sensitive
# but we want to be friendly to common spellings).
function _h5_attr_first(attrs, keys, default = nothing)
    keys_present = collect(string.(HDF5.keys(attrs)))
    lower = Dict(lowercase(k) => k for k in keys_present)
    for cand in keys
        k = lowercase(string(cand))
        if haskey(lower, k)
            try
                return read(attrs[lower[k]])
            catch
                continue
            end
        end
    end
    return default
end

function _build_wcs_from_h5_attrs(attrs, ndim::Int)
    axes = SimpleWCSAxis[]
    any_present = false
    for dim in 1:ndim
        ctype = _h5_attr_str(attrs, "CTYPE$(dim)", "")
        cunit = _h5_attr_str(attrs, "CUNIT$(dim)", "")
        crval = _h5_attr_float(attrs, "CRVAL$(dim)", 0.0)
        crpix = _h5_attr_float(attrs, "CRPIX$(dim)", 1.0)
        cdelt = _h5_attr_float(attrs, "CDELT$(dim)", 1.0)
        present = !isempty(ctype) ||
                  haskey(attrs, "CRVAL$(dim)") ||
                  haskey(attrs, "CDELT$(dim)")
        any_present |= present
        push!(axes, SimpleWCSAxis(ctype, cunit, crval, crpix, cdelt, present))
    end
    return any_present ? axes : SimpleWCSAxis[]
end

function _axis_labels_from_h5_attrs(attrs, ndim::Int)
    out = String[]
    for dim in 1:ndim
        name = _h5_attr_str(attrs, "AXIS$(dim)NAME", "")
        push!(out, isempty(name) ? "axis$(dim)" : name)
    end
    return out
end

function _is_healpix_h5(attrs)
    pixtype = uppercase(_h5_attr_str(attrs, "PIXTYPE", ""))
    pixtype == "HEALPIX" && return true
    v = _h5_attr_get(attrs, "healpix", nothing)
    if v !== nothing
        try
            return Bool(v)
        catch
            return false
        end
    end
    return false
end

"""
    load_hdf5(path, address; column=1, v0=0.0, dv=1.0, vunit="km/s")

Open `path` (HDF5 file), navigate to `address` (e.g. `/group/dataset`),
read the dataset and its attributes, and build an `AbstractCartaDataset`.
"""
function load_hdf5(
    path::AbstractString,
    address::AbstractString;
    column::Int = 1,
    v0::Real = 0.0,
    dv::Real = 1.0,
    vunit::AbstractString = "km/s",
)
    isfile(path) || throw(ArgumentError("HDF5 file not found: $(abspath(path))"))

    data, attrs_dict = h5open(path, "r") do f
        if !haskey(f, address)
            throw(ArgumentError("HDF5 address $(address) not found in $(abspath(path))"))
        end
        obj = f[address]
        ds = if obj isa HDF5.Dataset
            obj
        elseif obj isa HDF5.Group
            # Look for a single dataset child or an attribute "default_dataset".
            default = try
                read_attribute(obj, "default_dataset")
            catch
                nothing
            end
            if default !== nothing && haskey(obj, String(default))
                obj[String(default)]
            else
                children = collect(HDF5.keys(obj))
                ds_children = filter(k -> obj[k] isa HDF5.Dataset, children)
                length(ds_children) == 1 ||
                    throw(ArgumentError(
                        "HDF5 group $(address) is ambiguous: contains $(length(ds_children)) datasets " *
                        "($(ds_children)). Specify the full address."))
                obj[ds_children[1]]
            end
        else
            throw(ArgumentError("HDF5 object at $(address) is neither a dataset nor a group"))
        end
        # Snapshot attributes into a plain Dict so we can drop the file handle.
        attrs = HDF5.attributes(ds)
        attrs_snap = Dict{String,Any}()
        for k in HDF5.keys(attrs)
            try
                attrs_snap[String(k)] = read(attrs[k])
            catch
                attrs_snap[String(k)] = ""
            end
        end
        (read(ds), attrs_snap)
    end

    # Re-wrap snapshot in a tiny shim so the helper functions above keep working.
    return _build_dataset_from_h5(path, address, data, attrs_dict;
        column = column, v0 = v0, dv = dv, vunit = vunit)
end

# Wrapper around the snapshot dict that exposes the same `haskey`/getindex
# surface as `HDF5.attributes`. Lets us reuse `_h5_attr_*` helpers.
struct _AttrSnap
    d::Dict{String,Any}
end
HDF5.keys(a::_AttrSnap) = keys(a.d)
Base.haskey(a::_AttrSnap, k) = haskey(a.d, String(k))
Base.getindex(a::_AttrSnap, k) = _AttrShim(a.d[String(k)])

struct _AttrShim
    v::Any
end
Base.read(s::_AttrShim) = s.v

function _build_dataset_from_h5(path, address, data, attrs_dict;
        column::Int, v0::Real, dv::Real, vunit::AbstractString)

    attrs = _AttrSnap(attrs_dict)
    is_healpix_flag = _is_healpix_h5(attrs)

    unit_label = let s = _h5_attr_str(attrs, "units", _h5_attr_str(attrs, "UNITS",
                       _h5_attr_str(attrs, "bunit", _h5_attr_str(attrs, "BUNIT", ""))))
        isempty(s) ? "value" : s
    end

    sid_addr = replace(String(address), "/" => "_")
    sid_addr = lstrip(sid_addr, '_')
    source_id = String(splitext(basename(path))[1]) * "_" * sid_addr
    base_meta = Dict{Symbol,Any}(:hdf5_path => abspath(String(path)),
                                 :hdf5_address => String(address),
                                 :hdf5_attrs => attrs_dict)

    nd = ndims(data)

    # HEALPix routing
    if is_healpix_flag
        if nd == 1
            nside_guess = valid_healpix_npix(length(data))
            nside_guess > 0 || throw(ArgumentError(
                "HDF5 dataset $(address) flagged HEALPix but length $(length(data)) is not 12·nside²."))
            m = Healpix.HealpixMap{Float64,Healpix.RingOrder,Vector{Float64}}(Float64.(vec(data)))
            return HealpixMapDataset(m;
                column = column, unit_label = unit_label,
                source_id = source_id, metadata = base_meta)
        elseif nd == 2
            s = size(data)
            ns1 = valid_healpix_npix(s[1])
            ns2 = valid_healpix_npix(s[2])
            v0_eff = _h5_attr_float(attrs, "v0", v0)
            dv_eff = _h5_attr_float(attrs, "dv", dv)
            vunit_eff = let u = _h5_attr_str(attrs, "vunit", String(vunit))
                isempty(u) ? String(vunit) : u
            end
            if ns1 > 0
                cube = Float32.(data)
                return HealpixCubeDataset(cube; nside = ns1,
                    v0 = v0_eff, dv = dv_eff, vunit = vunit_eff,
                    unit_label = unit_label,
                    source_id = source_id, metadata = base_meta)
            elseif ns2 > 0
                cube = Float32.(permutedims(data))
                return HealpixCubeDataset(cube; nside = ns2,
                    v0 = v0_eff, dv = dv_eff, vunit = vunit_eff,
                    unit_label = unit_label,
                    source_id = source_id, metadata = base_meta)
            else
                throw(ArgumentError(
                    "HDF5 dataset $(address) flagged HEALPix but neither dim of $(s) is 12·nside²."))
            end
        else
            throw(ArgumentError(
                "HDF5 dataset $(address) flagged HEALPix but ndims=$(nd) is unsupported."))
        end
    end

    # Generic routing by ndims
    if nd == 1
        return VectorDataset(vec(data);
            axis_label = first(_axis_labels_from_h5_attrs(attrs, 1)),
            unit_label = unit_label,
            source_id = source_id,
            metadata = base_meta)
    elseif nd == 2
        return ImageDataset(data;
            axis_labels = _axis_labels_from_h5_attrs(attrs, 2),
            wcs = _build_wcs_from_h5_attrs(attrs, 2),
            unit_label = unit_label,
            source_id = source_id,
            metadata = base_meta)
    elseif nd == 3
        return CubeDataset(data;
            axis_labels = _axis_labels_from_h5_attrs(attrs, 3),
            wcs = _build_wcs_from_h5_attrs(attrs, 3),
            unit_label = unit_label,
            source_id = source_id,
            metadata = base_meta)
    else
        throw(ArgumentError("HDF5 dataset $(address) has unsupported ndims=$(nd) (size=$(size(data)))"))
    end
end
