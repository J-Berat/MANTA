# path: src/datasets/LoadDataset.jl
#
# Central dispatch for turning a user input into an `AbstractMANTADataset`.
# This is the layer between the public `manta(x)` entry point and the
# concrete loaders (FITSLoader, HDF5Loader, InMemoryLoader).
#
# Adding a new input type is a matter of writing one more `load_dataset`
# method — viewers see only `AbstractMANTADataset` and never the raw input.

"""
    load_dataset(x; kwargs...) -> AbstractMANTADataset

Dispatch table:

| Input                                       | Loader / result                         |
|---------------------------------------------|-----------------------------------------|
| `AbstractMANTADataset`                      | returned as-is                          |
| `"file.fits"` / `".fit"` / `".fits.gz"`     | `load_fits(path; kwargs...)`            |
| `"file.h5"` / `".hdf5"` / `".he5"`          | `load_hdf5(path; kwargs...)`            |
| `"file.h5:/group/dataset"`                  | `load_hdf5(path, address; kwargs...)`   |
| `AbstractVector` / `AbstractMatrix` / 3D    | in-memory image / cube / vector dataset |
| `NamedTuple` / `AbstractDict`               | `MultiChannelDataset`                   |
| `Healpix.HealpixMap`                        | `HealpixMapDataset`                     |

Unsupported inputs raise an `ArgumentError` with a clear message.
"""
load_dataset(ds::AbstractMANTADataset; kwargs...) = ds

function load_dataset(path::AbstractString; kwargs...)
    spec = parse_path_spec(path)
    kind = first(spec)
    if kind === :fits
        return load_fits(spec[2]; kwargs...)
    elseif kind === :hdf5
        # spec is either (:hdf5, file) or (:hdf5, file, address).
        if length(spec) >= 3
            return load_hdf5(spec[2], spec[3]; kwargs...)
        else
            return load_hdf5(spec[2]; kwargs...)
        end
    else
        throw(ArgumentError(
            "MANTA: cannot recognise file kind from path $(path). " *
            "Supported extensions: .fits, .fit, .fits.gz, .h5, .hdf5, .he5, " *
            "or the syntax \"file.h5:/group/dataset\"."))
    end
end

load_dataset(x::AbstractArray; kwargs...) = _dataset_from(x; kwargs...)
load_dataset(x::NamedTuple; kwargs...) = _dataset_from(x; kwargs...)
load_dataset(x::AbstractDict; kwargs...) = _dataset_from(x; kwargs...)
load_dataset(x::Healpix.HealpixMap; kwargs...) = _dataset_from(x; kwargs...)
