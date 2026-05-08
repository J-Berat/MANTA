# path: src/loaders/FITSLoader.jl
#
# Convert a FITS file path into an `AbstractCartaDataset`.
#
# Centralizes FITS reads previously inlined in `MANTA.manta(filepath)`
# and at the top of the HEALPix viewers. The dispatch order — HEALPix-PPV
# 2D cube first, then HEALPix 1D map, then 3D cube — and all error texts
# match the original behavior exactly. The `"integration: carta smoke and
# errors"` testset is the regression contract.

using FITSIO

# Public entry. `kwargs` mirror the `carta(filepath)` knobs that the loader
# itself needs (HEALPix column, spectral-axis defaults). Viewer-only kwargs
# (cmap, vmin, …) are not handled here.
function load_fits(
    filepath::AbstractString;
    column::Int = 1,
    v0::Real = 0.0,
    dv::Real = 1.0,
    vunit::AbstractString = "km/s",
)
    isfile(filepath) || throw(ArgumentError("FITS file not found: $(abspath(filepath))"))

    # Read primary HDU once. Tolerates an empty primary (HEALPix BinTable
    # files have no image in HDU 1).
    header = nothing
    raw = try
        FITS(filepath) do f
            header = try
                read_header(f[1])
            catch
                nothing
            end
            read(f[1])
        end
    catch
        nothing
    end

    # 1) HEALPix-PPV 2D cube: tested BEFORE is_healpix_fits because such cubes
    #    embed PIXTYPE=HEALPIX in their primary header.
    if raw !== nothing && ndims(raw) == 2
        s = size(raw)
        if valid_healpix_npix(s[1]) > 0 || valid_healpix_npix(s[2]) > 0
            return _load_healpix_cube_fits(filepath, raw, header, v0, dv, vunit)
        end
    end

    # 2) HEALPix 1D map (BinTable).
    if is_healpix_fits(filepath)
        return _load_healpix_map_fits(filepath; column = column)
    end

    # 3) 3D cube.
    raw === nothing && throw(ArgumentError("Failed to read primary HDU of $(abspath(filepath))."))
    if ndims(raw) != 3
        throw(ArgumentError(
            "Expected a 3D FITS cube, got ndims=$(ndims(raw)) and size=$(size(raw))."))
    end
    return _load_cube_fits(filepath, raw, header)
end

# ---- internal helpers (one per dataset kind) ----

function _load_cube_fits(filepath::AbstractString, raw, header)
    data = Float32.(raw)
    wcs = header === nothing ? SimpleWCSAxis[] : read_simple_wcs(header, 3)
    unit_label = data_unit_label(header; fallback = "value")
    fname_full = basename(filepath)
    fname = String(replace(fname_full, r"\.fits$" => ""))
    return CubeDataset(data;
        axis_labels = ["axis1", "axis2", "axis3"],
        wcs = wcs,
        unit_label = unit_label,
        source_id = fname,
        metadata = Dict{Symbol,Any}(:fits_header => header,
                                    :fits_path => abspath(String(filepath))),
    )
end

function _load_healpix_map_fits(filepath::AbstractString; column::Int = 1)
    m, hdr = read_healpix_map(filepath; column = column)
    fname = String(replace(basename(filepath), r"\.fits(\.gz)?$" => ""))
    unit_str = strip(String(get(hdr, "TUNIT$column", get(hdr, "BUNIT", ""))))
    unit_label = isempty(unit_str) ? "value" : String(unit_str)
    return HealpixMapDataset(m;
        column = column,
        unit_label = unit_label,
        source_id = fname,
        metadata = Dict{Symbol,Any}(:fits_header => hdr,
                                    :fits_path => abspath(String(filepath))),
    )
end

# Mirrors the prologue in `carta_healpix_cube`: detects which dim is the
# velocity axis, orients the cube to (npix, nv), and computes (v0, dv, vunit).
function _load_healpix_cube_fits(
    filepath::AbstractString, raw, header,
    v0::Real, dv::Real, vunit::AbstractString,
)
    ndims(raw) == 2 || throw(ArgumentError(
        "Expected 2D array (npix×nv), got ndims=$(ndims(raw))"))
    data_unit = data_unit_label(header; fallback = "value")

    s = size(raw)
    nside1 = valid_healpix_npix(s[1])
    nside2 = valid_healpix_npix(s[2])

    user_set_wcs = !(v0 == 0.0 && dv == 1.0)
    wcs_info = user_set_wcs ? nothing : detect_velocity_axis(filepath, 2)

    nside, npix, nv, vaxis, v0_eff, dv_eff, vunit_eff = if wcs_info !== nothing
        (vax, v0_h, dv_h, unit_h) = wcs_info
        hpix_dim = vax == 1 ? 2 : 1
        nside_h = valid_healpix_npix(s[hpix_dim])
        nside_h == 0 && throw(ArgumentError(
            "Header indique CTYPE$(vax) spectral mais NAXIS$(hpix_dim)=$(s[hpix_dim]) " *
            "n'est pas un npix HEALPix valide (12·nside²)."))
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

    no_wcs = (wcs_info === nothing) && !user_set_wcs
    if no_wcs
        vunit_eff = "channel"
        v0_eff = 1.0
        dv_eff = 1.0
    end
    @info "HEALPix PPV cube" path=abspath(filepath) nside npix nv vaxis v0=v0_eff dv=dv_eff unit=vunit_eff

    cube = vaxis === :last ? Float32.(raw) : Float32.(permutedims(raw))
    fname = String(replace(basename(filepath), r"\.fits(\.gz)?$" => ""))

    return HealpixCubeDataset(cube;
        nside = nside,
        v0 = v0_eff, dv = dv_eff, vunit = vunit_eff,
        unit_label = data_unit,
        source_id = fname,
        metadata = Dict{Symbol,Any}(:fits_header => header,
                                    :fits_path => abspath(String(filepath))),
    )
end
