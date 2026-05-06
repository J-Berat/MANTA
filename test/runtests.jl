# path: test/runtests.jl
using Test

# load the local module
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using CartaViewer

# deps used by the helpers
using Observables
using Makie
using LaTeXStrings
using ColorTypes
using FITSIO
using Statistics: mean
using Healpix

@testset "helpers: scaling" begin
    A = Float32.([1, 10, 100, 0, -1])
    lin = CartaViewer.apply_scale(A, :lin)
    log10v = CartaViewer.apply_scale(A, :log10)
    lnv = CartaViewer.apply_scale(A, :ln)

    @test eltype(lin) == Float32
    @test eltype(log10v) == Float32
    @test eltype(lnv) == Float32

    @test lin[1:3] == A[1:3]
    @test isapprox(log10v[1], 0f0; atol=1e-6)
    @test isapprox(log10v[2], 1f0; atol=1e-6)
    @test isfinite(lnv[1])
    @test !isfinite(lnv[4]) && !isfinite(lnv[5])

    mn, mx = CartaViewer.clamped_extrema(Float32.([1, 2, 3]))
    @test mn == 1f0 && mx == 3f0

    mn2, mx2 = CartaViewer.clamped_extrema(Float32.([5, 5, 5]))
    @test mn2 < 5.0f0 && mx2 > 5.0f0

    mn3, mx3 = CartaViewer.clamped_extrema(Float32.([NaN32, NaN32]))
    @test mn3 == 0f0 && mx3 == 1f0

    mn4, mx4 = CartaViewer.clamped_extrema(Float32.([]))
    @test mn4 == 0f0 && mx4 == 1f0

    p1, p99 = CartaViewer.percentile_clims(Float32.(1:100), 1, 99)
    @test p1 >= 1f0 && p99 <= 100f0 && p1 < p99

    hx, hy = CartaViewer.histogram_counts(Float32.(1:10); bins = 5)
    @test length(hx) == 5
    @test length(hy) == 5
    @test sum(hy) == 10f0

    levels = CartaViewer.automatic_contour_levels(Float32.(1:100); n = 6)
    @test length(levels) == 6
    @test issorted(levels)
end

@testset "helpers: mapping" begin
    # bijection uv <-> ijk depending on the axis
    for axis in 1:3
        i, j, k = 3, 2, 1
        u, v = CartaViewer.ijk_to_uv(i, j, k, axis)
        ii, jj, kk = CartaViewer.uv_to_ijk(u, v, axis, axis == 1 ? i : axis == 2 ? j : k)
        @test (ii, jj, kk) == (i, j, k)
    end

    # get_slice dims and type
    data = Array{Float32}(undef, 7, 5, 4)
    fill!(data, 1f0)
    s1 = CartaViewer.get_slice(data, 1, 2)
    s2 = CartaViewer.get_slice(data, 2, 3)
    s3 = CartaViewer.get_slice(data, 3, 1)
    @test size(s1) == (size(data, 2), size(data, 3))
    @test size(s2) == (size(data, 1), size(data, 3))
    @test size(s3) == (size(data, 1), size(data, 2))
    @test eltype(s1) == Float32 && eltype(s2) == Float32 && eltype(s3) == Float32

    box_uv = CartaViewer.region_uv_indices(10, 10, 2, 3, 4, 5, :box)
    @test (3, 2) in box_uv
    @test (5, 4) in box_uv

    circle_uv = CartaViewer.region_uv_indices(10, 10, 5, 5, 7, 5, :circle)
    @test (5, 5) in circle_uv
    @test (5, 7) in circle_uv
    @test (1, 1) ∉ circle_uv

    cube = reshape(Float32.(1:24), 2, 3, 4)
    spec = CartaViewer.mean_region_spectrum(cube, 3, [(1, 1), (2, 1)])
    @test length(spec) == 4
    @test spec[1] == mean(Float32[cube[1, 1, 1], cube[2, 1, 1]])
end

@testset "healpix: mollweide graticule geometry" begin
    for lon in (-120, -30, 0, 45, 150), lat in (-60, -15, 0, 35, 70)
        p = CartaViewer.mollweide_lonlat_to_xy(lon, lat)
        @test p !== nothing
        ll = CartaViewer.mollweide_xy_to_lonlat(p[1], p[2])
        @test ll !== nothing
        lon2, lat2 = ll
        @test isapprox(lon2, lon; atol=1e-4)
        @test isapprox(lat2, lat; atol=1e-4)
    end
end

@testset "healpix: projected regions" begin
    grid = Int32[
        0 1 1 2
        3 3 4 0
        5 6 6 7
    ]
    box_ips = CartaViewer.projected_region_ipix(grid, -2, -1, 2, 1, :box)
    @test box_ips == [1, 2, 3, 4, 5, 6, 7]

    circle_ips = CartaViewer.projected_region_ipix(grid, 0, 0, 1, 0, :circle)
    @test all(>(0), circle_ips)
    @test issorted(circle_ips)

    vals = Float32[10, 20, NaN32, 40]
    @test CartaViewer.healpix_region_mean(vals, [1, 2, 3]) == 15f0

    cube = Float32[
        1 10 100
        3 30 300
        NaN 40 400
    ]
    spec = CartaViewer.healpix_region_mean_spectrum(cube, [1, 2, 3], 3)
    @test spec == Float32[2, 80 / 3, 800 / 3]
end

@testset "helpers: latex" begin
    s = CartaViewer.make_info_tex(1, 2, 3, 4, 5, 6f0)
    t1 = CartaViewer.make_slice_title("fname", 3, 10)
    t2 = CartaViewer.make_spec_title(1, 2, 3)

    @test s isa LaTeXString
    @test t1 isa LaTeXString
    @test t2 isa LaTeXString

    # No LaTeX line breaks: forbid "\\ " and "\\\n"
    raw_s  = String(s)
    raw_t1 = String(t1)
    raw_t2 = String(t2)
    for raw in (raw_s, raw_t1, raw_t2)
        @test !occursin("\\\\ ", raw)
        @test !occursin("\\\\\\n", raw)
    end

    # Expect inline LaTeX (e.g., \\, for thin space)
    @test occursin("\\,", raw_s) || occursin("\\,", raw_t1) || occursin("\\,", raw_t2)
    @test occursin("intensity", lowercase(raw_s))
end

@testset "helpers: io" begin
    # to_cmap
    cm = CartaViewer.to_cmap(:viridis)
    @test length(cm) > 0
    @test cm[1] isa ColorTypes.Colorant

    # get_box_str via mock (no Makie Textbox available)
    struct MockTB
        stored_string::Observable{String}
    end
    tb = MockTB(Observable("   hello world   "))
    @test CartaViewer.get_box_str(tb) == "hello world"
    
    struct MockDisplayTB
        displayed_string::Observable{String}
    end
    tb2 = MockDisplayTB(Observable("   fallback value   "))
    @test CartaViewer.get_box_str(tb2) == "fallback value"
end

@testset "helpers: ui" begin
    # explicit override
    @test CartaViewer._pick_fig_size((111, 222)) == (111, 222)
    # default when no explicit size is provided
    @test CartaViewer._pick_fig_size(nothing) == (1800, 1000)
end

@testset "helpers: validation" begin
    ok, use_manual, clims, msg = CartaViewer.parse_manual_clims("1.5", "2.5")
    @test ok && use_manual
    @test clims == (1.5f0, 2.5f0)
    @test !isempty(msg)

    ok2, use_manual2, _, _ = CartaViewer.parse_manual_clims("", "")
    @test ok2 && !use_manual2

    ok3, use_manual3, _, _ = CartaViewer.parse_manual_clims("9", "2")
    @test ok3 && use_manual3

    ok4, _, _, _ = CartaViewer.parse_manual_clims("a", "2")
    @test !ok4

    gok, frames, fps, _ = CartaViewer.parse_gif_request("1", "5", "2", "10", 10)
    @test gok
    @test frames == [1, 3, 5]
    @test fps == 10

    gok2, frames2, _, _ = CartaViewer.parse_gif_request("5", "1", "2", "12", 10; pingpong = true)
    @test gok2
    @test frames2 == [1, 3, 5, 3]

    gok3, _, _, _ = CartaViewer.parse_gif_request("1", "5", "0", "12", 10)
    @test !gok3

    cok, cmanual, clevels, _ = CartaViewer.parse_contour_levels("1, 2  3")
    @test cok && cmanual
    @test clevels == Float32[1, 2, 3]

    cok2, cmanual2, _, _ = CartaViewer.parse_contour_levels("")
    @test cok2 && !cmanual2

    cok3, _, _, _ = CartaViewer.parse_contour_levels("1, nope")
    @test !cok3

    sok, smanual, slevels, scolors, _ = CartaViewer.parse_contour_specs("3:blue, 1:red, 2:#00ffaa")
    @test sok && smanual
    @test slevels == Float32[1, 2, 3]
    @test scolors == ["red", "#00ffaa", "blue"]
    @test CartaViewer.format_contour_specs(slevels, scolors) == "1:red, 2:#00ffaa, 3:blue"

    color_values = CartaViewer.contour_color_values(scolors, length(slevels), RGBAf(0, 0, 0, 1))
    @test length(color_values) == 3

    bad_color, _, _, _, _ = CartaViewer.parse_contour_specs("1:not_a_color")
    @test !bad_color
end

@testset "helpers: simple wcs" begin
    header = Dict{String,Any}(
        "CTYPE1" => "RA---TAN",
        "CUNIT1" => "deg",
        "CRVAL1" => 120.0,
        "CRPIX1" => 1.0,
        "CDELT1" => -0.5,
        "CTYPE2" => "DEC--TAN",
        "CUNIT2" => "deg",
        "CRVAL2" => -30.0,
        "CRPIX2" => 2.0,
        "CDELT2" => 0.25,
    )
    wcs = CartaViewer.read_simple_wcs(header, 3)
    @test CartaViewer.has_wcs(wcs, 1)
    @test CartaViewer.has_wcs(wcs, 2)
    @test !CartaViewer.has_wcs(wcs, 3)
    @test CartaViewer.world_coord(wcs, 1, 3) == 119.0
    @test occursin("RA", String(CartaViewer.wcs_axis_label(wcs, 1)))
    @test occursin("RA---TAN", CartaViewer.format_world_coord(wcs, 1, 1))
    @test CartaViewer.data_unit_label(Dict{String,Any}("BUNIT" => "K")) == "K"
    @test CartaViewer.data_unit_label(Dict{String,Any}("BUNIT" => "   ")) == "value"
    @test CartaViewer.data_unit_label(nothing) == "value"
end

@testset "helpers: settings io" begin
    mktempdir() do tmp
        settings_path = joinpath(tmp, "viewer_settings.toml")
        payload = Dict{String,Any}(
            "axis" => 2,
            "img_scale" => "log10",
            "colormap" => "viridis",
            "use_manual_clims" => true,
            "clim_min" => 0.1,
            "clim_max" => 42.0,
        )
        CartaViewer.save_viewer_settings(settings_path, payload)
        @test isfile(settings_path)

        restored = CartaViewer.load_viewer_settings(settings_path)
        @test restored["axis"] == 2
        @test restored["img_scale"] == "log10"
        @test restored["use_manual_clims"] == true
    end
end

@testset "integration: carta smoke and errors" begin
    mktempdir() do tmp
        cube_path = joinpath(tmp, "cube3d.fits")
        cube = reshape(Float32.(1:60), 3, 4, 5)
        FITS(cube_path, "w") do f
            write(f, cube)
        end

        fig = CartaViewer.carta(
            cube_path;
            activate_gl = false,
            display_fig = false,
            save_dir = tmp,
            settings_path = joinpath(tmp, "state.toml"),
            figsize = (800, 500),
        )
        @test fig isa Makie.Figure
        CartaViewer.forget!(fig)

        hpix_ppv_path = joinpath(tmp, "healpix_ppv.fits")
        hpix_ppv = reshape(Float32.(1:48), 12, 4)
        FITS(hpix_ppv_path, "w") do f
            write(f, hpix_ppv)
        end
        fig_hpix = CartaViewer.carta_healpix_cube(
            hpix_ppv_path;
            activate_gl = false,
            display_fig = false,
            save_dir = tmp,
            nx = 100,
            ny = 50,
            figsize = (800, 560),
        )
        @test fig_hpix isa Makie.Figure
        CartaViewer.forget!(fig_hpix)

        hpix_map_path = joinpath(tmp, "healpix_map.fits")
        hpix_map = HealpixMap{Float64,RingOrder,Vector{Float64}}(collect(1.0:12.0))
        Healpix.saveToFITS(hpix_map, hpix_map_path; unit = "K")
        fig_map = CartaViewer.carta_healpix(
            hpix_map_path;
            activate_gl = false,
            display_fig = false,
            save_dir = tmp,
            nx = 100,
            ny = 50,
            figsize = (800, 560),
        )
        @test fig_map isa Makie.Figure
        CartaViewer.forget!(fig_map)

        missing = joinpath(tmp, "does_not_exist.fits")
        @test_throws ArgumentError CartaViewer.carta(missing; activate_gl = false, display_fig = false)

        bad_path = joinpath(tmp, "cube2d.fits")
        FITS(bad_path, "w") do f
            write(f, reshape(Float32.(1:12), 3, 4))
        end
        err = try
            CartaViewer.carta(bad_path; activate_gl = false, display_fig = false)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("Expected a 3D FITS cube", sprint(showerror, err))
    end
end
