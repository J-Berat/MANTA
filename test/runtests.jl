# path: test/runtests.jl
using Test

# load the local module
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MANTA

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
    lin = MANTA.apply_scale(A, :lin)
    log10v = MANTA.apply_scale(A, :log10)
    lnv = MANTA.apply_scale(A, :ln)

    @test eltype(lin) == Float32
    @test eltype(log10v) == Float32
    @test eltype(lnv) == Float32

    @test lin[1:3] == A[1:3]
    @test isapprox(log10v[1], 0f0; atol=1e-6)
    @test isapprox(log10v[2], 1f0; atol=1e-6)
    @test isfinite(lnv[1])
    @test !isfinite(lnv[4]) && !isfinite(lnv[5])

    mn, mx = MANTA.clamped_extrema(Float32.([1, 2, 3]))
    @test mn == 1f0 && mx == 3f0

    mn2, mx2 = MANTA.clamped_extrema(Float32.([5, 5, 5]))
    @test mn2 < 5.0f0 && mx2 > 5.0f0

    mn3, mx3 = MANTA.clamped_extrema(Float32.([NaN32, NaN32]))
    @test mn3 == 0f0 && mx3 == 1f0

    mn4, mx4 = MANTA.clamped_extrema(Float32.([]))
    @test mn4 == 0f0 && mx4 == 1f0

    p1, p99 = MANTA.percentile_clims(Float32.(1:100), 1, 99)
    @test p1 >= 1f0 && p99 <= 100f0 && p1 < p99

    hx, hy = MANTA.histogram_counts(Float32.(1:10); bins = 5)
    @test length(hx) == 5
    @test length(hy) == 5
    @test sum(hy) == 10f0

    smoothed = MANTA.nan_gaussian_filter(Float32[NaN 1 1; NaN 1 1; NaN NaN NaN], 1.0)
    @test size(smoothed) == (3, 3)
    @test isfinite(smoothed[2, 2])
    @test isnan(MANTA.nan_gaussian_filter(fill(NaN32, 3, 3), 1.0)[2, 2])

    levels = MANTA.automatic_contour_levels(Float32.(1:100); n = 6)
    @test length(levels) == 6
    @test issorted(levels)
end

@testset "helpers: mapping" begin
    # bijection uv <-> ijk depending on the axis
    for axis in 1:3
        i, j, k = 3, 2, 1
        u, v = MANTA.ijk_to_uv(i, j, k, axis)
        ii, jj, kk = MANTA.uv_to_ijk(u, v, axis, axis == 1 ? i : axis == 2 ? j : k)
        @test (ii, jj, kk) == (i, j, k)
    end

    # get_slice dims and type
    data = Array{Float32}(undef, 7, 5, 4)
    fill!(data, 1f0)
    s1 = MANTA.get_slice(data, 1, 2)
    s2 = MANTA.get_slice(data, 2, 3)
    s3 = MANTA.get_slice(data, 3, 1)
    @test size(s1) == (size(data, 2), size(data, 3))
    @test size(s2) == (size(data, 1), size(data, 3))
    @test size(s3) == (size(data, 1), size(data, 2))
    @test eltype(s1) == Float32 && eltype(s2) == Float32 && eltype(s3) == Float32

    box_uv = MANTA.region_uv_indices(10, 10, 2, 3, 4, 5, :box)
    @test (3, 2) in box_uv
    @test (5, 4) in box_uv

    circle_uv = MANTA.region_uv_indices(10, 10, 5, 5, 7, 5, :circle)
    @test (5, 5) in circle_uv
    @test (5, 7) in circle_uv
    @test (1, 1) ∉ circle_uv

    cube = reshape(Float32.(1:24), 2, 3, 4)
    spec = MANTA.mean_region_spectrum(cube, 3, [(1, 1), (2, 1)])
    @test length(spec) == 4
    @test spec[1] == mean(Float32[cube[1, 1, 1], cube[2, 1, 1]])
end

@testset "helpers: products" begin
    @test MANTA.moments(Float32[-1, 0, 2, 3]; x = Float32[10, 20, 30, 40]) == (5.0, 36.0, sqrt(24.0))
    @test all(isnan, MANTA.moments(Float32[-1, 0]; x = Float32[1, 2]))

    a = Float32[2 4; 6 8]
    b = Float32[1 2; 0 4]
    @test MANTA.dual_view_product(a, b, :A) == a
    @test MANTA.dual_view_product(a, b, :B) == b
    @test MANTA.dual_view_product(a, b, :diff) == Float32[1 2; 6 4]
    ratio = MANTA.dual_view_product(a, b, :ratio)
    @test ratio[1, 1] == 2f0
    @test isnan(ratio[2, 1])
    resid = MANTA.dual_view_product(a, b, :residuals)
    @test isapprox(mean(vec(resid)), 0f0; atol = 1f-6)

    cube = Array{Float32}(undef, 2, 2, 3)
    cube[:, :, 1] .= 1f0
    cube[:, :, 2] .= 2f0
    cube[:, :, 3] .= 3f0
    m0 = MANTA.moment_map(cube, 3, 0; coords = Float32[10, 20, 30])
    m1 = MANTA.moment_map(cube, 3, 1; coords = Float32[10, 20, 30])
    M0, M1, M2 = MANTA.moments_map(cube, Float32[10, 20, 30])
    @test all(==(6f0), m0)
    @test M0 == m0
    @test M1 == m1
    @test all(isfinite, M2)
    @test all(isapprox.(m1, Ref(Float32((10 + 40 + 90) / 6)); atol = 1f-5))
    mv0, mv1, mv2 = MANTA.moment_vectors(Float32[0 2 3; -1 0 4], Float32[1, 2, 3])
    @test mv0 == Float32[5, 4]
    @test isfinite(mv1[1]) && isfinite(mv2[1])
    @test MANTA.filtered_cube_by_slice(cube, 3, 0) == cube
end

@testset "healpix: mollweide graticule geometry" begin
    for lon in (-120, -30, 0, 45, 150), lat in (-60, -15, 0, 35, 70)
        p = MANTA.mollweide_lonlat_to_xy(lon, lat)
        @test p !== nothing
        ll = MANTA.mollweide_xy_to_lonlat(p[1], p[2])
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
    box_ips = MANTA.projected_region_ipix(grid, -2, -1, 2, 1, :box)
    @test box_ips == [1, 2, 3, 4, 5, 6, 7]

    circle_ips = MANTA.projected_region_ipix(grid, 0, 0, 1, 0, :circle)
    @test all(>(0), circle_ips)
    @test issorted(circle_ips)

    vals = Float32[10, 20, NaN32, 40]
    @test MANTA.healpix_region_mean(vals, [1, 2, 3]) == 15f0

    cube = Float32[
        1 10 100
        3 30 300
        NaN 40 400
    ]
    spec = MANTA.healpix_region_mean_spectrum(cube, [1, 2, 3], 3)
    @test spec == Float32[2, 80 / 3, 800 / 3]
end

@testset "helpers: latex" begin
    s = MANTA.make_info_tex(1, 2, 3, 4, 5, 6f0)
    t1 = MANTA.make_slice_title("fname", 3, 10)
    t2 = MANTA.make_spec_title(1, 2, 3)

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
    cm = MANTA.to_cmap(:viridis)
    @test length(cm) > 0
    @test cm[1] isa ColorTypes.Colorant

    # get_box_str via mock (no Makie Textbox available)
    struct MockTB
        stored_string::Observable{String}
    end
    tb = MockTB(Observable("   hello world   "))
    @test MANTA.get_box_str(tb) == "hello world"
    
    struct MockDisplayTB
        displayed_string::Observable{String}
    end
    tb2 = MockDisplayTB(Observable("   fallback value   "))
    @test MANTA.get_box_str(tb2) == "fallback value"
end

@testset "RGB helpers and direct viewers" begin
    r = Float32[-1 0; 1 2]
    g = Float32[0 1; 2 3]
    b = Float32[3 2; 1 0]
    rgb = MANTA.rgb_image(r, g, b)
    @test size(rgb) == (2, 2)
    @test eltype(rgb) <: ColorTypes.Colorant

    stack_last = zeros(Float32, 2, 3, 3)
    stack_last[:, :, 1] .= 1
    img_last = MANTA.as_rgb_image(stack_last)
    @test size(img_last) == (2, 3)

    stack_first = zeros(Float32, 3, 2, 3)
    stack_first[2, :, :] .= 1
    img_first = MANTA.as_rgb_image(stack_first)
    @test size(img_first) == (2, 3)

    hpix_rgb = [RGBf(i / 12, 0.25, 1 - i / 12) for i in 1:12]
    @test length(MANTA.as_rgb_pixels(hpix_rgb)) == 12
    @test length(MANTA.as_rgb_pixels(Float32.(reshape(1:36, 12, 3)) ./ 36)) == 12

    fig_rgb = MANTA.manta(rgb; activate_gl=false, display_fig=false, figsize=(500, 400))
    @test fig_rgb isa Makie.Figure
    MANTA.forget!(fig_rgb)

    fig_panels = MANTA.manta_panels(rgb, r; activate_gl=false, display_fig=false, figsize=(700, 400))
    @test fig_panels isa Makie.Figure
    MANTA.forget!(fig_panels)

    fig_hpix_rgb = MANTA.manta_healpix(hpix_rgb; activate_gl=false, display_fig=false, nx=60, ny=30, figsize=(500, 320))
    @test fig_hpix_rgb isa Makie.Figure
    MANTA.forget!(fig_hpix_rgb)

    fig_hpix_panels = MANTA.manta_healpix_panels(hpix_rgb, Float32.(1:12); activate_gl=false, display_fig=false, nx=60, ny=30, figsize=(700, 320))
    @test fig_hpix_panels isa Makie.Figure
    MANTA.forget!(fig_hpix_panels)
end

@testset "helpers: ui" begin
    # explicit override
    @test MANTA._pick_fig_size((111, 222)) == (111, 222)
    # default when no explicit size is provided
    @test MANTA._pick_fig_size(nothing) == (1800, 1000)
end

@testset "helpers: validation" begin
    ok, use_manual, clims, msg = MANTA.parse_manual_clims("1.5", "2.5")
    @test ok && use_manual
    @test clims == (1.5f0, 2.5f0)
    @test !isempty(msg)

    ok2, use_manual2, _, _ = MANTA.parse_manual_clims("", "")
    @test ok2 && !use_manual2

    ok3, use_manual3, _, _ = MANTA.parse_manual_clims("9", "2")
    @test ok3 && use_manual3

    ok4, _, _, _ = MANTA.parse_manual_clims("a", "2")
    @test !ok4

    gok, frames, fps, _ = MANTA.parse_gif_request("1", "5", "2", "10", 10)
    @test gok
    @test frames == [1, 3, 5]
    @test fps == 10

    gok2, frames2, _, _ = MANTA.parse_gif_request("5", "1", "2", "12", 10; pingpong = true)
    @test gok2
    @test frames2 == [1, 3, 5, 3]

    gok3, _, _, _ = MANTA.parse_gif_request("1", "5", "0", "12", 10)
    @test !gok3

    cok, cmanual, clevels, _ = MANTA.parse_contour_levels("1, 2  3")
    @test cok && cmanual
    @test clevels == Float32[1, 2, 3]

    cok2, cmanual2, _, _ = MANTA.parse_contour_levels("")
    @test cok2 && !cmanual2

    cok3, _, _, _ = MANTA.parse_contour_levels("1, nope")
    @test !cok3

    sok, smanual, slevels, scolors, _ = MANTA.parse_contour_specs("3:blue, 1:red, 2:#00ffaa")
    @test sok && smanual
    @test slevels == Float32[1, 2, 3]
    @test scolors == ["red", "#00ffaa", "blue"]
    @test MANTA.format_contour_specs(slevels, scolors) == "1:red, 2:#00ffaa, 3:blue"

    color_values = MANTA.contour_color_values(scolors, length(slevels), RGBAf(0, 0, 0, 1))
    @test length(color_values) == 3

    bad_color, _, _, _, _ = MANTA.parse_contour_specs("1:not_a_color")
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
    wcs = MANTA.read_simple_wcs(header, 3)
    @test MANTA.has_wcs(wcs, 1)
    @test MANTA.has_wcs(wcs, 2)
    @test !MANTA.has_wcs(wcs, 3)
    @test MANTA.world_coord(wcs, 1, 3) == 119.0
    @test occursin("RA", String(MANTA.wcs_axis_label(wcs, 1)))
    @test occursin("RA---TAN", MANTA.format_world_coord(wcs, 1, 1))
    @test MANTA.data_unit_label(Dict{String,Any}("BUNIT" => "K")) == "K"
    @test MANTA.data_unit_label(Dict{String,Any}("BUNIT" => "   ")) == "value"
    @test MANTA.data_unit_label(nothing) == "value"
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
        MANTA.save_viewer_settings(settings_path, payload)
        @test isfile(settings_path)

        restored = MANTA.load_viewer_settings(settings_path)
        @test restored["axis"] == 2
        @test restored["img_scale"] == "log10"
        @test restored["use_manual_clims"] == true
    end
end

@testset "integration: manta smoke and errors" begin
    mktempdir() do tmp
        cube_path = joinpath(tmp, "cube3d.fits")
        cube = reshape(Float32.(1:60), 3, 4, 5)
        FITS(cube_path, "w") do f
            write(f, cube)
        end

        fig = MANTA.manta(
            cube_path;
            activate_gl = false,
            display_fig = false,
            save_dir = tmp,
            settings_path = joinpath(tmp, "state.toml"),
            figsize = (800, 500),
        )
        @test fig isa Makie.Figure
        MANTA.forget!(fig)

        rgb_path = joinpath(tmp, "rgb_stack.fits")
        rgb_stack = zeros(Float32, 2, 3, 3)
        rgb_stack[:, :, 1] .= 1
        FITS(rgb_path, "w") do f
            write(f, rgb_stack)
        end
        fig_rgb_path = MANTA.manta(
            rgb_path;
            rgb = true,
            activate_gl = false,
            display_fig = false,
            figsize = (500, 360),
        )
        @test fig_rgb_path isa Makie.Figure
        MANTA.forget!(fig_rgb_path)

        hpix_ppv_path = joinpath(tmp, "healpix_ppv.fits")
        hpix_ppv = reshape(Float32.(1:48), 12, 4)
        FITS(hpix_ppv_path, "w") do f
            write(f, hpix_ppv)
        end
        fig_hpix = MANTA.manta_healpix_cube(
            hpix_ppv_path;
            activate_gl = false,
            display_fig = false,
            save_dir = tmp,
            nx = 100,
            ny = 50,
            figsize = (800, 560),
        )
        @test fig_hpix isa Makie.Figure
        MANTA.forget!(fig_hpix)

        hpix_map_path = joinpath(tmp, "healpix_map.fits")
        hpix_map = HealpixMap{Float64,RingOrder,Vector{Float64}}(collect(1.0:12.0))
        Healpix.saveToFITS(hpix_map, hpix_map_path; unit = "K")
        fig_map = MANTA.manta_healpix(
            hpix_map_path;
            activate_gl = false,
            display_fig = false,
            save_dir = tmp,
            nx = 100,
            ny = 50,
            figsize = (800, 560),
        )
        @test fig_map isa Makie.Figure
        MANTA.forget!(fig_map)

        missing = joinpath(tmp, "does_not_exist.fits")
        @test_throws ArgumentError MANTA.manta(missing; activate_gl = false, display_fig = false)

        image2d_path = joinpath(tmp, "image2d.fits")
        FITS(image2d_path, "w") do f
            write(f, reshape(Float32.(1:12), 3, 4))
        end
        fig_2d = MANTA.manta(
            image2d_path;
            activate_gl = false,
            display_fig = false,
            save_dir = tmp,
            figsize = (700, 450),
        )
        @test fig_2d isa Makie.Figure
        MANTA.forget!(fig_2d)

        fig_2d_direct = MANTA.manta(
            reshape(Float32.(1:12), 3, 4);
            activate_gl = false,
            display_fig = false,
            figsize = (700, 450),
        )
        @test fig_2d_direct isa Makie.Figure
        MANTA.forget!(fig_2d_direct)

        bad_path = joinpath(tmp, "cube4d.fits")
        FITS(bad_path, "w") do f
            write(f, reshape(Float32.(1:24), 2, 3, 2, 2))
        end
        err = try
            MANTA.manta(bad_path; activate_gl = false, display_fig = false)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("Expected a 3D FITS cube", sprint(showerror, err))
    end
end

@testset "helpers: power spectrum" begin
    # --- Kronecker delta: flat raw |F|² (no window, no demean, no padding).
    @testset "delta is flat" begin
        N = 16
        A = zeros(Float64, N, N)
        A[N ÷ 2 + 1, N ÷ 2 + 1] = 1.0
        res = MANTA.power_spectrum_2d(A; window = :none, demean = false, pad_pow2 = false)
        # All bins identical for a delta input.
        @test maximum(res.P2d) ≈ minimum(res.P2d) atol = 1e-9
        # MASTER-light divisor is ⟨1²⟩ = 1 here, so |F|² should be 1 everywhere.
        @test all(p -> isapprox(p, 1.0; atol = 1e-9), res.P2d)
        @test res.f_sky == 1.0
        @test res.window === :none
        @test !res.padded
    end

    # --- Pure cosine peaks at the expected radial bin.
    @testset "sinusoid peaks at expected k" begin
        N = 64
        m = 5
        A = [cos(2π * m * (j - 1) / N) for i in 1:N, j in 1:N]
        res = MANTA.power_spectrum_2d(A; window = :none, demean = false, pad_pow2 = false)
        radii, prof = MANTA.power_spectrum_1d_radial(res.P2d)
        # Peak should be at the bin matching the spatial frequency m.
        @test argmax(prof) - 1 == m
        # Other bins (excluding immediate neighbours) are much smaller.
        peak_val = prof[m + 1]
        for b in 1:length(prof)
            (b == m || b == m + 1 || b == m + 2) && continue
            @test prof[b] < peak_val * 1e-3
        end
    end

    # Deterministic pseudo-random surface (LCG-shaped sums of trig modes) so the
    # test suite does not need Random/MersenneTwister as a stdlib dependency.
    deterministic_field(N::Int, M::Int) = Float64[
        sin(0.123 * i + 0.371 * j) +
        0.5 * cos(0.7 * i - 0.3 * j) +
        0.25 * sin(0.05 * i * j)
        for i in 1:N, j in 1:M
    ]

    # --- Parseval (window=:none, no padding, no demean): sum(|F|²) = N²·sum(|A|²).
    @testset "parseval no-window no-pad" begin
        A = deterministic_field(8, 8)
        res = MANTA.power_spectrum_2d(A; window = :none, demean = false, pad_pow2 = false)
        # power_spectrum_2d divides by ⟨W²⟩ = 1 here, so it stores |F|² directly.
        # Parseval for non-unitary FFT used by FFTW: sum(|F|²) = N² · sum(|A|²).
        @test isapprox(sum(res.P2d), length(A) * sum(abs2, A); rtol = 1e-9)
    end

    # --- Padding to next power of 2 produces correct effective size.
    @testset "pad to next pow2" begin
        A = deterministic_field(7, 11)
        res = MANTA.power_spectrum_2d(A; window = :none, demean = false, pad_pow2 = true)
        @test res.ny_eff == 8 && res.nx_eff == 16
        @test res.padded
        @test size(res.P2d) == (8, 16)
        # Already-pow2 input does not grow.
        B = deterministic_field(8, 8)
        res2 = MANTA.power_spectrum_2d(B; window = :none, demean = false, pad_pow2 = true)
        @test !res2.padded
        @test (res2.ny_eff, res2.nx_eff) == (8, 8)
    end

    # --- NaN apodization runs and reports a coherent f_sky.
    @testset "NaN apodization" begin
        N = 32
        A = deterministic_field(N, N)
        # Carve a NaN strip on the left.
        A[:, 1:6] .= NaN
        res = MANTA.power_spectrum_2d(A; window = :hann,
                                       apodize_nan = true, nan_taper = 3)
        @test res.apodized
        @test isapprox(res.f_sky, (N * (N - 6)) / (N * N); atol = 1e-9)
        @test all(isfinite, res.P2d)
        @test res.w_norm > 0
    end

    # --- Hamming window is applied (different result from :none).
    @testset "window kinds differ" begin
        A = deterministic_field(16, 16)
        r_none    = MANTA.power_spectrum_2d(A; window = :none, demean = false)
        r_hann    = MANTA.power_spectrum_2d(A; window = :hann, demean = false)
        r_hamming = MANTA.power_spectrum_2d(A; window = :hamming, demean = false)
        @test r_none.P2d != r_hann.P2d
        @test r_hann.P2d != r_hamming.P2d
    end

    # --- Log-log slope fit recovers an injected power law.
    @testset "fit_loglog_slope" begin
        k = collect(1.0:1.0:50.0)
        # P = 10 * k^(-2.7)
        p = 10.0 .* (k .^ -2.7)
        slope, intercept, n = MANTA.fit_loglog_slope(k, p; kmin = 2.0, kmax = 40.0)
        @test isapprox(slope, -2.7; atol = 1e-9)
        @test isapprox(intercept, log10(10.0); atol = 1e-9)
        @test n == count(ki -> 2.0 <= ki <= 40.0, k)

        # Empty band -> NaN, n = 0.
        s2, i2, n2 = MANTA.fit_loglog_slope(k, p; kmin = 1e9, kmax = 1e10)
        @test isnan(s2) && isnan(i2) && n2 == 0

        # Drops non-positive p (log undefined).
        kn = [1.0, 2.0, 4.0, 8.0]
        pn = [1.0, 0.0, 1 / 16, 1 / 64]   # zero is dropped
        s3, _, n3 = MANTA.fit_loglog_slope(kn, pn)
        @test n3 == 3
        @test isapprox(s3, -2.0; atol = 1e-9)
    end
end

# ----------------------------------------------------------------------------
# Refactor regression tests (Phase 1)
# ----------------------------------------------------------------------------

@testset "helpers: as_float32 and get_slice variants" begin
    # No-op on already-Float32 dense arrays.
    A32 = rand(Float32, 4, 5)
    @test MANTA.as_float32(A32) === A32

    # Conversion path is allocated, but only when needed.
    A64 = rand(Float64, 4, 5)
    A32_from_64 = MANTA.as_float32(A64)
    @test eltype(A32_from_64) === Float32
    @test size(A32_from_64) == size(A64)

    # get_slice_view returns a view; get_slice_copy returns an independent
    # buffer.
    cube = reshape(collect(Float32, 1:24), 2, 3, 4)
    sv1 = MANTA.get_slice_view(cube, 1, 1)
    sv2 = MANTA.get_slice_view(cube, 2, 1)
    sv3 = MANTA.get_slice_view(cube, 3, 1)
    @test size(sv1) == (3, 4)
    @test size(sv2) == (2, 4)
    @test size(sv3) == (2, 3)
    @test sv1 isa SubArray
    @test sv2 isa SubArray
    @test sv3 isa SubArray

    sc1 = MANTA.get_slice_copy(cube, 1, 1)
    @test sc1 == sv1
    sc1[1, 1] = -99f0
    @test cube[1, 1, 1] != -99f0   # the cube was not mutated by editing the copy.

    @test_throws ArgumentError MANTA.get_slice_view(cube, 4, 1)
    @test_throws BoundsError MANTA.get_slice_view(cube, 1, 99)
end

@testset "helpers: parse_path_spec" begin
    @test first(MANTA.parse_path_spec("foo.fits")) === :fits
    @test first(MANTA.parse_path_spec("foo.fit")) === :fits
    @test first(MANTA.parse_path_spec("foo.FITS.GZ")) === :fits
    @test first(MANTA.parse_path_spec("foo.h5")) === :hdf5
    @test first(MANTA.parse_path_spec("foo.hdf5")) === :hdf5

    # path:address syntax
    kind, p, addr = MANTA.parse_path_spec("file.h5:/group/ds")
    @test kind === :hdf5
    @test p == "file.h5"
    @test addr == "/group/ds"

    # Windows-drive letter is NOT treated as an HDF5 spec.
    @test first(MANTA.parse_path_spec("C:/data/foo.fits")) === :fits

    # Unknown extension.
    @test first(MANTA.parse_path_spec("notes.txt")) === :unknown
end

@testset "datasets: load_dataset (in-memory)" begin
    # 2D array → ImageDataset
    img32 = rand(Float32, 10, 20)
    ds_img = MANTA.load_dataset(img32)
    @test ds_img isa MANTA.ImageDataset
    @test eltype(ds_img.data) === Float32
    @test size(ds_img.data) == (10, 20)

    # 3D array → CubeDataset
    cube32 = rand(Float32, 10, 20, 30)
    ds_cube = MANTA.load_dataset(cube32)
    @test ds_cube isa MANTA.CubeDataset
    @test eltype(ds_cube.data) === Float32

    # Float64 input is preserved by the type alias rule of in-memory loader
    # (no implicit Float32 cast in this path — display-time conversion is the
    # viewer's responsibility via `as_float32`).
    cube64 = rand(Float64, 4, 5, 6)
    ds_cube64 = MANTA.load_dataset(cube64)
    @test ds_cube64 isa MANTA.CubeDataset
    @test eltype(ds_cube64.data) === Float64

    # NamedTuple → MultiChannelDataset
    Q = rand(Float32, 8, 8, 5)
    U = rand(Float32, 8, 8, 5)
    ds_mc = MANTA.load_dataset((Q = Q, U = U))
    @test ds_mc isa MANTA.MultiChannelDataset
    @test haskey(ds_mc.channels, :Q)
    @test haskey(ds_mc.channels, :U)
    @test ds_mc.kind === :cube

    # Idempotence: load_dataset(ds) === ds
    @test MANTA.load_dataset(ds_img) === ds_img

    # stable_source_id is deterministic per (typeof, size)
    sid1 = MANTA.stable_source_id(rand(Float32, 4, 5))
    sid2 = MANTA.stable_source_id(rand(Float32, 4, 5))
    @test sid1 == sid2
    @test sid1 != MANTA.stable_source_id(rand(Float32, 4, 6))

    # source_id override
    ds_named = MANTA.load_dataset(rand(Float32, 3, 3); source_id = "my_custom_id")
    @test ds_named.source_id == "my_custom_id"

    # Unsupported ndims → ArgumentError with a clear message.
    @test_throws ArgumentError MANTA.load_dataset(rand(Float32, 2, 2, 2, 2))
end

@testset "datasets: load_dataset (paths)" begin
    @test_throws ArgumentError MANTA.load_dataset("does_not_exist.fits")
    @test_throws ArgumentError MANTA.load_dataset("does_not_exist.h5")
    @test_throws ArgumentError MANTA.load_dataset("unrecognised.txt")
end

@testset "abstract type rename" begin
    # AbstractMANTADataset is the new name; the carta alias must still resolve
    # to the same type to avoid breaking external code.
    @test MANTA.AbstractCartaDataset === MANTA.AbstractMANTADataset
    @test MANTA.ImageDataset <: MANTA.AbstractMANTADataset
    @test MANTA.CubeDataset <: MANTA.AbstractMANTADataset
    @test MANTA.HealpixMapDataset <: MANTA.AbstractMANTADataset
    @test MANTA.HealpixCubeDataset <: MANTA.AbstractMANTADataset
    @test MANTA.MultiChannelDataset <: MANTA.AbstractMANTADataset
end

@testset "dispatch: manta on datasets" begin
    # Cube dispatch (in-memory). The full interactive viewer now runs even
    # without a backing FITS file. Headless flags keep it CI-safe: no GL
    # context, no window display, just figure construction + return.
    cube_ds = MANTA.load_dataset(rand(Float32, 6, 6, 4))
    @test cube_ds isa MANTA.CubeDataset
    fig_cube = MANTA.manta(cube_ds; activate_gl = false, display_fig = false)
    @test fig_cube isa Makie.Figure

    # Same outcome via the user-facing 3D-array entry point — routes through
    # load_dataset → CubeDataset → _view_cube.
    fig_arr = MANTA.manta(rand(Float32, 6, 6, 4);
                          activate_gl = false, display_fig = false)
    @test fig_arr isa Makie.Figure

    # `view_cube` is the exported alias for `_view_cube` and accepts the same
    # CubeDataset directly.
    fig_vc = MANTA.view_cube(cube_ds; activate_gl = false, display_fig = false)
    @test fig_vc isa Makie.Figure
    @test MANTA.view_cube === MANTA._view_cube

    # VectorDataset is still explicitly unsupported.
    vds = MANTA.load_dataset(rand(Float32, 16))
    @test vds isa MANTA.VectorDataset
    @test_throws ErrorException MANTA.manta(vds; activate_gl = false, display_fig = false)
end

@testset "view_cube: respects dataset fields" begin
    # Build a CubeDataset with explicit unit_label/source_id and confirm
    # the viewer constructs a figure that uses them (we can't easily probe
    # the labels post-construction, but the call should succeed and return
    # a Makie.Figure).
    data = rand(Float32, 5, 4, 3)
    ds = MANTA.CubeDataset(data;
        axis_labels = ["RA", "Dec", "v"],
        unit_label  = "Jy/beam",
        source_id   = "synthetic_cube",
    )
    fig = MANTA.view_cube(ds; activate_gl = false, display_fig = false)
    @test fig isa Makie.Figure

    # Float64 input gets coerced to Float32 by `as_float32`, without a crash.
    ds64 = MANTA.CubeDataset(rand(Float64, 4, 4, 3))
    fig64 = MANTA.view_cube(ds64; activate_gl = false, display_fig = false)
    @test fig64 isa Makie.Figure
end

