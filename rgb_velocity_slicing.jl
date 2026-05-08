# ============================================================
# RGB velocity slicing — ABC flow
# Each velocity component (u, v, w) is mapped to one RGB channel
# ============================================================
# using Pkg; Pkg.add(["CairoMakie", "LaTeXStrings", "Colors"])

using CairoMakie, LaTeXStrings
using Colors

# --- ABC flow (Arnold–Beltrami–Childress) ---
# Steady solution of the Euler equations, classical chaotic flow.
abstract type FlowField end

struct ABCFlow <: FlowField
    A::Float64
    B::Float64
    C::Float64
end

# Multiple-dispatch interface: velocity(flow, x, y, z) -> (u, v, w)
function velocity(f::ABCFlow, x, y, z)
    u = f.A * sin(z) + f.C * cos(y)
    v = f.B * sin(x) + f.A * cos(z)
    w = f.C * sin(y) + f.B * cos(x)
    return (u, v, w)
end

# --- Slice the 3 channels of the velocity at z = z₀ ---
function velocity_slice(flow::FlowField, xs, ys, z₀)
    Nx, Ny = length(xs), length(ys)
    U = Matrix{Float64}(undef, Ny, Nx)   # heatmap order (y, x)
    V = similar(U)
    W = similar(U)
    @inbounds for j in 1:Nx, i in 1:Ny
        u, v, w = velocity(flow, xs[j], ys[i], z₀)
        U[i, j] = u; V[i, j] = v; W[i, j] = w
    end
    return U, V, W
end

# Normalize a channel to [0, 1] (symmetric around 0 → mid-grey at v=0)
normalize_sym(x) = begin
    M = maximum(abs, x)
    M == 0 ? fill(0.5, size(x)) : 0.5 .+ 0.5 .* x ./ M
end

# Build the RGB image: each velocity component → one channel
function rgb_from_velocity(U, V, W)
    R = normalize_sym(U)
    G = normalize_sym(V)
    B = normalize_sym(W)
    return [RGB(R[i, j], G[i, j], B[i, j]) for i in axes(R, 1), j in axes(R, 2)]
end

# --- Parameters ---
const A_, B_, C_ = √3, √2, 1.0
flow = ABCFlow(A_, B_, C_)

N  = 400
xs = LinRange(0, 2π, N)
ys = LinRange(0, 2π, N)
z₀ = π/4

U, V, W  = velocity_slice(flow, xs, ys, z₀)
img_rgb  = rgb_from_velocity(U, V, W)
speed    = @. sqrt(U^2 + V^2 + W^2)

# --- Visualization: RGB composite + 3 separate channels ---
latex_fmt(vals) = [L"%$(round(v/π, digits=2))\pi" for v in vals]

fig = Figure(size=(1100, 850))

# (1,1) Composite RGB image — color = velocity direction
ax_rgb = Axis(fig[1, 1],
    xlabel=L"x", ylabel=L"y",
    title=L"\text{RGB composite: } (R,G,B) = (u,v,w)",
    xticksmirrored=true, yticksmirrored=true,
    xtickformat=latex_fmt, ytickformat=latex_fmt,
    aspect=DataAspect())
image!(ax_rgb, xs, ys, permutedims(img_rgb))   # CairoMakie expects (x, y)

# (1,2) Speed magnitude |v|
ax_s = Axis(fig[1, 2],
    xlabel=L"x", ylabel=L"y",
    title=L"\|\mathbf{v}\| = \sqrt{u^2+v^2+w^2}",
    xticksmirrored=true, yticksmirrored=true,
    xtickformat=latex_fmt, ytickformat=latex_fmt,
    aspect=DataAspect())
hm = heatmap!(ax_s, xs, ys, permutedims(speed), colormap=:magma)
Colorbar(fig[1, 3], hm, label=L"\|\mathbf{v}\|")

# (2,1..3) Three individual channels — R=u, G=v, B=w
chan_titles = (L"u(x,y,z_0) \; [R]", L"v(x,y,z_0) \; [G]", L"w(x,y,z_0) \; [B]")
chan_data   = (U, V, W)
chan_cmaps  = (:Reds, :Greens, :Blues)

for (k, (data, ttl, cmap)) in enumerate(zip(chan_data, chan_titles, chan_cmaps))
    ax = Axis(fig[2, k],
        xlabel=L"x", ylabel=L"y", title=ttl,
        xticksmirrored=true, yticksmirrored=true,
        xtickformat=latex_fmt, ytickformat=latex_fmt,
        aspect=DataAspect())
    M = maximum(abs, data)
    heatmap!(ax, xs, ys, permutedims(data),
             colormap=cmap, colorrange=(-M, M))
end

Label(fig[0, :],
    L"\text{ABC flow — RGB slicing of the velocity field at } z_0=\pi/4",
    fontsize=18)

display(fig)
save("/sessions/dazzling-awesome-hypatia/mnt/MANTA.jl/rgb_velocity_slicing.png", fig; px_per_unit=2)
