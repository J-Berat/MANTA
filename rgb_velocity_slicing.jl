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
axis_render_height(axis) = lift(axis.scene.viewport) do rect
    max(1, rect.widths[2])
end

function render_size()
    if length(ARGS) >= 2
        return (parse(Int, ARGS[1]), parse(Int, ARGS[2]))
    end
    w = parse(Int, get(ENV, "FIG_W", "1400"))
    h = parse(Int, get(ENV, "FIG_H", "900"))
    return (w, h)
end

fig_w, fig_h = render_size()
usable_h = max(420, fig_h - 90)
top_axis_size = Int(floor(max(180, min((fig_w - 520) / 2, 0.52 * usable_h - 72))))
bottom_axis_size = Int(floor(max(150, min((fig_w - 520) / 3, 0.48 * usable_h - 72))))
tick_vals = [0, π / 2, π, 3π / 2, 2π]

fig = Figure(size=(fig_w, fig_h))
top_grid = fig[1, 1] = GridLayout(; halign=:center, valign=:center)
bottom_grid = fig[2, 1] = GridLayout(; halign=:center, valign=:center)
colgap!(top_grid, 0)
colgap!(bottom_grid, 0)
rowgap!(fig.layout, 8)

# (1,1) Composite RGB image — color = velocity direction
ax_rgb = Axis(top_grid[1, 1],
    xlabel=L"x", ylabel=L"y",
    title=L"\text{RGB composite: } (R,G,B) = (u,v,w)",
    xticks=tick_vals, yticks=tick_vals,
    xticksmirrored=true, yticksmirrored=true,
    xtickformat=latex_fmt, ytickformat=latex_fmt,
    width=top_axis_size, height=top_axis_size,
    titlesize=15,
    aspect=DataAspect())
image!(ax_rgb, (first(xs), last(xs)), (first(ys), last(ys)), permutedims(img_rgb))

# (1,2) Speed magnitude |v|
speed_grid = top_grid[1, 3] = GridLayout()
colgap!(speed_grid, -8)
ax_s = Axis(speed_grid[1, 1],
    xlabel=L"x", ylabel=L"y",
    title=L"\text{speed } |v| = \sqrt{u^2+v^2+w^2}",
    xticks=tick_vals, yticks=tick_vals,
    xticksmirrored=true, yticksmirrored=true,
    xtickformat=latex_fmt, ytickformat=latex_fmt,
    width=top_axis_size, height=top_axis_size,
    titlesize=15,
    aspect=DataAspect())
hm = heatmap!(ax_s, xs, ys, permutedims(speed), colormap=:magma)
Colorbar(speed_grid[1, 2], hm,
    label=L"\text{speed } |v|",
    width=18,
    height=axis_render_height(ax_s),
    tellheight=false,
    valign=:center)

# (2,1..3) Three individual channels — R=u, G=v, B=w
chan_titles = (L"u(x,y,z_0) \; [R]", L"v(x,y,z_0) \; [G]", L"w(x,y,z_0) \; [B]")
chan_data   = (U, V, W)
chan_cmaps  = (:Reds, :Greens, :Blues)

for (k, (data, ttl, cmap)) in enumerate(zip(chan_data, chan_titles, chan_cmaps))
    col = 2k - 1
    ax = Axis(bottom_grid[1, col],
        xlabel=L"x", ylabel=L"y", title=ttl,
        xticks=tick_vals, yticks=tick_vals,
        xticksmirrored=true, yticksmirrored=true,
        xtickformat=latex_fmt, ytickformat=latex_fmt,
        width=bottom_axis_size, height=bottom_axis_size,
        titlesize=15,
        aspect=DataAspect())
    M = maximum(abs, data)
    heatmap!(ax, xs, ys, permutedims(data),
             colormap=cmap, colorrange=(-M, M))
end

colsize!(top_grid, 2, Fixed(96))
colsize!(bottom_grid, 2, Fixed(86))
colsize!(bottom_grid, 4, Fixed(86))
rowsize!(fig.layout, 1, Relative(0.52))
rowsize!(fig.layout, 2, Relative(0.48))

Label(fig[0, :],
    L"\text{ABC flow — RGB slicing of the velocity field at } z_0=\pi/4",
    fontsize=18)

display(fig)
save(joinpath(@__DIR__, "rgb_velocity_slicing.png"), fig; px_per_unit=1)
