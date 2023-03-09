using LinearAlgebra
using CairoMakie 
#using Interpolations
using Polynomials

# honeycomb lattice. spatial lattice basis

# scaling vs α
αs = 3:0.2:5
fs = f_density_honeycomb_α.(12, βc, αs)
f0 = f_density_honeycomb(12, βc)
fig_α = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 400))
ax1 = Axis(fig_α[1, 1], xscale=log10, yscale=log10)
sc1 = scatter!(ax1, 1 .- tanh.(αs), fs .- f0, marker=:dot, markersize=10)
@show fig_α

# scaling vs N
Ns = 8:8:120
f1s = f_density_honeycomb.(Ns, βc)
f2s = f_density_honeycomb.(Ns, βc-0.1)
f3s = f_density_honeycomb.(Ns, βc+0.1)
fig_N = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 400))
ax1 = Axis(fig_N[1, 1], yscale=log10)
sc1 = lines!(ax1, Ns, f1s .- f1s[end] .+ 1e-3, marker=:dot, markersize=10, label=L"\beta=\beta_c")
sc2 = lines!(ax1, Ns, f2s .- f2s[end] .+ 1e-3, marker=:dot, markersize=10, label=L"\beta=\beta_c-0.1")
sc3 = lines!(ax1, Ns, f3s .- f3s[end] .+ 1e-3, marker=:dot, markersize=10, label=L"\beta=\beta_c+0.1")
axislegend(ax1, position=:rt)
@show fig_N

N = 100
Δβ = 0.0025
βs = Vector(βc .+ Δβ .* (-20:20))
fs = f_density_honeycomb.(N, βs)
Es = (fs[3:end] - fs[1:end-2]) / (2*Δβ) .* βs[2:end-1]
Cvs = -(Es[3:end] - Es[1:end-2]) / (2*Δβ) .* (βs[3:end-2] .^ 2)

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 800))

ax1 = Axis(fig[1, 1], 
        xlabel = L"\beta",
        ylabel = L"f", 
        )
ax2 = Axis(fig[2, 1], 
        xlabel = L"\beta",
        ylabel = L"E", 
        )
ax3 = Axis(fig[3, 1], 
        xlabel = L"\beta",
        ylabel = L"C_V", 
        )
#itp = interpolate((βs,), fs, Gridded(Linear()));
#Es = only.(Interpolations.gradient.(Ref(itp), βs)) .* βs
#itp2 = interpolate((βs,), Es, Gridded(Linear()));
#Cvs = only.(Interpolations.gradient.(Ref(itp2), βs)) .* (-βs .^ 2)

sc1 = scatter!(ax1, βs, fs, markersize=10, label="N=$(N)")
sc2 = scatter!(ax2, βs[2:end-1], Es, markersize=10, label="N=$(N)")
sc3 = scatter!(ax3, βs[3:end-2], Cvs, markersize=10, label="N=$(N)")

sc2 = lines!(ax2, fill(βc, 2), [0.1, 0.9], color=:grey)
sc2 = lines!(ax3, fill(βc, 2), [0.5, 5], color=:grey)

axislegend(ax1)
axislegend(ax2)
axislegend(ax3)
@show fig

save("honeycomb_lattice_exact.pdf", fig)

################# triangular


βc = asinh(1/sqrt(3)) / 2

# scaling vs N
Ns = 8:8:120
f1s = f_density_triangular.(Ns, βc)
f2s = f_density_triangular.(Ns, βc-0.1)
f3s = f_density_triangular.(Ns, βc+0.1)
fig_N = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 400))
ax1 = Axis(fig_N[1, 1], yscale=log10)
sc1 = lines!(ax1, Ns, f1s .- f1s[end] .+ 1e-3, marker=:dot, markersize=10, label=L"\beta=\beta_c")
sc2 = lines!(ax1, Ns, f2s .- f2s[end] .+ 1e-3, marker=:dot, markersize=10, label=L"\beta=\beta_c-0.1")
sc3 = lines!(ax1, Ns, f3s .- f3s[end] .+ 1e-3, marker=:dot, markersize=10, label=L"\beta=\beta_c+0.1")
axislegend(ax1, position=:rt)
@show fig_N

N = 100
Δβ = 0.0025
βs = Vector(βc .+ Δβ .* (-10:10))
fs = f_density_triangular.(N, βs)
Es = (fs[3:end] - fs[1:end-2]) / (2*Δβ) .* βs[2:end-1]
Cvs = -(Es[3:end] - Es[1:end-2]) / (2*Δβ) .* (βs[3:end-2] .^ 2)

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 800))

ax1 = Axis(fig[1, 1], 
        xlabel = L"\beta",
        ylabel = L"f", 
        )
ax2 = Axis(fig[2, 1], 
        xlabel = L"\beta",
        ylabel = L"E", 
        )
ax3 = Axis(fig[3, 1], 
        xlabel = L"\beta",
        ylabel = L"C_V", 
        )
#itp = interpolate((βs,), fs, Gridded(Linear()));
#Es = only.(Interpolations.gradient.(Ref(itp), βs)) .* βs
#itp2 = interpolate((βs,), Es, Gridded(Linear()));
#Cvs = only.(Interpolations.gradient.(Ref(itp2), βs)) .* (-βs .^ 2)

sc1 = scatter!(ax1, βs, fs, markersize=10, label="N=$(N)")
sc2 = scatter!(ax2, βs[2:end-1], Es, markersize=10, label="N=$(N)")
sc3 = scatter!(ax3, βs[3:end-2], Cvs, markersize=10, label="N=$(N)")

lines!(ax2, fill(βc, 2), [0.1, 2.9], color=:grey)
lines!(ax3, fill(βc, 2), [0.5, 2], color=:grey)

axislegend(ax1)
axislegend(ax2)
axislegend(ax3)
@show fig

save("triangular_lattice_exact.pdf", fig)