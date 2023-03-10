using fermionizing_ising
using LinearAlgebra
using LsqFit
using CairoMakie 
#using Interpolations

βc = βc_honeycomb()

# scaling vs α
αs = 3:0.2:5
fs = f_density_honeycomb_α.(12, βc, αs)
f0 = f_density_honeycomb(12, βc) # extrapolated
fig_α = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 800))
ax1 = Axis(fig_α[1, 1], xlabel=L"1-\tanh(\alpha)", ylabel=L"f")
sc1 = scatter!(ax1, 1 .- tanh.(αs), fs, marker=:dot, markersize=10)
ax2 = Axis(fig_α[2, 1], xscale=log10, yscale=log10, xlabel=L"1-\tanh(\alpha)", ylabel=L"f - f_0")
sc2 = scatter!(ax2, 1 .- tanh.(αs), fs .- f0, marker=:dot, markersize=10)
@show fig_α
save("demos/honeycomb_scaling_vs_alpha.pdf", fig_α)

# scaling vs N
### compare with critical and off-critical 
Ns = 40:8:200
fcs = f_density_honeycomb.(Ns, βc)
fps = f_density_honeycomb.(Ns, βc-0.1)
fms = f_density_honeycomb.(Ns, βc+0.1)
fig_N = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 800))
ax1 = Axis(fig_N[1, 1], yscale=log10, xlabel=L"N", ylabel=L"f - f_{N=200}")
sc1 = lines!(ax1, Ns, abs.(fcs .- fcs[end]) .+ 1e-12, marker=:dot, markersize=10, label=L"\beta=\beta_c")
sc2 = lines!(ax1, Ns, abs.(fps .- fps[end]) .+ 1e-12, marker=:dot, markersize=10, label=L"\beta=\beta_c-0.1")
sc3 = lines!(ax1, Ns, abs.(fms .- fms[end]) .+ 1e-12, marker=:dot, markersize=10, label=L"\beta=\beta_c+0.1")
axislegend(ax1, position=:rb)
@show fig_N
### critical fit as f = a N^b + f0, log-log plot
@. f_fit(x, p) = p[1]*x^p[2] + p[3]
p0 = [1.0, -1.0, last(fcs) + 1e-6]
fit_outcome = curve_fit(f_fit, Ns, fcs, p0)
pf = coef(fit_outcome)
ax2 = Axis(fig_N[2, 1], xlabel=L"N", ylabel=L"f-f_\infty", yscale=log10, xscale=log10)
sc1 = scatter!(ax2, Ns, abs.(fcs .- pf[3]) , marker=:dot, markersize=10, label=L"\beta=\beta_c")
@show fig_N
@show pf[3]
save("demos/honeycomb_scaling_vs_N.pdf", fig_N)

# free-energy, energy and specific heat
N = 200
Δβ = 0.00125
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

lines!(ax2, fill(βc, 2), [minimum(Es), maximum(Es)], color=:grey)
lines!(ax3, fill(βc, 2), [minimum(Cvs), maximum(Cvs) + 0.2], color=:grey)

axislegend(ax1)
axislegend(ax2)
axislegend(ax3)
@show fig

save("demos/honeycomb_observables.pdf", fig)
