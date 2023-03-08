using LinearAlgebra
using CairoMakie 
#using Interpolations
using Polynomials

# honeycomb lattice. spatial lattice basis

function f_density_honeycomb_α(N::Int, β::Real, α::Real=5)

    β1 = -0.5 * log(tanh(β)) # β^\ast

    M1 = zeros(ComplexF64, 2*N, 2*N)
    M2 = zeros(ComplexF64, 2*N, 2*N)
    M3 = zeros(ComplexF64, 2*N, 2*N)

    for ix in 1:2:N
        M1[2*ix-1, 2*ix] = 2*β1
        M1[2*ix, 2*ix-1] = -2*β1
        M3[2*ix-1, 2*ix] = 2*α
        M3[2*ix, 2*ix-1] = -2*α
    end

    for ix in 2:2:N
        M3[2*ix-1, 2*ix] = 2*β1
        M3[2*ix, 2*ix-1] = -2*β1
        M1[2*ix-1, 2*ix] = 2*α
        M1[2*ix, 2*ix-1] = -2*α
    end

    for ix in 1:N
        M2[2*ix, (2*ix+1) % (2*N)] = 2*β
        M2[(2*ix+1) % (2*N), 2*ix] = -2*β
    end

    RT = exp(-im * M3 / 2) * exp(-im * M2) * exp(-im * M1) * exp(-im * M2) * exp(-im * M3 / 2)

    Λ, _ = eigen(Hermitian(RT))

    ϵs = log.(Λ[N+1:end])
    #ϵ1s = -log.(Λ[1:N]) # the positive entries are more stable and more accurate

    f = -sum(ϵs) / (4*N*β) - log(2*sinh(2*β)) / (4*β) + log(cosh(α)) / (2*β)
    return f
end

function f_density_honeycomb(N::Int, β::Real)
    αs = 3:0.5:5
    fs = f_density_honeycomb_α.(N, β, αs)
    fit_linf = Polynomials.fit(1 .- tanh.(αs), fs, 1)
    return fit_linf[0]
end

βc = asinh(sqrt(3)) / 2 

# scaling vs α
αs = 3:0.2:5
fs = f_density_honeycomb_α.(12, βc, αs)
fig_α = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 400))
ax1 = Axis(fig_α[1, 1],)
sc1 = scatter!(ax1, 1 .- tanh.(αs), fs, marker=:dot, markersize=10)
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

function f_density_triangular(N::Int, β::Real)
    β1 = -0.5 * log(tanh(β)) # β^\ast

    f_honeycomb = f_density_honeycomb(N, β1)
    f_triangular = (3*log(cosh(β)) + log(2) + (-β1)*f_honeycomb - 3*β1 / 2) / (-β)
    return f_triangular
end

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