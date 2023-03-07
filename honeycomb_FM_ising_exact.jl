using LinearAlgebra
using CairoMakie 
using Interpolations

# honeycomb lattice. spatial lattice basis

function free_energy_density_honeycomb(N::Int, β::Real, α::Real=5)

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
    #ϵ1s = -log.(Λ[1:N]) # ϵs is more stable and more accurate

    f = -sum(ϵs) / (4*N*β) - log(2*sinh(2*β)) / (4*β) + log(cosh(α)) / (2*β)
    #f1 = -sum(ϵ1s) / (4*N*β) - log(2*sinh(2*β)) / (4*β) + log(cosh(α)) / (2*β)
    #@show log.(Λ)
    return f
end

β = 1
βc = asinh(sqrt(3)) / 2 

αs = 3:0.2:7
fs = free_energy_density_honeycomb.(12, βc-0.1, αs)
f1s = free_energy_density_honeycomb.(12, βc+0.1, αs)

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 400))
ax1 = Axis(fig[1, 1],)
sc1_orig = scatter!(ax1, 1 .- tanh.(αs), fs, marker=:dot, markersize=10)
sc1_orig = scatter!(ax1, 1 .- tanh.(αs), f1s, marker=:dot, markersize=10)
@show fig

for N in 8:8:100
    @show free_energy_density_honeycomb(N, βc-0.1)
end

N = 50

βs = Vector(βc .+ 0.01 .* (-20:20))
fs = free_energy_density_honeycomb.(N, βs, 7)

itp = interpolate((βs,), fs, Gridded(Linear()));
Es = only.(Interpolations.gradient.(Ref(itp), βs)) .* βs
itp2 = interpolate((βs,), Es, Gridded(Linear()));
Cvs = only.(Interpolations.gradient.(Ref(itp2), βs)) .* (-βs .^ 2)

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 800))

ax1 = Axis(fig[1, 1], 
        xlabel = L"\beta",
        ylabel = L"f", 
        )

sc1_orig = scatter!(ax1, βs, fs, marker=:circle, markersize=15)
@show fig

ax2 = Axis(fig[2, 1], 
        xlabel = L"\beta",
        ylabel = L"E", 
        )

sc2_orig = scatter!(ax2, βs, Es, marker=:o, markersize=15)
@show fig

ax3 = Axis(fig[3, 1], 
        xlabel = L"\beta",
        ylabel = L"C_V", 
        )

sc3_orig = scatter!(ax3, βs, Cvs, marker=:o, markersize=15)
@show fig
