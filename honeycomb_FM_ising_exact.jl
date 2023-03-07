using LinearAlgebra
using CairoMakie 

# honeycomb lattice. spatial lattice basis

function free_energy_density_sp(N::Int, β::Real)

    β1 = -0.5 * log(tanh(β)) # β^\ast

    M1 = zeros(ComplexF64, 2*N, 2*N)
    M2 = zeros(ComplexF64, 2*N, 2*N)
    M3 = zeros(ComplexF64, 2*N, 2*N)

    for ix in 1:2:N
        M1[2*ix-1, 2*ix] = 2*β1
        M1[2*ix, 2*ix-1] = -2*β1
    end

    for ix in 2:2:N
        M3[2*ix-1, 2*ix] = 2*β1
        M3[2*ix, 2*ix-1] = -2*β1
    end

    for ix in 1:N
        M2[2*ix, (2*ix+1) % (2*N)] = 2*β
        M2[(2*ix+1) % (2*N), 2*ix] = -2*β
    end

    RT = exp(-im * M3 / 2) * exp(-im * M2) * exp(-im * M1) * exp(-im * M2) * exp(-im * M3 / 2)

    Λ, _ = eigen(Hermitian(RT))

    ϵs = log.(Λ)[N+1:end]

    f = sum(ϵs) / (4*N*β) - log(2*sinh(β)) / (2*β)
    return f
end

β = 1
for N in 8:8:40
    @show N, free_energy_density_sp(N, β)
end

N = 100

free_energy_density_sp(N, β)

βc = 1/1.51883

βs = Vector(βc .+ 0.02 .* (-10:10))
fs = free_energy_density_sp.(N, βs)

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 500))

ax1 = Axis(fig[1, 1], 
        xlabel = L"\beta",
        ylabel = L"f", 
        )

sc1_orig = scatter!(ax1, βs, fs, color=:darkgreen, marker=:o, markersize=15)
@show fig
