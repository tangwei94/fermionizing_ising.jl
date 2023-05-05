using LinearAlgebra
using CairoMakie
using LsqFit
using Revise 
using fermionizing_ising

function ReC_gs(N::Int, β::Real)

    β1 = -0.5 * log(tanh(β)) # β^\ast

    M1 = zeros(ComplexF64, 2*N, 2*N)
    M2 = zeros(ComplexF64, 2*N, 2*N)

    for ix in 1:N
        M1[2*ix-1, 2*ix] = 2*β1
        M1[2*ix, 2*ix-1] = -2*β1
    end

    for ix in 1:N-1
        M2[2*ix, 2*ix+1] = 2*β
        M2[2*ix+1, 2*ix] = -2*β
    end
    #M2[2*N, 1] = -2*β
    #M2[1, 2*N] = 2*β

    RT = exp(-im * M1 / 2) * exp(-im * M2) * exp(-im * M1 / 2)
    MT = real.(-im*log(RT))
   
    # covariance matrix of ground state
    ReC = covariance_matrix(MT, Inf)
    return ReC
end

function entanglement_entropy(ReC, subN)
    ReCsub = ReC[1:2*subN, 1:2*subN]
    λs = skew_canonical(ReCsub)[3]
    SE2 = 0
    for λ in λs
        if λ < 1
            SE2 += log(2) - 0.5*(log(1+λ) * (1+λ) + log(1-λ) * (1-λ))
        end
    end
    return SE2
end

N = 20
βc = asinh(1) / 2

M1 = zeros(Float64, 2*N, 2*N)

for ix in 1:N
    M1[2*ix-1, 2*ix] = 2
    M1[2*ix, 2*ix-1] = -2
end
    
M2 = zeros(Float64, 2*N, 2*N)

for ix in 1:N-1
    M2[2*ix, 2*ix+1] = 2
    M2[2*ix+1, 2*ix] = -2
end

ReC = ReC_gs(N, βc)

Z = - 0.25 * tr(ReC * M1) / N

SEs = Float64[]
Zs = Float64[]
τs = -3:0.1:3
for τ in τs 
    ReC_Uτ = covariance_matrix(M1*τ, 1)
    #ReC_τ = real.(product_rule(product_rule(ReC_Uτ, ReC), ReC_Uτ))
    ReC_τ = real.(ReC_Uτ ∘ ReC ∘ ReC_Uτ)

    SE = entanglement_entropy(ReC_τ, N÷2)
    Z = - 0.25 * tr(ReC_τ * M1) / N
    @show τ, SE, Z
    push!(Zs, Z)
    push!(SEs, SE)
end

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 800))

ax1 = Axis(fig[1, 1], 
        xlabel = L"\tau",
        ylabel = L"S",
        yscale = log10, 
        )

sc1 = scatter!(ax1, τs, SEs, markersize=10)
@show fig

ax2 = Axis(fig[2, 1], 
        xlabel = L"\tau",
        ylabel = L"\langle \sigma^z \rangle",
        )

sc2 = scatter!(ax2, τs, Zs, markersize=10)
@show fig
save("gauge_z_ising.pdf", fig)

XX = - 0.25 * tr(ReC * M2) / N

SEs = Float64[]
XXs = Float64[]
τs = -3:0.1:3
for τ in τs 
    ReC_Uτ = covariance_matrix(M2*τ, 1)
    #ReC_τ = real.(product_rule(product_rule(ReC_Uτ, ReC), ReC_Uτ))
    ReC_τ = real.(ReC_Uτ ∘ ReC ∘ ReC_Uτ)

    SE = entanglement_entropy(ReC_τ, N÷2)
    XX = - 0.25 * tr(ReC_τ * M1) / N
    @show τ, SE, XX
    push!(XXs, Z)
    push!(SEs, SE)
end

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 800))

ax1 = Axis(fig[1, 1], 
        xlabel = L"\tau",
        ylabel = L"S",
        yscale = log10, 
        )

sc1 = scatter!(ax1, τs, SEs, markersize=10)
@show fig

ax2 = Axis(fig[2, 1], 
        xlabel = L"\tau",
        ylabel = L"\langle \sigma^z \rangle",
        )

sc2 = scatter!(ax2, τs, XXs, markersize=10)
@show fig
save("gauge_xx_ising.pdf", fig)