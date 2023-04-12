using LinearAlgebra, TensorKit, MPSKit
using LsqFit
using CairoMakie
using JLD2
using Revise
using fermionizing_ising

include("utils.jl");

βc = βc_triangular()
#β = 0.1
β = βc

# MPO for the transfer matrix
function triangular_mpo(β::Real)
    Mdat = [exp(β) exp(-β) ; exp(-β) exp(β)]
    M = TensorMap(Mdat, ℂ^2, ℂ^2)
    sqrtM = sqrt(M)

    δdat = zeros(ComplexF64, 2, 2, 2, 2, 2, 2)
    δdat[1, 1, 1, 1, 1, 1] = δdat[2, 2, 2, 2, 2, 2] = 1
    δ = TensorMap(δdat, ℂ^2*ℂ^2*ℂ^2, ℂ^2*ℂ^2*ℂ^2)

    @tensor T[-1, -2, -3; -4, -5, -6] := sqrtM[-1, 1] * sqrtM[-2, 2] * sqrtM[-3, 3] * sqrtM[4, -4] * sqrtM[5, -5] * sqrtM[6, -6] * δ[1, 2, 3, 4, 5, 6]

    U, S, V = tsvd(T, (1, 2, 4), (3, 5, 6); trunc=truncerr(1e-10))
    T1 = permute(U * sqrt(S), (1, 2), (3, 4))
    T2 = permute(sqrt(S) * V, (1, 2), (3, 4))

    δm = isomorphism(ℂ^4, ℂ^2*ℂ^2)
    @tensor M1[-1 -2 ; -3 -4] := T1[3 1 ; -3 5] * T2[2 -2 ; 1 4] * δm[-1; 2 3] * δm'[4 5 ; -4]
    @tensor M2[-1 -2 ; -3 -4] := T2[3 1 ; -3 5] * T1[2 -2 ; 1 4] * δm[-1; 2 3] * δm'[4 5 ; -4]

    return T1, T2, M1, M2
end

T1, T2, M1, M2 = triangular_mpo(β)

𝕋 = MPOMultiline([T1 T2 ; T2 T1])
𝕋blk = DenseMPO([M1, M2])

χs = 8:8:40
fχs = Float64[]
ovlps = Float64[]
ψs = MPSMultiline[]
χ = 8
ψ0 = MPSMultiline([ℂ^2 ℂ^2; ℂ^2 ℂ^2], [ℂ^χ ℂ^χ; ℂ^χ ℂ^χ])
expand_alg = OptimalExpand(truncdim(8))
for χ in χs
    global ψ0, fχs, ovlps, ψs
    ψb, envsb, _ = leading_boundary(ψ0, 𝕋, VUMPS(tol_galerkin=1e-12, maxiter=2000)); 
    ψb1 = ψb.data[1]
    ψt1 = InfiniteMPS([ψb1.AL[2], ψb1.AL[1]])
    push!(fχs, real(log(dot(ψt1, 𝕋blk, ψb1)) - log(dot(ψt1, ψb1))) / (-β) / 2)
    push!(ovlps, real(log(dot(ψt1, ψb1))))
    push!(ψs, ψb)

    ψ0, envs = changebonds(ψb, 𝕋, expand_alg, envsb)
end

# exact solution. extrapolated from finite size results
Ns = 200:200:4000
fNs = f_density_triangular_analytic_sol.(Ns, β)
fNs1 = f_density_triangular.(Ns[1:4], β)
@. f_fit(x, p) = p[1]*x^p[2] + p[3]
#@. f_fit(x, p) = p[1]*exp(-p[2]*x) + p[3]
p0 = [1.0, -1.0, last(fNs) + 1e-6]
fit_outcome = curve_fit(f_fit, Ns, fNs, p0)
pf = coef(fit_outcome)
f0 = pf[3]

@save "VUMPS/VUMPS_multiline_results_beta$(β).jld2" fχs ovlps
@load "VUMPS/VUMPS_multiline_results_beta$(β).jld2" fχs ovlps

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 400))
ax1 = Axis(fig[1, 1], xlabel=L"1/\chi \text{ or } 50 / N", ylabel=L"|f-f_{\mathrm{exact}}|", yscale=log10, xscale=log10)
scatt_χ = scatter!(ax1, 1 ./ χs, abs.(fχs .- f0) .+ 1e-16, marker=:circle, markersize=10, label="VUMPS")
scatt_N = scatter!(ax1, 50 ./ Ns, abs.(fNs .- f0) .+ 1e-16, marker=:circle, markersize=10, label="exact #1")
scatt_N = scatter!(ax1, 50 ./ Ns[1:4], abs.(fNs1 .- f0) .+ 1e-16, marker='X', markersize=10, label="exact #2")
axislegend(ax1; position=:rb)
@show fig

save("VUMPS/triangular_Ising_multiline_results_beta$(β).pdf", fig)