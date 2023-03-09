using LinearAlgebra, TensorKit, MPSKit
using CairoMakie

include("utils.jl");

βc = asinh(1/sqrt(3)) / 2
#β = 0.2
β = βc

# MPO for the transfer matrix

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

𝕋 = DenseMPO([M1, M2])

# test at finite size
fNs = Float64[]
fN2s = Float64[]
for N in [2, 3, 4, 5]
    𝕋N = reduce(vcat, fill([M1, M2], N))
    𝕋N_mat = convert_to_mat(𝕋N)
    Λ, _ = eigen(𝕋N_mat)
    fN = - log(real(Λ.data[end])) / (2*N*β)
    fN2 = f_density_triangular(N, β)
    push!(fNs, fN)
    push!(fN2s, fN2)
    println("$(N), $(fN), $(fN2), $(fN - fN2)")
end

# hermicity
𝕋N = reduce(vcat, fill([M1, M2], 4));
𝕋N_mat = convert_to_mat(𝕋N);
### not hermitian
@show norm(𝕋N_mat - 𝕋N_mat') 
### not normal
@show norm(𝕋N_mat * 𝕋N_mat' - 𝕋N_mat' * 𝕋N_mat)

# VUMPS
χs = [8, 16, 24, 32, 40]
fχs = Float64[]
for χ in χs
    A = TensorMap(rand, ComplexF64, ℂ^χ*ℂ^2, ℂ^χ);
    B = TensorMap(rand, ComplexF64, ℂ^χ*ℂ^2, ℂ^χ);
    ψ0 = InfiniteMPS([A, B]);

    ψ, envs, _ = leading_boundary(ψ0, 𝕋, VUMPS(tol_galerkin=1e-10, maxiter=10000)); 
    push!(fχs, real(log(dot(ψ, 𝕋, ψ))) / (-β) / 2)
end

Ns = [80, 160, 240, 320, 400]
fNs = f_density_triangular.(Ns, β)

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 400))
ax1 = Axis(fig[1, 1])
sc1 = scatter!(ax1, 1 ./ χs[2:end], fχs[2:end], marker=:dot, markersize=10)
sc2 = scatter!(ax1, 20 ./ Ns[2:end], fNs[2:end], marker=:dot, markersize=10)
@show fig

save("triangular_Ising_results.pdf", fig)