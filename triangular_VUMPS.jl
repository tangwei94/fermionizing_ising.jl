using LinearAlgebra, TensorKit, MPSKit

β = 1
βc = asinh(1/sqrt(3)) / 2
β = βc

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
@tensor M1[-1 -2 ; -3 -4] := T1[2, 1, -3, 4] * T2[1, -2, 1, 3] * δm[-1, 1, 2] * δm'[3, 4, -4]
@tensor M2[-1 -2 ; -3 -4] := T2[2, 1, -3, 4] * T1[1, -2, 1, 3] * δm[-1, 1, 2] * δm'[3, 4, -4]

𝕋 = DenseMPO([M1, M2])
χ = 2
A = TensorMap(rand, ComplexF64, ℂ^χ*ℂ^2, ℂ^χ)
B = TensorMap(rand, ComplexF64, ℂ^χ*ℂ^2, ℂ^χ)
ψ0 = InfiniteMPS([A, B])

ψ, envs, _ = leading_boundary(ψ0, 𝕋, VUMPS(tol_galerkin=1e-10)); 
@show log(dot(ψ, 𝕋, ψ)) / (-β) / 2
dot(ψ, 𝕋, ψ) |> sqrt
exp(-β *f_density_triangular(100, β))