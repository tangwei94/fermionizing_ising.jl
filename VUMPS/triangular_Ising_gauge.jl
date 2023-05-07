using Optim
using fermionizing_ising
using LinearAlgebra, TensorKit, MPSKit, KrylovKit

include("utils.jl");

βc = βc_triangular()
#β = 0.2
β = βc

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

    return M1, M2
end

# MPO for the transfer matrix
M1, M2 = triangular_mpo(β)

𝕋 = DenseMPO([M1, M2])

@tensor t1[-1; -2] := M1[1 -1 -2 1]
@tensor t2[-1; -2] := M2[1 -1 -2 1]
@tensor t0[-1 -2; -3 -4] := M1[1 -1; -3 2] * M2[2 -2; -4 1]

Λ1, P1 = eigen(t1)
Pinv1 = inv(P1)

hs = zeros(4)
hs[1] = log(Pinv1.data' * Pinv1.data)[1] |> real
hs[2] = log(Pinv1.data' * Pinv1.data)[1,2] |> real
hs[3] = log(Pinv1.data' * Pinv1.data)[1,2] |> imag
hs[4] = log(Pinv1.data' * Pinv1.data)[4] |> real
@show [hs[1] hs[2] + im*hs[3] ; hs[2] - im*hs[3] hs[4]] - log(Pinv1.data' * Pinv1.data)
hs

M1_dag = mpotensor_dag(M1)
M2_dag = mpotensor_dag(M2)

𝕋dag = DenseMPO([M1_dag, M2_dag])

function AAprime_straight(𝕋::DenseMPO, 𝕋dag::DenseMPO, hs::Vector{<:Real})
    Hmat = [hs[1] hs[2] + im*hs[3] ; hs[2] - im*hs[3] hs[4]]
    H = TensorMap(Hmat, ℂ^2, ℂ^2)
    H = H + H'
    G = exp(H)
    Ginv = exp(-H)

    Hmat2 = [hs[5] hs[6] + im*hs[7] ; hs[6] - im*hs[7] hs[8]]
    H2 = TensorMap(Hmat2, ℂ^2, ℂ^2)
    H2 = H2 + H2'
    G2 = exp(H2)
    Ginv2 = exp(-H2)

    𝔾 = DenseMPO([add_util_leg(G), add_util_leg(G2)])
    𝔾inv = DenseMPO([add_util_leg(Ginv), add_util_leg(Ginv2)])

    ψ1 = convert(InfiniteMPS, 𝔾 * 𝕋 * 𝔾inv * 𝕋dag * 𝔾)
    ψ2 = convert(InfiniteMPS, 𝕋dag * 𝔾 * 𝕋)

    -norm(dot(ψ1, ψ2))
end

function g_AAprime_straight!(ghs::Vector{<:Real}, hs::Vector{<:Real})
    ghs .= grad(central_fdm(5, 1), x -> AAprime_straight(𝕋, 𝕋dag, x), hs)[1]
end

hs = zeros(8)

AAprime_straight(𝕋, 𝕋dag, hs)
using FiniteDifferences, Optim

#res = optimize(x -> AAprime_straight(𝕋, 𝕋dag, x), g_AAprime_straight!, hs, LBFGS(), Optim.Options(show_trace = true))
res = optimize(x -> AAprime_straight(𝕋, 𝕋dag, x), g_AAprime_straight!, hs, GradientDescent(), Optim.Options(show_trace = true))

@show Optim.minimum(res)
hs = Optim.minimizer(res)
AAprime_straight(𝕋, 𝕋dag, hs)

using JLD2
@save "hs.jld2" hs