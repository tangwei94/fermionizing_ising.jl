using LinearAlgebra, TensorKit, MPSKit
using LsqFit
using fermionizing_ising
using CairoMakie

include("utils.jl");

βc = βc_triangular()
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
fN1s = Float64[]
fN2s = Float64[]
for N in [2, 3, 4, 5, 6]
    𝕋N = reduce(vcat, fill([M1, M2], N))
    𝕋N_mat = convert_to_mat(𝕋N)
    Λ, _ = eigen(𝕋N_mat)
    fN = - log(real(Λ.data[end])) / (2*N*β)
    fN2 = f_density_triangular(N, β)
    push!(fN1s, fN)
    push!(fN2s, fN2)
    println("$(N), $(fN), $(fN2), $(fN - fN2)")
end

𝕋N = reduce(vcat, fill([M1, M2], 4))
𝕋N_mat = convert_to_mat(𝕋N)
Λ, _ = eigen(𝕋N_mat)
Λs =  Λ.data |> diag
Λm = norm(last(Λs))

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (400, 400))
ax1 = Axis(fig[1, 1])
xlims!(ax1, (-Λm, Λm))
ylims!(ax1, (-Λm, Λm))
scatt_N = scatter!(ax1, real.(Λs), imag.(Λs), marker=:circle, markersize=10)
@show fig

#check. if f_density_triangular has errors 
Ns = 40:8:200
f_imaging(x) = -0.80 * x^(-2) + 1 + 1e-7 *rand() * x

@. f_fit(x, p) = p[1]*x^p[2] + p[3]
p0 = [1.0, -1.0, 1.1]
fit_outcome_aaa = curve_fit(f_fit, Ns, f_imaging.(Ns), p0)
pf_aaa = coef(fit_outcome_aaa)

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 400))
ax1 = Axis(fig[1, 1], xlabel=L"1/\chi \text{ or } 10 / N", ylabel=L"|f-f_{\mathrm{exact}}|", yscale=log10, xscale=log10)
scatt_N = scatter!(ax1, 10 ./ Ns, abs.(f_imaging.(Ns) .- pf_aaa[3]) .+ 1e-14, marker=:circle, markersize=10)
@show fig

# hermicity
𝕋N = reduce(vcat, fill([M1, M2], 4));
𝕋N_mat = convert_to_mat(𝕋N);
### not hermitian
@show norm(𝕋N_mat - 𝕋N_mat') 
### not normal
@show norm(𝕋N_mat * 𝕋N_mat' - 𝕋N_mat' * 𝕋N_mat)

# VUMPS
χs = 4:4:40
fχs = Float64[]
for χ in χs
    A = TensorMap(rand, ComplexF64, ℂ^χ*ℂ^2, ℂ^χ);
    B = TensorMap(rand, ComplexF64, ℂ^χ*ℂ^2, ℂ^χ);
    ψ0 = InfiniteMPS([A, B]);

    ψ, envs, _ = leading_boundary(ψ0, 𝕋, VUMPS(tol_galerkin=1e-10, maxiter=10000)); 
    push!(fχs, real(log(dot(ψ, 𝕋, ψ))) / (-β) / 2)
end

# exact solution. extrapolated from finite size results
Ns = 40:8:200
fNs = f_density_triangular.(Ns, β)
@. f_fit(x, p) = p[1]*x^p[2] + p[3]
p0 = [1.0, -1.0, last(fNs) + 1e-6]
fit_outcome = curve_fit(f_fit, Ns, fNs, p0)
pf = coef(fit_outcome)
f0 = pf[3]

fit_outcome2 = curve_fit(f_fit, χs[3:end], fχs[3:end], p0)
f02 = coef(fit_outcome2)[3]

@show f02 - f0

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 400))
ax1 = Axis(fig[1, 1], xlabel=L"1/\chi \text{ or } 10 / N", ylabel=L"|f-f_{\mathrm{exact}}|", yscale=log10, xscale=log10)
scatt_χ = scatter!(ax1, 2 ./ χs, abs.(fχs .- f0), marker=:circle, markersize=10)
scatt_N = scatter!(ax1, 10 ./ Ns, abs.(fNs .- f0), marker=:circle, markersize=10)
@show fig

save("VUMPS/triangular_Ising_results.pdf", fig)