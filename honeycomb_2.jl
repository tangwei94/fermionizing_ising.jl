using TensorKit, TensorOperations

function kron4(A, B, C, D)
    @tensor T[-1 -2 -3 -4; -5 -6 -7 -8] := A[-1; -5] * B[-2; -6] * C[-3; -7] * D[-4; -8] 
    return T
end

βs = 0.1:0.01:1.0
β1s = kramers.(βs)

f1s, f2s = Float64[], Float64[]

for β in βs
    Tbond_dat = zeros(ComplexF64, 2, 2)
    Tbond_dat[1, 1] = Tbond_dat[2, 2] = exp(β)
    Tbond_dat[1, 2] = Tbond_dat[2, 1] = exp(-β)
    Tbond = TensorMap(Tbond_dat, ℂ^2, ℂ^2)

    Tempty_dat = zeros(ComplexF64, 2, 2)
    Tempty_dat[1, 1] = Tempty_dat[2, 2] = 1
    Tempty_dat[1, 2] = Tempty_dat[2, 1] = 1
    Tempty = TensorMap(Tempty_dat, ℂ^2, ℂ^2)

    σz_dat = zeros(ComplexF64, 2, 2)
    σz_dat[1, 1] = 1
    σz_dat[2, 2] = -1
    σz = TensorMap(σz_dat, ℂ^2, ℂ^2)

    Id = id(ℂ^2)

    T2 = kron4(σz, σz, Id, Id) + kron4(Id, σz, σz, Id) + kron4(Id, Id, σz, σz) + kron4(σz, Id, Id, σz) #- 4* kron4(Id, Id, Id, Id)
    T2 = exp(β * T2)

    T1 = kron4(Tbond, Tempty, Tbond, Tempty)
    T3 = kron4(Tempty, Tbond, Tempty, Tbond)

    Es, _ = eigen(T2 * T1 * T2 * T3)
    Es = Es.data |> diag
    E = last(Es)

    push!(f1s, - real(log(E)) / 4 / 2 / β) #- 8 / 4 /2)
    push!(f2s, f_density_honeycomb(4, β))
end

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 400))
ax1 = Axis(fig[1, 1])
sc1 = scatter!(ax1, βs, f1s, marker=:dot, markersize=10)
sc2 = lines!(ax1, βs, f2s)
@show fig