@testset "4 x infty lattice" for β in 0.1:0.1:1.0

    function kron4(A, B, C, D)
        @tensor T[-1 -2 -3 -4; -5 -6 -7 -8] := A[-1; -5] * B[-2; -6] * C[-3; -7] * D[-4; -8] 
        return T
    end

    β=0.6

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
    E = Es.data |> diag |> last |> real

    f1 = - log(E) / 4 / 2 / β
    f2 = f_density_honeycomb(4, β)

    @test isapprox(f1, f2, rtol=1e-6)

end