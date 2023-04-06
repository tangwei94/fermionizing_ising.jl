@testset "test triangular lattice free energy, N=3" for β in 0.05:0.05:0.4

    T1_dat = zeros(2, 2, 2, 2, 2, 2, 2, 2)

    function ⨇(x, y, β)
        if x == y 
            return exp(β)
        else 
            return exp(-β)
        end
    end

    for x1 in 1:2, x2 in 1:2, x3 in 1:2, x4 in 1:2
        for y1 in 1:2, y2 in 1:2, y3 in 1:2, y4 in 1:2 
            T1_dat[x1, x2, x3, x4, y1, y2, y3, y4] = ⨇(x1, y1, β) * ⨇(x1, y4, β) * ⨇(x2, y2, β) * ⨇(x2, y1, β) * ⨇(x3, y3, β) * ⨇(x3, y2, β) * ⨇(x4, y4, β) * ⨇(x4, y3, β) 
            T1_dat[x1, x2, x3, x4, y1, y2, y3, y4] *= ⨇(y1, y2, β) * ⨇(y2, y3, β) * ⨇(y3, y4, β) * ⨇(y4, y1, β)
        end
    end

    E = eigvals(reshape(T1_dat, 16, 16)) |> last |> real
    f1 = -log(E) / β / 4
    f2 = f_density_triangular(4, β)
    @test isapprox(f1, f2, rtol=1e-6)
end

@testset "test analytic sol" for N in 8:8:64

    βc = βc_triangular()
    for β in [0.05:0.05:0.4 ; βc]
        f1 = f_density_triangular(N, β)
        f2 = f_density_triangular_analytic_sol(N, β)
        @show N, β, f1 - f2
        @test isapprox(f1, f2, rtol=1e-6)
    end
end