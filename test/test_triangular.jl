@testset "test triangular lattice free energy, N=3" for β in 0.05:0.05:0.4

    T1_dat = zeros(2, 2, 2, 2, 2, 2)

    function ⨇(x, y, β)
        if x == y 
            return exp(β)
        else 
            return exp(-β)
        end
    end

    for x1 in 1:2 
        for x2 in 1:2
            for x3 in 1:2
                for y1 in 1:2 
                    for y2 in 1:2
                        for y3 in 1:2
                            T1_dat[x1, x2, x3, y1, y2, y3] = ⨇(x1, y1, β) * ⨇(x1, y3, β) * ⨇(x2, y2, β) * ⨇(x2, y1, β) * ⨇(x3, y3, β) * ⨇(x3, y2, β) 
                            T1_dat[x1, x2, x3, y1, y2, y3] *= ⨇(y1, y2, β) * ⨇(y2, y3, β) * ⨇(y3, y1, β)
                        end
                    end
                end
            end
        end
    end

    E = eigvals(reshape(T1_dat, 8, 8)) |> last |> real
    f1 = -log(E) / β / 3
    f2 = f_density_triangular(3, β)
    @test isapprox(f1, f2, rtol=1e-6)
end