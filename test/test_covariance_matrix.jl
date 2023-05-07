@testset "compare momentum basis and spatial basis results" for β in 0.1:0.1:1.0

    β1 = -0.5 * log(tanh(β)) 

    N = 20
    M1 = zeros(Float64, 2*N, 2*N)
    M2 = zeros(Float64, 2*N, 2*N)

    for ix in 1:N
        M1[2*ix-1, 2*ix] = 2*β1
        M1[2*ix, 2*ix-1] = -2*β1
    end

    for ix in 1:N-1
        M2[2*ix, 2*ix+1] = 2*β
        M2[2*ix+1, 2*ix] = -2*β
    end

    RT = exp(-im * M1 / 2) * exp(-im * M2) * exp(-im * M1 / 2)
    MT = real.(-im*log(RT))

    ReC1 = covariance_matrix(-M1/2, 1)
    ReC2 = covariance_matrix(-M2, 1)

    ReCtot = real.(product_rule(product_rule(ReC1, ReC2), ReC1))
    ReCtot1 = covariance_matrix(MT, 1)

    @test norm(ReCtot - ReCtot1) < 1e-10
end