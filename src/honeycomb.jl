function kramers(β::Real)
    return -0.5 * log(tanh(β))
end

function βc_honeycomb()
    return asinh(sqrt(3)) / 2 
end
"""
    f_density_honeycomb_α(N::Int, β::Real, α::Real=5)

- Compute the free energy density of FM classical Ising model on the honeycomb lattice.
- The lattice has a size of `N` and periodic boundary conditions in one direction, while the other direction is infinite in length.
- Requires that the system size `N` to be a multiple of 4.
- `β` is the inverse temperature. 
- When fermionizing the transfer matrix, we approximate `[1 1; 1 1]` as `exp(α σ^x)/cosh(α)`.
- When fermionizing the transfer matrix, we also need to choose the fermion `parity` sector (NS/R). As we are only interested in the dominant eigenvector, we set the parity to be 1. 
"""
function f_density_honeycomb_α(N::Int, β::Real, α::Real=5)

    @assert (N % 4) == 0

    M_even =Float64[0  1 0  0 0  0 0 1;
                    -1 0 0  0 0  0 -1 0;
                    0  0 0  1 0  1 0  0;
                    0  0 -1 0 -1 0 0  0;
                    0  0 0  1 0  1 0  0;
                    0  0 -1 0 -1 0 0  0;
                    0  1 0  0 0  0 0  1;
                    -1 0 0  0 0  0 -1 0]
                            
    M_odd = Float64[0  1  0  0  0  0  0  -1;
                    -1 0  0  0  0  0  1  0;
                    0  0  0  1  0  -1 0  0;
                    0  0  -1 0  1  0  0  0;
                    0  0  0  -1 0  1  0  0;
                    0  0  1  0  -1 0  0  0;
                    0  -1 0  0  0  0  0  1;
                    1  0  0  0  0  0  -1 0]

    fMx(k) = Float64[0   -2*cos(k) 2*sin(k) 0   0   0  0   0;
                  2*cos(k)  0   0  -2*sin(k) 0   0  0   0;
                  -2*sin(k) 0   0  -2*cos(k) 0   0  0   0;
                  0   2*sin(k)  2*cos(k) 0   0   0  0   0;
                  0   0   0  0   0   2*cos(k) 2*sin(k)  0;
                  0   0   0  0   -2*cos(k) 0  0   -2*sin(k);
                  0   0   0  0   -2*sin(k) 0  0   2*cos(k);
                  0   0   0  0   0   2*sin(k) -2*cos(k) 0]

    β1 = kramers(β) # β^\ast
    M1 = β1 * M_odd + α * M_even
    M3 = β1 * M_even + α * M_odd 

    sumϵs = 0
    for kx in 1:N÷4
        k = (2*kx-1) * pi / N
        Mx = β * fMx(k)

        RT = exp(-im*M3/2) * exp(-im*Mx) * exp(-im*M1) * exp(-im*Mx) * exp(-im*M3/2) #/ cosh(α)^2 

        Λ, _ = eigen(Hermitian(RT))
        sumϵs += sum(log.(abs.(Λ[5:end])))
    end

    f = -sumϵs / (4*N*β) - log(2*sinh(2*β)) / (4*β) + log(cosh(α)) / (2*β)
    return f
end
"""
    f_density_honeycomb_α_spatial(N::Int, β::Real, α::Real=5; parity::Real=1)

- Compute the free energy density of FM classical Ising model on the honeycomb lattice.
- Solved in the spatical basis, hence is slower.
- The lattice has a size of `N` and periodic boundary conditions in one direction, while the other direction is infinite in length.
- `β` is the inverse temperature. 
- When fermionizing the transfer matrix, we approximate `[1 1; 1 1]` as `exp(α σ^x)/cosh(α)`.
- When fermionizing the transfer matrix, we also need to choose the fermion `parity` sector (NS/R). As we are only interested in the dominant eigenvector, we set the parity to be 1. 
"""
function f_density_honeycomb_α_spatial(N::Int, β::Real, α::Real=5)

    β1 = kramers(β) # β^\ast

    M1 = zeros(ComplexF64, 2*N, 2*N)
    M2 = zeros(ComplexF64, 2*N, 2*N)
    M3 = zeros(ComplexF64, 2*N, 2*N)

    for ix in 1:2:N
        M1[2*ix-1, 2*ix] = 2*β1
        M1[2*ix, 2*ix-1] = -2*β1
        M3[2*ix-1, 2*ix] = 2*α
        M3[2*ix, 2*ix-1] = -2*α
    end

    for ix in 2:2:N
        M3[2*ix-1, 2*ix] = 2*β1
        M3[2*ix, 2*ix-1] = -2*β1
        M1[2*ix-1, 2*ix] = 2*α
        M1[2*ix, 2*ix-1] = -2*α
    end

    for ix in 1:N-1
        M2[2*ix, 2*ix+1] = 2*β
        M2[2*ix+1, 2*ix] = -2*β
    end
    M2[2*N, 1] = -2*β
    M2[1, 2*N] = 2*β

    Id = Matrix{ComplexF64}(I, 2*N, 2*N)
    Δ = 2*α

    RT = exp(-im*M3/2 - Δ*Id/2) * exp(-im * M2) * exp(-im * M1 - Δ*Id) * exp(-im * M2) * exp(-im * M3 / 2 - Δ*Id/2);

    Λ, _ = eigen(Hermitian(RT));

    ϵs = log.(Λ[N+1:end]) .+ 2*Δ
    #ϵ1s = -log.(Λ[1:N]) # the positive entries are more stable and more accurate

    f = -sum(ϵs) / (4*N*β) - log(2*sinh(2*β)) / (4*β) + log(cosh(α)) / (2*β)
    return f
end

"""
    f_density_honeycomb(N::Int, β::Real)

- Compute the free energy density of FM classical Ising model on the honeycomb lattice.
- The lattice has a size of `N` and periodic boundary conditions in one direction, while the other direction is infinite in length.
- `β` is the inverse temperature.
- Compared to `f_density_honeycomb_α``, this function computes the free energy using different values of `α`, and performs a linear extrapolation with respect to `1-tanh(α)` to obtain the final result. 
"""
function f_density_honeycomb(N::Int, β::Real)
    αs = 5:0.25:6#4:0.5:5
    fs = f_density_honeycomb_α.(N, β, αs)
    fit_linf = Polynomials.fit(1 .- tanh.(αs), fs, 1)
    return fit_linf(0)
end