"""
    skew_canonical(A::Matrix{<:Real})
    
    Given a real skew-symmetric matrix `A`, transform it to the canonical form.
    Returns  
    - `Λ` is the canonical form.
    - `U` is the orthogonal transformation: `A = U * Λ * U'`. 
    - `Σ` is contain all the eigenvalues.
"""
function skew_canonical(A::Matrix{<:Real})
    N = size(A)[1] ÷ 2

    Q, H = hessenberg(A)
    P = zeros(Float64, 2*N, 2*N)
    for ix in 1:N
        P[2*ix-1, ix] = 1
        P[2*ix, N+ix] = 1
    end
    J = -(P' * H * P)[N+1:2*N, 1:N]
    V, Σ, W = svd(J)
    WV = zeros(Float64, 2*N, 2*N)
    WV[1:N, 1:N] = W 
    WV[N+1:2*N, N+1:2*N] = V
    U = Q * P * WV * P'
    Λ = U' * A * U

    return Λ, U, Σ
end

"""
    covariance_matrix(M::Matrix{<:Real}, β::Real)

    Given a quadratic fermion Hermitian parametrized by the skew-symmetric matrix `M`, compute the covariance matrix at temperature `β`. 
    Returns the real part of the covariance matrix. 

"""
function covariance_matrix(M::Matrix{<:Real}, β::Real)
    _, U, Σ = skew_canonical(M)
    N = size(M)[1] ÷ 2
    ReC̃ = zeros(Float64, size(M))
    for ix in 1:N
        ReC̃[2*ix-1, 2*ix] = tanh(β*Σ[ix]/2)
        ReC̃[2*ix, 2*ix-1] = -tanh(β*Σ[ix]/2)
    end
    ReC = U * ReC̃ * U' 
    return ReC
end

"""
    product_rule(ReC1::Matrix{<:Number}, ReC2::Matrix{<:Number})

    The product rule for `exp(-H1) * exp(-H2)`. Here `H1` and `H2` are quadratic Hamiltonians, whose covariance matrcies are `im * ReC1` and `im * ReC2`. 

    Returns `-im * C`. Here `C` is the new covariance matrix. 
"""
function product_rule(ReC1::Matrix{<:Number}, ReC2::Matrix{<:Number})
    Id = Matrix{Float64}(I, size(ReC1))
    C3 =Id - (Id - im * ReC2) * inv(Id - ReC1 * ReC2) * (Id - im * ReC1)
    return -im * C3
end
"""
    ∘(ReC1::Matrix{<:Number}, ReC2::Matrix{<:Number})

    product_rule(ReC1::Matrix{<:Number}, ReC2::Matrix{<:Number})
"""
function Base.:∘(ReC1::Matrix{<:Number}, ReC2::Matrix{<:Number})
    return product_rule(ReC1, ReC2)
end