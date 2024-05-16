"""
    convert_to_mat(𝕋::DenseMPO)

    convert a (pressumed to be finite) MPO into a big matrix. 
    Be careful about the MPO's length!
"""
function convert_to_mat(𝕋::DenseMPO)

    L = length(𝕋)

    ncon_contraction_order = [[ix, -2*ix+1, -2*ix, ix+1] for ix in 1:L] 
    ncon_contraction_order[end][end] = 1
    permutation_orders = Tuple(2 .* (1:L) .- 1), Tuple(2 .* (1:L))

    𝕋mat = permute(ncon(𝕋.opp, ncon_contraction_order), permutation_orders...)   

    return 𝕋mat
end
function convert_to_mat(Ts::Vector{<:MPSKit.MPOTensor})

    L = length(Ts)

    ncon_contraction_order = [[ix, -2*ix+1, -2*ix, ix+1] for ix in 1:L] 
    ncon_contraction_order[end][end] = 1
    permutation_orders = Tuple(2 .* (1:L) .- 1), Tuple(2 .* (1:L))

    𝕋mat = permute(ncon(Ts, ncon_contraction_order), permutation_orders...)   

    return 𝕋mat
end
"""
    mpotensor_dag(T::MPOTensor)

    Generate the hermitian conjugate of a MPO.
"""
function mpotensor_dag(T::MPSKit.MPOTensor)
    T_data = reshape(T.data, (dims(codomain(T))..., dims(domain(T))...))
    Tdag_data = permutedims(conj.(T_data), (1, 3, 2, 4))
    
    return TensorMap(Tdag_data, space(T))
end

function ChainRulesCore.rrule(::typeof(TensorKit.exp), K::TensorMap)
    W, UR = eig(K)
    UL = inv(UR)
    Ws = []

    if W.data isa Matrix 
        Ws = diag(W.data)
    elseif W.data isa TensorKit.SortedVectorDict
        Ws = vcat([diag(values) for (_, values) in W.data]...)
    end

    expK = UR * exp(W) * UL

    function exp_pushback(f̄wd)
        ēxpK = f̄wd 
       
        K̄ = zero(K)

        if ēxpK != ZeroTangent()
            if W.data isa TensorKit.SortedVectorDict
                # TODO. symmetric tensor
                error("symmetric tensor. not implemented")
            end
            function coeff(a::Number, b::Number) 
                if a ≈ b
                    return exp(a)
                else 
                    return (exp(a) - exp(b)) / (a - b)
                end
            end
            M = UR' * ēxpK * UL'
            M1 = similar(M)
            copyto!(M1.data, M.data .* coeff.(Ws', conj.(Ws)))
            K̄ += UL' * M1 * UR'# - tr(ēxpK * expK') * expK'
        end
        return NoTangent(), K̄
    end 
    return expK, exp_pushback
end