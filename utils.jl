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