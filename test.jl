using MPSKit,MPSKitModels,TensorKit, KrylovKit

#function TensorKit.dot(a::InfiniteMPS, mpo::DenseMPO, b::InfiniteMPS;krylovdim = 30)
#    _firstspace = MPSKit._firstspace
#    function fill_data!(a::TensorMap,dfun)
#    for (k,v) in blocks(a)
#        map!(x->dfun(typeof(x)),v,v);
#    end
#
#    a
#    end
#    randomize!(a::TensorMap) = fill_data!(a,randn)
#
#    init = similar(a.AL[1], _firstspace(b.AL[1])*_firstspace(mpo.opp[1]) ← _firstspace(a.AL[1]))
#    randomize!(init)
#
#    (vals,vecs,convhist) = eigsolve(TransferMatrix(b.AL, mpo.opp, a.AL),init,1,:LM,Arnoldi(krylovdim=krylovdim))
#    convhist.converged == 0 && @info "dot mps not converged"
#    return vals[1]
#end

mpo = nonsym_ising_mpo();

state = InfiniteMPS([ℂ^2],[ℂ^10]);
(state,envs,_) = leading_boundary(state,mpo,VUMPS(tol_galerkin=1e-10));

@show dot(state, mpo, state)

