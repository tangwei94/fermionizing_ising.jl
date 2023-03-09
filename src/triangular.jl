function βc_triangular()
    return asinh(1/sqrt(3)) / 2
end

"""
    f_density_triangular(N::Int, β::Real)

- Compute the free energy density of FM classical Ising model on the triangular lattice.
- The lattice has a size of `N` and periodic boundary conditions in one direction, while the other direction is infinite in length.
- `β` is the inverse temperature.
"""
function f_density_triangular(N::Int, β::Real)
    β1 = kramers(β) # β^\ast

    f_honeycomb = f_density_honeycomb(2*N, β1)
    #f_triangular1 = (3*log(cosh(β)) + log(2) + (-2*β1)*f_honeycomb - 3*β1) / (-β)
    f_triangular = ((-2*β1)*f_honeycomb - log(2) / 2 + 3 * log(sinh(2*β)) / 2 ) / (-β)
    return f_triangular
end