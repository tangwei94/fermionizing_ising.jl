function βc_triangular()
    return asinh(1/sqrt(3)) / 2
end

"""
    f_density_triangular(N::Int, β::Real)

- Compute the free energy density of FM classical Ising model on the triangular lattice.
- The lattice has a size of `N` and periodic boundary conditions in one direction, while the other direction is infinite in length.
- `β` is the inverse temperature.
"""
function f_density_triangular(N::Int, β::Real; αs=(5:0.25:6))
    β1 = kramers(β) # β^\ast

    f_honeycomb = f_density_honeycomb(2*N, β1; αs)
    #f_triangular1 = (3*log(cosh(β)) + log(2) + (-2*β1)*f_honeycomb - 3*β1) / (-β)
    f_triangular = ((-2*β1)*f_honeycomb - log(2) / 2 + 3 * log(sinh(2*β)) / 2 ) / (-β)
    return f_triangular
end

"""
    f_density_triangular_analytic_sol(N::Int, β::Real)

- Analytic solution for the triangular lattice Ising model on a infinite cylinder of width `N` at inverse temperature `β`
- T. M. Liaw, M. C. Huang, S. C. Lin, and M. C. Wu Phys. Rev. B 60, 12994 (1999)
"""
function f_density_triangular_analytic_sol(N::Int, β::Real)
    t = tanh(β)

    ps = (1:N) .+ (1/2)

    A0 = 1 + 3*t^2 + 3 * t^4 + t^6 + 8 * t^3
    A1 = 2*t*(1-2*t^2+t^4)

    RT = 2/(1-t^2)^(3/2)
    C1 = log(RT)

    function p_integral(p)
        g(ϕ) = (1/2/pi) * log(A0 - A1 * cos(2*pi*p/N) - A1 * cos(ϕ) - A1 * cos(2*pi*p/N - ϕ))
        return quadgk(g, 0, 2*pi)[1]
    end

    fexact = (-C1 - 1/2/N * sum(p_integral.(ps))) / β
    return fexact
end