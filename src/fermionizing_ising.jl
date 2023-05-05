__precompile__()

module fermionizing_ising

    using LinearAlgebra
    using Polynomials
    using QuadGK

    export skew_canonical, covariance_matrix, product_rule, ∘
    export kramers, βc_honeycomb, f_density_honeycomb_α, f_density_honeycomb
    export βc_triangular, f_density_triangular, f_density_triangular_analytic_sol

    include("covariance_matrix.jl")
    include("honeycomb.jl");
    include("triangular.jl");

end
